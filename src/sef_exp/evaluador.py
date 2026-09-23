"""Evaluador comun SEF x MRAP.

Folds fijos, WOE por fold con OptBinning fijo, logistica MLE.
Sin regularizacion, signos libres. Rechazo solo por filtros
estadisticos, no convergencia o valores no finitos.
"""
import warnings

import numpy as np
import polars as pl
from optbinning import OptimalBinning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED_FOLDS = 42
N_SPLITS = 5
K_VARS = 6
IV_MIN = 0.1
CORR_MAX = 0.7
VIF_MAX = 5.0
BIN_CFG = {
    "dtype": "numerical",
    "solver": "cp",
    "max_n_bins": 5,
    "min_prebin_size": 0.05,
}

NULOS = {
    "CVPRPAYMNT11": [-7.0, -6.0, -2.0],
    "CVPRTRV10": [-7.0, -6.0, -2.0, -1.0],
    "CVPRRVDEXQ4": [-9.0, -7.0, -6.0, -2.0, -1.0],
    "CVPRNONMTTRD": [-7.0, -6.0],
    "CVPRTRD": [-7.0],
    "CVSSG990S": [-4.0, -7.0],
    "CVSSG960S": [-4.0, -7.0],
    "IN20": [-1.0],
    "AM20": [-1.0],
    "AT20": [-1.0],
    "GO158": [-6.0],
    "PR12": [999.0],
}


def cargar_datos(data_dir="data"):
    xtr = pl.read_csv(data_dir + "/X_train.csv").drop("")
    ytr = pl.read_csv(data_dir + "/y_train.csv").select("target")
    xte = pl.read_csv(data_dir + "/X_test.csv").drop("")
    yte = pl.read_csv(data_dir + "/y_test.csv").select("target")
    return {
        "xtr": aplicar_nulos(xtr),
        "ytr": ytr["target"].cast(pl.Int64).to_numpy().ravel(),
        "xte": aplicar_nulos(xte),
        "yte": yte["target"].cast(pl.Int64).to_numpy().ravel(),
        "vars": [c for c in xtr.columns],
    }


def aplicar_nulos(df):
    exprs = []
    for c in df.columns:
        if c in NULOS:
            exprs.append(
                pl.when(pl.col(c).is_in(NULOS[c]))
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
            )
        else:
            exprs.append(pl.col(c))
    return df.with_columns(exprs)


def columna(df, var):
    return df.select(var).to_series().cast(pl.Float64).to_numpy().ravel()


def hacer_folds(y, n=N_SPLITS, seed=SEED_FOLDS):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


def _iv_total(opt):
    bt = opt.binning_table.build()
    ivs = bt["IV"].to_numpy(dtype=float)
    bins = bt["Bin"].astype(str).to_numpy()
    tot = bins == "Totals"
    if tot.any():
        return float(ivs[tot][0])
    return float(np.nansum(ivs))


def precomputar(xtr, ytr, folds, var_list=None, log=None):
    var_list = var_list or [c for c in xtr.columns]
    pre = {}
    for r, (tr, va) in enumerate(folds):
        for j, var in enumerate(var_list):
            x = columna(xtr, var)
            opt = OptimalBinning(**BIN_CFG)
            opt.fit(x[tr], ytr[tr])
            pre[(r, var)] = {"opt": opt, "iv": _iv_total(opt)}
            if log is not None:
                log(r, j, len(var_list), var)
    return pre


def precomputar_woe(xtr, folds, pre, var_list):
    """Cache the fixed fold transformations once for repeated candidate evaluations."""
    return {
        (r, var): pre[(r, var)]["opt"].transform(columna(xtr, var), metric="woe").ravel()
        for r in range(len(folds))
        for var in var_list
    }


def _vif_max(W):
    if W.shape[1] == 1:
        return 1.0
    R = np.corrcoef(W, rowvar=False)
    if R.shape != (W.shape[1], W.shape[1]) or not np.all(np.isfinite(R)):
        return float("inf")
    try:
        inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return float("inf")
    d = np.diag(inv)
    if not np.all(np.isfinite(d)):
        return float("inf")
    return float(np.max(d))


def evaluar(S, xtr, ytr, folds, pre, woe_cache=None, k=K_VARS):
    if len(S) != k:
        return {"estado": "rechazado", "causa": "k_distinto", "J": None, "aucs": []}
    aucs = []
    for r, (tr, va) in enumerate(folds):
        cols_tr, cols_va = [], []
        for v in S:
            iv = pre[(r, v)]["iv"]
            if not np.isfinite(iv) or iv <= IV_MIN:
                return {"estado": "rechazado", "causa": "IV_%s_fold%d" % (v, r),
                        "J": None, "aucs": aucs}
            if woe_cache is None:
                opt = pre[(r, v)]["opt"]
                x = columna(xtr, v)
                cols_tr.append(opt.transform(x[tr], metric="woe").ravel())
                cols_va.append(opt.transform(x[va], metric="woe").ravel())
            else:
                woe = woe_cache[(r, v)]
                cols_tr.append(woe[tr])
                cols_va.append(woe[va])
        Wtr = np.column_stack(cols_tr)
        Wva = np.column_stack(cols_va)
        if not np.all(np.isfinite(Wtr)) or not np.all(np.isfinite(Wva)):
            return {"estado": "rechazado", "causa": "woe_no_finito_fold%d" % r,
                    "J": None, "aucs": aucs}
        if any(np.std(Wtr[:, j]) == 0.0 for j in range(Wtr.shape[1])):
            return {"estado": "rechazado", "causa": "constante_fold%d" % r,
                    "J": None, "aucs": aucs}
        if len(S) == 1:
            cmax = 0.0
        else:
            R = np.corrcoef(Wtr, rowvar=False)
            off = R - np.eye(R.shape[0])
            cmax = float(np.nanmax(np.abs(off)))
        if not np.isfinite(cmax) or cmax >= CORR_MAX:
            return {"estado": "rechazado", "causa": "corr_fold%d_%.3f" % (r, cmax),
                    "J": None, "aucs": aucs}
        vmax = _vif_max(Wtr)
        if not np.isfinite(vmax) or vmax >= VIF_MAX:
            return {"estado": "rechazado", "causa": "vif_fold%d_%.2f" % (r, vmax),
                    "J": None, "aucs": aucs}
        clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                clf.fit(Wtr, ytr[tr])
        except ConvergenceWarning:
            return {"estado": "rechazado", "causa": "no_converge_fold%d" % r,
                    "J": None, "aucs": aucs}
        if not np.all(np.isfinite(clf.coef_)) or not np.isfinite(clf.intercept_).all():
            return {"estado": "rechazado", "causa": "coef_no_finito_fold%d" % r,
                    "J": None, "aucs": aucs}
        p = clf.predict_proba(Wva)[:, 1]
        aucs.append(float(roc_auc_score(ytr[va], p)))
    J = float(np.mean(aucs))
    return {"estado": "ok", "causa": "", "J": J, "aucs": aucs}
