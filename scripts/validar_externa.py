"""Validacion externa unica: abre el test una vez por track.

Finalistas y regla fijados en docs/validacion-externa.md antes de
correrlo. Scorecards: binning+WOE en train completo, logistica, AUC
test. Series: ARIMA re-ajustado en train, RMSE/MAE a 24 meses mas
naive de contexto. Escribe outputs/validacion_externa.json.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sef_exp.evaluador import BIN_CFG, aplicar_nulos, cargar_datos, columna
from sef_exp.evaluador_series import evaluar_serie
from sef_exp.series import CFG as SERIES_CFG

ROOT = Path(__file__).resolve().parents[1]

FINALISTAS = {
    "fase3b": ("data", ("AT23", "CVPRTRV10", "CVSC150", "CVSSG990S",
                         "IN20", "PR09"), 0.713542318119154),
    "fase4": ("data/taiwan", ("EDUCATION", "LIMIT_BAL", "PAY_0", "PAY_4",
                              "PAY_AMT2", "PAY_AMT4"), 0.7615923356582383),
}


def validar_scorecard(nombre, data_dir, variables, j_sel):
    d = cargar_datos(str(ROOT / data_dir))
    xtr, ytr, xte, yte = d["xtr"], d["ytr"], d["xte"], d["yte"]
    Wtr, Wte = [], []
    for v in variables:
        opt = OptimalBinning(**BIN_CFG)
        opt.fit(columna(xtr, v), ytr)
        Wtr.append(opt.transform(columna(xtr, v), metric="woe").ravel())
        Wte.append(opt.transform(columna(xte, v), metric="woe").ravel())
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000)
    clf.fit(np.column_stack(Wtr), ytr)
    auc_tr = float(roc_auc_score(ytr, clf.predict_proba(np.column_stack(Wtr))[:, 1]))
    auc_te = float(roc_auc_score(yte, clf.predict_proba(np.column_stack(Wte))[:, 1]))
    return {"J_seleccion": j_sel, "AUC_train_completo": auc_tr,
            "AUC_test": auc_te, "caida": auc_tr - auc_te,
            "n_test": len(yte)}


def validar_series():
    base = ROOT / "data/series"
    x = pl.read_csv(base / "X_train.csv").drop("")
    y = pl.read_csv(base / "y_train.csv")["target"].cast(pl.Float64).to_numpy().ravel()
    xt = pl.read_csv(base / "X_test.csv").drop("")
    yt = pl.read_csv(base / "y_test.csv")["target"].cast(pl.Float64).to_numpy().ravel()
    medias = {c: x[c].mean() for c in x.columns}
    desvs = {c: x[c].std() or 1.0 for c in x.columns}
    xs = x.with_columns(((pl.col(c) - medias[c]) / desvs[c]).alias(c) for c in x.columns)
    xts = xt.with_columns(((pl.col(c) - medias[c]) / desvs[c]).alias(c) for c in x.columns)
    from statsmodels.tsa.arima.model import ARIMA
    S = ("COBRE", "DESEMPLEO", "p3q0")
    r = evaluar_serie(S, y, xs, d=0)
    assert r["estado"] == "ok", r
    res = ARIMA(np.asarray(y, dtype=float),
                exog=np.asarray(xs[["COBRE", "DESEMPLEO"]], dtype=float),
                order=(3, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False).fit()
    pron = np.asarray(res.get_forecast(
        steps=len(yt),
        exog=np.asarray(xts[["COBRE", "DESEMPLEO"]], dtype=float),
    ).predicted_mean).ravel()
    naive = np.full(len(yt), y[-1])
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    return {"J_seleccion": -549.0535808233672, "S": "|".join(S),
            "RMSE_test": rmse(pron, yt), "MAE_test": mae(pron, yt),
            "RMSE_naive": rmse(naive, yt), "MAE_naive": mae(naive, yt),
            "n_test": len(yt)}


def main():
    out = {}
    for nombre, (data_dir, variables, j_sel) in FINALISTAS.items():
        out[nombre] = validar_scorecard(nombre, data_dir, variables, j_sel)
        print(nombre, out[nombre], flush=True)
    out["series"] = validar_series()
    print("series", out["series"], flush=True)
    destino = ROOT / "outputs/validacion_externa.json"
    destino.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {destino}", flush=True)


if __name__ == "__main__":
    main()
