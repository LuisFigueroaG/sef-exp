"""Evaluador HGB para el track hostil: AUC medio de 5 folds.

Sin WOE ni filtros de scorecard. J = media de AUC con el mismo
HGB fijo para todos los conjuntos. Determinista con semilla y
un hilo.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PARAMS = {
    "max_iter": 50, "max_leaf_nodes": 31, "learning_rate": 0.1,
    "min_samples_leaf": 200, "l2_regularization": 1.0,
    "random_state": 7,
}


def evaluar_hgb(S, Xtr, ytr, folds):
    cols = list(S)
    try:
        X = np.asarray(Xtr.select(cols).to_numpy(), dtype=float)
    except Exception:
        return {"estado": "rechazado", "causa": "columnas", "J": None, "aucs": []}
    aucs = []
    for tr, va in folds:
        try:
            clf = HistGradientBoostingClassifier(**PARAMS)
            clf.fit(X[tr], ytr[tr])
            p = clf.predict_proba(X[va])[:, 1]
        except Exception as e:
            return {"estado": "rechazado",
                    "causa": "no_ajusta_%s" % type(e).__name__,
                    "J": None, "aucs": aucs}
        if not np.all(np.isfinite(p)):
            return {"estado": "rechazado", "causa": "p_no_finito",
                    "J": None, "aucs": aucs}
        aucs.append(float(roc_auc_score(ytr[va], p)))
    return {"estado": "ok", "causa": "", "J": float(np.mean(aucs)), "aucs": aucs}
