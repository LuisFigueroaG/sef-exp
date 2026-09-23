"""Evaluador ARIMAX para el track de series: J = -AICc en train.

Sin WOE, IV, correlacion ni VIF. Factibilidad: convergencia del MLE
mas raices AR/MA fuera del circulo unidad. Todo determinista.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss

EXOGENAS = ("IPC", "TPM", "DOLAR", "COBRE", "DESEMPLEO")


def decodificar(S):
    p = q = None
    exogenas = []
    for token in S:
        if token.startswith("p") and "q" in token:
            medio = token.index("q")
            p, q = int(token[1:medio]), int(token[medio + 1:])
        else:
            exogenas.append(token)
    return p, q, tuple(sorted(exogenas))


def orden_diferencias(y, p_umbral=0.05, d_max=2):
    serie = np.asarray(y, dtype=float)
    for d in range(d_max + 1):
        try:
            p_valor = kpss(serie, regression="c", nlags="auto",
                         store=False)[1]
        except Exception:
            return min(d + 1, d_max)
        if p_valor > p_umbral or d == d_max:
            return d
        serie = np.diff(serie)
    return d_max


def aicc(res):
    n = int(res.nobs)
    k = len(res.params)
    if n - k - 1 <= 0:
        return float("inf")
    return float(res.aic + 2 * k * (k + 1) / (n - k - 1))


def fuera_circulo(raices):
    return all(abs(r) > 1.0 for r in np.atleast_1d(raices))


def evaluar_serie(S, ytr, Xtr, d):
    p, q, exogenas = decodificar(S)
    if p is None or q is None or len(exogenas) != 2:
        return {"estado": "rechazado", "causa": "estructura", "J": None, "aucs": []}
    if any(e not in EXOGENAS for e in exogenas):
        return {"estado": "rechazado", "causa": "exogena_desconocida",
                "J": None, "aucs": []}
    y = np.asarray(ytr, dtype=float)
    X = np.asarray(Xtr[list(exogenas)], dtype=float)
    try:
        res = ARIMA(y, exog=X, order=(p, d, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit()
    except Exception as e:
        return {"estado": "rechazado",
                "causa": "no_converge_%s" % type(e).__name__,
                "J": None, "aucs": []}
    if not bool(res.mle_retvals.get("converged", True)):
        return {"estado": "rechazado", "causa": "no_converge", "J": None, "aucs": []}
    if p > 0 and not fuera_circulo(res.arroots):
        return {"estado": "rechazado", "causa": "no_estacionario",
                "J": None, "aucs": []}
    if q > 0 and not fuera_circulo(res.maroots):
        return {"estado": "rechazado", "causa": "no_invertible",
                "J": None, "aucs": []}
    valor = aicc(res)
    if not np.isfinite(valor):
        return {"estado": "rechazado", "causa": "aicc_no_finito",
                "J": None, "aucs": []}
    return {"estado": "ok", "causa": "", "J": -valor, "aucs": []}
