"""Demo sobre Home Credit: mismo motor, espacio grande.

66 variables numericas (fuera las con >50% nulos), cada una su propia
familia, k=6: C(66,6) ~ 90 millones de conjuntos. Split local en
data/homecredit/split (ver scripts/partir_homecredit.py), test cerrado
hasta el reporte.

Uso:
    uv run python scripts/demo_homecredit.py --trials 100 --semillas 0,1
    uv run python scripts/demo_homecredit.py --hostil --trials 100
"""

import argparse
import contextlib
import io
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import demo_simple as D
from sef_exp import fase3b
from sef_exp.evaluador import hacer_folds, precomputar
from sef_exp.fase2 import ROOT

SPLIT = ROOT / "data/homecredit/split"


def cargar_split():
    vars_ = (SPLIT / "variables.txt").read_text().split()
    xtr = pl.read_csv(SPLIT / "X_train.csv")
    xte = pl.read_csv(SPLIT / "X_test.csv")
    ytr = pl.read_csv(SPLIT / "y_train.csv")["TARGET"].to_numpy()
    yte = pl.read_csv(SPLIT / "y_test.csv")["TARGET"].to_numpy()
    return {"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte, "vars": vars_}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--semillas", default="0,1,2")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--muestra", type=int, default=None)
    parser.add_argument("--hostil", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    D.configurar_log(logging.DEBUG if args.verbose else logging.INFO)
    log = D.log

    D.WORK = ROOT / ("work/demo_hc_hostil_runs" if args.hostil
                     else "work/demo_hc_runs")
    destino = ROOT / ("outputs/demo_homecredit_hostil.html" if args.hostil
                      else "outputs/demo_homecredit.html")
    if args.muestra is None:
        args.muestra = 8000 if args.hostil else 20000
    modo = "MLP(32)x58" if args.hostil else f"logit(C={args.c})"
    log.info("demo_homecredit modo=%s trials=%s semillas=%s muestra=%s",
             modo, args.trials, args.semillas, args.muestra)

    D.WORK.mkdir(parents=True, exist_ok=True)
    for viejo in D.WORK.glob("*.csv"):
        viejo.unlink()
    datos = D.submuestra(cargar_split(), args.muestra)
    log.info("Train de busqueda: %d filas, %d variables",
             len(datos["ytr"]), len(datos["vars"]))
    folds = hacer_folds(datos["ytr"])
    familias = [(v, (v,)) for v in datos["vars"]]
    if args.hostil:
        evaluar = lambda S: D.ajuste_mlp_cv(S, datos, folds)  # noqa
        puntuar = lambda S: D.test_score_mlp(S, datos)  # noqa
        iv = D.ranking_univariado(datos, folds)
    else:
        pre = precomputar(datos["xtr"], datos["ytr"], folds, datos["vars"])
        iv = {v: statistics.mean(pre[(r, v)]["iv"]
                                 for r in range(len(folds)))
              for v in datos["vars"]}
        evaluar = lambda S: D.ajuste_cv(S, datos, folds, pre, args.c)  # noqa
        puntuar = lambda S: D.test_score(S, datos, args.c)  # noqa
    cfg = fase3b.Config(nombre="homecredit", familias=familias, k=D.K,
                        presupuestos=(args.trials,))
    t0 = time.perf_counter()
    silencio = contextlib.nullcontext() if args.verbose \
        else contextlib.redirect_stdout(io.StringIO())

    with silencio:
        for metodo in D.METODOS_MOTOR:
            for seed in [int(x) for x in args.semillas.split(",")]:
                t = time.perf_counter()
                fits = D.correr_motor(metodo, seed, args.trials, cfg,
                                      lambda S, c, g: evaluar(S), iv)
                df = pl.read_csv(
                    D.WORK / f"{metodo if metodo != 'sef' else 'sef_completo'}_{seed:02d}.csv",
                    columns=["trial", "mejor_J", "mejor_S"])
                df = df.with_columns(
                    pl.col("mejor_J").cast(pl.Float64, strict=False))
                df = df.filter(pl.col("mejor_J").is_not_null())
                if len(df):
                    top = df.sort("trial").tail(1)
                    log.info("%s semilla=%s finished with value=%.4f "
                             "(trial %d/%d) en %.1fs, %d ajustes",
                             metodo, seed, top["mejor_J"][0],
                             top["trial"][0], args.trials,
                             time.perf_counter() - t, fits)
                else:
                    log.info("%s semilla=%s sin conjunto factible en %.1fs, "
                             "%d ajustes", metodo, seed,
                             time.perf_counter() - t, fits)

        finales = {}
        for nombre, fn in (("forward", lambda: D.correr_forward(cfg, evaluar)),
                           ("hibrido", lambda: D.correr_hibrido(cfg, evaluar))):
            t = time.perf_counter()
            _, S = fn()
            J = evaluar(S) if S is not None and len(S) == D.K else None
            finales[nombre] = (J, S)
            log.info("%s finished with value=%s en %.1fs, S=%s", nombre,
                     f"{J:.4f}" if J is not None else "-",
                     time.perf_counter() - t,
                     "|".join(S) if S else "-")

    log.info("Abriendo test una vez para el reporte")
    test_rows = []
    for metodo, dfs in D.leer_trazas().items():
        mejor_cv, mejor_s, ensayos = None, None, 0
        for df in dfs:
            ensayos += len(df)
            fila = df.filter(pl.col("mejor_J").is_not_null()).sort("trial").tail(1)
            if len(fila) and (mejor_cv is None or fila["mejor_J"][0] > mejor_cv):
                mejor_cv = fila["mejor_J"][0]
                mejor_s = tuple(fila["mejor_S"][0].split("|"))
        if metodo in finales:
            Jf_, Sf_ = finales[metodo]
            if Jf_ is not None and Sf_ is not None and len(Sf_) == D.K:
                mejor_cv, mejor_s = Jf_, Sf_
        if mejor_s is None or len(mejor_s) != D.K:
            log.info("%s: sin conjunto completo, se omite test", metodo)
            continue
        te = puntuar(mejor_s)
        test_rows.append((metodo, mejor_cv, te, ensayos, "|".join(mejor_s)))
        log.info("%s best CV=%.4f test=%.4f S=%s",
                 metodo, mejor_cv, te, "|".join(mejor_s))
    subtitulo = (f"modo={modo}, trials={args.trials}, "
                 f"semillas={args.semillas}, "
                 f"train de busqueda={len(datos['ytr'])} filas, "
                 f"test completo={len(datos['yte'])} filas.")
    destino.write_text(D.figura_html(D.leer_trazas(), sorted(
        test_rows, key=lambda r: -r[1]), subtitulo,
        titulo="Demo Home Credit · 66 variables sin cache"))
    log.info("Fin en %.1fs -> %s", time.perf_counter() - t0, destino)


if __name__ == "__main__":
    main()
