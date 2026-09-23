"""Barrido de exogenas en series: n=0..5 con (p,q) libre (k=1+n).

Mismo motor, pool de 7 exogenas sin lags. Solo lee train.
"""

import argparse

from sef_exp import fase3b
from sef_exp.fase2 import ROOT
from sef_exp.series import (
    bits_series, calcular_hk, cargar_series, evaluar_set_series,
    familias_series, preparar_series,
)
from sef_exp.series2 import EXOGENAS2

NS = (0, 1, 2, 3, 4, 5)


def config_x(n):
    return fase3b.Config(
        nombre=f"seriesx{n}",
        familias=familias_series(EXOGENAS2, (0,), 3, 2),
        k=1 + n,
        presupuestos=(25, 50, 75, 100),
        granular_desde=10,
        epsilon=2.0,
        etiqueta_y="-AICc en train",
        titulo_j="Mediana de -AICc",
        exogenas=EXOGENAS2,
        lags=(0,),
        p_max=3,
        q_max=2,
        datos=(ROOT / "data/series/X_train.csv",
               ROOT / "data/series/y_train.csv"),
        codigo=(
            ROOT / "src/sef_exp/espacio.py",
            ROOT / "src/sef_exp/evaluador_series.py",
            ROOT / "src/sef_exp/fase3b.py",
            ROOT / "src/sef_exp/series.py",
            ROOT / "scripts/barrido_x.py",
            ROOT / "scripts/elegir_x.py",
        ),
        cargar=cargar_series,
        mensaje_catalogo=f"Precompute seriesx{n}: train cerrado test sin abrir",
        evaluar_set=evaluar_set_series,
        preparar_fn=preparar_series,
        bits_fn=bits_series,
        archivo_evaluador=ROOT / "src/sef_exp/evaluador_series.py",
        detalle_manifiesto={"metrica": "-AICc", "d": "KPSS en train",
                            "n_exogenas": n},
    )


def correr(n, solo_graficos=False):
    cfg = config_x(n)
    if solo_graficos:
        fase3b.graficar(fase3b.metricas(cfg), cfg)
        return
    fase3b.ejecutar(False, 0, (), render=False, cfg=cfg)
    calcular_hk(cfg, n)
    fase3b.graficar(fase3b.metricas(cfg), cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", default=",".join(map(str, NS)))
    parser.add_argument("--solo-graficos", action="store_true")
    args = parser.parse_args()
    for n in [int(x) for x in args.ns.split(",")]:
        print(f"=== n={n} ===", flush=True)
        correr(n, args.solo_graficos)


if __name__ == "__main__":
    main()
