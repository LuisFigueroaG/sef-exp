"""Barrido de k en Taiwan: mismo motor, k=2..8 (k=6 es fase4 y se reusa).

Corre busqueda completa + forward + metricas + figura por k con
nombre fase4k{k}. Costoso pero paralelizable por k a mano con
--ks. Solo lee train.
"""

import argparse

from sef_exp import fase3b
from sef_exp.fase2 import ROOT
from sef_exp.fase4 import FAMILIAS_TAIWAN, cargar_taiwan

KS = (2, 3, 4, 5, 7, 8)
EPSILON = 0.001


def config_k(k):
    return fase3b.Config(
        nombre=f"fase4k{k}",
        familias=FAMILIAS_TAIWAN,
        k=k,
        iv_min=0.02,
        datos=(ROOT / "data/taiwan/X_train.csv", ROOT / "data/taiwan/y_train.csv"),
        codigo=(
            ROOT / "src/sef_exp/espacio.py",
            ROOT / "src/sef_exp/fase3b.py",
            ROOT / "src/sef_exp/fase4.py",
            ROOT / "scripts/barrido_k.py",
            ROOT / "scripts/elegir_k.py",
        ),
        cargar=cargar_taiwan,
        mensaje_catalogo=f"Precompute Taiwan k={k}: train cerrado test sin abrir",
    )


def correr(k, solo_graficos=False):
    cfg = config_k(k)
    if solo_graficos:
        fase3b.graficar(fase3b.metricas(cfg), cfg)
        return
    fase3b.ejecutar(False, 0, (), render=False, cfg=cfg)
    fase3b.calcular_forward(cfg)
    fase3b.graficar(fase3b.metricas(cfg), cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    parser.add_argument("--solo-graficos", action="store_true")
    args = parser.parse_args()
    for k in [int(x) for x in args.ks.split(",")]:
        print(f"=== k={k} ===", flush=True)
        correr(k, args.solo_graficos)


if __name__ == "__main__":
    main()
