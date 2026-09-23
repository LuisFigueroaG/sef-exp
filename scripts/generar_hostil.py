"""Genera el dataset hostil: 3 pares de interaccion plantados.

10 familias x 3 variables N(0,1). Y depende solo de productos de
pares de distintas familias; marginales ~0. Train 150k, test 50k,
semillas fijas. Solo este script conoce el patron verdadero.
"""

import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/hostil"
N_TRAIN, N_TEST = 150_000, 50_000
FAMILIAS = tuple((f"F{i}", (f"v{i}a", f"v{i}b", f"v{i}c"))
                 for i in range(1, 11))
# pares vivos: (familia, opcion, familia, opcion)
PARES = ((("F1", "v1b"), ("F4", "v4a")),
         (("F2", "v2c"), ("F7", "v7b")),
         (("F5", "v5a"), ("F9", "v9c")))


def generar(n, seed):
    rng = np.random.default_rng(seed)
    cols = {}
    for _, opciones in FAMILIAS:
        for v in opciones:
            cols[v] = rng.normal(size=n)
    logit = np.zeros(n)
    for (f1, a), (f2, b) in PARES:
        logit += cols[a] * cols[b]
    logit *= 3.0
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.int64)
    return cols, y


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for nombre, n, seed in (("train", N_TRAIN, 7), ("test", N_TEST, 99)):
        cols, y = generar(n, seed)
        variables = [v for _, ops in FAMILIAS for v in ops]
        x = pl.DataFrame({c: cols[c] for c in variables})
        x.with_row_index("").write_csv(OUT / f"X_{nombre}.csv")
        pl.DataFrame({"": range(n), "target": y}).write_csv(OUT / f"y_{nombre}.csv")
        print(f"{nombre}: n={n} tasa={y.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
