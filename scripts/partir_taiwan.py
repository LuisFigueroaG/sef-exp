"""Particion 80/20 estratificada del dataset Taiwan; el test queda cerrado.

Lee data/taiwan/taiwan.csv (descarga UCI) y escribe X/y de train y test
en el mismo formato que data/ de MRAP: primera columna de indice sin
nombre en X, columna "target" en y. Solo este script abre el test.
"""

import random
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/taiwan/taiwan.csv"
OUT = ROOT / "data/taiwan"
TARGET = "default payment next month"
SEED = 42


def main():
    df = pl.read_csv(SRC)
    df.columns = [str(c).strip() for c in df.row(0)]
    df = df.slice(1).with_columns(pl.all().cast(pl.Int64, strict=False))
    df = df.with_row_index("fila")
    rng = random.Random(SEED)
    test_filas = set()
    for cls in (0, 1):
        ids = df.filter(pl.col(TARGET) == cls)["fila"].to_list()
        rng.shuffle(ids)
        test_filas.update(ids[:len(ids) // 5])
    tr = df.filter(~pl.col("fila").is_in(test_filas))
    te = df.filter(pl.col("fila").is_in(test_filas))
    assert len(tr) + len(te) == len(df) and len(te) == len(test_filas)
    variables = [c for c in df.columns if c not in ("fila", "ID", TARGET)]
    for nombre, parte in (("train", tr), ("test", te)):
        x = parte.select(["fila", *variables])
        x.columns = ["", *variables]
        x.write_csv(OUT / f"X_{nombre}.csv")
        y = parte.select(["fila", TARGET])
        y.columns = ["", "target"]
        y.write_csv(OUT / f"y_{nombre}.csv")
    print(f"train={len(tr)} test={len(te)} "
          f"tasa_test={te[TARGET].mean():.4f} -> {OUT}")


if __name__ == "__main__":
    main()
