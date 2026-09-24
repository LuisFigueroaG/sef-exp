"""Parte Home Credit application_train en train/test para la demo.

Reglas: solo numericas con a lo mas 50% de nulos (las categoricas y el
bloque habitacional con 70% de huecos quedan fuera), split 80/20
estratificado semilla 42. El crudo se baja con el CLI de kaggle
(competition home-credit-default-risk); el split es local y no se
versiona.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CRUD0 = ROOT / "data/homecredit/application_train.csv"
DEST = ROOT / "data/homecredit/split"

SEED = 42
MAX_NULOS = 0.50


def main():
    df = pl.read_csv(CRUD0)
    num = [c for c, t in df.schema.items()
           if t.is_numeric() and c not in ("SK_ID_CURR", "TARGET")]
    buenas = [c for c in num if df[c].null_count() <= MAX_NULOS * len(df)]
    print(f"numericas: {len(num)}, sobreviven: {len(buenas)}", flush=True)
    y = df["TARGET"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx_tr, idx_te = [], []
    for cls in (0, 1):
        ids = np.flatnonzero(y == cls)
        rng.shuffle(ids)
        n_te = int(round(0.2 * len(ids)))
        idx_te.extend(ids[:n_te].tolist())
        idx_tr.extend(ids[n_te:].tolist())
    idx_tr, idx_te = sorted(idx_tr), sorted(idx_te)
    DEST.mkdir(parents=True, exist_ok=True)
    df.select(buenas).with_row_index("i").filter(
        pl.col("i").is_in(idx_tr)).drop("i").write_csv(DEST / "X_train.csv")
    df.select(buenas).with_row_index("i").filter(
        pl.col("i").is_in(idx_te)).drop("i").write_csv(DEST / "X_test.csv")
    np.savetxt(DEST / "y_train.csv", y[np.array(idx_tr)], fmt="%d",
               header="TARGET", comments="")
    np.savetxt(DEST / "y_test.csv", y[np.array(idx_te)], fmt="%d",
               header="TARGET", comments="")
    print(f"train={len(idx_tr)} test={len(idx_te)} -> {DEST}", flush=True)
    (DEST / "variables.txt").write_text("\n".join(buenas) + "\n")


if __name__ == "__main__":
    sys.exit(main())
