"""Fase 4: replica externa del protocolo 3b en el dataset Taiwan.

Mismas reglas (seis variables, una por familia, filtros solo en train
de cada fold, presupuestos y semillas), con IV_MIN 0.02 documentado en
docs/fase4.md. El test de data/taiwan/ nunca se abre aqui.
"""

import polars as pl

from sef_exp import fase3b
from sef_exp.evaluador import aplicar_nulos
from sef_exp.fase2 import ROOT

FAMILIAS_TAIWAN = (
    ("cupo", ("LIMIT_BAL",)),
    ("sexo", ("SEX",)),
    ("educacion", ("EDUCATION",)),
    ("estado_civil", ("MARRIAGE",)),
    ("edad", ("AGE",)),
    ("pago_reciente", ("PAY_0", "PAY_2", "PAY_3")),
    ("pago_antiguo", ("PAY_4", "PAY_5", "PAY_6")),
    ("facturacion_reciente", ("BILL_AMT1", "BILL_AMT2", "BILL_AMT3")),
    ("facturacion_antigua", ("BILL_AMT4", "BILL_AMT5", "BILL_AMT6")),
    ("monto_reciente", ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3")),
    ("monto_antiguo", ("PAY_AMT4", "PAY_AMT5", "PAY_AMT6")),
)


def cargar_taiwan():
    base = ROOT / "data/taiwan"
    x = pl.read_csv(base / "X_train.csv").drop("")
    y = pl.read_csv(base / "y_train.csv")["target"].cast(pl.Int64).to_numpy().ravel()
    return aplicar_nulos(x), y


CFG = fase3b.Config(
    nombre="fase4",
    familias=FAMILIAS_TAIWAN,
    iv_min=0.02,
    datos=(ROOT / "data/taiwan/X_train.csv", ROOT / "data/taiwan/y_train.csv"),
    codigo=(
        ROOT / "src/sef_exp/espacio.py",
        ROOT / "src/sef_exp/fase3b.py",
        ROOT / "src/sef_exp/fase4.py",
        ROOT / "scripts/partir_taiwan.py",
    ),
    cargar=cargar_taiwan,
    mensaje_catalogo="Precompute Taiwan train: 23 variables x 5 folds; test reservado sin abrir",
)


def main():
    fase3b.main(CFG)


if __name__ == "__main__":
    main()
