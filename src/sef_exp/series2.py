"""Track series2: ARIMAX ampliado (p 0-5, q 0-3, lags, 7 exogenas).

Reutiliza el motor de series.py con otra Config: 24 combinaciones pq
mas siete familias exogenas (IPC, TPM, DOLAR, COBRE, DESEMPLEO, UF,
EURO) con opciones de nivel, lag 1 y lag 3. k=3. Espacio jugable
4536 conjuntos; el oraculo exhaustivo cubre el estructural completo.
El test de data/series/ nunca se abre aqui.
"""

from sef_exp import fase3b
from sef_exp.fase2 import ROOT
from sef_exp.series import (
    bits_series, calcular_hk, cargar_series, construir_oraculo,
    evaluar_set_series, familias_series, preparar_series,
)

EXOGENAS2 = ("IPC", "TPM", "DOLAR", "COBRE", "DESEMPLEO", "UF", "EURO")
LAGS2 = (0, 1, 3)

CFG2 = fase3b.Config(
    nombre="series2",
    familias=familias_series(EXOGENAS2, LAGS2, 5, 3),
    k=3,
    presupuestos=(100, 250, 500, 1000),
    granular_desde=25,
    epsilon=2.0,
    etiqueta_y="-AICc en train",
    titulo_j="Mediana de -AICc",
    exogenas=EXOGENAS2,
    lags=LAGS2,
    p_max=5,
    q_max=3,
    datos=(ROOT / "data/series/X_train.csv", ROOT / "data/series/y_train.csv"),
    codigo=(
        ROOT / "src/sef_exp/espacio.py",
        ROOT / "src/sef_exp/evaluador_series.py",
        ROOT / "src/sef_exp/fase3b.py",
        ROOT / "src/sef_exp/series.py",
        ROOT / "src/sef_exp/series2.py",
        ROOT / "scripts/descargar_mindicador.py",
    ),
    cargar=cargar_series,
    mensaje_catalogo="Precompute series2 train: IMACEC + 7 exogenas con lags; test cerrado",
    evaluar_set=evaluar_set_series,
    preparar_fn=preparar_series,
    bits_fn=bits_series,
    archivo_evaluador=ROOT / "src/sef_exp/evaluador_series.py",
    detalle_manifiesto={"metrica": "-AICc", "d": "KPSS repetidos en train",
                        "exogenas": "z-score en train",
                        "lags": "0,1,3 (train pierde 3 meses de cabeza)"},
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piloto", action="store_true")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--semillas", default="0")
    parser.add_argument("--solo-graficos", action="store_true")
    parser.add_argument("--sin-graficos", action="store_true")
    parser.add_argument("--hk-ref", action="store_true")
    parser.add_argument("--oraculo", action="store_true")
    args = parser.parse_args()
    if args.hk_ref:
        calcular_hk(CFG2)
    elif args.oraculo:
        construir_oraculo(CFG2)
    elif args.solo_graficos:
        fase3b.graficar(fase3b.metricas(CFG2), CFG2)
    else:
        seeds = tuple(int(x) for x in args.semillas.split(","))
        fase3b.ejecutar(args.piloto, args.trials, seeds,
                         render=not args.sin_graficos, cfg=CFG2)


if __name__ == "__main__":
    main()
