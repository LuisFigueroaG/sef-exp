"""Track hostil: HGB sobre interacciones plantadas.

10 familias x 3 opciones, k=6, J = AUC medio de 5 folds con HGB
fijo. Sin filtros de scorecard. El test de data/hostil/ nunca se
abre aqui.
"""

import time

import numpy as np
import polars as pl

from sef_exp import fase3b
from sef_exp.evaluador import hacer_folds
from sef_exp.evaluador_hgb import evaluar_hgb
from sef_exp.fase2 import ROOT

FAMILIAS_HOSTIL = tuple((f"F{i}", (f"v{i}a", f"v{i}b", f"v{i}c"))
                        for i in range(1, 11))


def cargar_hostil():
    base = ROOT / "data/hostil"
    x = pl.read_csv(base / "X_train.csv").drop("")
    y = pl.read_csv(base / "y_train.csv")["target"].cast(pl.Int64).to_numpy().ravel()
    return x, y


def preparar_hostil(cfg):
    print(cfg.mensaje_catalogo, flush=True)
    start = time.monotonic()
    x, y = cfg.cargar()
    if set(x.columns) != set(cfg.espacio.variables):
        raise RuntimeError("Las columnas train no coinciden con el contrato")
    folds = hacer_folds(y)
    iv = {}
    for v in cfg.espacio.variables:
        r = evaluar_hgb((v,), x, y, folds)
        iv[v] = r["J"] if r["J"] is not None else 0.5
    print(f"precompute={time.monotonic() - start:.1f}s", flush=True)
    return (x, y, folds), iv, time.monotonic() - start


def evaluar_set_hgb(S, context, cfg):
    x, y, folds = context
    return evaluar_hgb(S, x, y, folds)


def forward_hgb(context, cfg):
    """Greedy sobre ganancia HGB: el control que deberia perder."""
    espacio = cfg.espacio
    cache = fase3b.Evaluaciones(cfg=cfg)
    elegidas, usadas, ensayos = (), set(), 0
    for _ in range(cfg.k):
        opciones = []
        for name, choices in espacio.familias:
            if name in usadas:
                continue
            for v in choices:
                S = tuple(sorted((*elegidas, v)))
                row, _ = cache.consulta(S, context)
                ensayos += 1
                J = float(row["J"]) if row["J"] else None
                opciones.append((J, S, name))
        validas = [o for o in opciones if o[0] is not None]
        if not validas:
            return None, elegidas, ensayos
        _, elegidas, name = max(validas, key=lambda o: (o[0], o[1]))
        usadas.add(name)
    row, _ = cache.consulta(tuple(sorted(elegidas)), context)
    J = float(row["J"]) if row["J"] else None
    return J, tuple(sorted(elegidas)), ensayos


def calcular_forward(cfg):
    previos = fase3b.aplicar_umbrales(cfg)
    try:
        context, _, _ = (cfg.preparar_fn or fase3b.preparar)(cfg)
        t = time.monotonic()
        J, S, ensayos = forward_hgb(context, cfg)
        modelo = time.monotonic() - t
    finally:
        fase3b.restaurar_umbrales(previos)
    import csv
    with cfg.referencias_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fase3b.REFERENCIA_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "metodo": "forward", "J": "" if J is None else repr(J),
            "S": "" if not S else "|".join(S),
            "ensayos": ensayos, "pared_modelo_s": f"{modelo:.6f}",
        })
    print(f"Forward referencia: J={J} S={'|'.join(S) if S else '-'} "
          f"ensayos={ensayos} -> {cfg.referencias_path}", flush=True)
    return J


CFG = fase3b.Config(
    nombre="hostil",
    familias=FAMILIAS_HOSTIL,
    k=6,
    presupuestos=(50, 100, 250),
    semillas=tuple(range(10)),
    granular_desde=25,
    epsilon=0.005,
    etiqueta_y="AUC medio de cinco folds",
    titulo_j="J mediana en CV",
    datos=(ROOT / "data/hostil/X_train.csv", ROOT / "data/hostil/y_train.csv"),
    codigo=(
        ROOT / "src/sef_exp/espacio.py",
        ROOT / "src/sef_exp/evaluador_hgb.py",
        ROOT / "src/sef_exp/fase3b.py",
        ROOT / "src/sef_exp/hostil.py",
        ROOT / "scripts/generar_hostil.py",
    ),
    cargar=cargar_hostil,
    mensaje_catalogo="Precompute hostil: HGB univariado x30; test cerrado",
    evaluar_set=evaluar_set_hgb,
    preparar_fn=preparar_hostil,
    archivo_evaluador=ROOT / "src/sef_exp/evaluador_hgb.py",
    detalle_manifiesto={"metrica": "AUC medio 5 folds", "modelo": "HGB",
                        "hgb_params": {"max_iter": 50, "max_leaf_nodes": 31,
                                       "learning_rate": 0.1,
                                       "min_samples_leaf": 200,
                                       "l2_regularization": 1.0},
                        "filtros": "ninguno salvo ajuste"},
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piloto", action="store_true")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--semillas", default="0")
    parser.add_argument("--solo-graficos", action="store_true")
    parser.add_argument("--sin-graficos", action="store_true")
    parser.add_argument("--forward-ref", action="store_true")
    args = parser.parse_args()
    if args.forward_ref:
        calcular_forward(CFG)
    elif args.solo_graficos:
        fase3b.graficar(fase3b.metricas(CFG), CFG)
    else:
        seeds = tuple(int(x) for x in args.semillas.split(","))
        fase3b.ejecutar(args.piloto, args.trials, seeds,
                         render=not args.sin_graficos, cfg=CFG)


if __name__ == "__main__":
    main()
