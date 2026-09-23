"""Track series: ARIMAX sobre IMACEC con exogenas macro chilenas.

Espacio: familias P={p0..p3}, Q={q0..q2} mas cinco exogenas singleton;
k=4 (un P, un Q, dos exogenas). d fijo por KPSS repetidos. J=-AICc.
El test de data/series/ nunca se abre aqui.
"""

import time

import numpy as np
import polars as pl

from sef_exp import fase3b
from sef_exp.evaluador_series import (
    EXOGENAS, evaluar_serie, orden_diferencias, parte_exogena,
)
from sef_exp.fase2 import ROOT


def opciones_exogenas(exogenas, lags):
    opciones = []
    for e in exogenas:
        for lag in lags:
            opciones.append(e if lag == 0 else f"{e}_L{lag}")
    return opciones


def familias_series(exogenas, lags, p_max, q_max):
    fams = [("pq", tuple(f"p{p}q{q}" for p in range(p_max + 1)
                         for q in range(q_max + 1)))]
    for e in exogenas:
        fams.append((e.lower(), tuple(opciones_exogenas((e,), lags))))
    return tuple(fams)


FAMILIAS_SERIES = familias_series(EXOGENAS, (0,), 3, 2)


def cargar_series():
    base = ROOT / "data/series"
    x = pl.read_csv(base / "X_train.csv").drop("")
    y = pl.read_csv(base / "y_train.csv")["target"].cast(pl.Float64).to_numpy().ravel()
    return x, y


def preparar_series(cfg):
    print(cfg.mensaje_catalogo, flush=True)
    start = time.monotonic()
    x, y = cfg.cargar()
    if not set(cfg.exogenas) <= set(x.columns):
        raise RuntimeError("Faltan exogenas del contrato en train")
    x = x.select(list(cfg.exogenas))
    max_lag = max(cfg.lags)
    cols = {}
    for e in cfg.exogenas:
        for lag in cfg.lags:
            nombre = e if lag == 0 else f"{e}_L{lag}"
            cols[nombre] = x[e].shift(lag)
    xl = pl.DataFrame(cols).slice(max_lag)
    yl = np.asarray(y, dtype=float)[max_lag:]
    medias = {c: xl[c].mean() for c in xl.columns}
    desvs = {c: xl[c].std() or 1.0 for c in xl.columns}
    xstd = xl.with_columns(
        ((pl.col(c) - medias[c]) / desvs[c]).alias(c) for c in xl.columns)
    d = orden_diferencias(yl)
    iv = {}
    for v in cfg.espacio.variables:
        if v == "pq" or v.startswith("p"):
            iv[v] = 0.0
            continue
        base, _ = parte_exogena(v)
        otra = next(e for e in cfg.exogenas if e != base)
        r = evaluar_serie(("p0q0", v, otra), yl, xstd, d)
        iv[v] = r["J"] if r["J"] is not None else float("-inf")
    print(f"d por KPSS={d} precompute={time.monotonic() - start:.1f}s", flush=True)
    return (yl, xstd, d), iv, time.monotonic() - start


def evaluar_set_series(S, context, cfg):
    ytr, Xtr, d = context
    return evaluar_serie(S, ytr, Xtr, d)


def bits_series(S, cfg):
    """One-hot sobre las opciones del espacio (pq + exogenas con lag).

    Se probo con dummies por familia + p,q enteros, pero los triples
    sin token pq (estructuralmente validos, siempre rechazados)
    rompian la matriz del sustituto con largos distintos y Nones.
    El one-hot es uniforme para todo lo que la bolsa propone.
    """
    selected = set(S)
    return [int(v in selected) for v in cfg.espacio.variables]


def vecinos_pq(p, q, cfg):
    for dp, dq in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        a, b = p + dp, q + dq
        if 0 <= a <= cfg.p_max and 0 <= b <= cfg.q_max:
            yield a, b


def mejor_pq(exogenas, context, cfg, cache, ensayos):
    p, q = 0, 0
    mejor = None
    for _ in range(2 * (cfg.p_max + cfg.q_max + 2)):
        candidatos = []
        for a, b in vecinos_pq(p, q, cfg):
            S = tuple(sorted((f"p{a}q{b}", *exogenas)))
            row, _ = cache.consulta(S, context)
            ensayos.add(S)
            J = float(row["J"]) if row["J"] else None
            if J is not None:
                candidatos.append((J, a, b))
        if not candidatos:
            return mejor
        J, p, q = max(candidatos)
        if mejor is None or J > mejor[0]:
            mejor = (J, p, q)
        else:
            break
    return mejor


def hk_forward(context, iv, cfg, n_exogenas=2):
    """Greedy sobre exogenas con (p,q) re-entonado por paso HK local.

    Arranca del mejor conjunto univariado y prueba intercambios
    simples. Determinista, sin presupuesto: referencia descriptiva.
    """
    cache = fase3b.Evaluaciones(cfg=cfg)
    ensayos = set()
    opciones = [v for v in cfg.espacio.variables if not v.startswith("p")]
    ranking = sorted(
        ((iv[v], v) for v in opciones), reverse=True)
    elegidas = []
    for _, v in ranking:
        if all(parte_exogena(v)[0] != parte_exogena(e)[0] for e in elegidas):
            elegidas.append(v)
        if len(elegidas) == n_exogenas:
            break
    elegidas = tuple(sorted(elegidas))
    local = mejor_pq(elegidas, context, cfg, cache, ensayos)
    if local is None:
        return None, None, len(ensayos)
    mejor = (local[0], elegidas, local[1], local[2])
    for _ in range(3 * (n_exogenas + 1)):
        candidatas = []
        for fuera in mejor[1]:
            for _, dentro in ranking:
                if dentro in mejor[1]:
                    continue
                resto = [e for e in mejor[1] if e != fuera]
                if any(parte_exogena(dentro)[0] == parte_exogena(e)[0]
                       for e in resto):
                    continue
                exogenas = tuple(sorted((dentro, *resto)))
                local = mejor_pq(exogenas, context, cfg, cache, ensayos)
                if local is not None:
                    candidatas.append((local[0], exogenas, local[1], local[2]))
        if not candidatas:
            break
        candidato = max(candidatas)
        if candidato[0] <= mejor[0]:
            break
        mejor = candidato
    J, elegidas, p, q = mejor
    return J, tuple(sorted((f"p{p}q{q}", *elegidas))), len(ensayos)


def calcular_hk(cfg, n_exogenas=2):
    previos = fase3b.aplicar_umbrales(cfg)
    try:
        context, iv, _ = (cfg.preparar_fn or fase3b.preparar)(cfg)
        t = time.monotonic()
        J, S, ensayos = hk_forward(context, iv, cfg, n_exogenas)
        modelo = time.monotonic() - t
    finally:
        fase3b.restaurar_umbrales(previos)
    import csv
    with cfg.referencias_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fase3b.REFERENCIA_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "metodo": "hk_forward", "J": "" if J is None else repr(J),
            "S": "" if S is None else "|".join(S),
            "ensayos": ensayos, "pared_modelo_s": f"{modelo:.6f}",
        })
    print(f"HK referencia: J={J} S={'|'.join(S) if S else '-'} "
          f"ensayos={ensayos} -> {cfg.referencias_path}", flush=True)
    return J


CFG = fase3b.Config(
    nombre="series",
    familias=FAMILIAS_SERIES,
    k=3,
    presupuestos=(25, 50, 75, 100),
    granular_desde=10,
    epsilon=2.0,
    etiqueta_y="-AICc en train",
    titulo_j="Mediana de -AICc",
    exogenas=EXOGENAS,
    lags=(0,),
    p_max=3,
    q_max=2,
    datos=(ROOT / "data/series/X_train.csv", ROOT / "data/series/y_train.csv"),
    codigo=(
        ROOT / "src/sef_exp/espacio.py",
        ROOT / "src/sef_exp/evaluador_series.py",
        ROOT / "src/sef_exp/fase3b.py",
        ROOT / "src/sef_exp/series.py",
        ROOT / "scripts/descargar_mindicador.py",
    ),
    cargar=cargar_series,
    mensaje_catalogo="Precompute series train: IMACEC + 5 exogenas; test reservado sin abrir",
    evaluar_set=evaluar_set_series,
    preparar_fn=preparar_series,
    bits_fn=bits_series,
    archivo_evaluador=ROOT / "src/sef_exp/evaluador_series.py",
    detalle_manifiesto={"metrica": "-AICc", "d": "KPSS repetidos en train",
                        "exogenas": "z-score en train"},
)


def construir_oraculo(cfg):
    """Evalua los 130 conjuntos exhaustivamente via cache compartido."""
    import csv as csv_mod

    previos = fase3b.aplicar_umbrales(cfg)
    try:
        context, _, _ = (cfg.preparar_fn or fase3b.preparar)(cfg)
        cache = fase3b.Evaluaciones(cfg=cfg)
        rng = __import__("random").Random(0)
        faltantes = 0
        while len(cache.rows) < cfg.espacio.total:
            S = cfg.espacio.muestra(rng)
            if S not in cache.rows:
                cache.consulta(S, context)
                faltantes += 1
        destino = cfg.output("oraculo.csv")
        with destino.open("w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fase3b.CACHE_COLUMNS,
                                        lineterminator="\n")
            writer.writeheader()
            for S in sorted(cache.rows):
                writer.writerow(cache.rows[S])
    finally:
        fase3b.restaurar_umbrales(previos)
    optimas = [r for r in cache.rows.values() if r["J"]]
    mejor = max(optimas, key=lambda r: float(r["J"]))
    print(f"Oraculo: {len(cache.rows)}/{cfg.espacio.total} "
          f"nuevos={faltantes} mejor J={mejor['J']} S={mejor['S']} -> {destino}",
          flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piloto", action="store_true")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--semillas", default="0")
    parser.add_argument("--solo-graficos", action="store_true")
    parser.add_argument("--sin-graficos", action="store_true")
    parser.add_argument("--hk-ref", action="store_true")
    parser.add_argument("--oraculo", action="store_true")
    args = parser.parse_args()
    if args.hk_ref:
        calcular_hk(CFG)
    elif args.oraculo:
        construir_oraculo(CFG)
    elif args.solo_graficos:
        fase3b.graficar(fase3b.metricas(CFG), CFG)
    else:
        seeds = tuple(int(x) for x in args.semillas.split(","))
        fase3b.ejecutar(args.piloto, args.trials, seeds,
                         render=not args.sin_graficos, cfg=CFG)


if __name__ == "__main__":
    main()
