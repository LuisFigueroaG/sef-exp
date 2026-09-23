"""Baseline TPE sobre series2: estudio separado que reusa su cache.

Optuna TPESampler propone una opcion por familia (pq + 7 exogenas
con "ausente" permitido); si no salen exactamente 1 pq + 2 exogenas
se repara a k exacto descartando/eligiendo al azar con la semilla
de la corrida (misma filosofia que la reparacion del cruce).
Duplicados consumen propuestas, unicos consumen trials, igual que
los demas metodos. 20 semillas, presupuestos de series2.
Escribe work/tpe_series2_runs/ y outputs/tpe_series2_resumen.csv.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import optuna

from sef_exp import fase3b
from sef_exp.series2 import CFG2

RUNS_DIR = CFG2.work("tpe_runs")
SUMMARY_PATH = Path(str(CFG2.summary_path).replace("series2_resumen", "tpe_series2_resumen"))


def reparar(base, fams, rng):
    """Ajusta a 1 pq + 2 exogenas descartando o sorteando opciones."""
    por_fam = {f: vs for f, vs in fams}
    es_pq = lambda v: v is not None and v.startswith("p")
    pq = [v for v in base.values() if es_pq(v)]
    ex = [(f, v) for f, v in base.items()
          if v is not None and not es_pq(v)]
    while len(pq) > 1:
        pq.pop(rng.randrange(len(pq)))
    while len(ex) > 2:
        ex.pop(rng.randrange(len(ex)))
    fams_pq = [f for f, _ in fams if f == "pq"]
    fams_ex = [f for f, _ in fams if f != "pq"]
    if not pq:
        f = rng.choice(fams_pq)
        pq = [rng.choice(por_fam[f])]
    while len(ex) < 2:
        disponibles = [f for f in fams_ex if f not in {f for f, _ in ex}]
        f = rng.choice(disponibles)
        ex.append((f, rng.choice(por_fam[f])))
    return tuple(sorted((*pq, *(v for _, v in ex))))


def opciones(cfg):
    return [(name, ("ausente", *choices)) for name, choices in cfg.espacio.familias]


def correr(seed, limit, context, cache):
    cfg = CFG2
    espacio = cfg.espacio
    fams = opciones(cfg)
    rng = random.Random(seed + 99_000)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    estudio = optuna.create_study(direction="maximize", sampler=sampler)
    archive, attempts, duplicates = {}, 0, 0
    model_seconds = 0.0
    best_j, best_s = None, None
    trial = 0
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with (RUNS_DIR / f"tpe_{seed:02d}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fase3b.TRIAL_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        while len(archive) < limit:
            trial_opt = estudio.ask()
            base = {name: trial_opt.suggest_categorical(name, list(vals))
                    for name, vals in fams}
            base = {k: (None if v == "ausente" else v) for k, v in base.items()}
            S = reparar(base, fams, rng)
            attempts += 1
            if not espacio.valido(S):
                continue
            if S in archive:
                duplicates += 1
                continue
            row, _ = cache.consulta(S, context)
            archive[S] = row
            J = float(row["J"]) if row["J"] else None
            model_seconds += float(row["pared_s"])
            estudio.tell(trial_opt, -1e9 if J is None else J)
            trial += 1
            if J is not None and (best_j is None or J > best_j):
                best_j, best_s = J, S
            writer.writerow({
                "metodo": "tpe", "semilla": seed, "trial": trial,
                "propuestas": attempts, "duplicados": duplicates,
                "S": "|".join(S), "fuente": "tpe",
                "estado": row["estado"], "causa": row["causa"],
                "J": row["J"],
                "mejor_J": "" if best_j is None else repr(best_j),
                "mejor_S": "" if best_s is None else "|".join(best_s),
                "pred_J": "", "pared_modelo_s": f"{model_seconds:.6f}",
            })
    return {"metodo": "tpe", "semilla": seed, "trials": trial,
            "propuestas": attempts, "duplicados": duplicates,
            "mejor_J": best_j, "mejor_S": best_s,
            "pared_modelo_s": model_seconds}


def resumir(filas):
    import statistics
    por_metodo = {}
    for metodo in ("aleatorio", "local", "ga_local", "sef_completo", "tpe"):
        if metodo == "tpe":
            js = [f["mejor_J"] for f in filas if f["mejor_J"] is not None]
        else:
            with open(f"outputs/series2_resumen.csv", newline="") as fh:
                base = [r for r in csv.DictReader(fh)
                        if r["metodo"] == metodo
                        and int(r["presupuesto"]) == 1000 and r["mejor_J"]]
            js = [float(r["mejor_J"]) for r in base]
        por_metodo[metodo] = {
            "corridas": len(js),
            "J_mediana": statistics.median(js) if js else None,
            "dentro_2": sum((-535.4400277398996 - j) <= 2.0 + 1e-12
                            for j in js),
        }
    return por_metodo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semillas", default=",".join(map(str, range(20))))
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()
    previos = fase3b.aplicar_umbrales(CFG2)
    try:
        context, _, _ = (CFG2.preparar_fn or fase3b.preparar)(CFG2)
        cache = fase3b.Evaluaciones(cfg=CFG2)
        t0 = time.monotonic()
        filas = []
        for i, seed in enumerate(int(x) for x in args.semillas.split(",")):
            print(f"[{i + 1}] tpe semilla={seed}", flush=True)
            filas.append(correr(seed, args.trials, context, cache))
        with SUMMARY_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=("metodo", "semilla", "trials", "propuestas",
                               "duplicados", "mejor_J", "mejor_S",
                               "pared_modelo_s"), lineterminator="\n")
            writer.writeheader()
            for r in filas:
                r = dict(r)
                r["mejor_J"] = "" if r["mejor_J"] is None else repr(r["mejor_J"])
                r["mejor_S"] = "" if r["mejor_S"] is None else "|".join(r["mejor_S"])
                writer.writerow(r)
        print(f"TPE listo en {time.monotonic() - t0:.1f}s -> {SUMMARY_PATH}",
              flush=True)
        for metodo, m in resumir(filas).items():
            print(metodo, m, flush=True)
    finally:
        fase3b.restaurar_umbrales(previos)


if __name__ == "__main__":
    main()
