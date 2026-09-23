"""SEF and ablations on the fixed, fully enumerated phase 2 subspace."""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

from sef_exp.evaluador import hacer_folds, precomputar
from sef_exp.fase2 import (
    BUDGETS,
    FAMILIAS,
    FINGERPRINT_PATH,
    ORACLE_PATH,
    ROOT,
    SEEDS,
    abrir_oraculo,
    cargar_train,
    catalogo,
    firma,
    vecindarios,
)

VARIABLES = tuple(sorted(v for _, choices in FAMILIAS for v in choices))
FAMILY_OF = {v: family for family, choices in FAMILIAS for v in choices}
FAMILY_CHOICES = {family: choices for family, choices in FAMILIAS}
BATCH_SIZE = 16
POPULATION_SIZE = 64
ELITES = 8
POOL_GENETIC = 96
POOL_RANDOM = 16
POOL_LOCAL = 16
FIT_EVERY_BATCHES = 4
TRIALS_PATH = ROOT / "work/fase3_trials.csv"
RUNS_DIR = ROOT / "work/fase3_runs"
SUMMARY_PATH = ROOT / "outputs/fase3_resumen.csv"
METRICS_PATH = ROOT / "outputs/fase3_metricas.csv"
FIGURE_PATH = ROOT / "outputs/fase3_curvas.png"
ABLATION_FIGURE_PATH = ROOT / "outputs/fase3_ablaciones.png"
MANIFEST_PATH = ROOT / "outputs/fase3_manifest.json"

TRIAL_COLUMNS = (
    "variante", "semilla", "trial", "propuestas", "duplicados", "estructurales_rechazados",
    "reparaciones", "reinicios", "S", "fuente", "estado", "causa", "J", "mejor_J",
    "mejor_S", "pred_J", "pred_desacuerdo", "pared_modelo_s",
)
SUMMARY_COLUMNS = (
    "variante", "semilla", "presupuesto", "trials_reales", "propuestas", "duplicados",
    "estructurales_rechazados", "reparaciones", "reinicios", "factibles", "rechazados",
    "mejor_J", "mejor_S", "pared_modelo_s", "pared_algoritmo_s", "cpu_algoritmo_s",
    "ajustes_sustituto", "pared_sustituto_s", "diversidad_variables", "diversidad_familias",
    "predicciones_validas", "mae_prequential", "rho_prequential",
    "T_005", "T_002", "T_001", "T_optimo",
)


@dataclass(frozen=True)
class Variante:
    name: str
    local: bool = True
    surrogate: bool = True
    doubles: bool = True
    restart: bool = True
    guided: bool = True
    repair: bool = True


VARIANTES = (
    Variante("ga_basico", local=False, surrogate=False),
    Variante("ga_local", surrogate=False),
    Variante("sef_completo"),
    Variante("sef_sin_local", local=False),
    Variante("sef_sin_dobles", doubles=False),
    Variante("sef_sin_reinicios", restart=False),
    Variante("sef_sin_guiada", guided=False),
    Variante("sef_sin_reparacion", repair=False),
)


def leer_oraculo(universe):
    if not ORACLE_PATH.exists() or not FINGERPRINT_PATH.exists():
        raise RuntimeError("Primero complete la fase 2")
    rows = abrir_oraculo(firma(), universe)
    if len(rows) != len(universe):
        raise RuntimeError("El oraculo de fase 2 esta incompleto")
    return {tuple(row["S"].split("|")): row for row in rows}


def iv_por_variable():
    start = time.monotonic()
    print("Precompute de IV por variable y fold, solo train", flush=True)
    x, y = cargar_train()
    folds = hacer_folds(y)
    pre = precomputar(x, y, folds, list(VARIABLES))
    iv = {v: statistics.mean(pre[(r, v)]["iv"] for r in range(len(folds))) for v in VARIABLES}
    elapsed = time.monotonic() - start
    print(f"IV listo en {elapsed:.1f}s", flush=True)
    return iv, elapsed


def genes(S):
    selected = set(S)
    return tuple(next((v for v in choices if v in selected), None) for _, choices in FAMILIAS)


def features(S):
    selected = set(S)
    return [int(v in selected) for v in VARIABLES]


def guided_candidate(iv, rng):
    ranked = []
    for family, choices in FAMILIAS:
        chosen = max(choices, key=lambda v: (iv[v] + rng.gauss(0.0, 0.05), v))
        ranked.append((iv[chosen] + rng.gauss(0.0, 0.05), family, chosen))
    ranked.sort(reverse=True)
    return tuple(sorted(item[2] for item in ranked[:6]))


def distance(a, b):
    return len(set(a) - set(b))


def mutate(S, rng, doubles=True):
    modes = ("same", "family", "double") if doubles else ("same", "family")
    weights = (0.5, 0.3, 0.2) if doubles else (0.625, 0.375)
    mode = rng.choices(modes, weights=weights, k=1)[0]

    def one_move(current):
        selected = set(current)
        active = [family for family, choices in FAMILIAS if any(v in selected for v in choices)]
        inactive = [family for family, _ in FAMILIAS if family not in active]
        change_family = rng.random() < 0.375 and bool(inactive)
        family = rng.choice(active)
        old = next(v for v in FAMILY_CHOICES[family] if v in selected)
        selected.remove(old)
        if change_family:
            new_family = rng.choice(inactive)
            selected.add(rng.choice(FAMILY_CHOICES[new_family]))
        else:
            selected.add(next(v for v in FAMILY_CHOICES[family] if v != old))
        return tuple(sorted(selected))

    if mode == "double":
        return one_move(one_move(S))
    selected = set(S)
    active = [family for family, choices in FAMILIAS if any(v in selected for v in choices)]
    family = rng.choice(active)
    old = next(v for v in FAMILY_CHOICES[family] if v in selected)
    selected.remove(old)
    if mode == "same":
        selected.add(next(v for v in FAMILY_CHOICES[family] if v != old))
    else:
        inactive = [family for family, _ in FAMILIAS if family not in active]
        selected.add(rng.choice(FAMILY_CHOICES[rng.choice(inactive)]))
    return tuple(sorted(selected))


def crossover(a, b, rng, repair=True):
    ga, gb = genes(a), genes(b)
    child = [x if x == y else rng.choice((x, y)) for x, y in zip(ga, gb)]
    active = [i for i, value in enumerate(child) if value is not None]
    changed = len(active) != 6
    if changed and not repair:
        return None, False
    while len(active) > 6:
        removable = [i for i in active if ga[i] != gb[i]] or active
        i = rng.choice(removable)
        child[i] = None
        active.remove(i)
    while len(active) < 6:
        empty = [i for i, value in enumerate(child) if value is None]
        i = rng.choice(empty)
        parents = [v for v in (ga[i], gb[i]) if v is not None]
        child[i] = rng.choice(parents) if parents else rng.choice(FAMILIAS[i][1])
        active.append(i)
    return tuple(sorted(v for v in child if v is not None)), changed


def select_parent(population, valid, rng, other=None):
    candidates = rng.sample(population, min(3, len(population)))
    top = max(valid[S] for S in candidates)
    near_top = [S for S in candidates if top - valid[S] <= 0.0005]
    if other is None:
        return max(near_top, key=lambda S: (valid[S], S))
    return max(near_top, key=lambda S: (distance(S, other), valid[S], S))


def survivor_population(old, new_valid, valid, rng):
    elite = sorted(valid, key=lambda S: (valid[S], S), reverse=True)[:ELITES]
    pool = list(dict.fromkeys((*old, *new_valid)))
    selected = list(elite)
    pool = [S for S in pool if S not in selected]
    while pool and len(selected) < POPULATION_SIZE:
        tournament = rng.sample(pool, min(3, len(pool)))
        top = max(valid[S] for S in tournament)
        close = [S for S in tournament if top - valid[S] <= 0.0005]
        winner = max(close, key=lambda S: (
            min(distance(S, current) for current in selected) if selected else 6,
            valid[S], S,
        ))
        selected.append(winner)
        pool.remove(winner)
    return selected


def population_diversity(population):
    if len(population) < 2:
        return None, None
    variable, family = [], []
    for a, b in itertools.combinations(population, 2):
        sa, sb = set(a), set(b)
        fa, fb = {FAMILY_OF[v] for v in a}, {FAMILY_OF[v] for v in b}
        variable.append(1 - len(sa & sb) / len(sa | sb))
        family.append(1 - len(fa & fb) / len(fa | fb))
    return statistics.mean(variable), statistics.mean(family)


def run_variant(variant, seed, universe, oracle, near, iv, trial_writer, trial_file,
                optimum, trial_limit=None):
    rng_init = random.Random(seed + 10_000)
    rng_guided = random.Random(seed + 15_000)
    rng_gen = random.Random(seed + 20_000)
    rng_random = random.Random(seed + 30_000)
    rng_local = random.Random(seed + 40_000)
    rng_survival = random.Random(seed + 50_000)
    archive, valid = {}, {}
    population = []
    best_j, best_s = None, None
    last_result = ""
    stats = {
        "propuestas": 0, "duplicados": 0, "estructurales_rechazados": 0,
        "reparaciones": 0, "reinicios": 0, "factibles": 0, "rechazados": 0,
        "pared_modelo_s": 0.0, "ajustes_sustituto": 0, "pared_sustituto_s": 0.0,
    }
    predictions = []
    first_hit = {0.005: None, 0.002: None, 0.001: None, 0.0: None}
    summaries = []
    start_wall, start_cpu = time.monotonic(), time.process_time()

    def snapshot(budget):
        div_var, div_family = population_diversity(population)
        pred_x = [x for x, _ in predictions]
        pred_y = [y for _, y in predictions]
        rho = None
        if len(predictions) >= 3 and len(set(pred_x)) > 1 and len(set(pred_y)) > 1:
            rho = float(spearmanr(pred_x, pred_y).statistic)
            if not math.isfinite(rho):
                rho = None
        row = {
            "variante": variant.name, "semilla": seed, "presupuesto": budget,
            "trials_reales": len(archive), "propuestas": stats["propuestas"],
            "duplicados": stats["duplicados"],
            "estructurales_rechazados": stats["estructurales_rechazados"],
            "reparaciones": stats["reparaciones"], "reinicios": stats["reinicios"],
            "factibles": stats["factibles"], "rechazados": stats["rechazados"],
            "mejor_J": "" if best_j is None else repr(best_j),
            "mejor_S": "" if best_s is None else "|".join(best_s),
            "pared_modelo_s": f"{stats['pared_modelo_s']:.6f}",
            "pared_algoritmo_s": f"{time.monotonic() - start_wall:.6f}",
            "cpu_algoritmo_s": f"{time.process_time() - start_cpu:.6f}",
            "ajustes_sustituto": stats["ajustes_sustituto"],
            "pared_sustituto_s": f"{stats['pared_sustituto_s']:.6f}",
            "diversidad_variables": "" if div_var is None else repr(div_var),
            "diversidad_familias": "" if div_family is None else repr(div_family),
            "predicciones_validas": len(predictions),
            "mae_prequential": "" if not predictions else repr(statistics.mean(
                abs(x - y) for x, y in predictions)),
            "rho_prequential": "" if rho is None else repr(rho),
            "T_005": "" if first_hit[0.005] is None else first_hit[0.005],
            "T_002": "" if first_hit[0.002] is None else first_hit[0.002],
            "T_001": "" if first_hit[0.001] is None else first_hit[0.001],
            "T_optimo": "" if first_hit[0.0] is None else first_hit[0.0],
        }
        summaries.append(row)
        trial_file.flush()
        eta_model = (len(universe) - len(archive)) * stats["pared_modelo_s"] / max(len(archive), 1)
        print(f"  {variant.name} s={seed} {len(archive)}/{len(universe)} "
              f"ultimo={last_result} mejor={best_j} propuestas={stats['propuestas']} "
              f"rechazos={stats['rechazados']} reinicios={stats['reinicios']} "
              f"ETA_modelo={eta_model / 60:.1f}m", flush=True)

    def evaluate(S, source, pred=None):
        nonlocal best_j, best_s, last_result
        if S in archive:
            raise AssertionError("Una evaluacion duplicada no debe consumir trial")
        row = oracle[S]
        archive[S] = row
        J = float(row["J"]) if row["J"] else None
        stats["pared_modelo_s"] += float(row["pared_s"])
        last_result = f"J={J:.6f}" if J is not None else row["causa"]
        if J is None:
            stats["rechazados"] += 1
        else:
            valid[S] = J
            stats["factibles"] += 1
            if pred is not None:
                predictions.append((pred[0], J))
            if best_j is None or J > best_j:
                best_j, best_s = J, S
                print(f"  {variant.name} s={seed} nuevo mejor J={J:.6f} "
                      f"trial={len(archive)} S={'|'.join(S)}", flush=True)
            for epsilon in first_hit:
                if first_hit[epsilon] is None and optimum - J <= epsilon + 1e-12:
                    first_hit[epsilon] = len(archive)
        trial_writer.writerow({
            "variante": variant.name, "semilla": seed, "trial": len(archive),
            "propuestas": stats["propuestas"], "duplicados": stats["duplicados"],
            "estructurales_rechazados": stats["estructurales_rechazados"],
            "reparaciones": stats["reparaciones"], "reinicios": stats["reinicios"],
            "S": "|".join(S), "fuente": source, "estado": row["estado"],
            "causa": row["causa"], "J": "" if J is None else repr(J),
            "mejor_J": "" if best_j is None else repr(best_j),
            "mejor_S": "" if best_s is None else "|".join(best_s),
            "pred_J": "" if pred is None else repr(pred[0]),
            "pred_desacuerdo": "" if pred is None else repr(pred[1]),
            "pared_modelo_s": f"{stats['pared_modelo_s']:.6f}",
        })
        if len(archive) in BUDGETS:
            snapshot(len(archive))
        return J

    def offer(S, source, bag, channel):
        stats["propuestas"] += 1
        if S is None:
            stats["estructurales_rechazados"] += 1
            return False
        if S in archive or S in bag:
            stats["duplicados"] += 1
            return False
        bag[S] = source
        channel.append(S)
        return True

    random_order = rng_init.sample(universe, len(universe))
    initial = {}
    initial_random = []
    for S in random_order[:48]:
        offer(S, "init_random", initial, initial_random)
    initial_guided = []
    if variant.guided:
        attempts = 0
        while len(initial_guided) < 16 and attempts < 2048:
            attempts += 1
            offer(guided_candidate(iv, rng_guided), "init_iv", initial, initial_guided)
    for S in random_order[48:]:
        if len(initial) >= POPULATION_SIZE:
            break
        offer(S, "init_random", initial, initial_guided)
    for S, source in initial.items():
        evaluate(S, source)
    population = survivor_population([], list(valid), valid, rng_survival)
    significant_best = best_j
    stagnant_batches = 0
    restart_remaining = 0
    model = None
    last_fit_batch = -FIT_EVERY_BATCHES
    batch_index = 0
    limit = min(max(BUDGETS), len(universe), trial_limit or len(universe))

    while len(archive) < limit:
        batch_index += 1
        if variant.surrogate and len(valid) >= 64 and (
            model is None or batch_index - last_fit_batch >= FIT_EVERY_BATCHES
        ):
            model = ExtraTreesRegressor(
                n_estimators=32, min_samples_leaf=2, max_features=0.8,
                random_state=seed * 100_000 + batch_index, n_jobs=1,
            )
            ordered = sorted(valid)
            t = time.monotonic()
            model.fit(np.asarray([features(S) for S in ordered], dtype=np.float32),
                      np.asarray([valid[S] for S in ordered]))
            stats["pared_sustituto_s"] += time.monotonic() - t
            stats["ajustes_sustituto"] += 1
            last_fit_batch = batch_index

        bag = {}
        genetic, immigrants, local = [], [], []
        gen_attempts = 0
        while len(genetic) < POOL_GENETIC and gen_attempts < 384 and population:
            gen_attempts += 1
            first = select_parent(population, valid, rng_gen)
            if len(population) > 1 and rng_gen.random() < 0.6:
                second = select_parent(population, valid, rng_gen, first)
                child, repaired = crossover(first, second, rng_gen, variant.repair)
                if repaired:
                    stats["reparaciones"] += 1
                if child is not None and rng_gen.random() < 0.8:
                    child = mutate(child, rng_gen, variant.doubles)
                source = "cruce"
            else:
                child = mutate(first, rng_gen, variant.doubles)
                source = "mutacion"
            offer(child, source, bag, genetic)

        remaining = [S for S in universe if S not in archive and S not in bag]
        for S in rng_random.sample(remaining, min(POOL_RANDOM, len(remaining))):
            offer(S, "inmigrante", bag, immigrants)

        if variant.local and population:
            parents = sorted(population, key=lambda S: (valid[S], S), reverse=True)[:8]
            targets = ((1, 8), (2, 8)) if variant.doubles else ((1, 16),)
            local_attempts = 0
            for dist, target in targets:
                found = 0
                for parent in parents:
                    neighbors = near[parent][dist - 1].copy()
                    rng_local.shuffle(neighbors)
                    for S in neighbors:
                        if found >= target or local_attempts >= 128:
                            break
                        local_attempts += 1
                        if offer(S, f"local_{dist}", bag, local):
                            found += 1
                    if found >= target or local_attempts >= 128:
                        break

        pred_map = {}
        if model is not None and bag:
            bag_order = list(bag)
            X = np.asarray([features(S) for S in bag_order], dtype=np.float32)
            tree_preds = np.asarray([tree.predict(X) for tree in model.estimators_])
            means, stds = tree_preds.mean(axis=0), tree_preds.std(axis=0)
            pred_map = {S: (float(means[i]), float(stds[i])) for i, S in enumerate(bag_order)}

        selected = []

        def take(candidates, count):
            for S in candidates:
                if len(selected) >= BATCH_SIZE or count <= 0:
                    break
                if S not in selected:
                    selected.append(S)
                    count -= 1

        random_quota = max(3, min(8, restart_remaining))
        if model is not None:
            take(sorted(genetic, key=lambda S: (pred_map[S][0], S), reverse=True), 6)
            take(sorted(genetic, key=lambda S: (sum(pred_map[S]), S), reverse=True), 3)
        else:
            take(genetic, 9)
        if not variant.local:
            take(genetic, 4)
        take(immigrants, random_quota)
        if variant.local:
            take(local, 4)
        leftovers = [S for S in bag if S not in selected]
        rng_gen.shuffle(leftovers)
        take(leftovers, BATCH_SIZE - len(selected))
        if len(selected) < BATCH_SIZE:
            fallback = [S for S in universe if S not in archive and S not in bag]
            rng_random.shuffle(fallback)
            for S in fallback[:BATCH_SIZE - len(selected)]:
                offer(S, "respaldo_aleatorio", bag, selected)
        selected = selected[:min(BATCH_SIZE, limit - len(archive))]
        if not selected:
            raise RuntimeError("Quedan conjuntos sin evaluar, pero ningun canal produjo candidatos")

        new_valid = []
        for S in selected:
            if S not in pred_map and model is not None:
                X = np.asarray([features(S)], dtype=np.float32)
                tree_preds = np.asarray([tree.predict(X)[0] for tree in model.estimators_])
                pred_map[S] = (float(tree_preds.mean()), float(tree_preds.std()))
            source = bag[S]
            J = evaluate(S, source, pred_map.get(S))
            if J is not None:
                new_valid.append(S)
            if restart_remaining and source in ("inmigrante", "respaldo_aleatorio"):
                restart_remaining -= 1
        population = survivor_population(population, new_valid, valid, rng_survival)

        if significant_best is None or (best_j is not None and best_j >= significant_best + 0.0005):
            significant_best, stagnant_batches = best_j, 0
        else:
            stagnant_batches += 1
        if variant.restart and stagnant_batches >= 10:
            elite = sorted(valid, key=lambda S: (valid[S], S), reverse=True)[:ELITES]
            non_elite = [S for S in population if S not in elite]
            keep = rng_survival.sample(non_elite, len(non_elite) // 2)
            population = elite + keep
            restart_remaining = 32
            stagnant_batches = 0
            stats["reinicios"] += 1

    for budget in BUDGETS:
        if budget > len(archive):
            snapshot(budget)
    print(f"  {variant.name} s={seed} fin | mejor={best_j:.6f} "
          f"trials={len(archive)} propuestas={stats['propuestas']} "
          f"sustituto={stats['ajustes_sustituto']}", flush=True)
    return summaries


def manifest_payload():
    phase2 = json.loads(FINGERPRINT_PATH.read_text())
    source = Path(__file__).read_bytes()
    payload = {
        "phase2_sha256": phase2["sha256"],
        "fase3_codigo_sha256": hashlib.sha256(source).hexdigest(),
        "variantes": [variant.__dict__ for variant in VARIANTES],
        "semillas": SEEDS,
        "presupuestos": BUDGETS,
        "lote": BATCH_SIZE,
        "poblacion": POPULATION_SIZE,
        "elite": ELITES,
        "pool": {"genetico": POOL_GENETIC, "aleatorio": POOL_RANDOM, "local": POOL_LOCAL},
        "sustituto": {"arboles": 32, "min_leaf": 2, "max_features": 0.8,
                       "cada_lotes": FIT_EVERY_BATCHES, "min_factibles": 64},
        "versiones": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "optbinning", "polars", "plotly")
        },
    }
    return json.loads(json.dumps(payload))


def ejecutar_benchmark():
    universe = catalogo()
    oracle = leer_oraculo(universe)
    optimum = max(float(row["J"]) for row in oracle.values() if row["J"])
    payload = manifest_payload()
    if MANIFEST_PATH.exists():
        if json.loads(MANIFEST_PATH.read_text()) != payload:
            raise RuntimeError("La fase 3 existente usa otro codigo o configuracion; resguardar los resultados antes de repetir")
    else:
        if SUMMARY_PATH.exists():
            raise RuntimeError("Hay un resumen de fase 3 sin manifiesto compatible")
        MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    iv, precompute_time = iv_por_variable()
    near = vecindarios(universe)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open(newline="") as f:
            existing = list(csv.DictReader(f))
    else:
        existing = []
        with SUMMARY_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, lineterminator="\n").writeheader()
    complete = {(r["variante"], int(r["semilla"])) for r in existing if int(r["presupuesto"]) == 2000}
    total = len(VARIANTES) * len(SEEDS)
    finished = len(complete)
    started = time.monotonic()
    with SUMMARY_PATH.open("a", newline="") as summary_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS,
                                        lineterminator="\n")
        for variant in VARIANTES:
            for seed in SEEDS:
                if (variant.name, seed) in complete:
                    continue
                run_path = RUNS_DIR / f"{variant.name}_{seed:02d}.csv"
                print(f"[{finished + 1}/{total}] {variant.name} semilla={seed} "
                      f"mejor global J*={optimum:.6f}", flush=True)
                with run_path.open("w", newline="") as f:
                    trial_writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS,
                                                  lineterminator="\n")
                    trial_writer.writeheader()
                    summaries = run_variant(variant, seed, universe, oracle, near, iv,
                                            trial_writer, f, optimum)
                summary_writer.writerows(summaries)
                summary_file.flush()
                finished += 1
                elapsed = time.monotonic() - started
                eta = elapsed / max(finished - len(complete), 1) * (total - finished)
                print(f"Corridas completas {finished}/{total} | "
                      f"ETA algoritmo={eta / 60:.1f} min | "
                      f"precompute IV={precompute_time:.1f}s", flush=True)
    print(f"Benchmark listo: {SUMMARY_PATH}", flush=True)


def calcular_metricas():
    with SUMMARY_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (ROOT / "outputs/fase2_resumen.csv").open(newline="") as f:
        baseline = list(csv.DictReader(f))
    oracle = leer_oraculo(catalogo())
    optimum = max(float(row["J"]) for row in oracle.values() if row["J"])
    metrics = []
    methods = ("aleatorio", "local", *(v.name for v in VARIANTES))
    for method in methods:
        source = baseline if method in ("aleatorio", "local") else rows
        key = "metodo" if source is baseline else "variante"
        for budget in BUDGETS:
            selected = [r for r in source if r[key] == method and int(r["presupuesto"]) == budget]
            if not selected:
                raise RuntimeError(f"Faltan resultados para {method} B={budget}")
            gaps = [optimum - float(r["mejor_J"]) for r in selected if r["mejor_J"]]
            result = {
                "metodo": method, "presupuesto": budget, "corridas": len(selected),
                "trials_reales_min": min(int(r["trials_reales"]) for r in selected),
                "trials_reales_max": max(int(r["trials_reales"]) for r in selected),
                "sin_solucion": len(selected) - len(gaps),
                "brecha_media": "" if not gaps else repr(statistics.mean(gaps)),
                "brecha_mediana": "" if not gaps else repr(statistics.median(gaps)),
                "exitos_optimo": sum(g <= 1e-12 for g in gaps),
                "exitos_005": sum(g <= 0.005 for g in gaps),
                "exitos_002": sum(g <= 0.002 for g in gaps),
                "exitos_001": sum(g <= 0.001 for g in gaps),
                "propuestas_media": repr(statistics.mean(int(r[
                    "propuestas_totales" if source is baseline else "propuestas"
                ]) for r in selected)),
                "pared_modelo_media_s": repr(statistics.mean(float(r["pared_modelo_s"])
                                                              for r in selected)),
                "pared_algoritmo_media_s": repr(statistics.mean(float(r[
                    "pared_ejecucion_s" if source is baseline else "pared_algoritmo_s"
                ]) for r in selected)),
                "T_optimo_mediana_exitos": "",
                "mae_prequential_media": "",
                "rho_prequential_mediana": "",
            }
            if source is rows:
                hits = [int(r["T_optimo"]) for r in selected if r["T_optimo"]]
                maes = [float(r["mae_prequential"]) for r in selected if r["mae_prequential"]]
                rhos = [float(r["rho_prequential"]) for r in selected if r["rho_prequential"]]
                result["T_optimo_mediana_exitos"] = "" if not hits else repr(statistics.median(hits))
                result["mae_prequential_media"] = "" if not maes else repr(statistics.mean(maes))
                result["rho_prequential_mediana"] = "" if not rhos else repr(statistics.median(rhos))
            metrics.append(result)
    with METRICS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def referencia_forward():
    """Forward greedy de fase 2 como referencia descriptiva.

    Comparte subespacio, evaluador y folds con esta fase, asi que su J
    es comparable. No tiene presupuesto por trials: linea horizontal.
    Devuelve (brecha_en_milesimas, J, ensayos).
    """
    with (ROOT / "outputs/fase2_resumen.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["metodo"] == "forward"]
    if not rows:
        raise RuntimeError("Falta el baseline forward de fase 2")
    J = max(float(r["mejor_J"]) for r in rows if r["mejor_J"])
    ensayos = max(int(r["trials_reales"]) for r in rows)
    oracle = leer_oraculo(catalogo())
    optimum = max(float(row["J"]) for row in oracle.values() if row["J"])
    return 1000 * (optimum - J), J, ensayos


def graficar(metrics):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors = {
        "aleatorio": "#b65b38", "local": "#475c73", "ga_basico": "#ad8b22",
        "ga_local": "#667c2b", "sef_completo": "#087f70",
    }
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Brecha media al oraculo", "Corridas que encuentran el optimo"))
    for method in colors:
        series = [r for r in metrics if r["metodo"] == method]
        x = [r["trials_reales_max"] for r in series]
        gap = [1000 * float(r["brecha_media"]) for r in series]
        success = [r["exitos_optimo"] / r["corridas"] for r in series]
        for col, y in ((1, gap), (2, success)):
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers", name=method, legendgroup=method,
                showlegend=col == 1, line={"color": colors[method], "shape": "hv"},
                marker={"size": 7},
            ), row=1, col=col)
    fig.update_xaxes(title_text="Trials unicos evaluados", tickvals=[100, 250, 500, 1000, 1792])
    fig.update_yaxes(title_text="Brecha (milesimas de AUC)", rangemode="tozero", row=1, col=1)
    brecha_ref, _, ensayos_ref = referencia_forward()
    fig.add_trace(go.Scatter(
        x=[100, 1792], y=[brecha_ref, brecha_ref], mode="lines",
        name=f"forward (referencia, {ensayos_ref} ensayos)",
        line={"color": "#888888", "dash": "dash", "width": 2},
    ), row=1, col=1)
    fig.update_yaxes(title_text="Fraccion de corridas", range=[0, 1.05], row=1, col=2)
    fig.update_layout(width=1400, height=570, template="plotly_white",
                      margin={"l": 75, "r": 35, "t": 70, "b": 90},
                      legend={"orientation": "h", "y": -0.23, "x": 0.5, "xanchor": "center"})
    fig.write_image(FIGURE_PATH, scale=2)

    ablations = [v.name for v in VARIANTES]
    budgets = (100, 250, 500, 1000)
    values, labels = [], []
    for method in ablations:
        series = [next(r for r in metrics if r["metodo"] == method and r["presupuesto"] == b)
                  for b in budgets]
        values.append([r["exitos_optimo"] / r["corridas"] for r in series])
        labels.append([f"{r['exitos_optimo']}/{r['corridas']}" for r in series])
    heat = go.Figure(go.Heatmap(
        z=values, x=[str(b) for b in budgets], y=ablations,
        text=labels, texttemplate="%{text}",
        colorscale="Viridis", zmin=0, zmax=1,
        colorbar={"title": "Fraccion"},
    ))
    heat.update_layout(width=920, height=550, template="plotly_white",
                       xaxis_title="Presupuesto de trials unicos",
                       yaxis={"autorange": "reversed"},
                       margin={"l": 160, "r": 70, "t": 35, "b": 70})
    heat.write_image(ABLATION_FIGURE_PATH, scale=2)
    print(f"Metricas: {METRICS_PATH}; figuras: {FIGURE_PATH}, "
          f"{ABLATION_FIGURE_PATH}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-graficos", action="store_true")
    args = parser.parse_args()
    if not args.solo_graficos:
        ejecutar_benchmark()
    graficar(calcular_metricas())


if __name__ == "__main__":
    main()
