"""Full-catalogue search on train CV; the reserved test is never read."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import resource
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from sef_exp import evaluador as evaluador_mod
from sef_exp.espacio import Espacio
from sef_exp.evaluador import (
    BIN_CFG, CORR_MAX, IV_MIN, K_VARS, SEED_FOLDS, VIF_MAX,
    evaluar, hacer_folds, precomputar, precomputar_woe,
)
from sef_exp.familias import FAMILIAS_COMPLETAS
from sef_exp.fase2 import BUDGETS, ROOT, SEEDS, cargar_train
from sef_exp.fase3 import (
    BATCH_SIZE, ELITES, FIT_EVERY_BATCHES, POOL_GENETIC, POOL_LOCAL,
    POOL_RANDOM, POPULATION_SIZE, select_parent, survivor_population,
)


@dataclass
class Config:
    nombre: str
    familias: tuple
    k: int = K_VARS
    iv_min: float = IV_MIN
    corr_max: float = CORR_MAX
    vif_max: float = VIF_MAX
    datos: tuple = ()
    codigo: tuple = ()
    cargar: object = None
    mensaje_catalogo: str = ""
    granular_desde: int = 50
    presupuestos: tuple = BUDGETS
    epsilon: float = 0.001
    etiqueta_y: str = "AUC medio de cinco folds"
    titulo_j: str = "J mediana en CV"
    exogenas: tuple = ()
    lags: tuple = (0,)
    p_max: int = 3
    q_max: int = 2
    evaluar_set: object = None
    preparar_fn: object = None
    bits_fn: object = None
    archivo_evaluador: object = None
    detalle_manifiesto: dict = field(default_factory=dict)
    espacio: Espacio = field(init=False)

    def __post_init__(self):
        self.espacio = Espacio(self.familias, self.k)
        if self.archivo_evaluador is None:
            self.archivo_evaluador = ROOT / "src/sef_exp/evaluador.py"

    def work(self, name):
        return ROOT / "work" / f"{self.nombre}_{name}"

    def output(self, name):
        return ROOT / "outputs" / f"{self.nombre}_{name}"

    @property
    def cache_path(self):
        return self.work("evaluaciones.csv")

    @property
    def cache_fingerprint(self):
        return self.work("evaluaciones_firma.json")

    @property
    def runs_dir(self):
        return self.work("runs")

    @property
    def summary_path(self):
        return self.output("resumen.csv")

    @property
    def metrics_path(self):
        return self.output("metricas.csv")

    @property
    def stability_path(self):
        return self.output("estabilidad.csv")

    @property
    def figure_path(self):
        return self.output("curvas.png")

    @property
    def manifest_path(self):
        return self.output("manifest.json")

    @property
    def pilot_path(self):
        return self.work("piloto_resumen.csv")

    @property
    def referencias_path(self):
        return self.output("referencias.csv")


CFG = Config(
    nombre="fase3b",
    familias=FAMILIAS_COMPLETAS,
    datos=(ROOT / "data/X_train.csv", ROOT / "data/y_train.csv"),
    codigo=(
        ROOT / "src/sef_exp/espacio.py",
        ROOT / "src/sef_exp/familias.py",
        ROOT / "src/sef_exp/fase3.py",
        ROOT / "src/sef_exp/fase3b.py",
    ),
    cargar=cargar_train,
    mensaje_catalogo="Precompute train: 35 variables x 5 folds; test reservado sin abrir",
)
SPACE = CFG.espacio
METHODS = ("aleatorio", "local", "ga_local", "sef_completo")
CACHE_PATH = CFG.cache_path
CACHE_FINGERPRINT = CFG.cache_fingerprint
RUNS_DIR = CFG.runs_dir
SUMMARY_PATH = CFG.summary_path
METRICS_PATH = CFG.metrics_path
STABILITY_PATH = CFG.stability_path
FIGURE_PATH = CFG.figure_path
MANIFEST_PATH = CFG.manifest_path
PILOT_PATH = CFG.pilot_path
REFERENCIAS_PATH = CFG.referencias_path
GRANULAR_DESDE = CFG.granular_desde

CACHE_COLUMNS = ("S", "estado", "causa", "J", "aucs", "pared_s")
TRIAL_COLUMNS = (
    "metodo", "semilla", "trial", "propuestas", "duplicados", "S", "fuente",
    "estado", "causa", "J", "mejor_J", "mejor_S", "pred_J", "pared_modelo_s",
)
SUMMARY_COLUMNS = (
    "metodo", "semilla", "presupuesto", "trials_reales", "propuestas",
    "duplicados", "factibles", "rechazados", "causas", "mejor_J", "mejor_S",
    "pared_modelo_s", "pared_ajustes_reales_s", "pared_algoritmo_s",
    "pared_corrida_s", "cpu_algoritmo_s", "cache_hits",
    "ajustes_sustituto", "pared_sustituto_s", "diversidad_variables",
)
REFERENCIA_COLUMNS = (
    "metodo", "J", "S", "ensayos", "pared_modelo_s",
)


def aplicar_umbrales(cfg):
    """Fija los umbrales del evaluador segun cfg; devuelve los previos.

    El evaluador lee estos umbrales como constantes de modulo y su
    archivo no cambia entre fases, asi los fingerprints de fases
    anteriores siguen validos. Los valores usados quedan registrados
    en el manifiesto y la firma de cada fase.
    """
    previos = (evaluador_mod.IV_MIN, evaluador_mod.CORR_MAX, evaluador_mod.VIF_MAX)
    evaluador_mod.IV_MIN = cfg.iv_min
    evaluador_mod.CORR_MAX = cfg.corr_max
    evaluador_mod.VIF_MAX = cfg.vif_max
    return previos


def restaurar_umbrales(previos):
    evaluador_mod.IV_MIN, evaluador_mod.CORR_MAX, evaluador_mod.VIF_MAX = previos


def sha_archivos(paths):
    h = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def firma_evaluador(cfg=None):
    cfg = cfg or CFG
    firma = {
        "datos_y_evaluador_sha256": sha_archivos((
            *cfg.datos, cfg.archivo_evaluador,
        )),
        "fold_seed": SEED_FOLDS, "k": cfg.k, "iv_min": cfg.iv_min,
        "corr_max": cfg.corr_max, "vif_max": cfg.vif_max, "bin_cfg": BIN_CFG,
    }
    firma.update(cfg.detalle_manifiesto)
    return firma


def manifiesto(cfg=None):
    cfg = cfg or CFG
    return {
        "evaluador": firma_evaluador(cfg),
        "familias": cfg.espacio.familias,
        "espacio_estructural": cfg.espacio.total,
        "metodos": METHODS,
        "semillas": SEEDS,
        "presupuestos": cfg.presupuestos,
        "codigo_sha256": sha_archivos(cfg.codigo),
        "versiones": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "scikit-learn", "optbinning", "polars", "plotly",
        )},
    }


def preparar(cfg=None):
    cfg = cfg or CFG
    espacio = cfg.espacio
    print(cfg.mensaje_catalogo, flush=True)
    x, y = cfg.cargar()
    if set(x.columns) != set(espacio.variables):
        raise RuntimeError("El catalogo de familias no coincide con las columnas train")
    folds = hacer_folds(y)
    start = time.monotonic()

    def progreso(r, j, n, var):
        if (j + 1) % 5 == 0 or j + 1 == n:
            print(f"  binning fold {r + 1}/5 variable {j + 1}/{n} {var} "
                  f"{time.monotonic() - start:.1f}s", flush=True)

    pre = precomputar(x, y, folds, espacio.variables, log=progreso)
    woe = precomputar_woe(x, folds, pre, espacio.variables)
    iv = {v: statistics.mean(pre[(r, v)]["iv"] for r in range(len(folds)))
          for v in espacio.variables}
    elapsed = time.monotonic() - start
    print(f"Precompute completo en {elapsed:.1f}s", flush=True)
    return (x, y, folds, pre, woe), iv, elapsed


class Evaluaciones:
    def __init__(self, path=None, fingerprint_path=None, cfg=None):
        cfg = cfg or CFG
        self.cfg = cfg
        self.path = Path(path or cfg.cache_path)
        self.fingerprint_path = Path(fingerprint_path or cfg.cache_fingerprint)
        fingerprint = firma_evaluador(cfg)
        if self.path.exists():
            if not self.fingerprint_path.exists() or json.loads(
                self.fingerprint_path.read_text()
            ) != fingerprint:
                raise RuntimeError("Cache de evaluaciones incompatible; resguardar antes de repetir")
            with self.path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=CACHE_COLUMNS, lineterminator="\n").writeheader()
            rows = []
        self.rows = {tuple(row["S"].split("|")): row for row in rows}
        if len(self.rows) != len(rows):
            raise RuntimeError("Cache con conjuntos repetidos")
        self.fits = 0
        self.hits = 0

    def consulta(self, S, context):
        if S in self.rows:
            self.hits += 1
            return self.rows[S], True
        t = time.monotonic()
        evaluar_fn = self.cfg.evaluar_set or evaluar_por_defecto
        result = evaluar_fn(S, context, self.cfg)
        row = {
            "S": "|".join(S), "estado": result["estado"], "causa": result["causa"],
            "J": "" if result["J"] is None else repr(result["J"]),
            "aucs": json.dumps(result["aucs"]),
            "pared_s": f"{time.monotonic() - t:.6f}",
        }
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_COLUMNS, lineterminator="\n")
            writer.writerow(row)
        self.rows[S] = row
        self.fits += 1
        return row, False


def bits(S, variables):
    selected = set(S)
    return [int(v in selected) for v in variables]


def evaluar_por_defecto(S, context, cfg):
    x, y, folds, pre, woe = context
    return evaluar(S, x, y, folds, pre, woe, k=cfg.k)


def vector_por_defecto(S, cfg):
    return bits(S, cfg.espacio.variables)


def proponer(vecinos, vistos, rng, limit):
    candidates = list(dict.fromkeys(v for v in vecinos if v not in vistos))
    rng.shuffle(candidates)
    return candidates[:limit]


def run_method(method, seed, limit, context, iv, cache, writer, global_best,
               cfg=None):
    cfg = cfg or CFG
    espacio = cfg.espacio
    vector = cfg.bits_fn or vector_por_defecto
    rng_init = random.Random(seed + 10_000)
    rng_guided = random.Random(seed + 15_000)
    rng_gen = random.Random(seed + 20_000)
    rng_random = random.Random(seed + 30_000)
    rng_local = random.Random(seed + 40_000)
    rng_survival = random.Random(seed + 50_000)
    archive, valid = {}, {}
    population = []
    best_j, best_s = None, None
    attempts = duplicates = rejected = cache_hits = fits_surrogate = 0
    model_seconds = actual_fit_seconds = surrogate_seconds = 0.0
    causes = Counter()
    summaries = []
    start_wall, start_cpu = time.monotonic(), time.process_time()

    def snapshot(budget, last=""):
        elapsed = time.monotonic() - start_wall
        row = {
            "metodo": method, "semilla": seed, "presupuesto": budget,
            "trials_reales": len(archive), "propuestas": attempts,
            "duplicados": duplicates, "factibles": len(valid),
            "rechazados": rejected, "causas": json.dumps(causes, sort_keys=True),
            "mejor_J": "" if best_j is None else repr(best_j),
            "mejor_S": "" if best_s is None else "|".join(best_s),
            "pared_modelo_s": f"{model_seconds:.6f}",
            "pared_ajustes_reales_s": f"{actual_fit_seconds:.6f}",
            "pared_algoritmo_s": f"{max(0.0, elapsed - actual_fit_seconds):.6f}",
            "pared_corrida_s": f"{elapsed:.6f}",
            "cpu_algoritmo_s": f"{time.process_time() - start_cpu:.6f}",
            "cache_hits": cache_hits, "ajustes_sustituto": fits_surrogate,
            "pared_sustituto_s": f"{surrogate_seconds:.6f}",
            "diversidad_variables": "" if not population else espacio.diversidad(population),
        }
        if len(archive) in cfg.presupuestos or len(archive) == limit:
            summaries.append(row)
        eta = elapsed * (limit - len(archive)) / max(len(archive), 1)
        print(f"  {method} s={seed} {len(archive)}/{limit} ultimo={last} "
              f"mejor={best_j} S={'|'.join(best_s) if best_s else '-'} "
              f"global={global_best['J']} propuestas={attempts} "
              f"rechazos={rejected} causas={dict(causes)} "
              f"ETA={eta / 60:.1f}m", flush=True)

    def evaluate_candidate(S, source, prediction=None):
        nonlocal best_j, best_s, rejected, model_seconds, actual_fit_seconds, cache_hits
        if S in archive:
            raise AssertionError("Trial repetido")
        row, hit = cache.consulta(S, context)
        cache_hits += int(hit)
        archive[S] = row
        J = float(row["J"]) if row["J"] else None
        model_seconds += float(row["pared_s"])
        if not hit:
            actual_fit_seconds += float(row["pared_s"])
        last = f"J={J:.6f}" if J is not None else row["causa"]
        if J is None:
            rejected += 1
            causes[row["causa"].split("_")[0]] += 1
        else:
            valid[S] = J
            if best_j is None or J > best_j:
                best_j, best_s = J, S
                print(f"  {method} s={seed} nuevo mejor J={J:.6f} "
                      f"trial={len(archive)} S={'|'.join(S)}", flush=True)
            if global_best["J"] is None or J > global_best["J"]:
                global_best.update(J=J, S=S, metodo=method, semilla=seed)
                print(f"  mejor global observado J={J:.6f} "
                      f"{method} s={seed} S={'|'.join(S)}", flush=True)
        writer.writerow({
            "metodo": method, "semilla": seed, "trial": len(archive),
            "propuestas": attempts, "duplicados": duplicates, "S": "|".join(S),
            "fuente": source, "estado": row["estado"], "causa": row["causa"],
            "J": row["J"], "mejor_J": "" if best_j is None else repr(best_j),
            "mejor_S": "" if best_s is None else "|".join(best_s),
            "pred_J": "" if prediction is None else repr(prediction),
            "pared_modelo_s": f"{model_seconds:.6f}",
        })
        progress_every = 50 if limit <= 100 else 250
        if len(archive) in cfg.presupuestos or len(archive) % progress_every == 0 or len(archive) == limit:
            snapshot(len(archive), last)
        return J

    def offer(S, source, bag):
        nonlocal attempts, duplicates
        attempts += 1
        if not espacio.valido(S):
            raise AssertionError(f"Propuesta estructural invalida: {S}")
        if S in archive or S in bag:
            duplicates += 1
            return False
        bag[S] = source
        return True

    if method == "aleatorio":
        while len(archive) < limit:
            bag = {}
            while not bag:
                offer(espacio.muestra(rng_random), "aleatorio", bag)
            S = next(iter(bag))
            evaluate_candidate(S, bag[S])

    elif method == "local":
        incumbent = None
        while len(archive) < limit:
            if incumbent is None:
                bag = {}
                while not bag:
                    offer(espacio.muestra(rng_random), "reinicio", bag)
                incumbent = next(iter(bag))
                evaluate_candidate(incumbent, bag[incumbent])
                if incumbent not in valid:
                    incumbent = None
                continue
            base = valid[incumbent]
            neighbors = proponer(espacio.vecinos_simples(incumbent), archive, rng_local, 256)
            candidates = []
            for S in neighbors:
                if len(archive) >= limit:
                    break
                bag = {}
                if offer(S, "local_1", bag):
                    evaluate_candidate(S, "local_1")
                    if S in valid:
                        candidates.append(S)
            improving = [S for S in candidates if valid[S] > base]
            if not improving and len(archive) < limit:
                doubles = []
                for _ in range(512):
                    S = espacio.muta(incumbent, rng_local, dobles=True)
                    if len(set(incumbent) - set(S)) == 2:
                        doubles.append(S)
                for S in proponer(doubles, archive, rng_local, 64):
                    if len(archive) >= limit:
                        break
                    bag = {}
                    if offer(S, "local_2", bag):
                        evaluate_candidate(S, "local_2")
                        if S in valid:
                            candidates.append(S)
                improving = [S for S in candidates if valid[S] > base]
            incumbent = max(improving, key=lambda S: (valid[S], S)) if improving else None

    elif method in ("ga_local", "sef_completo"):
        initial = {}
        while len(initial) < 48:
            offer(espacio.muestra(rng_init), "init_random", initial)
        for _ in range(2048):
            if len(initial) >= POPULATION_SIZE:
                break
            offer(espacio.guiada(iv, rng_guided), "init_iv", initial)
        while len(initial) < POPULATION_SIZE:
            offer(espacio.muestra(rng_init), "init_random", initial)
        for S, source in initial.items():
            if len(archive) >= limit:
                break
            evaluate_candidate(S, source)
        population = survivor_population([], list(valid), valid, rng_survival)
        model = None
        last_fit_batch = -FIT_EVERY_BATCHES
        batch_index = stagnant = immigrants_left = 0
        significant_best = best_j
        while len(archive) < limit:
            batch_index += 1
            if method == "sef_completo" and len(valid) >= 64 and (
                model is None or batch_index - last_fit_batch >= FIT_EVERY_BATCHES
            ):
                t = time.monotonic()
                model = ExtraTreesRegressor(
                    n_estimators=32, min_samples_leaf=2, max_features=0.8,
                    random_state=seed * 100_000 + batch_index, n_jobs=1,
                )
                ordered = sorted(valid)
                model.fit(np.asarray([vector(S, cfg) for S in ordered], dtype=np.float32),
                          np.asarray([valid[S] for S in ordered]))
                surrogate_seconds += time.monotonic() - t
                fits_surrogate += 1
                last_fit_batch = batch_index

            bag = {}
            genetic, immigrants, local = [], [], []
            tries = 0
            while len(genetic) < POOL_GENETIC and tries < 384 and population:
                tries += 1
                a = select_parent(population, valid, rng_gen)
                if len(population) > 1 and rng_gen.random() < 0.6:
                    b = select_parent(population, valid, rng_gen, a)
                    child, _ = espacio.cruza(a, b, rng_gen)
                    if rng_gen.random() < 0.8:
                        child = espacio.muta(child, rng_gen)
                    source = "cruce"
                else:
                    child = espacio.muta(a, rng_gen)
                    source = "mutacion"
                if offer(child, source, bag):
                    genetic.append(child)
            for _ in range(POOL_RANDOM):
                for _ in range(100):
                    S = espacio.muestra(rng_random)
                    if offer(S, "inmigrante", bag):
                        immigrants.append(S)
                        break
            if population:
                for parent in sorted(population, key=lambda S: (valid[S], S),
                                     reverse=True)[:ELITES]:
                    for S in proponer(espacio.vecinos_simples(parent),
                                      set(archive) | set(bag), rng_local, 4):
                        if len(local) >= POOL_LOCAL:
                            break
                        if offer(S, "local_1", bag):
                            local.append(S)
                    if len(local) >= POOL_LOCAL:
                        break
                tries = 0
                while len(local) < POOL_LOCAL and tries < 128:
                    tries += 1
                    parent = rng_local.choice(population)
                    S = espacio.muta(parent, rng_local)
                    if offer(S, "local_2", bag):
                        local.append(S)

            predictions = {}
            if model is not None and bag:
                ordered = list(bag)
                X = np.asarray([vector(S, cfg) for S in ordered], dtype=np.float32)
                trees = np.asarray([tree.predict(X) for tree in model.estimators_])
                means, stds = trees.mean(axis=0), trees.std(axis=0)
                predictions = {S: (float(means[i]), float(stds[i]))
                               for i, S in enumerate(ordered)}
            selected = []

            def take(candidates, count):
                for S in candidates:
                    if len(selected) >= BATCH_SIZE or count <= 0:
                        break
                    if S not in selected:
                        selected.append(S)
                        count -= 1

            if model is None:
                take(genetic, 9)
            else:
                take(sorted(genetic, key=lambda S: (predictions[S][0], S),
                            reverse=True), 6)
                take(sorted(genetic, key=lambda S: (sum(predictions[S]), S),
                            reverse=True), 3)
            take(immigrants, max(3, min(8, immigrants_left)))
            take(local, 4)
            leftovers = [S for S in bag if S not in selected]
            rng_gen.shuffle(leftovers)
            take(leftovers, BATCH_SIZE - len(selected))
            while len(selected) < min(BATCH_SIZE, limit - len(archive)):
                S = espacio.muestra(rng_random)
                if offer(S, "respaldo", bag):
                    selected.append(S)
            new_valid = []
            for S in selected[:limit - len(archive)]:
                J = evaluate_candidate(S, bag[S], predictions.get(S, (None,))[0])
                if J is not None:
                    new_valid.append(S)
                if immigrants_left and bag[S] in ("inmigrante", "respaldo"):
                    immigrants_left -= 1
            population = survivor_population(population, new_valid, valid, rng_survival)
            if significant_best is None or (
                best_j is not None and best_j >= significant_best + 0.0005
            ):
                significant_best, stagnant = best_j, 0
            else:
                stagnant += 1
            if stagnant >= 10:
                elite = sorted(valid, key=lambda S: (valid[S], S), reverse=True)[:ELITES]
                other = [S for S in population if S not in elite]
                population = elite + rng_survival.sample(other, len(other) // 2)
                immigrants_left = 32
                stagnant = 0
    else:
        raise ValueError(method)

    return summaries


def forward_greedy(evaluar_tamano, cfg=None):
    """Forward greedy determinista sobre el espacio de cfg.

    evaluar_tamano(S, k) devuelve J o None. No usa presupuesto por
    trials: es una referencia descriptiva fuera de la comparacion.
    """
    espacio = (cfg or CFG).espacio
    chosen, usadas, ensayos = (), set(), 0
    for size in range(1, espacio.k + 1):
        opciones = []
        for name, choices in espacio.familias:
            if name in usadas:
                continue
            for v in choices:
                S = tuple(sorted((*chosen, v)))
                J = evaluar_tamano(S, size)
                ensayos += 1
                opciones.append((J, S, name))
        validas = [item for item in opciones if item[0] is not None]
        if not validas:
            print(f"Forward bloqueado en tamano {size - 1}", flush=True)
            return None, None, ensayos
        J, chosen, name = max(validas, key=lambda item: (item[0], item[1]))
        usadas.add(name)
    return J, chosen, ensayos


def calcular_forward(cfg=None):
    cfg = cfg or CFG
    espacio = cfg.espacio
    previos = aplicar_umbrales(cfg)
    try:
        return _calcular_forward(cfg, espacio)
    finally:
        restaurar_umbrales(previos)


def _calcular_forward(cfg, espacio):
    context, _, _ = preparar(cfg)
    cache = Evaluaciones(cfg=cfg)
    x, y, folds, pre, woe = context
    model_seconds = 0.0

    def evaluar_tamano(S, k):
        nonlocal model_seconds
        if k == espacio.k:
            row, _ = cache.consulta(S, context)
            J = float(row["J"]) if row["J"] else None
            model_seconds += float(row["pared_s"])
            return J
        t = time.monotonic()
        result = evaluar(S, x, y, folds, pre, woe, k=k)
        model_seconds += time.monotonic() - t
        return result["J"]

    J, S, ensayos = forward_greedy(evaluar_tamano, cfg)
    with cfg.referencias_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REFERENCIA_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "metodo": "forward", "J": "" if J is None else repr(J),
            "S": "" if S is None else "|".join(S),
            "ensayos": ensayos, "pared_modelo_s": f"{model_seconds:.6f}",
        })
    print(f"Forward referencia: J={J} S={'|'.join(S) if S else '-'} "
          f"ensayos={ensayos} modelo={model_seconds:.1f}s -> {cfg.referencias_path}",
          flush=True)
    return J


def leer_referencia(cfg=None):
    cfg = cfg or CFG
    if not cfg.referencias_path.exists():
        return None, None
    with cfg.referencias_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["J"]:
                return row["metodo"], float(row["J"])
    return None, None


def metricas(cfg=None):
    cfg = cfg or CFG
    espacio = cfg.espacio
    with cfg.summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(METHODS) * len(SEEDS) * len(cfg.presupuestos):
        raise RuntimeError(f"Faltan corridas o presupuestos en {cfg.nombre}")
    reference = max(float(r["mejor_J"]) for r in rows if r["mejor_J"])
    result, stability = [], []
    for method in METHODS:
        for budget in cfg.presupuestos:
            selected = [r for r in rows if r["metodo"] == method
                        and int(r["presupuesto"]) == budget]
            if len(selected) != len(SEEDS):
                raise RuntimeError(f"Faltan semillas para {method} B={budget}")
            js = [float(r["mejor_J"]) for r in selected if r["mejor_J"]]
            sets = [set(r["mejor_S"].split("|")) for r in selected if r["mejor_S"]]
            overlaps = [len(a & b) / len(a | b) for i, a in enumerate(sets)
                        for b in sets[i + 1:]]
            variable_counts = Counter(v for S in sets for v in S)
            family_counts = Counter(espacio.familia_de[v] for S in sets for v in S)
            for kind, names, counts in (
                ("variable", espacio.variables, variable_counts),
                ("familia", (name for name, _ in espacio.familias), family_counts),
            ):
                for name in names:
                    stability.append({
                        "metodo": method, "presupuesto": budget, "tipo": kind,
                        "nombre": name, "selecciones": counts[name],
                        "frecuencia": repr(counts[name] / len(selected)),
                    })
            result.append({
                "metodo": method, "presupuesto": budget,
                "corridas": len(selected), "sin_solucion": len(selected) - len(js),
                "J_media": "" if not js else repr(statistics.mean(js)),
                "J_mediana": "" if not js else repr(statistics.median(js)),
                "brecha_al_mejor_observado_media": "" if not js else repr(
                    statistics.mean(reference - j for j in js)),
                "dentro_001": sum(reference - j <= cfg.epsilon + 1e-12 for j in js),
                "mejor_observado": repr(reference),
                "jaccard_medio": "" if not overlaps else repr(statistics.mean(overlaps)),
                "propuestas_media": repr(statistics.mean(int(r["propuestas"])
                                                     for r in selected)),
                "rechazos_media": repr(statistics.mean(int(r["rechazados"])
                                                    for r in selected)),
                "modelo_estimado_media_s": repr(statistics.mean(float(r["pared_modelo_s"])
                                                             for r in selected)),
                "pared_algoritmo_media_s": repr(statistics.mean(float(r["pared_algoritmo_s"])
                                                              for r in selected)),
            })
    with cfg.metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    with cfg.stability_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(stability[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(stability)
    return result


def curvas_granulares(cfg=None):
    """Mediana del mejor J por trial sobre las trazas de work/.

    Empieza en cfg.granular_desde porque antes ninguna semilla tiene
    cobertura completa de factibles y la mediana es ruido.
    """
    cfg = cfg or CFG
    import polars as pl

    frames = []
    for path in sorted(cfg.runs_dir.glob("*.csv")):
        frames.append(pl.read_csv(
            path, columns=["metodo", "semilla", "trial", "mejor_J"]))
    runs = pl.concat(frames).with_columns(
        pl.col("mejor_J").cast(pl.Float64)).sort("metodo", "semilla", "trial")
    runs = runs.with_columns(
        pl.col("mejor_J").forward_fill().over(["metodo", "semilla"])
    ).filter(pl.col("trial") >= cfg.granular_desde)
    mediana = runs.group_by(["metodo", "trial"]).agg(
        pl.col("mejor_J").median().alias("j_mediana"))
    return mediana.sort("metodo", "trial")


def graficar(rows, cfg=None):
    cfg = cfg or CFG
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    colors = {
        "aleatorio": "#b65b38", "local": "#475c73",
        "ga_local": "#667c2b", "sef_completo": "#087f70",
    }
    mediana = curvas_granulares(cfg)
    titulo_dentro = f"A {cfg.epsilon:g} del mejor observado"
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(cfg.titulo_j, titulo_dentro))
    for method in METHODS:
        series = mediana.filter(pl.col("metodo") == method).sort("trial")
        trials = series["trial"].to_list()
        fig.add_trace(go.Scatter(
            x=trials, y=series["j_mediana"].to_list(), mode="lines",
            name=method, legendgroup=method, showlegend=True,
            line={"color": colors[method], "width": 2},
        ), row=1, col=1)
        dentro = [int(r["dentro_001"]) / int(r["corridas"])
                  for r in rows if r["metodo"] == method]
        presupuestos = [int(r["presupuesto"])
                        for r in rows if r["metodo"] == method]
        fig.add_trace(go.Scatter(
            x=presupuestos, y=dentro, mode="lines+markers", name=method,
            legendgroup=method, showlegend=False,
            line={"color": colors[method], "shape": "hv"},
            marker={"size": 7},
        ), row=1, col=2)
    referencia_metodo, referencia = leer_referencia(cfg)
    if referencia is not None:
        fig.add_trace(go.Scatter(
            x=[cfg.granular_desde, max(cfg.presupuestos)], y=[referencia, referencia],
            mode="lines", name=f"{referencia_metodo} (referencia, sin presupuesto)",
            line={"color": "#888888", "dash": "dash", "width": 2},
        ), row=1, col=1)
    fig.update_xaxes(type="log", tickvals=[50, 100, 500, 1000, 2000],
                     ticktext=["50", "100", "500", "1000", "2000"],
                     title_text="Trials unicos (log)", row=1, col=1)
    fig.update_xaxes(title_text="Trials unicos", tickvals=list(cfg.presupuestos), row=1, col=2)
    fig.update_yaxes(title_text=cfg.etiqueta_y, row=1, col=1)
    fig.update_yaxes(title_text="Fraccion de corridas", range=[0, 1.05], row=1, col=2)
    fig.update_layout(width=1400, height=570, template="plotly_white",
                      margin={"l": 75, "r": 35, "t": 70, "b": 90},
                      legend={"orientation": "h", "y": -0.23, "x": 0.5, "xanchor": "center"})
    fig.write_image(cfg.figure_path, scale=2)
    print(f"Metricas: {cfg.metrics_path}; figura: {cfg.figure_path}", flush=True)


def ejecutar(pilot=False, pilot_trials=100, pilot_seeds=(0,), render=True,
             cfg=None):
    cfg = cfg or CFG
    previos = aplicar_umbrales(cfg)
    try:
        return _ejecutar(pilot, pilot_trials, pilot_seeds, render, cfg)
    finally:
        restaurar_umbrales(previos)


def _ejecutar(pilot, pilot_trials, pilot_seeds, render, cfg):
    if pilot:
        if pilot_trials < 1 or pilot_trials > max(cfg.presupuestos):
            raise ValueError("El piloto supera el presupuesto maximo")
        methods, seeds, limit = METHODS, pilot_seeds, pilot_trials
        summary_path = cfg.pilot_path
        runs_dir = cfg.work("piloto_runs")
    else:
        payload = manifiesto(cfg)
        if cfg.manifest_path.exists():
            if json.loads(cfg.manifest_path.read_text()) != json.loads(json.dumps(payload)):
                raise RuntimeError("Fase 3b existente usa otro codigo o familias; resguardar antes")
        elif cfg.summary_path.exists():
            raise RuntimeError("Resumen de fase 3b sin manifiesto")
        else:
            cfg.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
        methods, seeds, limit = METHODS, SEEDS, min(max(cfg.presupuestos),
                                                    cfg.espacio.total)
        summary_path = cfg.summary_path
        runs_dir = cfg.runs_dir

    context, iv, precompute_seconds = (cfg.preparar_fn or preparar)(cfg)
    cache = Evaluaciones(cfg=cfg)
    runs_dir.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        with summary_path.open(newline="") as f:
            existing = list(csv.DictReader(f))
    else:
        existing = []
        with summary_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, lineterminator="\n").writeheader()
    completed = {(r["metodo"], int(r["semilla"])) for r in existing
                 if int(r["trials_reales"]) == limit}
    global_best = {"J": None, "S": None, "metodo": None, "semilla": None}
    for row in existing:
        if row["mejor_J"] and (
            global_best["J"] is None or float(row["mejor_J"]) > global_best["J"]
        ):
            global_best.update(J=float(row["mejor_J"]),
                               S=tuple(row["mejor_S"].split("|")),
                               metodo=row["metodo"], semilla=int(row["semilla"]))
    total = len(methods) * len(seeds)
    finished = len(completed)
    start = time.monotonic()
    initial_fits, initial_hits = cache.fits, cache.hits
    with summary_path.open("a", newline="") as summary_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS,
                                        lineterminator="\n")
        for method in methods:
            for seed in seeds:
                if (method, seed) in completed:
                    continue
                print(f"[{finished + 1}/{total}] {method} semilla={seed} "
                      f"presupuesto={limit} espacio={cfg.espacio.total}", flush=True)
                path = runs_dir / f"{method}_{seed:02d}.csv"
                with path.open("w", newline="") as trial_file:
                    writer = csv.DictWriter(trial_file, fieldnames=TRIAL_COLUMNS,
                                            lineterminator="\n")
                    writer.writeheader()
                    summaries = run_method(method, seed, limit, context, iv, cache,
                                           writer, global_best, cfg)
                summary_writer.writerows(summaries)
                summary_file.flush()
                finished += 1
                elapsed = time.monotonic() - start
                eta = elapsed / max(finished - len(completed), 1) * (total - finished)
                print(f"Corridas {finished}/{total} | mejor global={global_best['J']} "
                      f"ajustes nuevos={cache.fits - initial_fits} "
                      f"cache_hits={cache.hits - initial_hits} "
                      f"ETA={eta / 60:.1f}m", flush=True)
    print(f"Fin | precompute={precompute_seconds:.1f}s "
          f"corridas_pared={time.monotonic() - start:.1f}s "
          f"ajustes_nuevos={cache.fits - initial_fits} "
          f"cache_hits={cache.hits - initial_hits} "
          f"mejor_observado={global_best['J']} "
          f"RSS_pico_MiB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024):.1f}",
          flush=True)
    if not pilot:
        rows = metricas(cfg)
        if render:
            graficar(rows, cfg)


def main(cfg=None):
    cfg = cfg or CFG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piloto", action="store_true")
    parser.add_argument("--trials", type=int, default=100,
                        help="presupuesto por metodo y semilla del piloto")
    parser.add_argument("--semillas", default="0", help="semillas del piloto separadas por coma")
    parser.add_argument("--solo-graficos", action="store_true")
    parser.add_argument("--sin-graficos", action="store_true")
    parser.add_argument("--forward-ref", action="store_true",
                        help="forward greedy sobre el catalogo completo como "
                             "referencia descriptiva (fuera de la comparacion)")
    args = parser.parse_args()
    if args.forward_ref:
        calcular_forward(cfg)
    elif args.solo_graficos:
        graficar(metricas(cfg), cfg)
    else:
        seeds = tuple(int(x) for x in args.semillas.split(","))
        ejecutar(args.piloto, args.trials, seeds, render=not args.sin_graficos,
                 cfg=cfg)


if __name__ == "__main__":
    main()
