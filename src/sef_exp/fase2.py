"""Small exact oracle and first search baselines for the fixed SEF evaluator."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import polars as pl

from sef_exp.evaluador import (
    BIN_CFG,
    CORR_MAX,
    IV_MIN,
    K_VARS,
    SEED_FOLDS,
    VIF_MAX,
    aplicar_nulos,
    evaluar,
    hacer_folds,
    precomputar,
    precomputar_woe,
)

FAMILIAS = (
    ("scores", ("CVSC100", "CVSC150")),
    ("satisfactorias", ("AM03", "AM26")),
    ("totales", ("AT01", "AT30")),
    ("antiguedad", ("AT93", "IN20")),
    ("consultas", ("GO06", "GO243")),
    ("cv_uso", ("CVPRPAYMNT11", "CVPRTRD")),
    ("revolvente", ("RE03", "BC111")),
    ("cobranzas", ("PR09", "PR95")),
)
BUDGETS = (100, 250, 500, 1000, 2000)
SEEDS = tuple(range(20))
ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = ROOT / "outputs/fase2_oraculo.csv"
TRIALS_PATH = ROOT / "work/fase2_trials.csv"
SUMMARY_PATH = ROOT / "outputs/fase2_resumen.csv"
METRICS_PATH = ROOT / "outputs/fase2_metricas.csv"
FIGURE_PATH = ROOT / "outputs/fase2_curvas.png"
FINGERPRINT_PATH = ROOT / "outputs/fase2_fingerprint.json"
ORACLE_COLUMNS = ("numero", "S", "estado", "causa", "J", "aucs", "pared_s")
TRIAL_COLUMNS = ("metodo", "semilla", "trial", "propuestas_totales", "tipo", "S", "estado", "causa", "J", "mejor_J", "mejor_S", "pared_modelo_s")


def catalogo():
    result = []
    for selected in itertools.combinations(FAMILIAS, K_VARS):
        for choice in itertools.product(*(vars_ for _, vars_ in selected)):
            result.append(tuple(sorted(choice)))
    return sorted(result)


def firma():
    payload = {
        "familias": FAMILIAS,
        "k": K_VARS,
        "seed_folds": SEED_FOLDS,
        "iv_min": IV_MIN,
        "corr_max": CORR_MAX,
        "vif_max": VIF_MAX,
        "bin_cfg": BIN_CFG,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for path in (ROOT / "data/X_train.csv", ROOT / "data/y_train.csv", ROOT / "src/sef_exp/evaluador.py"):
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return {"sha256": h.hexdigest(), "config": json.loads(json.dumps(payload))}


def cargar_train():
    x = pl.read_csv(ROOT / "data/X_train.csv").drop("")
    y = pl.read_csv(ROOT / "data/y_train.csv")["target"].cast(pl.Int64).to_numpy().ravel()
    return aplicar_nulos(x), y


def preparar():
    print("Preparando train y 5 folds; test reservado sin abrir", flush=True)
    x, y = cargar_train()
    folds = hacer_folds(y)
    vars_ = [v for _, choices in FAMILIAS for v in choices]
    start = time.monotonic()

    def progreso(r, j, n, var):
        if j + 1 == n or (j + 1) % 4 == 0:
            print(f"  binning fold {r + 1}/5 variable {j + 1}/{n} {var} "
                  f"{time.monotonic() - start:.1f}s", flush=True)

    pre = precomputar(x, y, folds, vars_, log=progreso)
    print("Calculando WOE una vez por fold y variable", flush=True)
    woe = precomputar_woe(x, folds, pre, vars_)
    print(f"Precompute completo en {time.monotonic() - start:.1f}s", flush=True)
    return x, y, folds, pre, woe


def abrir_oraculo(fingerprint, universe):
    if ORACLE_PATH.exists():
        if not FINGERPRINT_PATH.exists() or json.loads(FINGERPRINT_PATH.read_text()) != fingerprint:
            raise RuntimeError("El oraculo existente usa otra configuracion o datos; resguardar sus archivos antes de repetir")
        with ORACLE_PATH.open(newline="") as f:
            rows = list(csv.DictReader(f))
        for i, row in enumerate(rows):
            if int(row["numero"]) != i + 1 or row["S"] != "|".join(universe[i]):
                raise RuntimeError("Checkpoint de oraculo incompleto o fuera de orden")
        return rows
    ORACLE_PATH.parent.mkdir(exist_ok=True)
    FINGERPRINT_PATH.write_text(json.dumps(fingerprint, indent=2) + "\n")
    with ORACLE_PATH.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=ORACLE_COLUMNS, lineterminator="\n").writeheader()
    return []


def ejecutar_oraculo(universe, fingerprint):
    rows = abrir_oraculo(fingerprint, universe)
    if len(rows) == len(universe):
        best = max((r for r in rows if r["J"]), key=lambda r: float(r["J"]))
        print(f"Oraculo recuperado: {len(rows)}/{len(universe)} | "
              f"factibles={sum(r['estado'] == 'ok' for r in rows)} | "
              f"mejor J={float(best['J']):.6f} S={best['S']}", flush=True)
        return rows
    x, y, folds, pre, woe = preparar()
    best = max((float(r["J"]) for r in rows if r["J"]), default=None)
    best_s = next((r["S"] for r in rows if r["J"] and float(r["J"]) == best), "")
    causes = Counter(r["causa"].split("_")[0] for r in rows if r["estado"] != "ok")
    start = time.monotonic()
    start_index = len(rows)
    with ORACLE_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ORACLE_COLUMNS, lineterminator="\n")
        for i in range(len(rows), len(universe)):
            S = universe[i]
            t = time.monotonic()
            result = evaluar(S, x, y, folds, pre, woe)
            elapsed = time.monotonic() - t
            row = {
                "numero": i + 1, "S": "|".join(S), "estado": result["estado"],
                "causa": result["causa"], "J": "" if result["J"] is None else repr(result["J"]),
                "aucs": json.dumps(result["aucs"]), "pared_s": f"{elapsed:.6f}",
            }
            writer.writerow(row)
            rows.append(row)
            if result["J"] is not None and (best is None or result["J"] > best):
                best, best_s = result["J"], row["S"]
                print(f"  nuevo mejor J={best:.6f} S={best_s}", flush=True)
            if result["estado"] != "ok":
                causes[result["causa"].split("_")[0]] += 1
            if (i + 1) % 25 == 0 or i + 1 == len(universe):
                f.flush()
                done_now = i + 1 - start_index
                eta = (len(universe) - i - 1) / max(done_now / max(time.monotonic() - start, 1e-9), 1e-9)
                print(f"Oraculo {i + 1}/{len(universe)} | ultimo={row['J'] or row['causa']} "
                      f"| mejor={best if best is not None else 'ninguno'} | rechazos={dict(causes)} "
                      f"| ETA={eta / 60:.1f} min", flush=True)
    print(f"Oraculo completo | factibles={sum(r['estado'] == 'ok' for r in rows)} "
          f"| mejor J={best:.6f} S={best_s}", flush=True)
    return rows


def vecindarios(universe):
    variable_bits = {v: 1 << i for i, v in enumerate(sorted({v for S in universe for v in S}))}
    masks = [sum(variable_bits[v] for v in S) for S in universe]
    result = {S: ([], []) for S in universe}
    for i, S in enumerate(universe):
        for j in range(i + 1, len(universe)):
            distance = (masks[i] ^ masks[j]).bit_count() // 2
            if distance in (1, 2):
                result[S][distance - 1].append(universe[j])
                result[universe[j]][distance - 1].append(S)
    return result


def baselines(universe, rows, ctx):
    table = {tuple(r["S"].split("|")): r for r in rows}
    assert len(table) == len(universe)
    x, y, folds, pre, woe = ctx
    near = vecindarios(universe)
    TRIALS_PATH.parent.mkdir(exist_ok=True)
    summaries = []
    with TRIALS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS, lineterminator="\n")
        writer.writeheader()

        def run(method, seed, proposals):
            seen = set()
            attempts = 0
            best, best_s = None, ""
            rejected = Counter()
            model_time = 0.0
            start_wall, start_cpu = time.monotonic(), time.process_time()
            for proposal in proposals:
                attempts += 1
                S, kind = proposal[:2]
                supplied = proposal[2] if len(proposal) > 2 else None
                S = tuple(sorted(S))
                if S in seen:
                    continue
                if len(seen) >= max(BUDGETS):
                    break
                seen.add(S)
                if len(S) == K_VARS:
                    row = table[S]
                    J = float(row["J"]) if row["J"] else None
                    cause = row["causa"]
                    state = row["estado"]
                    cost = float(row["pared_s"])
                else:
                    if supplied is None:
                        t = time.monotonic()
                        result = evaluar(S, x, y, folds, pre, woe, k=len(S))
                        cost = time.monotonic() - t
                    else:
                        result, cost = supplied
                    J, cause, state = result["J"], result["causa"], result["estado"]
                model_time += cost
                if state != "ok":
                    rejected[cause.split("_")[0]] += 1
                if len(S) == K_VARS and J is not None and (best is None or J > best):
                    best, best_s = J, "|".join(S)
                    print(f"  {method} semilla={seed} nuevo mejor J={best:.6f} S={best_s}", flush=True)
                writer.writerow({
                    "metodo": method, "semilla": seed, "trial": len(seen),
                    "propuestas_totales": attempts, "tipo": kind,
                    "S": "|".join(S), "estado": state, "causa": cause,
                    "J": "" if J is None else repr(J), "mejor_J": "" if best is None else repr(best),
                    "mejor_S": best_s, "pared_modelo_s": f"{model_time:.6f}",
                })
                if len(seen) in BUDGETS:
                    print(f"  {method} semilla={seed} {len(seen)}/{max(BUDGETS)} "
                          f"ultimo={J if J is not None else cause} mejor={best} "
                          f"propuestas={attempts} rechazos={dict(rejected)} "
                          f"tiempo_modelo={model_time:.1f}s", flush=True)
                    summaries.append((method, seed, len(seen), len(seen), attempts,
                                      best, best_s, model_time,
                                      time.monotonic() - start_wall, time.process_time() - start_cpu))
            for b in BUDGETS:
                if b > len(seen):
                    summaries.append((method, seed, b, len(seen), attempts,
                                      best, best_s, model_time,
                                      time.monotonic() - start_wall, time.process_time() - start_cpu))
            print(f"  {method} semilla={seed} fin | trials={len(seen)} "
                  f"propuestas={attempts} mejor={best} S={best_s or 'ninguno'}", flush=True)
            f.flush()

        for seed in SEEDS:
            rng = random.Random(seed)
            random_order = rng.sample(universe, len(universe))
            run("aleatorio", seed, ((S, "completo") for S in random_order))

        # Forward is deterministic. Its partial models consume the same trial budget.
        def forward():
            chosen = ()
            used_families = set()
            for size in range(1, K_VARS + 1):
                options = []
                for family, choices in FAMILIAS:
                    if family in used_families:
                        continue
                    for v in choices:
                        S = tuple(sorted((*chosen, v)))
                        if size == K_VARS:
                            row = table[S]
                            J = float(row["J"]) if row["J"] else None
                            supplied = None
                        else:
                            t = time.monotonic()
                            result = evaluar(S, x, y, folds, pre, woe, k=size)
                            supplied = (result, time.monotonic() - t)
                            J = result["J"]
                        options.append((J, S, family))
                        yield S, "parcial" if size < K_VARS else "completo", supplied
                valid = [item for item in options if item[0] is not None]
                if not valid:
                    print(f"Forward bloqueado en tamano {size - 1}", flush=True)
                    return
                _, chosen, family = max(valid, key=lambda item: (item[0], item[1]))
                used_families.add(family)

        run("forward", 0, forward())

        for seed in SEEDS:
            rng = random.Random(seed)

            def local():
                current = rng.choice(universe)
                visited = set()
                while len(visited) < len(universe):
                    if current not in visited:
                        visited.add(current)
                        yield current, "inicio" if len(visited) == 1 else "intercambio"
                    current_j = float(table[current]["J"]) if table[current]["J"] else -math.inf
                    moved = False
                    for distance in (1, 2):
                        candidates = near[current][distance - 1].copy()
                        rng.shuffle(candidates)
                        choices = []
                        for S in candidates:
                            if S not in visited:
                                visited.add(S)
                            yield S, f"intercambio_{distance}"
                            J = float(table[S]["J"]) if table[S]["J"] else -math.inf
                            if J > current_j:
                                choices.append((J, S))
                        if choices:
                            current = max(choices)[1]
                            moved = True
                            break
                    if not moved:
                        remaining = [S for S in universe if S not in visited]
                        if not remaining:
                            return
                        current = rng.choice(remaining)

            run("local", seed, local())

    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("metodo", "semilla", "presupuesto", "trials_reales", "propuestas_totales",
                         "mejor_J", "mejor_S", "pared_modelo_s", "pared_ejecucion_s", "cpu_ejecucion_s"))
        for method, seed, budget, actual, attempts, best, best_s, model_time, wall, cpu in summaries:
            writer.writerow((method, seed, budget, actual, attempts,
                             "" if best is None else repr(best), best_s,
                             f"{model_time:.6f}", f"{wall:.6f}", f"{cpu:.6f}"))
    print(f"Ensayos: {TRIALS_PATH}; resumen: {SUMMARY_PATH}", flush=True)


def analizar_y_graficar(rows):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    optimum = max(float(row["J"]) for row in rows if row["J"])
    with SUMMARY_PATH.open(newline="") as f:
        summaries = list(csv.DictReader(f))
    with TRIALS_PATH.open(newline="") as f:
        trials = list(csv.DictReader(f))
    by_run = {}
    for row in trials:
        by_run.setdefault((row["metodo"], int(row["semilla"])), []).append(row)

    metrics = []
    for method in ("aleatorio", "forward", "local"):
        for budget in BUDGETS:
            selected = [r for r in summaries if r["metodo"] == method and int(r["presupuesto"]) == budget]
            gaps = [optimum - float(r["mejor_J"]) for r in selected if r["mejor_J"]]
            row = {
                "metodo": method, "presupuesto": budget, "corridas": len(selected),
                "trials_reales_min": min(int(r["trials_reales"]) for r in selected),
                "trials_reales_max": max(int(r["trials_reales"]) for r in selected),
                "sin_solucion": len(selected) - len(gaps),
                "brecha_media": "" if not gaps else repr(statistics.mean(gaps)),
                "brecha_mediana": "" if not gaps else repr(statistics.median(gaps)),
                "exitos_optimo": sum(g <= 1e-12 for g in gaps),
            }
            for eps, label in ((0.005, "005"), (0.002, "002"), (0.001, "001")):
                row[f"exitos_{label}"] = sum(g <= eps for g in gaps)
                times = []
                for result in selected:
                    run = by_run[(method, int(result["semilla"]))]
                    t = next((int(item["trial"]) for item in run
                              if int(item["trial"]) <= budget and item["mejor_J"]
                              and optimum - float(item["mejor_J"]) <= eps), None)
                    if t is not None:
                        times.append(t)
                row[f"T_{label}_mediana_exitos"] = "" if not times else repr(statistics.median(times))
            metrics.append(row)

    with METRICS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Brecha media al oraculo", "Corridas que encuentran el optimo"))
    colors = {"aleatorio": "#b65b38", "forward": "#475c73", "local": "#17806d"}
    for method in ("aleatorio", "forward", "local"):
        series = [m for m in metrics if m["metodo"] == method]
        if method == "forward":
            series = series[:1]
        x = [m["trials_reales_max"] for m in series]
        gaps = [1000 * float(m["brecha_media"]) if m["brecha_media"] else None for m in series]
        success = [m["exitos_optimo"] / m["corridas"] for m in series]
        style = "dash" if method == "forward" else "solid"
        fig.add_trace(go.Scatter(x=x, y=gaps, name=method, mode="lines+markers",
                                 line={"color": colors[method], "dash": style, "shape": "hv"},
                                 marker={"size": 8}, legendgroup=method), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=success, name=method, mode="lines+markers",
                                 line={"color": colors[method], "dash": style, "shape": "hv"},
                                 marker={"size": 8}, legendgroup=method, showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Trials unicos evaluados", tickvals=[100, 250, 500, 1000, 1792])
    fig.update_yaxes(title_text="Brecha (milesimas de AUC)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Fraccion de corridas", range=[0, 1.05], row=1, col=2)
    fig.update_layout(width=1280, height=520, template="plotly_white",
                      margin={"l": 75, "r": 35, "t": 70, "b": 75},
                      legend={"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"})
    fig.write_image(FIGURE_PATH, scale=2)
    print(f"Metricas: {METRICS_PATH}; figura: {FIGURE_PATH}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-oraculo", action="store_true")
    parser.add_argument("--solo-figura", action="store_true")
    args = parser.parse_args()
    universe = catalogo()
    print(f"Subespacio: {len(universe)} conjuntos estructurales", flush=True)
    if len(universe) != 1792:
        raise RuntimeError("Cambio inesperado del subespacio")
    fingerprint = firma()
    if args.solo_figura:
        rows = abrir_oraculo(fingerprint, universe)
        if len(rows) != len(universe):
            raise RuntimeError("El oraculo debe estar completo para generar la figura")
        analizar_y_graficar(rows)
        return
    rows = ejecutar_oraculo(universe, fingerprint)
    if args.solo_oraculo:
        return
    ctx = preparar()
    baselines(universe, rows, ctx)
    analizar_y_graficar(rows)


if __name__ == "__main__":
    main()
