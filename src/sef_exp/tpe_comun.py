"""Motor TPE generico: muestrea una opcion por familia y repara a k.

Reutilizable por track via Config. Duplicados consumen propuestas,
unicos consumen trials, igual que el resto de metodos.
"""

import csv
import random
import time

import optuna

from sef_exp import fase3b


def reparar(base, fams, rng, k):
    """Ajusta a exactamente k familias con una opcion cada una."""
    elegidas = {f: v for f, v in base.items() if v is not None}
    fams = list(fams)
    while len(elegidas) > k:
        elegidas.pop(rng.choice(sorted(elegidas)))
    while len(elegidas) < k:
        faltan = [f for f, _ in fams if f not in elegidas]
        f = rng.choice(faltan)
        elegidas[f] = rng.choice(dict(fams)[f])
    return tuple(sorted(elegidas[f] for f in elegidas))


def opciones(cfg):
    return [(name, ("ausente", *choices)) for name, choices in cfg.espacio.familias]


def correr(cfg, seed, limit, context, cache, runs_dir):
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
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / f"tpe_{seed:02d}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fase3b.TRIAL_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        while len(archive) < limit:
            t = estudio.ask()
            base = {name: t.suggest_categorical(name, list(vals))
                    for name, vals in fams}
            base = {k: (None if v == "ausente" else v) for k, v in base.items()}
            S = reparar(base, fams, rng, cfg.k)
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
            estudio.tell(t, -1e9 if J is None else J)
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


def figura(cfg, tpe_runs_dir, destino):
    import plotly.graph_objects as go
    import polars as pl

    mediana = fase3b.curvas_granulares(cfg)
    frames = []
    for path in sorted(tpe_runs_dir.glob("*.csv")):
        frames.append(pl.read_csv(
            path, columns=["metodo", "semilla", "trial", "mejor_J"]
        ).with_columns(pl.col("mejor_J").cast(pl.Float64, strict=False)))
    runs = pl.concat(frames).sort("metodo", "semilla", "trial")
    runs = runs.with_columns(
        pl.col("mejor_J").forward_fill().over(["metodo", "semilla"])
    ).filter(pl.col("trial") >= cfg.granular_desde)
    tpe_med = runs.group_by(["metodo", "trial"]).agg(
        pl.col("mejor_J").median().alias("j_mediana")).sort("trial")
    colors = {
        "aleatorio": "#b65b38", "local": "#475c73",
        "ga_local": "#667c2b", "sef_completo": "#087f70",
        "tpe": "#7b2d8b",
    }
    fig = go.Figure()
    for method in ("aleatorio", "local", "ga_local", "sef_completo"):
        series = mediana.filter(pl.col("metodo") == method).sort("trial")
        if not len(series):
            continue
        fig.add_trace(go.Scatter(
            x=series["trial"].to_list(), y=series["j_mediana"].to_list(),
            mode="lines", name=method,
            line={"color": colors[method], "width": 2}))
    fig.add_trace(go.Scatter(
        x=tpe_med["trial"].to_list(), y=tpe_med["j_mediana"].to_list(),
        mode="lines", name="tpe",
        line={"color": colors["tpe"], "width": 2}))
    nombre_ref, valor_ref = fase3b.leer_referencia(cfg)
    if valor_ref is not None:
        fig.add_hline(y=valor_ref, line_dash="dash", line_color="#888888",
                      annotation_text=f"{nombre_ref} (referencia)")
    ticks = [cfg.granular_desde, *cfg.presupuestos]
    fig.update_xaxes(type="log", tickvals=ticks,
                     ticktext=[str(t) for t in ticks],
                     title_text="Trials unicos (log)")
    fig.update_yaxes(title_text=cfg.etiqueta_y)
    fig.update_layout(width=1400, height=570, template="plotly_white",
                      margin={"l": 75, "r": 35, "t": 40, "b": 90},
                      legend={"orientation": "h", "y": -0.23, "x": 0.5,
                              "xanchor": "center"})
    t0 = time.monotonic()
    fig.write_image(destino, scale=2)
    print(f"figura: {destino} ({time.monotonic() - t0:.1f}s)", flush=True)
