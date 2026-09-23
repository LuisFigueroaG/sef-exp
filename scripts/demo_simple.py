"""Demo personal: aleatorio, stepwise x3 y SEF sin cache, tiempo real.

Diferencias con los tracks del repo: sin cache de evaluaciones
(cada propuesta se ajusta de verdad y se mide pared real),
logistica CON regularizacion (C configurable), sin filtros
IV/corr/VIF (caso simple: todo conjunto se evalua), y score en
test al final. Solo train durante la busqueda; el test se abre
una vez para el reporte.

Uso:
    uv run python scripts/demo_simple.py --trials 200 --semillas 0,1,2
    uv run python scripts/demo_simple.py --trials 500 --semillas 0 --c 0.5
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import polars as pl
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sef_exp import fase3b
from sef_exp.evaluador import (
    BIN_CFG, cargar_datos, columna, hacer_folds, precomputar,
)
from sef_exp.familias import FAMILIAS_COMPLETAS
from sef_exp.fase2 import ROOT

K = 6
METODOS_MOTOR = ("aleatorio", "local", "ga_local", "sef")
METODOS_STEP = ("forward", "backward", "hibrido")
WORK = ROOT / "work/demo_runs"


class SinCache:
    """Misma interfaz que Evaluaciones pero ajusta siempre."""

    def __init__(self, f_ajuste):
        self.f_ajuste = f_ajuste
        self.fits = 0
        self.hits = 0

    def consulta(self, S, context):
        t = time.perf_counter()
        J = self.f_ajuste(S)
        pared = time.perf_counter() - t
        self.fits += 1
        return {"S": "|".join(S),
                "estado": "ok" if J is not None else "rechazado",
                "causa": "" if J is not None else "no_finito",
                "J": "" if J is None else repr(J), "aucs": "[]",
                "pared_s": f"{pared:.6f}"}, False


def ajuste_cv(S, datos, folds, pre, c):
    """AUC medio de 5 folds con logistica(C); None si algo falla."""
    try:
        aucs = []
        for r, (tr, va) in enumerate(folds):
            W = np.column_stack([
                pre[(r, v)]["opt"].transform(columna(datos["xtr"], v),
                                             metric="woe").ravel()
                for v in S])
            Wtr, Wva = W[tr], W[va]
            if not (np.all(np.isfinite(Wtr)) and np.all(np.isfinite(Wva))):
                return None
            clf = LogisticRegression(C=c, max_iter=5000)
            clf.fit(Wtr, datos["ytr"][tr])
            aucs.append(float(roc_auc_score(
                datos["ytr"][va], clf.predict_proba(Wva)[:, 1])))
        return float(np.mean(aucs))
    except Exception:
        return None


def correr_motor(metodo, seed, trials, cfg, evaluar_set, iv):
    metodo = "sef_completo" if metodo == "sef" else metodo
    cache = SinCache(lambda S: evaluar_set(S, None, None))
    path = WORK / f"{metodo}_{seed:02d}.csv"
    global_best = {"J": None, "S": None, "metodo": None, "semilla": None}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fase3b.TRIAL_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        fase3b.run_method(metodo, seed, trials, None, iv, cache, writer,
                          global_best, cfg)
    return cache.fits


def traza_step(metodo, puntos):
    """puntos: lista de (ensayo, pared_acum, J, S)."""
    path = WORK / f"{metodo}_00.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fase3b.TRIAL_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        mejor, mejor_s, pared = None, None, 0.0
        for i, (acum, J, S) in enumerate(puntos, 1):
            pared = acum
            if J is not None and (mejor is None or J > mejor):
                mejor, mejor_s = J, S
            writer.writerow({
                "metodo": metodo, "semilla": 0, "trial": i,
                "propuestas": i, "duplicados": 0, "S": "|".join(S),
                "fuente": metodo, "estado": "ok" if J is not None else "rechazado",
                "causa": "", "J": "" if J is None else repr(J),
                "mejor_J": "" if mejor is None else repr(mejor),
                "mejor_S": "" if mejor_s is None else "|".join(mejor_s),
                "pred_J": "", "pared_modelo_s": f"{pared:.6f}",
            })
    return mejor, mejor_s


def correr_forward(cfg, evaluar):
    ensayos = []
    def evaluar_tamano(S, k):
        t = time.perf_counter()
        J = evaluar(S)
        ensayos.append((time.perf_counter() - t, J, S))
        return J
    J, S, _ = fase3b.forward_greedy(evaluar_tamano, cfg)
    return traza_step("forward", puntos_reales(ensayos)), S


def puntos_reales(ensayos):
    puntos, acum = [], 0.0
    for dt, J, S in ensayos:
        acum += dt
        puntos.append((acum, J, S))
    return puntos


def correr_backward(cfg, evaluar, familias):
    espacio = cfg.espacio
    elegidas = tuple(sorted(espacio.variables))
    ensayos, vistos = [], set()

    def probar(S):
        t = time.perf_counter()
        J = evaluar(S)
        ensayos.append((time.perf_counter() - t, J, S))
        return J

    while len(elegidas) > K:
        opciones = []
        for v in elegidas:
            S = tuple(x for x in elegidas if x != v)
            opciones.append((probar(S), S))
        validas = [o for o in opciones if o[0] is not None]
        if not validas:
            break
        _, elegidas = max(validas, key=lambda o: (o[0], o[1]))
    elegidas = reparar_familias(elegidas, probar, espacio)
    return traza_step("backward", puntos_reales(ensayos)), elegidas


def reparar_familias(elegidas, evaluar, espacio):
    elegidas = set(elegidas)
    fams = {}
    for v in elegidas:
        fams.setdefault(espacio.familia_de[v], []).append(v)
    while len(elegidas) > K or any(len(v) > 1 for v in fams.values()):
        candidatos = []
        for v in sorted(elegidas):
            S = tuple(sorted(elegidas - {v}))
            candidatos.append((evaluar(S), v))
        validos = [(J, v) for J, v in candidatos if J is not None]
        if not validos:
            break
        _, fuera = max(validos)
        elegidas.remove(fuera)
        fams = {}
        for v in elegidas:
            fams.setdefault(espacio.familia_de[v], []).append(v)
    while len(elegidas) < K:
        usadas = {espacio.familia_de[v] for v in elegidas}
        resto = [v for v in espacio.variables
                 if espacio.familia_de[v] not in usadas]
        if not resto:
            break
        scored = [(evaluar(tuple(sorted((*elegidas, v)))), v) for v in resto]
        validos = [(J, v) for J, v in scored if J is not None]
        if not validos:
            break
        _, entra = max(validos)
        elegidas.add(entra)
    return tuple(sorted(elegidas))


def correr_hibrido(cfg, evaluar):
    espacio = cfg.espacio
    ensayos = []

    def probar(S):
        t = time.perf_counter()
        J = evaluar(S)
        ensayos.append((time.perf_counter() - t, J, S))
        return J

    elegidas, usadas = (), set()
    for _ in range(K):
        opciones = []
        for name, choices in espacio.familias:
            if name in usadas:
                continue
            for v in choices:
                S = tuple(sorted((*elegidas, v)))
                opciones.append((probar(S), S, name))
        validas = [o for o in opciones if o[0] is not None]
        if not validas:
            break
        _, elegidas, name = max(validas, key=lambda o: (o[0], o[1]))
        usadas.add(name)
        mejoro = True
        while mejoro and len(elegidas) > 1:
            mejoro = False
            bajas = [((probar(tuple(x for x in elegidas if x != v))), v)
                     for v in elegidas]
            validas_b = [(J, v) for J, v in bajas if J is not None]
            if validas_b:
                Jb, _ = max(validas_b)
                Ja = probar(elegidas)
                if Jb is not None and Ja is not None and Jb > Ja + 1e-12:
                    _, elegidas = max(
                        [(J, tuple(x for x in elegidas if x != v))
                         for J, v in validas_b])
                    elegidas = tuple(sorted(elegidas))
                    usadas = {espacio.familia_de[v] for v in elegidas}
                    mejoro = True
    elegidas = reparar_familias(elegidas, probar, espacio)
    return traza_step("hibrido", puntos_reales(ensayos)), elegidas


def submuestra(datos, n, seed=7):
    if n is None or n >= len(datos["ytr"]):
        return datos
    rng = np.random.default_rng(seed)
    idx = []
    for cls in (0, 1):
        ids = np.flatnonzero(datos["ytr"] == cls)
        rng.shuffle(ids)
        take = max(1, int(round(len(ids) * n / len(datos["ytr"]))))
        idx.extend(ids[:take].tolist())
    idx = sorted(idx)
    xtr = datos["xtr"].with_row_index("i").filter(
        pl.col("i").is_in(idx)).drop("i")
    return {"xtr": xtr, "ytr": datos["ytr"][np.array(idx)],
            "xte": datos["xte"], "yte": datos["yte"],
            "vars": datos["vars"]}


def test_score(S, datos, c):
    Wtr, Wte = [], []
    for v in S:
        opt = OptimalBinning(**BIN_CFG)
        opt.fit(columna(datos["xtr"], v), datos["ytr"])
        Wtr.append(opt.transform(columna(datos["xtr"], v),
                                 metric="woe").ravel())
        Wte.append(opt.transform(columna(datos["xte"], v),
                                 metric="woe").ravel())
    Wtr, Wte = np.column_stack(Wtr), np.column_stack(Wte)
    clf = LogisticRegression(C=c, max_iter=5000)
    clf.fit(Wtr, datos["ytr"])
    return float(roc_auc_score(datos["yte"],
                               clf.predict_proba(Wte)[:, 1]))


def leer_trazas():
    series = {}
    for path in sorted(WORK.glob("*.csv")):
        metodo = path.stem.rsplit("_", 1)[0]
        df = pl.read_csv(path, columns=["semilla", "trial", "mejor_J",
                                        "pared_modelo_s", "S", "mejor_S"])
        df = df.with_columns(
            pl.col("mejor_J").cast(pl.Float64, strict=False),
            pl.col("pared_modelo_s").cast(pl.Float64, strict=False))
        series.setdefault(metodo, []).append(df)
    return series


def figura_html(series, test_rows, subtitulo):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colores = {"aleatorio": "#b65b38", "local": "#475c73",
               "ga_local": "#667c2b", "sef_completo": "#087f70",
               "forward": "#a1a1aa", "backward": "#71717a",
               "hibrido": "#3f3f46"}
    nombres = {"aleatorio": "Aleatorio", "local": "Local",
               "ga_local": "GA + local", "sef_completo": "SEF",
               "forward": "Forward", "backward": "Backward",
               "hibrido": "Hibrido"}
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Por trials (ensayos)",
                                        "Por tiempo real (log)"))
    for metodo, dfs in series.items():
        juntos = pl.concat([d.with_row_index("s") for d in dfs])
        if len(dfs) > 1:
            g = juntos.group_by("trial").agg(
                pl.col("mejor_J").median().alias("j"),
                pl.col("pared_modelo_s").mean().alias("t")).sort("trial")
        else:
            g = dfs[0].select(
                pl.col("trial"), pl.col("mejor_J").alias("j"),
                pl.col("pared_modelo_s").alias("t")).sort("trial")
        g = g.filter(pl.col("j").is_not_null())
        if not len(g):
            continue
        fig.add_trace(go.Scatter(
            x=g["trial"].to_list(), y=g["j"].to_list(), mode="lines",
            name=nombres.get(metodo, metodo), legendgroup=metodo,
            showlegend=True,
            line={"color": colores.get(metodo, "#000"), "width": 2.5}),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[max(t, 1e-6) for t in g["t"].to_list()], y=g["j"].to_list(),
            mode="lines", name=nombres.get(metodo, metodo),
            legendgroup=metodo, showlegend=False,
            line={"color": colores.get(metodo, "#000"), "width": 2.5}),
            row=1, col=2)
    fig.update_xaxes(title_text="Ensayos (trials únicos)")
    fig.update_xaxes(type="log", title_text="Segundos reales (log)",
                     row=1, col=2)
    fig.update_yaxes(title_text="Mejor AUC de CV")
    fig.update_layout(height=440, template="plotly_white",
                      margin={"l": 60, "r": 20, "t": 40, "b": 40},
                      legend={"orientation": "h", "y": -0.25, "x": 0.5,
                              "xanchor": "center"},
                      hovermode="x unified")
    filas = "".join(
        f"<tr><td>{m}</td><td class='num'>{cv:.4f}</td>"
        f"<td class='num'>{te:.4f}</td><td class='num'>{n}</td>"
        f"<td class='vars'>{s}</td></tr>"
        for m, cv, te, n, s in test_rows)
    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Demo simple · sin cache, tiempo real</title>
    <style>
      :root {{ --ink: #18181b; --mut: #71717a; --line: #e4e4e7; }}
      body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
              background: #fafafa; color: var(--ink); margin: 0; }}
      main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
      h1 {{ font-size: 26px; margin: 0; }} .lede {{ color: var(--mut); }}
      .card {{ background: #fff; border: 1px solid var(--line);
               border-radius: 12px; padding: 20px 22px; margin-top: 24px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
      th, td {{ text-align: left; padding: 6px 10px;
                border-bottom: 1px solid var(--line); }}
      th {{ color: var(--mut); }} .num {{ text-align: right;
        font-variant-numeric: tabular-nums; }} .vars {{ font-size: 11px; }}
    </style></head><body><main>
    <h1>Demo simple · MRAP sin cache, tiempo real</h1>
    <p class="lede">Logistica con regularizacion, sin filtros, cada
    propuesta se ajusta de verdad. Test abierto una sola vez.<br>
    {subtitulo}</p>
    <section class="card"><h2>Mejor CV por metodo y score en test</h2>
    <table><thead><tr><th>Metodo</th><th>CV</th><th>Test</th>
    <th>Ensayos</th><th>Conjunto</th></tr></thead>
    <tbody>{filas}</tbody></table>
    {fig.to_html(full_html=False, include_plotlyjs=True)}
    </section></main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--semillas", default="0,1,2")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--muestra", type=int, default=20000,
                        help="filas de train para buscar (test siempre "
                             "completo); mismo recorte para todos")
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    for viejo in WORK.glob("*.csv"):
        viejo.unlink()
    print("Cargando train (test cerrado hasta el reporte)...", flush=True)
    datos = cargar_datos(str(ROOT / "data"))
    datos = submuestra(datos, args.muestra)
    print(f"Train de busqueda: {len(datos['ytr'])} filas", flush=True)
    folds = hacer_folds(datos["ytr"])
    pre = precomputar(datos["xtr"], datos["ytr"], folds,
                      list(datos["xtr"].columns))
    iv = {v: statistics.mean(pre[(r, v)]["iv"] for r in range(len(folds)))
          for v in datos["xtr"].columns}
    cfg = fase3b.Config(nombre="demo", familias=FAMILIAS_COMPLETAS, k=K,
                        presupuestos=(args.trials,))
    evaluar = lambda S: ajuste_cv(S, datos, folds, pre, args.c)  # noqa
    t0 = time.perf_counter()

    for metodo in METODOS_MOTOR:
        for seed in [int(x) for x in args.semillas.split(",")]:
            t = time.perf_counter()
            fits = correr_motor(metodo, seed, args.trials, cfg,
                                lambda S, c, g: evaluar(S), iv)
            print(f"{metodo} s={seed}: {fits} ajustes en "
                  f"{time.perf_counter() - t:.1f}s", flush=True)

    finales = {}
    _, Sf = correr_forward(cfg, evaluar)
    Jf = evaluar(Sf) if Sf is not None and len(Sf) == K else None
    finales["forward"] = (Jf, Sf)
    print(f"forward: J={Jf} S={'|'.join(Sf) if Sf else '-'}", flush=True)
    _, Sb = correr_backward(cfg, evaluar, FAMILIAS_COMPLETAS)
    Jb = evaluar(Sb) if Sb is not None and len(Sb) == K else None
    finales["backward"] = (Jb, Sb)
    print(f"backward: J={Jb} S={'|'.join(Sb) if Sb else '-'}", flush=True)
    _, Sh = correr_hibrido(cfg, evaluar)
    Jh = evaluar(Sh) if Sh is not None and len(Sh) == K else None
    finales["hibrido"] = (Jh, Sh)
    print(f"hibrido: J={Jh} S={'|'.join(Sh) if Sh else '-'}", flush=True)

    print("Abriendo test una vez para el reporte...", flush=True)
    test_rows = []
    for metodo, dfs in leer_trazas().items():
        mejor_cv, mejor_s, ensayos = None, None, 0
        for df in dfs:
            ensayos += len(df)
            fila = df.filter(pl.col("mejor_J").is_not_null()).sort("trial").tail(1)
            if len(fila) and (mejor_cv is None or fila["mejor_J"][0] > mejor_cv):
                mejor_cv = fila["mejor_J"][0]
                mejor_s = tuple(fila["mejor_S"][0].split("|"))
        if metodo in finales:
            Jf_, Sf_ = finales[metodo]
            if Jf_ is not None and Sf_ is not None and len(Sf_) == K:
                mejor_cv, mejor_s = Jf_, Sf_
        if mejor_s is None or len(mejor_s) != K:
            print(f"{metodo}: sin conjunto completo, se omite test",
                  flush=True)
            continue
        te = test_score(mejor_s, datos, args.c)
        test_rows.append((metodo, mejor_cv, te, ensayos, "|".join(mejor_s)))
        print(f"{metodo}: CV={mejor_cv:.4f} test={te:.4f} S={'|'.join(mejor_s)}",
              flush=True)
    destino = ROOT / "outputs/demo_simple.html"
    subtitulo = (f"C={args.c}, trials={args.trials}, semillas={args.semillas}, "
                 f"train de busqueda={len(datos['ytr'])} filas, "
                 f"test completo={len(datos['yte'])} filas.")
    destino.write_text(figura_html(leer_trazas(), sorted(
        test_rows, key=lambda r: -r[1]), subtitulo))
    print(f"Total {time.perf_counter() - t0:.1f}s -> {destino}", flush=True)


if __name__ == "__main__":
    main()
