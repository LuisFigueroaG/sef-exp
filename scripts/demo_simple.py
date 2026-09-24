"""Demo personal: aleatorio, stepwise x2 y SEF sin cache, tiempo real.

Diferencias con los tracks del repo: sin cache de evaluaciones
(cada propuesta se ajusta de verdad y se mide pared real),
logistica CON regularizacion (C configurable), sin filtros
IV/corr/VIF (caso simple: todo conjunto se evalua), y score en
test al final. Solo train durante la busqueda; el test se abre
una vez para el reporte.

Uso:
    uv run python scripts/demo_simple.py --trials 200 --semillas 0,1,2
    uv run python scripts/demo_simple.py --trials 500 --semillas 0 --c 0.5
    uv run python scripts/demo_simple.py --hostil --trials 100 --muestra 8000

Modo hostil (--hostil): cambia logistica+WOE por el MLP ganador del
notebook 05 de MRAP (una capa de 32, alpha y lr congelados, 58 epocas
por partial_fit) sobre variables crudas winsorizadas 1/99, imputadas
y escaladas. Sin WOE, sin filtros: mundo no lineal donde el greedy
sufre. El modo hostil necesita --muestra chica porque cada ajuste
cuesta segundos.
"""

import argparse
import contextlib
import csv
import io
import logging
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from optbinning import OptimalBinning
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sef_exp import fase3b
from sef_exp.evaluador import (
    BIN_CFG, cargar_datos, columna, hacer_folds, precomputar,
)
from sef_exp.familias import FAMILIAS_COMPLETAS
from sef_exp.fase2 import ROOT

K = 6
METODOS_MOTOR = ("aleatorio", "local", "ga_local", "sef")
METODOS_STEP = ("forward", "hibrido")
WORK = ROOT / "work/demo_runs"

log = logging.getLogger("demo")

# MLP ganador del notebook 05 de MRAP (configuracion congelada).
MLP_CFG = {"hidden_layer_sizes": (32,), "activation": "relu",
           "solver": "adam", "alpha": 0.009735125945191669,
           "learning_rate_init": 0.0005633564508332506,
           "batch_size": 1024, "max_iter": 1, "random_state": 42}
MLP_EPOCAS = 58


def configurar_log(nivel=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(levelname).1s %(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(nivel)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


def crudo(datos, v):
    return datos["xtr"][v].to_numpy().astype(np.float64)


def preparar_mlp(Atr, Ava):
    """Winsoriza 1/99, imputa mediana y escala; todo ajustado en train."""
    limites = np.nanquantile(Atr, [0.01, 0.99], axis=0)
    Atr = np.clip(Atr, limites[0], limites[1]).astype(np.float32)
    Ava = np.clip(Ava, limites[0], limites[1]).astype(np.float32)
    pipe = make_pipeline(SimpleImputer(strategy="median",
                                       add_indicator=True),
                         StandardScaler())
    return (pipe.fit_transform(Atr).astype(np.float32),
            pipe.transform(Ava).astype(np.float32))


def ajustar_mlp(Atr, ytr):
    clf = MLPClassifier(**MLP_CFG)
    for _ in range(MLP_EPOCAS):
        clf.partial_fit(Atr, ytr, classes=np.array([0, 1]))
    return clf


def ajuste_mlp_cv(S, datos, folds):
    """AUC medio de 5 folds con el MLP congelado; None si algo falla."""
    try:
        aucs = []
        for tr, va in folds:
            Atr, Ava = preparar_mlp(
                np.column_stack([crudo(datos, v)[tr] for v in S]),
                np.column_stack([crudo(datos, v)[va] for v in S]))
            if not (np.all(np.isfinite(Atr)) and np.all(np.isfinite(Ava))):
                return None
            clf = ajustar_mlp(Atr, datos["ytr"][tr])
            aucs.append(float(roc_auc_score(
                datos["ytr"][va], clf.predict_proba(Ava)[:, 1])))
        return float(np.mean(aucs))
    except Exception:
        return None


def ranking_univariado(datos, folds):
    """AUC univariado con logistica cruda: solo para sembrar el inicio."""
    log.info("Ranking univariado para la siembra inicial")
    rank = {}
    for v in datos["vars"]:
        try:
            aucs = []
            for tr, va in folds:
                x = crudo(datos, v)
                clf = LogisticRegression(max_iter=2000)
                clf.fit(x[tr].reshape(-1, 1), datos["ytr"][tr])
                aucs.append(float(roc_auc_score(
                    datos["ytr"][va],
                    clf.predict_proba(x[va].reshape(-1, 1))[:, 1])))
            rank[v] = float(np.mean(aucs))
        except Exception:
            rank[v] = 0.5
    return rank


class ProgressWriter:
    """Envuelve el writer y loguea el avance por cuartos, estilo optuna."""

    def __init__(self, writer, metodo, seed, trials):
        self.writer = writer
        self.metodo = metodo
        self.seed = seed
        self.trials = trials
        self.n = 0
        self.marcas = {trials // 4, trials // 2, 3 * trials // 4} - {0}

    def __getattr__(self, name):
        return getattr(self.writer, name)

    def writerow(self, row):
        self.n += 1
        self.writer.writerow(row)
        if self.n in self.marcas:
            try:
                mejor = f"{float(row.get('mejor_J') or ''):.4f}"
            except (TypeError, ValueError):
                mejor = "-"
            log.info("%s semilla=%s [%d/%d] mejor=%s", self.metodo,
                     self.seed, self.n, self.trials, mejor)


def correr_motor(metodo, seed, trials, cfg, evaluar_set, iv):
    metodo = "sef_completo" if metodo == "sef" else metodo
    cache = SinCache(lambda S: evaluar_set(S, None, None))
    path = WORK / f"{metodo}_{seed:02d}.csv"
    global_best = {"J": None, "S": None, "metodo": None, "semilla": None}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fase3b.TRIAL_COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        fase3b.run_method(metodo, seed, trials, None, iv, cache,
                          ProgressWriter(writer, metodo, seed, trials),
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
    mejor = {"J": None}

    def evaluar_tamano(S, k):
        t = time.perf_counter()
        J = evaluar(S)
        ensayos.append((time.perf_counter() - t, J, S))
        if J is not None and (mejor["J"] is None or J > mejor["J"]):
            mejor["J"] = J
        if len(ensayos) % 50 == 0:
            log.info("forward [%d ajustes] mejor=%s", len(ensayos),
                     f"{mejor['J']:.4f}" if mejor["J"] is not None else "-")
        return J
    J, S, _ = fase3b.forward_greedy(evaluar_tamano, cfg)
    return traza_step("forward", puntos_reales(ensayos)), S


def puntos_reales(ensayos):
    puntos, acum = [], 0.0
    for dt, J, S in ensayos:
        acum += dt
        puntos.append((acum, J, S))
    return puntos


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
    mejor = {"J": None}

    def probar(S):
        t = time.perf_counter()
        J = evaluar(S)
        ensayos.append((time.perf_counter() - t, J, S))
        if J is not None and (mejor["J"] is None or J > mejor["J"]):
            mejor["J"] = J
        if len(ensayos) % 50 == 0:
            log.info("hibrido [%d ajustes] mejor=%s", len(ensayos),
                     f"{mejor['J']:.4f}" if mejor["J"] is not None else "-")
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


def test_score_mlp(S, datos):
    Atr, Ate = preparar_mlp(
        np.column_stack([crudo(datos, v) for v in S]),
        np.column_stack(
            [datos["xte"][v].to_numpy().astype(np.float64) for v in S]))
    clf = ajustar_mlp(Atr, datos["ytr"])
    return float(roc_auc_score(datos["yte"],
                               clf.predict_proba(Ate)[:, 1]))


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


def figura_html(series, test_rows, subtitulo,
                titulo="Demo simple · MRAP sin cache, tiempo real"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colores = {"aleatorio": "#b65b38", "local": "#475c73",
               "ga_local": "#667c2b", "sef_completo": "#087f70",
               "forward": "#a1a1aa",
               "hibrido": "#3f3f46"}
    nombres = {"aleatorio": "Aleatorio", "local": "Local",
               "ga_local": "GA + local", "sef_completo": "SEF",
               "forward": "Forward",
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
    <h1>{titulo}</h1>
    <p class="lede">Cada propuesta se ajusta de verdad, sin cache.
    Test abierto una sola vez.<br>
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
    parser.add_argument("--muestra", type=int, default=None,
                        help="filas de train para buscar (test siempre "
                             "completo); mismo recorte para todos "
                             "(default: 20000, 8000 en hostil)")
    parser.add_argument("--hostil", action="store_true",
                        help="MLP congelado sobre crudas en vez de "
                             "logistica+WOE")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="muestra el detalle interno del motor")
    args = parser.parse_args()
    configurar_log(logging.DEBUG if args.verbose else logging.INFO)

    global WORK
    WORK = ROOT / ("work/demo_hostil_runs" if args.hostil
                   else "work/demo_runs")
    destino = ROOT / ("outputs/demo_simple_hostil.html" if args.hostil
                      else "outputs/demo_simple.html")
    if args.muestra is None:
        args.muestra = 8000 if args.hostil else 20000
    modo = "MLP(32)x58" if args.hostil else f"logit(C={args.c})"
    log.info("demo_simple modo=%s trials=%s semillas=%s muestra=%s",
             modo, args.trials, args.semillas, args.muestra)

    WORK.mkdir(parents=True, exist_ok=True)
    for viejo in WORK.glob("*.csv"):
        viejo.unlink()
    log.info("Cargando train (test cerrado hasta el reporte)")
    datos = cargar_datos(str(ROOT / "data"))
    datos = submuestra(datos, args.muestra)
    log.info("Train de busqueda: %d filas", len(datos["ytr"]))
    folds = hacer_folds(datos["ytr"])
    if args.hostil:
        evaluar = lambda S: ajuste_mlp_cv(S, datos, folds)  # noqa
        puntuar = lambda S: test_score_mlp(S, datos)  # noqa
        iv = ranking_univariado(datos, folds)
    else:
        pre = precomputar(datos["xtr"], datos["ytr"], folds,
                          list(datos["xtr"].columns))
        iv = {v: statistics.mean(pre[(r, v)]["iv"]
                                 for r in range(len(folds)))
              for v in datos["xtr"].columns}
        evaluar = lambda S: ajuste_cv(S, datos, folds, pre, args.c)  # noqa
        puntuar = lambda S: test_score(S, datos, args.c)  # noqa
    cfg = fase3b.Config(nombre="demo", familias=FAMILIAS_COMPLETAS, k=K,
                        presupuestos=(args.trials,))
    t0 = time.perf_counter()
    silencio = contextlib.nullcontext() if args.verbose \
        else contextlib.redirect_stdout(io.StringIO())

    with silencio:
        for metodo in METODOS_MOTOR:
            for seed in [int(x) for x in args.semillas.split(",")]:
                t = time.perf_counter()
                fits = correr_motor(metodo, seed, args.trials, cfg,
                                    lambda S, c, g: evaluar(S), iv)
                df = pl.read_csv(WORK / f"{metodo if metodo != 'sef' else 'sef_completo'}_{seed:02d}.csv",
                                 columns=["trial", "mejor_J", "mejor_S"])
                df = df.with_columns(
                    pl.col("mejor_J").cast(pl.Float64, strict=False))
                df = df.filter(pl.col("mejor_J").is_not_null())
                if len(df):
                    top = df.sort("trial").tail(1)
                    log.info("%s semilla=%s finished with value=%.4f "
                             "(trial %d/%d) en %.1fs, %d ajustes",
                             metodo, seed, top["mejor_J"][0],
                             top["trial"][0], args.trials,
                             time.perf_counter() - t, fits)
                else:
                    log.info("%s semilla=%s sin conjunto factible en %.1fs, "
                             "%d ajustes", metodo, seed,
                             time.perf_counter() - t, fits)

        finales = {}
        for nombre, fn in (("forward", lambda: correr_forward(cfg, evaluar)),
                           ("hibrido", lambda: correr_hibrido(cfg, evaluar))):
            t = time.perf_counter()
            _, S = fn()
            J = evaluar(S) if S is not None and len(S) == K else None
            finales[nombre] = (J, S)
            log.info("%s finished with value=%s en %.1fs, S=%s", nombre,
                     f"{J:.4f}" if J is not None else "-",
                     time.perf_counter() - t,
                     "|".join(S) if S else "-")

    log.info("Abriendo test una vez para el reporte")
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
            log.info("%s: sin conjunto completo, se omite test", metodo)
            continue
        te = puntuar(mejor_s)
        test_rows.append((metodo, mejor_cv, te, ensayos, "|".join(mejor_s)))
        log.info("%s best CV=%.4f test=%.4f S=%s",
                 metodo, mejor_cv, te, "|".join(mejor_s))
    subtitulo = (f"modo={modo}, trials={args.trials}, "
                 f"semillas={args.semillas}, "
                 f"train de busqueda={len(datos['ytr'])} filas, "
                 f"test completo={len(datos['yte'])} filas.")
    destino.write_text(figura_html(leer_trazas(), sorted(
        test_rows, key=lambda r: -r[1]), subtitulo))
    log.info("Fin en %.1fs -> %s", time.perf_counter() - t0, destino)


if __name__ == "__main__":
    main()
