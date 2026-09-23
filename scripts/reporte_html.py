"""Reporte HTML comparativo de todos los experimentos.

Por track: KPIs + curva granular (mediana del mejor J por trial)
con eje X en trials y otra en tiempo estimado de modelo acumulado.
Estilo: tarjetas claras, linea principal oscura, secundaria gris
(es-CL). Autocontenido (plotly inline). Solo lee work/ y outputs/.
"""

import csv
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]

COLORES = {
    "aleatorio": "#a1a1aa", "local": "#52525b",
    "ga_local": "#84cc16", "sef_completo": "#18181b", "tpe": "#7b2d8b",
}
NOMBRES = {
    "aleatorio": "Aleatorio", "local": "Local",
    "ga_local": "GA + local", "sef_completo": "SEF completo", "tpe": "TPE",
}

TRACKS = {
    "fase3b": {"titulo": "Fase 3b · Scorecard MRAP (k=6, 1,25M conjuntos)",
               "runs": "work/fase3b_runs", "tpe_runs": None,
               "ref": "outputs/fase3b_referencias.csv",
               "desde": 50, "mejor": 0.713542318119154},
    "fase4": {"titulo": "Fase 4 · Scorecard Taiwan (k=6, 26.262 conjuntos)",
              "runs": "work/fase4_runs", "tpe_runs": "work/tpe_fase4_runs",
              "ref": "outputs/fase4_referencias.csv",
              "desde": 50, "mejor": 0.7615923356582383},
    "series": {"titulo": "Series · ARIMAX IMACEC (130 conjuntos)",
               "runs": "work/series_runs", "tpe_runs": None,
               "ref": "outputs/series_referencias.csv",
               "desde": 10, "mejor": -549.0535808233672},
    "series2": {"titulo": "Series2 · ARIMAX ampliado (5.481 conjuntos)",
                "runs": "work/series2_runs", "tpe_runs": "work/series2_tpe_runs",
                "ref": "outputs/series2_referencias.csv",
                "desde": 25, "mejor": -535.4400277398996},
    "hostil": {"titulo": "Hostil · HGB + interacciones (153.090 conjuntos)",
               "runs": "work/hostil_runs", "tpe_runs": "work/tpe_hostil_runs",
               "ref": "outputs/hostil_referencias.csv",
               "desde": 25, "mejor": 0.9433257978758217},
}


def mediana_por_trial(runs_dir, desde):
    frames = []
    for path in sorted(Path(runs_dir).glob("*.csv")):
        try:
            frames.append(pl.read_csv(
                path, columns=["metodo", "semilla", "trial", "mejor_J",
                               "pared_modelo_s"]))
        except Exception:
            continue
    if not frames:
        return {}
    df = pl.concat(frames).with_columns(
        pl.col("mejor_J").cast(pl.Float64, strict=False),
        pl.col("pared_modelo_s").cast(pl.Float64, strict=False),
    ).sort("metodo", "semilla", "trial")
    df = df.with_columns(
        pl.col("mejor_J").forward_fill().over(["metodo", "semilla"]))
    df = df.filter(pl.col("trial") >= desde)
    series = {}
    for metodo in df["metodo"].unique().to_list():
        d = df.filter(pl.col("metodo") == metodo)
        g = d.group_by("trial").agg(
            pl.col("mejor_J").median().alias("j"),
            pl.col("pared_modelo_s").mean().alias("t")).sort("trial")
        series[metodo] = {"trial": g["trial"].to_list(),
                          "j": g["j"].to_list(),
                          "t": g["t"].to_list()}
    return series


def referencia(path):
    if not path.exists():
        return []
    out = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["J"]:
                out.append((row["metodo"], float(row["J"])))
    return out


def figura(track, series, refs, con_js):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Por trials únicos (log)",
                                        "Por tiempo de modelo (log)"))
    orden = [m for m in NOMBRES if m in series]
    for metodo in orden:
        s = series[metodo]
        fig.add_trace(go.Scatter(
            x=s["trial"], y=s["j"], mode="lines", name=NOMBRES[metodo],
            legendgroup=metodo, showlegend=True,
            line={"color": COLORES[metodo], "width": 2.5}), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[max(t, 1e-6) for t in s["t"]], y=s["j"], mode="lines",
            name=NOMBRES[metodo], legendgroup=metodo, showlegend=False,
            line={"color": COLORES[metodo], "width": 2.5}), row=1, col=2)
    for nombre, valor in refs:
        fig.add_hline(y=valor, line_dash="dash", line_color="#a1a1aa",
                      annotation_text=f"{nombre}",
                      annotation_font_size=11, row=1, col=1)
        fig.add_hline(y=valor, line_dash="dash", line_color="#a1a1aa",
                      row=1, col=2)
    fig.update_xaxes(type="log", title_text="Trials únicos")
    fig.update_xaxes(type="log", title_text="Segundos (log)", row=1, col=2)
    fig.update_yaxes(title_text="Mediana del mejor J")
    fig.update_layout(height=420, template="plotly_white",
                      margin={"l": 60, "r": 20, "t": 40, "b": 40},
                      legend={"orientation": "h", "y": -0.25, "x": 0.5,
                              "xanchor": "center"},
                      hovermode="x unified")
    return fig.to_html(full_html=False, include_plotlyjs=con_js,
                       div_id=f"fig-{track}")


def kpis(track, cfg, series):
    mejor = cfg["mejor"]
    filas = []
    for metodo, s in series.items():
        js = [j for j in s["j"] if j is not None]
        if not js:
            continue
        filas.append((NOMBRES[metodo], js[-1]))
    return sorted(filas, key=lambda r: -r[1]), mejor


def main():
    bloques = []
    primero = True
    for track, cfg in TRACKS.items():
        base = mediana_por_trial(ROOT / cfg["runs"], cfg["desde"])
        if cfg["tpe_runs"] and (ROOT / cfg["tpe_runs"]).exists():
            base.update(mediana_por_trial(ROOT / cfg["tpe_runs"], cfg["desde"]))
        refs = referencia(ROOT / cfg["ref"])
        tabla, mejor = kpis(track, cfg, base)
        filas_html = "".join(
            f"<tr><td>{m}</td><td class='num'>{v:.4f}</td>"
            f"<td class='num'>{(v - tabla[-1][1]):+.4f}</td></tr>"
            for m, v in tabla)
        bloques.append(f"""
        <section class="card">
          <header><h2>{cfg['titulo']}</h2>
          <p class="sub">Mejor observado: <b>{mejor:.4f}</b></p></header>
          <table><thead><tr><th>Método</th><th>J final</th>
          <th>vs peor</th></tr></thead><tbody>{filas_html}</tbody></table>
          {figura(track, base, refs, primero)}
        </section>""")
        primero = False
    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SEF · Resumen de experimentos</title>
    <style>
      :root {{ --ink: #18181b; --mut: #71717a; --line: #e4e4e7; --bg: #fafafa; }}
      body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
              background: var(--bg); color: var(--ink); margin: 0; }}
      main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
      h1 {{ font-size: 28px; margin: 0; }} .lede {{ color: var(--mut); }}
      .card {{ background: #fff; border: 1px solid var(--line);
               border-radius: 12px; padding: 20px 22px; margin-top: 24px;
               box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
      .card h2 {{ font-size: 18px; margin: 0; }}
      .sub {{ color: var(--mut); font-size: 13px; margin: 4px 0 12px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 13px;
               margin-bottom: 8px; }}
      th, td {{ text-align: left; padding: 6px 10px;
                border-bottom: 1px solid var(--line); }}
      th {{ color: var(--mut); font-weight: 600; }} .num {{ text-align: right;
        font-variant-numeric: tabular-nums; }}
    </style></head><body><main>
    <h1>SEF · Resumen de experimentos</h1>
    <p class="lede">Mediana del mejor J por trial y por tiempo estimado de
    modelo acumulado. Curvas granulares desde cobertura completa de
    factibles. Referencias punteadas fuera de presupuesto.</p>
    {''.join(bloques)}
    <p class="lede">Generado desde work/ y outputs/ del repo. Tiempo =
    pared_modelo_s acumulado medio (estimado aislado, mismo criterio
    que las metricas).</p>
    </main></body></html>"""
    destino = ROOT / "outputs/resumen_experimentos.html"
    destino.write_text(html)
    print(f"-> {destino} ({destino.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
