"""Aplica la regla de docs/seriesx-contrato.md: menor n con mediana
a epsilon (2 AICc) del mejor mediano global. Escribe
outputs/seriesx_eleccion.json y outputs/seriesx_codo.png.
"""

import csv
import json
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
NS = (0, 1, 2, 3, 4, 5)
EPSILON = 2.0


def elegir(medianas_por_n, epsilon=EPSILON):
    techo = max(medianas_por_n.values())
    for n in sorted(medianas_por_n):
        if techo - medianas_por_n[n] <= epsilon + 1e-12:
            return n, techo
    raise AssertionError("regla vacia")


def main():
    por_n = {}
    for n in NS:
        with (ROOT / "outputs" / f"seriesx{n}_metricas.csv").open(newline="") as f:
            metricas = list(csv.DictReader(f))
        with (ROOT / "outputs" / f"seriesx{n}_resumen.csv").open(newline="") as f:
            resumen = list(csv.DictReader(f))
        presupuestos = sorted({int(r["presupuesto"]) for r in metricas})
        top = max(presupuestos)
        medianas = [float(r["J_mediana"]) for r in metricas
                    if int(r["presupuesto"]) == top and r["J_mediana"]]
        mejores = [float(r["mejor_J"]) for r in resumen
                   if int(r["presupuesto"]) == top and r["mejor_J"]]
        por_n[n] = {"presupuesto": top,
                    "mediana_max_metodos": max(medianas),
                    "mejor_J": max(mejores)}
    medianas = {n: v["mediana_max_metodos"] for n, v in por_n.items()}
    elegido, techo = elegir(medianas)
    (ROOT / "outputs/seriesx_eleccion.json").write_text(json.dumps(
        {"epsilon": EPSILON, "techo": techo, "elegido": elegido,
         "por_n": por_n}, indent=2) + "\n")
    xs = sorted(por_n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=[por_n[n]["mediana_max_metodos"] for n in xs],
        mode="lines+markers", name="mejor mediana",
        line={"color": "#087f70", "width": 2}, marker={"size": 8}))
    fig.add_trace(go.Scatter(
        x=xs, y=[por_n[n]["mejor_J"] for n in xs],
        mode="lines+markers", name="mejor J",
        line={"color": "#b65b38", "dash": "dash", "width": 2},
        marker={"size": 8}))
    fig.add_vline(x=elegido, line={"color": "#888888", "dash": "dot"},
                  annotation_text=f"elegido n={elegido}")
    fig.update_xaxes(title_text="n (exogenas por conjunto)",
                     tickvals=list(xs))
    fig.update_yaxes(title_text="-AICc en train")
    fig.update_layout(width=900, height=550, template="plotly_white",
                      margin={"l": 75, "r": 35, "t": 40, "b": 70})
    fig.write_image(ROOT / "outputs/seriesx_codo.png", scale=2)
    print(f"elegido n={elegido} techo={techo:.4f} -> outputs/seriesx_eleccion.json",
          flush=True)


if __name__ == "__main__":
    main()
