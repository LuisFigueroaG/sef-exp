"""Aplica la regla de docs/fase4k-contrato.md: menor k con mediana
a epsilon del mejor mediano global (k=6 sale de fase4 tal cual).
Escribe outputs/fase4k_eleccion.json y outputs/fase4k_codo.png.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KS = (2, 3, 4, 5, 6, 7, 8)
EPSILON = 0.001


def metricas_de(k):
    base = "fase4" if k == 6 else f"fase4k{k}"
    with (ROOT / "outputs" / f"{base}_metricas.csv").open(newline="") as f:
        metricas = list(csv.DictReader(f))
    with (ROOT / "outputs" / f"{base}_resumen.csv").open(newline="") as f:
        resumen = list(csv.DictReader(f))
    return metricas, resumen


def elegir(medianas_por_k, epsilon=EPSILON):
    techo = max(medianas_por_k.values())
    for k in sorted(medianas_por_k):
        if techo - medianas_por_k[k] <= epsilon + 1e-12:
            return k, techo
    raise AssertionError("regla vacia")


def main():
    por_k = {}
    for k in KS:
        metricas, resumen = metricas_de(k)
        presupuestos = sorted({int(r["presupuesto"]) for r in metricas})
        top = max(presupuestos)
        medianas = [float(r["J_mediana"]) for r in metricas
                    if int(r["presupuesto"]) == top and r["J_mediana"]]
        mejores = [float(r["mejor_J"]) for r in resumen
                   if int(r["presupuesto"]) == top and r["mejor_J"]]
        por_k[k] = {"presupuesto": top,
                    "mediana_max_metodos": max(medianas),
                    "mejor_J": max(mejores)}
    medianas = {k: v["mediana_max_metodos"] for k, v in por_k.items()}
    elegido, techo = elegir(medianas)
    salida = {"epsilon": EPSILON, "techo": techo, "elegido": elegido,
              "por_k": por_k}
    (ROOT / "outputs/fase4k_eleccion.json").write_text(
        json.dumps(salida, indent=2) + "\n")

    import plotly.graph_objects as go
    xs = sorted(por_k)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=[por_k[k]["mediana_max_metodos"] for k in xs],
        mode="lines+markers", name="mejor mediana",
        line={"color": "#087f70", "width": 2}, marker={"size": 8}))
    fig.add_trace(go.Scatter(
        x=xs, y=[por_k[k]["mejor_J"] for k in xs],
        mode="lines+markers", name="mejor J",
        line={"color": "#b65b38", "dash": "dash", "width": 2},
        marker={"size": 8}))
    fig.add_vline(x=elegido, line={"color": "#888888", "dash": "dot"},
                  annotation_text=f"elegido k={elegido}")
    fig.update_xaxes(title_text="k (variables por conjunto)",
                     tickvals=list(xs))
    fig.update_yaxes(title_text="AUC medio de cinco folds")
    fig.update_layout(width=900, height=550, template="plotly_white",
                      margin={"l": 75, "r": 35, "t": 40, "b": 70})
    fig.write_image(ROOT / "outputs/fase4k_codo.png", scale=2)
    print(f"elegido k={elegido} techo={techo:.6f} -> outputs/fase4k_eleccion.json",
          flush=True)


if __name__ == "__main__":
    main()
