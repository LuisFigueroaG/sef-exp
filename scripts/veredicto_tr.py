"""Tiempos a epsilon y brechas integradas del protocolo.

Para cada track con trazas por trial: T(eps) = primer trial con
brecha <= eps (None si censura a maximo presupuesto) y R = media
de la brecha en 1..B. Referencia: optimo de oraculo (series,
series2) o mejor observado (fase3b, fase4). Escribe
outputs/veredicto_tr.csv e imprime el resumen.
"""

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPS_AUC = (0.005, 0.002, 0.001)

TRACKS = {
    "fase3b": {"runs": "work/fase3b_runs", "ref": 0.713542318119154,
               "eps": EPS_AUC},
    "fase4": {"runs": "work/fase4_runs", "ref": 0.7615923356582383,
              "eps": EPS_AUC},
    "series": {"runs": "work/series_runs", "ref": -549.0535808233672,
               "eps": (2.0,)},
    "series2": {"runs": "work/series2_runs", "ref": -535.4400277398996,
                "eps": (2.0,)},
}


def curva(path):
    js = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["mejor_J"]:
                js.append(float(row["mejor_J"]))
            elif js:
                js.append(js[-1])
    return js


def main():
    filas = []
    for track, cfg in TRACKS.items():
        for path in sorted((ROOT / cfg["runs"]).glob("*.csv")):
            metodo = path.stem.rsplit("_", 1)[0]
            js = curva(path)
            if not js:
                continue
            brechas = [cfg["ref"] - j for j in js]
            fila = {"track": track, "metodo": metodo,
                    "semilla": path.stem.rsplit("_", 1)[1],
                    "trials": len(js),
                    "R_media": statistics.mean(brechas)}
            for eps in cfg["eps"]:
                t = next((i + 1 for i, r in enumerate(brechas) if r <= eps),
                         None)
                fila[f"T_{eps}"] = "" if t is None else t
            filas.append(fila)
    destino = ROOT / "outputs/veredicto_tr.csv"
    campos = ["track", "metodo", "semilla", "trials", "R_media",
              "T_0.005", "T_0.002", "T_0.001", "T_2.0"]
    with destino.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)
    print(f"{len(filas)} corridas -> {destino}", flush=True)
    for track in TRACKS:
        for metodo in ("aleatorio", "local", "ga_local", "sef_completo"):
            sel = [r for r in filas if r["track"] == track
                   and r["metodo"] == metodo]
            if not sel:
                continue
            r_media = statistics.mean(float(r["R_media"]) for r in sel)
            cols_t = [c for c in sel[0] if c.startswith("T_")]
            ts = {c: (sum(1 for r in sel if r[c] == ""),
                      [int(r[c]) for r in sel if r[c] != ""]) for c in cols_t}
            linea = {c: f"cens={v[0]} med={statistics.median(v[1]) if v[1] else '-'}" for c, v in ts.items()}
            print(track, metodo, f"R={r_media:.5f}", linea, flush=True)


if __name__ == "__main__":
    main()
