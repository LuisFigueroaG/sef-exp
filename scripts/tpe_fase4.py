"""TPE sobre fase4 (Taiwan): mismo motor generico, misma cache.

20 semillas x 2000 trials. Escribe work/tpe_fase4_runs/ y
outputs/tpe_fase4_resumen.csv + outputs/tpe_fase4_curvas.png.
Estudio separado: no toca el manifiesto de fase4.
"""

import argparse
import csv
import time
from pathlib import Path

from sef_exp import fase3b, tpe_comun
from sef_exp.fase4 import CFG

RUNS_DIR = Path(str(CFG.runs_dir).replace("fase4_runs", "tpe_fase4_runs"))
SUMMARY_PATH = Path(str(CFG.summary_path).replace("fase4_resumen",
                                                  "tpe_fase4_resumen"))
FIGURE_PATH = Path(str(CFG.figure_path).replace("fase4_curvas",
                                                "tpe_fase4_curvas"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semillas", default=",".join(map(str, range(20))))
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--solo-figura", action="store_true")
    args = parser.parse_args()
    if args.solo_figura:
        tpe_comun.figura(CFG, RUNS_DIR, FIGURE_PATH)
        return
    previos = fase3b.aplicar_umbrales(CFG)
    try:
        context, _, _ = (CFG.preparar_fn or fase3b.preparar)(CFG)
        cache = fase3b.Evaluaciones(cfg=CFG)
        t0 = time.monotonic()
        filas = []
        for i, seed in enumerate(int(x) for x in args.semillas.split(",")):
            print(f"[{i + 1}] tpe semilla={seed}", flush=True)
            filas.append(tpe_comun.correr(CFG, seed, args.trials,
                                          context, cache, RUNS_DIR))
        with SUMMARY_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=("metodo", "semilla", "trials", "propuestas",
                               "duplicados", "mejor_J", "mejor_S",
                               "pared_modelo_s"), lineterminator="\n")
            writer.writeheader()
            for r in filas:
                r = dict(r)
                r["mejor_J"] = "" if r["mejor_J"] is None else repr(r["mejor_J"])
                r["mejor_S"] = "" if r["mejor_S"] is None else "|".join(r["mejor_S"])
                writer.writerow(r)
        print(f"TPE listo en {time.monotonic() - t0:.1f}s -> {SUMMARY_PATH}",
              flush=True)
        tpe_comun.figura(CFG, RUNS_DIR, FIGURE_PATH)
    finally:
        fase3b.restaurar_umbrales(previos)


if __name__ == "__main__":
    main()
