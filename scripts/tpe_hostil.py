"""TPE sobre hostil: motor generico, misma cache.

10 semillas x 250 trials. Escribe work/tpe_hostil_runs/ y
outputs/tpe_hostil_resumen.csv + outputs/tpe_hostil_curvas.png.
Estudio separado: no toca el manifiesto de hostil.
"""

import argparse
import csv
import time
from pathlib import Path

from sef_exp import fase3b, tpe_comun
from sef_exp.hostil import CFG

RUNS_DIR = Path(str(CFG.runs_dir).replace("hostil_runs", "tpe_hostil_runs"))
SUMMARY_PATH = Path(str(CFG.summary_path).replace("hostil_resumen",
                                                  "tpe_hostil_resumen"))
FIGURE_PATH = Path(str(CFG.figure_path).replace("hostil_curvas",
                                                "tpe_hostil_curvas"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semillas", default=",".join(map(str, range(10))))
    parser.add_argument("--trials", type=int, default=250)
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
