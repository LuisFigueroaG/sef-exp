"""Descarga IMACEC + 5 exogenas desde mindicador.cl y arma la particion.

Fuente: mindicador.cl (API abierta, sin clave; republica datos del
Banco Central). Ventana comun 2013-01 a 2026-07 (~163 meses): el cobre
empieza en oct-2012 y el desempleo en 2010. Agregacion mensual:
dolar y cobre por media, tpm por ultimo valor del mes; imacec, ipc
y desempleo ya vienen mensuales. Train hasta 2024-07, test cerrado
ultimos 24 meses. Cachea crudos en data/series/crudo/ para correr
offline y reproducible.
"""

import json
import time
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CRUDO = ROOT / "data/series/crudo"
OUT = ROOT / "data/series"

MENSUALES = ("imacec", "ipc", "tasa_desempleo")
MEDIA_DIARIA = ("dolar", "libra_cobre", "uf", "euro")
ULTIMO_DIARIO = ("tpm",)
INDICADORES = (*MENSUALES, *MEDIA_DIARIA, *ULTIMO_DIARIO)
ANIOS = range(2013, 2027)
TEST_MESES = 24


def traer(codigo, anio):
    destino = CRUDO / f"{codigo}_{anio}.json"
    if destino.exists():
        return json.loads(destino.read_text())
    url = f"https://mindicador.cl/api/{codigo}/{anio}"
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read())
    destino.write_text(json.dumps(payload))
    time.sleep(0.5)
    return payload


def mensualizar(codigo):
    filas = []
    for anio in ANIOS:
        for punto in traer(codigo, anio).get("serie", []):
            filas.append((punto["fecha"][:7], float(punto["valor"])))
    df = pl.DataFrame({"mes": [f for f, _ in filas],
                       "valor": [v for _, v in filas]})
    if codigo in MENSUALES:
        return df.group_by("mes").agg(pl.col("valor").last().alias(codigo)).sort("mes")
    if codigo in MEDIA_DIARIA:
        return df.group_by("mes").agg(pl.col("valor").mean().alias(codigo)).sort("mes")
    return df.group_by("mes").agg(pl.col("valor").last().alias(codigo)).sort("mes")


def main():
    CRUDO.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    base = None
    for codigo in INDICADORES:
        mensual = mensualizar(codigo)
        base = mensual if base is None else base.join(mensual, on="mes", how="inner")
    base = base.sort("mes")
    assert base.height >= 150, f"ventana corta: {base.height} meses"
    nulos = base.drop("mes").null_count().to_dicts()[0]
    if any(nulos.values()):
        raise RuntimeError(f"Huecos en la ventana comun: {nulos}")
    nombres = {"imacec": "IMACEC", "ipc": "IPC", "tasa_desempleo": "DESEMPLEO",
               "dolar": "DOLAR", "libra_cobre": "COBRE", "tpm": "TPM",
               "uf": "UF", "euro": "EURO"}
    base = base.rename(nombres)
    tr, te = base.slice(0, base.height - TEST_MESES), base.slice(-TEST_MESES)
    for nombre, parte in (("train", tr), ("test", te)):
        x = parte.select(["mes", *[c for c in parte.columns
                                    if c not in ("mes", "IMACEC")]])
        x.columns = ["", *[c for c in parte.columns
                            if c not in ("mes", "IMACEC")]]
        x.write_csv(OUT / f"X_{nombre}.csv")
        y = parte.select(["mes", "IMACEC"])
        y.columns = ["", "target"]
        y.write_csv(OUT / f"y_{nombre}.csv")
    print(f"ventana={base['mes'][0]}..{base['mes'][-1]} "
          f"train={tr.height} test={te.height} -> {OUT}")


if __name__ == "__main__":
    main()
