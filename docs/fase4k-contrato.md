# Barrido de k en Taiwan: elegir cantidad de variables

## Contrato previo a resultados

Fecha: 2026-09-23. Todo lo hecho fija k por decreto (6 en
scorecards, 4/3 en series). Este experimento deja que los datos
hablen: se corre el mismo motor en Taiwan para k=2..8 y una regla
escrita antes elige. k=6 no se recorre: es fase4 y se reutilizan
sus outputs tal cual.

## Espacio por k

Mismas 11 familias y mismo IV 0.02. Estructurales: k=2: 235,
k=3: 1405, k=4: 5450, k=5: 14374, k=6: 26262 (fase4), k=7: 33210,
k=8: 28485. Presupuestos (100, 250, 500, 1000, 2000) con tope en
el espacio; 20 semillas; mismos 4 metodos mas forward por k.

## Regla de eleccion (fijada antes de correr)

Entre los mejores medianos a maximo presupuesto de cada k, se
elige el **menor k cuya mediana este a 0,001 o menos del mejor
mediano global**. Parsimonia con tolerancia: si k=3 empata con
k=8 dentro de una milesima, gana k=3. Sin empates a mano ni
segundas vueltas: lo que salga se reporta.

## Ejecucion

```sh
uv run python scripts/barrido_k.py
uv run python scripts/elegir_k.py
```

`scripts/barrido_k.py` corre motor + forward + metricas + figura
por k con nombre `fase4k{k}`. `scripts/elegir_k.py` aplica la
regla, escribe `outputs/fase4k_eleccion.json` y dibuja
`outputs/fase4k_codo.png` (mediana y mejor J por k, elegido
marcado). Los J son comparables entre k: mismo evaluador, mismos
datos, misma definicion de J.

## Resultados

Regla aplicada: **elegido k=5** (mediana 0,76146 a 0,00013 del
techo 0,76159 de k=6). Codo clasico: 0,7414 (k=2), 0,7533 (k=3),
0,7599 (k=4), 0,7615 (k=5), techo en k=6, leve caida en k=7
(0,76148) y k=8 (0,76138): mas variables no pagan bajo estos
filtros. Los espacios chicos se agotan (k=2: 235/235 ajustes,
k=3: 1405/1405), asi que esos techos son casi-oraculos.

Incidente: `curvas_granulares` fallaba al concatenar trazas con
`mejor_J` inferido como String (primeras 100 filas vacias en
k=8). Se castea por archivo con strict=False antes de
concatenar; vale para todos los tracks.
