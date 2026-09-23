# Barrido de exogenas en series: cuantas variables usar

## Contrato previo a resultados

Fecha: 2026-09-23. Espejo del barrido de k en Taiwan: aca la
pregunta es cuantas exogenas (0 a 5) con (p,q) libre. Pool de 7
candidatas sin lags (los lags confundirian la pregunta).
n=2 con este pool NO es series v1 (que usaba 5): se recorre
fresco. k=1+n por conteo.

Estructurales (pq 12 + 7 exogenas): n=0: 12, n=1: 84, n=2: 287,
n=3: 455, n=4: 441, n=5: 259. Los sin-pq se rechazan por
estructura. Presupuestos (25, 50, 75, 100) con tope; 20 semillas;
4 metodos mas hk con n concordante.

## Regla de eleccion (misma forma, otro epsilon)

Menor n con mediana a maximo presupuesto a **2 puntos AICc** o
menos del mejor mediano global (epsilon del track series, no
0,001 de scorecards). Lo que salga se reporta.

Nota: con k=1 la inicializacion guiada propone exogenas sueltas
(todas rechazadas); el azar cubre los 12 pq en pocos trials.
Espacio tan chico que el resultado roza el oraculo: se reporta
igual, la pregunta es el codo, no el ganador.

## Ejecucion

```sh
uv run python scripts/barrido_x.py
uv run python scripts/elegir_x.py
```

`outputs/seriesx{n}_*` por conteo, `outputs/seriesx_eleccion.json`
y `outputs/seriesx_codo.png` con la regla.

## Resultados

Regla aplicada: **elegido n=2** (mediana -549,05 a 0,82 del techo
-548,23 de n=3). Curva con joroba: -557,97 (n=0), -551,80 (n=1),
-549,05 (n=2), techo -548,23 (n=3), caida a -549,55 (n=4) y
-551,84 (n=5). El AICc penaliza el exceso de exogenas tal como
predica la teoria. Todos los espacios se agotaron (oraculos
efectivos), asi que cada techo es el optimo de su conteo.

Incidente: el motor asumia espacios de 48+ conjuntos (inicial
sin tope y bolsa de respaldo sin salida). En n=0 (19 sets) el ga
giraba infinito. Se acoto la inicial a min(48, total), la
repoblacion con intentos acotados y la bolsa con cierre de
corrida; mas test de terminacion en espacio chico. Vale para
todos los tracks sin cambiar su comportamiento.
