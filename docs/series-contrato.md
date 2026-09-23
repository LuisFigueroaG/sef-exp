# Track series: ARIMAX sobre IMACEC

## Contrato previo a resultados

Fecha: 2026-09-23. Track separado (no fase 5): otro evaluador, otra
pregunta. Se busca (p,q) + 2 exogenas que minimicen AICc sobre el
IMACEC mensual chileno. El test (ultimos 24 meses) queda cerrado
para validacion externa posterior; la busqueda usa solo train.

## Datos

`scripts/descargar_mindicador.py` trae IMACEC, IPC, TPM, dolar,
cobre y desempleo desde mindicador.cl (API abierta sin clave, que
republica al Banco Central; la BDE directa estaba en mantencion).
Ventana comun 2013-01 a 2025-12: 156 meses. El cobre empieza en
oct-2012 y el desempleo en 2010, por eso la ventana no arranca
antes. Agregacion mensual: dolar y cobre por media, tpm por ultimo
valor del mes; imacec, ipc y desempleo ya son mensuales. Train
132 meses hasta 2023-12, test 24 meses cerrado. Crudos en
`data/series/crudo/` (ignorados), particion versionada.

## Espacio

Siete familias: `pq` con 12 combinaciones (p 0..3, q 0..2) mas
cinco exogenas singleton. k=3: un token pq + 2 exogenas. Numero
estructural: **130** conjuntos, de los cuales 10 no traen token pq
y se rechazan siempre por estructura (8% del espacio; consumen
trial como cualquier rechazo). El oraculo exhaustivo cubre los 130.

Los lags de exogenas no se buscan en v1 (documentado como
extension futura: cada familia ganaria opciones).

## Evaluador

`src/sef_exp/evaluador_series.py`, sin WOE/IV/corr/VIF. J=-AICc
(el motor maximiza). d fijo por KPSS repetidos (0..2) solo en
train, registrado en el manifiesto. Exogenas en z-score de train.
Factibilidad: convergencia del MLE mas raices AR/MA fuera del
circulo unidad; si no, rechazo con causa. Todo determinista
(statsmodels). En este train el KPSS no pidio diferencias: d=0.

## Comparacion fijada

Metodos: aleatorio, local, ga_local y sef_completo con el mismo
motor parametrizado, mas `hk_forward` como referencia
descriptiva: greedy sobre exogenas con (p,q) re-entonado por paso
HK local en cada evaluacion, determinista, sin presupuesto,
ensayos contados. Presupuestos 25, 50, 75 y 100 trials unicos
porque el espacio es chico; 20 semillas. El sustituto usa dummies
de exogenas mas p,q enteros (decision v1 documentada).

El piloto corre 25 trials con semillas 0 y 1 para medir costo y
mezcla de rechazos. No se elige ganador ni se ajustan parametros
segun el piloto.

## Ejecucion

```sh
uv run python scripts/descargar_mindicador.py
uv run python -m sef_exp.series --piloto --trials 25 --semillas 0,1
uv run python -m sef_exp.series --sin-graficos
uv run python -m sef_exp.series --hk-ref
uv run python -m sef_exp.series --solo-graficos
```

`work/series_*` desechable. `outputs/series_resumen.csv`,
`series_metricas.csv`, `series_estabilidad.csv`,
`series_referencias.csv`, `series_oraculo.csv`, `series_curvas.png` y
`series_manifest.json` versionables. La columna `dentro_001` de
metricas significa "a epsilon del mejor", con epsilon=2 puntos AICc
en este track.

## Resultados

Piloto (25 trials, semillas 0 y 1): precompute 0,1 s con d=0 por
KPSS; 8 corridas en 3,5 s. Mezcla de rechazos: estructura y no
convergencia; el sustituto aun no activa a 25 trials.

Corrida completa: 80/80 en 2,4 s de pared porque piloto y HK ya
habian cubierto casi todo el espacio de 130 (15 ajustes nuevos,
7985 consultas al cache). Oraculo 130/130: el mejor observado
J=-549,0536 (`COBRE|DESEMPLEO|p3q0`, AICc 549,05) es el optimo
exhaustivo. El HK queda en -549,31 (AICc 549,31, 79 ensayos):
primera replica donde el greedy NO alcanza el pico, por 0,25
puntos. Local llega a la mediana optima ~trial 40, ga/sef ~75-100,
aleatorio 19/20 a epsilon 2 en 100.
