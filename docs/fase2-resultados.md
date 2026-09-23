# Resultados de fase 2

Corrida del 2026-09-22 con Python 3.12.13, `uv` y la especificacion de
`docs/fase2.md`. El test reservado no se abrio.

## Oraculo

El espacio tiene 1792 conjuntos estructurales. Pasaron los filtros 416
(23,2 %); los otros 1376 se rechazaron por correlacion de WOE en train de
algun fold. El mejor conjunto fue `AM03, CVPRPAYMNT11, CVSC150, GO243, IN20,
PR09`, con J de CV igual a **0,7112047307**. Sus AUC por fold fueron 0,7186493,
0,7078068, 0,7090980, 0,7098134 y 0,7106562.

La suma de los tiempos medidos por candidato fue 161,34 s, mas 4,6 s para
binning y WOE en la corrida inicial. La mediana por candidato fue 0,0094 s.
Los candidatos factibles tardaron en promedio 0,357 s porque ajustan cinco
logisticas; los rechazados tardaron 0,0094 s en promedio.

## Comparacion

La tabla muestra corridas que encontraron exactamente el mejor J del oraculo
y brecha media `J* - mejor J` entre las 20 semillas. Forward es determinista
y tiene una sola corrida.

| Trials unicos disponibles | Aleatorio, optimo | Local, optimo | Brecha media aleatorio | Brecha media local |
|---:|---:|---:|---:|---:|
| 100 | 1/20 | 15/20 | 0,0006725 | 0,0002345 |
| 250 | 1/20 | 19/20 | 0,0004958 | 0,0000542 |
| 500 | 4/20 | 20/20 | 0,0003177 | 0 |
| 1000 | 9/20 | 20/20 | 0,0001540 | 0 |
| 1792 | 20/20 | 20/20 | 0 | 0 |

Forward termino tras 66 trials, incluidos modelos parciales, con J de
0,7105176946. La brecha fue 0,0006870361. No alcanzo el optimo exacto.

El tiempo mediano hasta el optimo exacto fue 84,5 evaluaciones unicas para
local y 1077,5 para aleatorio. Local necesito una mediana de 105 propuestas
hasta ese punto, frente a 1077,5 de aleatorio. Al agotar todo el espacio,
local hizo en promedio 23.835 propuestas por corrida; la revision de
vecindarios repetidos tiene un costo real aunque la cache evita nuevos
ajustes. En el presupuesto 100, local hizo 121,7 propuestas en promedio.

El presupuesto nominal 2000 excede el espacio. Las corridas de conjuntos
completos se detuvieron en 1792 trials y forward en 66. La figura usa esos
conteos reales en el eje X.

## Interpretacion y limites

En este subespacio, local encontro el mejor conjunto con menos evaluaciones
de modelos que aleatorio. El subespacio incluyo deliberadamente un conjunto
factible conocido por la verificacion manual de fase 1. Esta seleccion
exploratoria limita cualquier extrapolacion al catalogo de 35 variables.
El resultado no prueba una ventaja para SEF: el
algoritmo evolutivo todavia no se implemento. Tampoco prueba mejor
generalizacion; J se calculo en los folds train y el test sigue reservado.

Los baselines consultaron una tabla de resultados del oraculo solo para
sus propuestas. `pared_modelo_s` suma el tiempo observado por candidato en
la enumeracion; sirve como aproximacion del costo de ajustes, no como tiempo
de reloj de una ejecucion independiente. `pared_ejecucion_s` mide el buscador
sobre esa tabla. Forward si ajusto directamente sus modelos parciales.
La metrica de brecha integrada `R` queda pendiente: antes de la primera
solucion factible, `b(t)` no existe y asignarle un valor artificial cambiaria
la comparacion. Se reportan por separado primer conjunto factible, brecha y
tiempos de llegada con censura.

Archivos: `outputs/fase2_oraculo.csv`, `outputs/fase2_resumen.csv`,
`outputs/fase2_metricas.csv`, `outputs/fase2_curvas.png` y
`work/fase2_trials.csv`. `work/` es desechable; los CSV y PNG en `outputs/`
quedan versionables. La siguiente fase compara SEF con estas referencias
bajo el mismo evaluador y el mismo subespacio antes de ampliar familias.
