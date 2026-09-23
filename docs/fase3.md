# Fase 3: SEF en el subespacio de fase 2

## Contrato previo a resultados

Fecha: 2026-09-22. Se reutilizan los 16 predictores, ocho familias, cinco
folds y umbrales de `docs/fase2.md`. El algoritmo lee el resultado del
oraculo solo para un conjunto que ya decidio evaluar. El codigo de metricas
lee el maximo global como referencia, pero ese valor no entra en la
generacion, filtrado ni orden de propuestas. El test reservado sigue cerrado. Esta tarea es
exploratoria porque el subespacio ya incorporaba un conjunto conocido en
fase 1.

Cada metodo usa semillas 0 a 19 y presupuestos 100, 250, 500, 1000 y 2000.
Como solo hay 1792 conjuntos estructurales, 2000 queda censurado en 1792.
Un trial unico consume el resultado del evaluador comun, incluso si este
rechaza el conjunto. Duplicados y propuestas estructuralmente invalidas
no consumen trial, pero se cuentan. El tiempo de modelo se estima sumando
los tiempos medidos al construir el oraculo; el tiempo de algoritmo se mide
durante la consulta a la tabla. Se informan por separado.

## Genetico por familias

Cada individuo es una tupla de ocho posiciones. `-1` significa familia
ausente y `0` o `1` el representante elegido. Exactamente seis posiciones
estan activas. Los operadores conservan la cuota de una por familia. No hay
pares incompatibles adicionales. El cruce conserva elecciones compartidas
entre padres y completa desde su union; una reparacion estructural aleatoria
ajusta la cardinalidad a seis. La mutacion cambia representante en la misma
familia con probabilidad 0,5, sustituye una familia con 0,3 o hace dos
cambios con 0,2. Si un movimiento no cambia el individuo, se vuelve a
proponer dentro de un limite de intentos.

La inicializacion evalua 64 conjuntos: 48 aleatorios sin reemplazo y 16
guiados por IV medio de los cinco train folds. Los primeros 48 son iguales
para todas las variantes con una misma semilla. Los candidatos guiados se
construyen con un ranking de IV perturbado con ruido normal de escala 0,05,
sin usar J.
La ablacion de inicializacion usa 64 aleatorios y conserva los mismos 48
primeros. El costo de calcular IV es parte del precompute comun y se
informa. La poblacion mantiene hasta 64 individuos con J valido. Ocho
plazas son elites; las demas se completan mediante torneos de tres, con
preferencia por diversidad si la diferencia de J no supera 0,0005.

Se forman lotes de 16 trials desde una bolsa de hasta 128 propuestas
estructurales. La bolsa pequena adapta las 640 propuestas de la propuesta
original al espacio de solo 1792 conjuntos. Sin sustituto, nueve plazas
vienen de descendencia genetica, tres de inmigrantes aleatorios y cuatro
de vecinos locales cuando el metodo tiene busqueda local. Sin local, sus
cuatro plazas pasan a descendencia genetica. Si un canal no ofrece
candidatos, las plazas se llenan con propuestas aleatorias no evaluadas.

La busqueda local usa vecinos a distancia uno o dos de los mejores
individuos. La ablacion sin dobles limita tanto la mutacion como esta
busqueda a movimientos simples. Tras diez lotes sin mejorar el mejor J al
menos 0,0005, el reinicio descarta la mitad de la poblacion no elite y
prioriza 32 inmigrantes aleatorios evaluados en lotes posteriores. La
ablacion sin reinicios desactiva esta regla. No hay parada temprana por
estancamiento.

## Sustituto

Despues de observar 64 conjuntos factibles, un `ExtraTreesRegressor` de
32 arboles aprende J a partir de los 16 indicadores binarios de inclusion.
Usa `min_samples_leaf=2` y `max_features=0.8`. Se actualiza cada cuatro
lotes para limitar costo. En cada lote con modelo entrenado, seis plazas
priorizan mayor J predicho y tres priorizan J predicho mas la desviacion
entre arboles. La desviacion es una heuristica, no un intervalo calibrado.
Los otros siete lugares conservan tres inmigrantes y cuatro vecinos.

Antes de evaluar cada candidato priorizado se guarda su prediccion. El
error absoluto y el ranking se calculan despues solo sobre candidatos
con J valido. Las predicciones nunca sustituyen una evaluacion observada.

## Variantes y comparacion

| Variante | Local | Sustituto | Dobles | Reinicio | IV guiado | Reparacion |
|---|---:|---:|---:|---:|---:|---:|
| `ga_basico` | No | No | Si | Si | Si | Si |
| `ga_local` | Si | No | Si | Si | Si | Si |
| `sef_completo` | Si | Si | Si | Si | Si | Si |
| `sef_sin_local` | No | Si | Si | Si | Si | Si |
| `sef_sin_dobles` | Si | Si | No | Si | Si | Si |
| `sef_sin_reinicios` | Si | Si | Si | No | Si | Si |
| `sef_sin_guiada` | Si | Si | Si | Si | No | Si |
| `sef_sin_reparacion` | Si | Si | Si | Si | Si | No |

`ga_local` es tambien la ablacion sin sustituto. `sef_sin_reparacion`
descarta un cruce con cardinalidad incorrecta y registra el descarte;
las mutaciones y las propuestas aleatorias siguen siendo estructuralmente
validas. Dado que no hay conflictos `E`, esta comparacion solo mide la
reparacion de cardinalidad del cruce.

Las comparaciones principales son brecha al oraculo, tasa de llegada al
optimo y a epsilon 0,005, 0,002 y 0,001, trials unicos, propuestas totales,
rechazos, tiempo estimado de modelo, tiempo de algoritmo y diversidad.
Se comparan ademas con aleatorio y local de fase 2 a igual presupuesto y
semilla. Las 20 semillas son repeticiones del optimizador sobre una sola
tarea; no se trataran como 20 datasets independientes.

La decision de conservar sustituto o reparacion depende del rendimiento
y su costo, no de que el algoritmo sea mas complejo. Los resultados de CV
no establecen generalizacion ni una politica de scorecard operacional.
