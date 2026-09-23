# Resultados de fase 3

Corrida del 2026-09-22 con Python 3.12.13 y `uv`, segun el contrato
previo de `docs/fase3.md`. Se ejecutaron ocho variantes por 20 semillas:
160 corridas. Cada una llego a los 1792 conjuntos estructurales. El
presupuesto nominal 2000 se censuro en 1792. Se reutilizo el oraculo de
fase 2 como tabla de evaluaciones; esta corrida no volvio a ajustar los
modelos de CV. El test reservado sigue cerrado.

## Llegada al optimo

El optimo de referencia es J = 0,7112047307. La brecha media de la tabla
esta en milesimas de AUC. `T` es la mediana de trials unicos hasta el
optimo entre las 20 corridas completas, no solo entre exitos a B=100.

| Metodo | Optimo B=100 | Brecha B=100 | Optimo B=250 | T mediana |
|---|---:|---:|---:|---:|
| Aleatorio (fase 2) | 1/20 | 0,6725 | 1/20 | 1077,5 |
| Local (fase 2) | 15/20 | 0,2345 | 19/20 | 84,5 |
| `ga_basico` | 4/20 | 0,3079 | 18/20 | 162 |
| `ga_local` | 5/20 | 0,2541 | 20/20 | 156 |
| `sef_completo` | 5/20 | 0,2541 | 20/20 | 156 |
| `sef_sin_local` | 4/20 | 0,3079 | 20/20 | 147 |
| `sef_sin_dobles` | 5/20 | 0,2798 | 20/20 | 123,5 |
| `sef_sin_reinicios` | 5/20 | 0,2541 | 20/20 | 156 |
| `sef_sin_guiada` | 5/20 | 0,4128 | 20/20 | 137 |
| `sef_sin_reparacion` | 4/20 | 0,3048 | 20/20 | 129,5 |

A B=500 todas las variantes evolutivas llegaron 20/20. El baseline local
llego 20/20 y el aleatorio 4/20. Los umbrales epsilon 0,005 y 0,002
saturaron casi desde B=100; el optimo exacto discrimina mejor aqui. Las
curvas y la tabla completa conservan tambien epsilon 0,001.

## Aporte de los componentes

En `sef_completo` la primera prediccion del sustituto ocurrio entre los
trials 161 y 193 (mediana 169), una vez reunidos 64 conjuntos factibles.
Por eso `ga_local` y `sef_completo` son identicos hasta B=100. En 14
semillas ambos encontraron el optimo antes de usar el sustituto. En las
otras seis, `sef_completo` llego antes que `ga_local`; no llego despues
en ninguna semilla. No cambia la tasa de exito a B=250, pero si la
trayectoria de esas seis corridas. La mediana de T sigue en 156 para
ambos. A B=250, el error absoluto prequential medio del sustituto fue
0,000771 y la correlacion de Spearman prequential mediana, 0,727. Es
una prediccion util para ordenar parte de la bolsa, no una estimacion
calibrada de incertidumbre.

Quitar movimientos dobles mejoro T en 11 semillas, empato en cinco y
empeoro en cuatro frente a `sef_completo`; su mediana fue 123,5. Quitar
la reparacion de cruces mejoro T en 12, empato en cuatro y empeoro en
cuatro, pero genero 198.302 propuestas estructuralmente invalidas en
las 20 corridas completas, frente a cero con reparacion. A B=250 hizo
4385 propuestas por corrida en promedio, frente a 3510 del metodo
completo. Por eso no es una simplificacion eficiente aunque llegue antes
al optimo en algunas semillas. Sin reinicios, T fue identico en las 20
semillas: los reinicios sucedieron demasiado tarde para ayudar en este
subespacio. Las otras ablaciones tienen efectos mixtos. Las comparaciones
de variantes comparten semilla e inicializacion aleatoria de 48
conjuntos, pero al cambiar operadores tambien cambian las trayectorias
posteriores; estas diferencias no identifican efectos universales.

## Recursos y decision

A B=100, el local simple necesita en promedio 122 propuestas y encuentra
el optimo en 15/20; `sef_completo` necesita 735 propuestas y llega en
5/20. Su menor tiempo estimado de modelo en ese punto (13,73 s frente a
15,62 s) no compensa la menor tasa de exito. A B=250, el local simple
llega en 19/20 con 299 propuestas y 34,62 s de modelo estimado; el SEF
completo llega en 20/20 con 3510 propuestas, 38,51 s de modelo estimado
y 0,076 s de algoritmo sobre la tabla. `ga_local` tambien llega 20/20,
con 3373 propuestas y 39,90 s de modelo estimado. El sustituto agrega
entrenamientos y complejidad para un adelanto en seis trayectorias, no
una ventaja clara en el checkpoint principal.

**Decision:** conservar el local simple como referencia fuerte y el
genetico local como comparador de menor complejidad. Conservar SEF como
prototipo experimental, sin declararlo superior ni llevarlo directamente
a un scorecard. Antes de abrir el test, la siguiente decision es fijar
un conjunto finalista y un protocolo externo sin elegirlo por resultados
del test. Ampliar el catalogo de familias requeriria otro experimento de
CV; el oraculo de este subespacio no cubre las 35 variables.

## Lectura de costos y limites

`pared_modelo_s` suma tiempos observados al construir el oraculo. No es
la duracion de pared de cada nueva corrida. `pared_algoritmo_s` mide
busqueda y entrenamiento del sustituto al consultar la tabla. El
precompute de WOE/IV y la enumeracion del oraculo quedan fuera del
presupuesto del buscador; el costo de IV guiado es comun, no gratuito en
un despliegue nuevo. No se midio memoria pico por corrida, asi que no
hay conclusion sobre ese recurso. Una unica tarea con 20 semillas mide
variabilidad del optimizador, no variabilidad entre datasets. El
subespacio se eligio con conocimiento exploratorio de fase 1; las tasas
de exito no se extrapolan al catalogo completo. No se abrio el test ni
se midio generalizacion.

Archivos versionables: `outputs/fase3_resumen.csv`,
`outputs/fase3_metricas.csv`, `outputs/fase3_curvas.png`,
`outputs/fase3_ablaciones.png` y `outputs/fase3_manifest.json`. Las
trazas por trial estan en `work/fase3_runs/` y son desechables. El
manifiesto registra configuracion, versiones y hash del codigo; tras la
corrida solo cambio el eje categorico de la figura de ablaciones, y se
actualizo ese hash. Para reproducir desde cero hacen falta los mismos
datos, dependencias y oraculo de fase 2.
