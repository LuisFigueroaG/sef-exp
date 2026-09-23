# Veredicto: que muestran los experimentos

## Metodo de este documento (fijado antes de los numeros nuevos)

Se agrega el baseline TPE que la propuesta pedia y nunca se
implemento, sobre series2 (oraculo conocido, barato): muestrea
una opcion por familia con Optuna (`TPESampler` por semilla),
repara a k exacto como el cruce del genetico (documentado abajo)
y cuenta igual que los demas (duplicados consumen propuestas,
unicos consumen trials). No entra al manifiesto de series2:
estudio separado que reusa su cache.

Se computan las metricas del protocolo (`docs/protocolo.md`)
que faltaban en todos los tracks: tiempo a epsilon T (primer
trial con brecha <= epsilon, censurado si no llega) y brecha
integrada R (media de la brecha en 1..B). Referencia: optimo de
oraculo donde existe (series, series2), mejor observado en
3b y 4. Epsilons del protocolo (0.005, 0.002, 0.001) en
scorecards; 2 puntos AICc en series.

## Resultados TPE

En series2 a 1000 trials: mediana -535,44 (el optimo), 20/20 a
epsilon 2. Empata arriba con los tres guiados. Su T_2.0 mediano
es 86, el mas rapido de los cinco (local 108, sef 119, ga 126,
aleatorio 129). Paga 476 s de overhead Optuna por corrida
contra centavos de los demas: velocidad de trials, no de reloj.

## Tiempos y brechas por track

Medianas de T (censuras entre parentesis) y R media:

- fase3b: T_0.001: aleat 746 (17 cens), local 343, ga 206,
  sef 185. R: 0,00215 / 0,00077 / 0,00058 / 0,00052.
- fase4: T_0.001: aleat 485 (11 cens), local 151, ga 52,
  sef 52. R: 0,00242 / 0,00084 / 0,00042 / 0,00041.
- series (eps 2): T: aleat 17,5, local 24, ga 32,5, sef 32,5.
  R: 1,61 / 1,93 / 2,67 / 2,67.
- series2 (eps 2): T: aleat 129 (1 cens), local 108, ga 125,5,
  sef 119, tpe 86. R: 1,24 / 0,50 / 0,42 / 0,38.

## Veredicto

1. Ningun metodo guiado se separa de otro de forma relevante
   en ningun track. Las diferencias (sef primero en 3b por 21
   trials, ultimo en series, primero en R de series2) son de
   segundo orden y cambian de signo entre tracks.
2. El greedy empata en scorecards (3b, 4: mismo conjunto) y
   pierde en series (0,25 y 0,5 AICc). La forma del paisaje
   manda, no el algoritmo.
3. El sustituto no decide ningun resultado: ga_local y
   sef_completo empatan en medianas de todos los tracks. TPE,
   el otro metodo con modelo, empata arriba y llega antes
   pero a costo de reloj muy superior.
4. Aleatorio es el unico siempre peor, y ni a 2000 converge
   del todo (19/20 en series2, 17 censuras en 3b a 0,001).
   La busqueda guiada si importa; cual de ellas, no tanto.
5. Generalizacion: los finalistas de scorecards aguantan en
   test (+0,001 ambas); series empata al naive con n=24.

Pregunta abierta que estos datos no cierran: evaluaciones
caras (minutos por trial) con interacciones plantadas, donde
el sustituto ahorraria horas y no centavos. Ese experimento
hostil por diseño es el unico que podria revertir el punto 3.
