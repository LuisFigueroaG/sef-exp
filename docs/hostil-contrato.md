# Track hostil: interacciones plantadas + evaluacion cara

## Contrato previo a resultados

Fecha: 2026-09-23. El veredicto (`docs/veredicto.md`, punto 3)
dice que el sustituto no decide nada en paisajes dociles y
baratos. Este track lo pone a prueba en el unico escenario que
podria revertirlo: interacciones que el greedy no ve y trials
que duelen.

## Datos sinteticos (generador versionado)

`scripts/generar_hostil.py` (semilla fija): 10 familias x 3
opciones continuas N(0,1), 150k train + 50k test. Senal solo en
3 pares de distintas familias:
`logit = 3*(x1*x2 + x3*x4 + x5*x6) + ruido`. Las 24 variables
restantes son ruido puro. Marginales ~0 por construccion: el
greedy univariado no tiene de donde agarrarse. Las 6 variables
vivas estan en 6 de las 10 familias; 4 familias son señuelo.

## Espacio y evaluador

k=6 sobre 10 familias: C(10,6)*3^6 = **153.090** conjuntos.
Evaluador HGB (`HistGradientBoostingClassifier` de sklearn, sin
deps nuevas): `max_iter=50, max_leaf_nodes=31,
learning_rate=0.1, min_samples_leaf=200`, 5-fold CV AUC con
folds fijos. Sin filtros IV/corr/VIF (un arbol no los necesita);
factibilidad = converge + finito. J = AUC medio. Cada trial
cuesta ~2 s: ahi el sustituto tiene plata que ahorrar.

Enmienda pre-resultados: el piloto con max_iter=200 midio 8-11 s
por trial (inviable para 80k nominales). Se baja a max_iter=50
y presupuestos a (50, 100, 250, 500). El cache del piloto viejo
se resguarda y se parte de cero: otro evaluador, otros numeros.

## Comparacion fijada

Metodos: aleatorio, local, ga_local, sef_completo, tpe
(generico) y forward-greedy-sobre-ganancia como referencia
determinista (se espera que pierda: es el control que valida
que lo plantado es hostil de verdad). Presupuestos 50, 100
y 250; 10 semillas (0-9). Enmienda pre-resultados 2: la corrida
a 500 trials proyectaba 10-14 h; se recorta (~2-3 h). Sin
oraculo (153k no se enumeran); referencia = mejor observado.

El piloto corre 50 trials con semillas 0 y 1 para medir costo
por trial y fijar si los presupuestos son viables.

## Resultados

40 corridas en 3,4 h (6573 ajustes nuevos). Mejor J=0,9433
(`v1b|v2c|v4a|v5a|v7b|v9c`: los 3 pares plantados). A 250:
sef mediana 0,9433 (9/10 a 0,005), ga 0,8519 (3/10), local
0,8511 (1/10), aleatorio 0,79 (0/10). Forward 0,8486 con 135
ensayos: el control pierde como se diseño. TPE 3/10 al optimo
(mediana 0,85): lidera temprano (~45-60) pero se estanca en 2
pares; el SEF es el unico que junta los 3 como conducta
mediana. Primera victoria real del sustituto: revierte el
punto 3 del veredicto (ver enmienda alla). Si un trial
mediano supera 5 s, se bajan presupuestos a (50, 100, 250, 500)
antes de la corrida (unico ajuste permitido, documentado aqui
si ocurre).

## Ejecucion

```sh
uv run python scripts/generar_hostil.py
uv run python -m sef_exp.hostil --piloto --trials 50 --semillas 0,1
uv run python -m sef_exp.hostil --sin-graficos
uv run python scripts/tpe_hostil.py
uv run python -m sef_exp.hostil --solo-graficos
```

`data/hostil/` NO se versiona (85 MB): se regenera identico con
`scripts/generar_hostil.py` de semillas fijas antes de correr.
`work/hostil_*` desechable. `outputs/hostil_*` versionables.
