# Fase 4: replica externa en Taiwan

## Contrato previo a resultados

Fecha: 2026-09-23. Replica del protocolo de fase 3b sobre el dataset
publico Default of Credit Card Clients (Taiwan, UCI, 30.000 filas).
Motivo: en fase 3b el forward greedy alcanzo el mismo mejor conjunto
que los tres metodos guiados, lo que sugiere un paisaje aditivo. Una
sola tarea no dirime si la ventaja del SEF generaliza; hace falta otro
dataset con otra estructura de correlacion.

Split 80/20 estratificado con semilla 42: 24.001 filas train, 5.999
test. El test queda cerrado igual que MRAP; solo el script de particion
lo abre una vez para generarlo. Folds: los mismos cinco de
`hacer_folds` sobre train.

## Familias y espacio

Once familias por decision del usuario (reciente vs antiguo separados):

- `cupo`: LIMIT_BAL
- `sexo`: SEX
- `educacion`: EDUCATION
- `estado_civil`: MARRIAGE
- `edad`: AGE
- `pago_reciente`: PAY_0, PAY_2, PAY_3
- `pago_antiguo`: PAY_4, PAY_5, PAY_6
- `facturacion_reciente`: BILL_AMT1, BILL_AMT2, BILL_AMT3
- `facturacion_antigua`: BILL_AMT4, BILL_AMT5, BILL_AMT6
- `monto_reciente`: PAY_AMT1, PAY_AMT2, PAY_AMT3
- `monto_antiguo`: PAY_AMT4, PAY_AMT5, PAY_AMT6

Cada conjunto tiene seis variables y a lo sumo una por familia.
Numero estructural: **26.262** conjuntos (coeficiente de grado seis).
No se enumeraran ni se llamara optimo al mejor observado.

## Diferencia justificada con fase 3b: umbral IV

Con IV_MIN 0.1 de MRAP, 200/200 conjuntos piloto mueren por IV en
train: la senal individual de Taiwan es mas debil (solo cupo, pagos y
montos superan 0.1; sexo, educacion, edad y facturacion estan entre
0.007 y 0.04). Se fija IV_MIN 0.02 para esta fase: deja 8 de 11
familias jugables y filtra ruido puro (estado civil, 0.006).
CORR_MAX, VIF_MAX, binning y regresion no cambian. La comparacion
entre metodos sigue siendo justa (mismo filtro para todos); los J
absolutos no son comparables entre datasets porque son tareas
distintas.

## Comparacion fijada

Metodos: aleatorio uniforme, local con reinicios, genetico local sin
sustituto y SEF completo, mas forward greedy determinista como
referencia descriptiva fuera de la comparacion. Cada metodo usa
semillas 0 a 19 y presupuestos de 100, 250, 500, 1000 y 2000
conjuntos unicos, con las mismas reglas de fase 3b (duplicados
consumen propuestas, rechazos consumen trial). El motor de busqueda
es el mismo codigo parametrizado; solo cambian familias, datos,
umbral IV y rutas.

El piloto ejecuta los cuatro metodos con 100 trials y dos semillas
para medir costo y mezcla de rechazos. No se elige ganador ni se
ajustan parametros segun el piloto.

Piloto observado (semillas 0 y 1): precompute 2,5 s; 8 corridas en
10,7 s de pared con 554 ajustes nuevos. Todos los rechazos son por
IV (aleatorio 91-93 de 100, local 53-56, geneticos 64-68). Local
arranca mejor (44-47 factibles) por sus reinicios guiados; los dos
geneticos son identicos a 100 trials porque el sustituto aun no
activa, igual que en fase 3b. Mejor J del piloto: 0,7616. Solo se
leyo train.

## Ejecucion

```sh
uv run python scripts/partir_taiwan.py
uv run python -m sef_exp.fase4 --piloto --trials 100 --semillas 0,1
uv run python -m sef_exp.fase4 --sin-graficos
uv run python -m sef_exp.fase4 --forward-ref
uv run python -m sef_exp.fase4 --solo-graficos
```

`data/taiwan/` guarda el CSV original y la particion train/test.
`work/fase4_*` son checkpoints desechables.
`outputs/fase4_resumen.csv`, `fase4_metricas.csv`,
`fase4_estabilidad.csv`, `fase4_referencias.csv`,
`fase4_curvas.png` y `fase4_manifest.json` son versionables.

## Resultados

80 corridas en 269,9 s de pared (precompute 2,8 s), 22.891 ajustes
nuevos y 137.109 consultas al cache: el espacio de 26.262 conjuntos
se agota y los metodos se pisan. Todos los rechazos son por IV;
la correlacion no muerde en este dataset.

Mejor observado: J=0,7615923356582383 con
`EDUCATION|LIMIT_BAL|PAY_0|PAY_4|PAY_AMT2|PAY_AMT4`. Lo encuentran
los tres metodos guiados (Jaccard 1,0 desde 500 trials) y tambien
el forward con 101 ensayos y 2,5 s de modelo. Segunda replica
donde el greedy alcanza el mismo pico que la busqueda presupuestada.

Convergencia (corridas a 0,001 del mejor): ga_local y sef_completo
20/20 desde 100 trials, local 20/20 desde 250, aleatorio 9/20 a
2000. La diferencia entre metodos esta en velocidad (100-250),
no en calidad final. El sustituto no se distingue del genetico
sin sustituto en este paisaje: ambos clavan la mediana desde 100.
