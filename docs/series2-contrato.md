# Track series2: ARIMAX ampliado

## Contrato previo a resultados

Fecha: 2026-09-23. El track series (130 conjuntos) se agoto: a 100
trials los metodos ya ven el 77% del espacio. Para darle al
sustituto un espacio donde perderse o lucirse se amplia: ordenes
p 0-5 y q 0-3 (24 combinaciones), lags 0/1/3 por exogena y dos
exogenas nuevas (UF, euro). Se eligen 2 de 7 familias.

## Espacio

Ocho familias: `pq` con 24 tokens mas siete exogenas con 3
opciones (nivel, L1, L3). k=3. Estructural: **5481** conjuntos
(4536 jugables + 945 sin token pq que se rechazan por
estructura). El oraculo exhaustivo cubre el estructural completo.

Lags: la columna L* usa el valor de hace L meses; el train pierde
los 3 primeros meses de cabeza (ventana efectiva desde 2013-04).
Estandarizacion z-score de train sobre columnas ya lageadas.
Agregacion mensual igual que series (UF y euro por media).
d por KPSS en train (resulto 0 tambien aqui; queda registrado
en el manifiesto).

## Comparacion fijada

Metodos: aleatorio, local, ga_local y sef_completo, mas
`hk_forward` (greedy con intercambios simples respetando una
familia por pick, (p,q) re-entonado en cada evaluacion).
Presupuestos 100, 250, 500, 1000 y 2000 trials unicos (2000
agregado post-hoc para verificar si aleatorio alcanza: a 1000 iba
18/20); 20 semillas.
Epsilon 2 puntos AICc para la fraccion de cercania. El sustituto
usa 7 dummies + 2 lags + p,q enteros.

El piloto corre 100 trials con semillas 0 y 1. Mezcla observada:
~30% no-convergencia, ~15% estructura. Mejor del piloto: -535,44
(AICc 535,4, `COBRE_L1|DOLAR|p2q3`). Solo train.

## Ejecucion

```sh
uv run python -m sef_exp.series2 --piloto --trials 100 --semillas 0,1
uv run python -m sef_exp.series2 --oraculo
uv run python -m sef_exp.series2 --sin-graficos
uv run python -m sef_exp.series2 --hk-ref
uv run python -m sef_exp.series2 --solo-graficos
```

`work/series2_*` desechable. `outputs/series2_resumen.csv`,
`series2_metricas.csv`, `series2_estabilidad.csv`,
`series2_referencias.csv`, `series2_oraculo.csv`,
`series2_curvas.png` y `series2_manifest.json` versionables.

## Resultados

Oraculo 5481/5481: mejor J=-535,44 (`COBRE|DESEMPLEO|p5q0`).
El optimo toca el borde p=5: si se amplian ordenes de nuevo,
hay que subir p_max (registrado como deuda).

80 corridas en 15,2 s sobre el cache completo. Convergencia a
epsilon 2: a 250 trials sef 20/20, local y ga 19/20; a 500 los
tres 20/20; aleatorio 18/20 a 1000. HK con 964 ensayos queda en
-535,97 (`COBRE|TPM_L1|p5q0`): segunda replica donde el greedy
no alcanza el pico, por 0,5 puntos AICc. Mismo pq, distinta
segunda exogena.

Incidente: el sustituto con features densos rompia la matriz
ante triples sin pq (largos y Nones mezclados). Se cambio a
one-hot sobre opciones con respaldo del caso, y los outputs
previos se resguardaron en work/respaldo-series2-sustituto/.

Nota: los outputs de series v1 corresponden al split de 5
exogenas; al agregar UF/euro cambio el hash de datos y su
fingerprint ya no se reproduce. Quedan como registro congelado.
