# Vision: libreria SEF para scorecards (futuro, no ahora)

Fecha: 2026-09-23. Estado: intencion documentada durante la fase
de investigacion. No se escribe codigo de libreria todavia; el
repo sigue siendo experimentacion.

## A quien sirve

Modeladores de default en bancos: reciben cientos o miles de
variables de feature engineering y necesitan un conjunto que
arme un buen modelo cumpliendo IV, VIF, correlacion y criterio
de negocio, con trazabilidad auditable.

## API imaginada (una funcion)

`sef.seleccionar(X, y, ...)` donde se indica:

- `objetivo`: funcion que recibe un conjunto y devuelve un
  numero (default: AUC medio de 5 folds con la receta scorecard).
- `woe`: si/no, con OptBinning simplificado (pocos parametros,
  defaults sanos para riesgo de credito) y soporte de nulos en
  binning y WOE.
- `restricciones`: IV minimo, correlacion maxima, VIF maximo.
- `k`: fijo (6) o rango a barrer (4 a 12) con regla de
  parsimonia escrita (la de fase4k).
- `familias`: diccionario {familia: [variables]} o columna de
  familia en el mismo dataframe; maximo por familia (default 1).
- `modelos`: scorecard logistica primero; despues XGB/LGBM/HGB
  y redes, con sus propios filtros (un arbol no necesita VIF).

Devuelve: mejor conjunto, WOE tablas, fit final, reporte de
metricas, perdidas por trial (trazas), manifiesto de
reproducibilidad y cache reutilizable.

## Despues (mucho despues)

Restricciones de sentido de negocio por variable: WOE
monotono segun bad-rate, signos esperados por dominio,
vetos y obligatorias. DSL chico, no un framework.

## Principios heredados de la investigacion

Contratos previos, fingerprints, presupuestos por trials
unicos, duplicados que no consumen evaluacion, test cerrado,
sin re-seleccion post-hoc. Todo lo que un area de riesgo
modelo (SR 11-7 mediante) exige para auditar.
