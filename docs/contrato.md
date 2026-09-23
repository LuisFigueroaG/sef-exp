# Contrato del problema SEF x MRAP automotriz

Fuente: adaptacion de docs/propuesta-original.tex a datos locales.
Estado: decisiones cerradas por grilling el 2026-09-22. Sin resultados empiricos.

## Objetivo

Comparar algoritmos de seleccion por eficiencia de busqueda.
Gana el mayor AUC promedio de CV. KS, PR-AUC, calibracion y
estabilidad quedan solo como diagnosticos en test reservado.
No hay regla operacional de preferencia. No elegimos scorecard
para operar.

## Datos

- Train: X_train.csv + y_train.csv, 241836 filas, 35 predictoras.
  Positivos 16611, tasa 6.87 por ciento.
- Test: X_test.csv + y_test.csv, 103644 filas.
  Positivos 7154, tasa 6.90 por ciento.
- Test reservado puro hasta fase 4. No se usa para familias,
  binning ni tuning.
- Sin columna de fecha en CSV actuales. Sin prueba temporal
  en fases 2 y 3.
- Nulos segun data/diccionario_variables_train.md:
  especiales CreditVision a null, defaults no regulares
  IN20, AM20, AT20, GO158, PR12 a null, ceros tradicionales
  se mantienen, CVSC100 igual 0 y CVSC150 igual 0 se mantienen
  como valor con bin propio y documentacion de limite.

## Unidad de busqueda y evaluacion

Unidad de busqueda: conjunto completo S de k igual 6 variables.
Unidad de evaluacion: 5-fold estratificado con semillas fijas,
mismos folds para todos los candidatos.

Representacion: z en {0,1} con suma igual 6 y maximo una por
familia. Incompatibilidades E como z_i mas z_j menor o igual 1.
Sin cuotas por dominio en fase 2. Sin variables obligatorias
ni prohibidas salvo politica de sensibles por definir.

## Evaluador comun

1. Por fold y por variable: OptBinning con configuracion fija
   preregistrada, ajustado solo con train del fold.
2. WOE igual log(p_bad dividio p_good) con suavizado fijo,
   calculado solo con train del fold. Tablas precomputadas
   una vez por variable y fold, reutilizadas por lookup.
3. Filtros por fold train: IV mayor a 0.1, correlacion Pearson
   sobre WOE absoluta menor a 0.7, VIF menor a 5, constante
   rechaza. Interseccion estricta entre folds para elegibilidad.
4. Logistica aditiva WOE sin regularizacion, maxima verosimilitud.
   Signos libres por decision de libreria futura. Solo rechazo
   por no convergencia, estimaciones no finitas o singularidad.
5. Objetivo J(S) igual media de AUC de validacion en K folds
   validos. AUC con medio punto por empates. Misma orientacion
   de score para todos.

Formulas base en docs/protocolo.md.

## Eficiencia y consola

Precompute de WOE una vez. Cada candidato cuesta K ajustes
logisticos de 6 columnas mas lookups. Cache por conjunto.
Conteos separados: propuestas totales, unicos, descartes
previos, evaluaciones completas, CPU, pared y memoria.
Consola rica viva: barra por presupuesto, mejor acumulado,
ultimo resultado y tabla de trials. Detalle en protocolo.

## Fases

1. Evaluador comun y precompute verificable.
2. Oraculo chico mas aleatorio, forward e intercambios.
3. Genetico por familias, luego local y sustituto, ablaciones.
4. Generalizacion en test reservado. Sin temporal.

La superioridad queda abierta hasta completar experimentos.
