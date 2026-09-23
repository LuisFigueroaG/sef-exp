# Validacion externa: finalistas y regla

Fecha: 2026-09-23. Se abre el test por primera vez bajo esta regla
fijada antes de verlo. Un finalista por track, sin re-seleccion:
lo que salga se reporta, gane o pierda contra el J de seleccion.

## Finalistas (mejor observado = consenso de metodos)

- fase3b (MRAP): `AT23|CVPRTRV10|CVSC150|CVSSG990S|IN20|PR09`,
  J de seleccion 0,713542318119154 (AUC medio de 5 folds en train).
- fase4 (Taiwan): `EDUCATION|LIMIT_BAL|PAY_0|PAY_4|PAY_AMT2|PAY_AMT4`,
  J de seleccion 0,7615923356582383.
- series (IMACEC): `COBRE|DESEMPLEO|p3q0`, J de seleccion -549,0536
  (-AICc en train, d=0). Pronostico a 24 meses.

## Regla

Scorecards: binning + WOE ajustados en train completo, regresion
logistica, AUC en test. Se reporta AUC test junto al J de
seleccion; exito descriptivo si la caida es chica, sin umbral
mecanico. Series: ARIMA re-ajustado en train completo con exogenas
en z-score de train, RMSE y MAE a 24 meses contra naive de ultima
observacion como contexto (el naive no selecciona, no hay peeking).

No se prueba ningun otro conjunto en el test. Si un finalista
colapsa fuera de muestra, se reporta como tal y no se reemplaza.

Este script no tiene test automatizado a proposito: ejecutarlo
abre el test, y la disciplina del repo lo prohibe en cada corrida.

## Resultados (test abierto una vez el 2026-09-23)

`outputs/validacion_externa.json`.

- fase3b: AUC test **0,7146** vs J 0,7135 (sube +0,0011, n=103.644).
  No hay degradacion; el conjunto aguanta fuera de muestra.
- fase4: AUC test **0,7629** vs J 0,7616 (sube +0,0013, n=5.999).
  Tampoco degrada.
- series: RMSE **3,69** vs naive 3,66; MAE **3,13** vs naive 3,32
  (n=24 meses). No colapsa, pero no bate al naive en RMSE: con
  24 puntos el ruido domina y el modelo no demuestra ventaja
  predictiva fuera de muestra. Se reporta sin reemplazar.
