# Resumen de la propuesta

La propuesta plantea SEF, selector evolutivo por familias, para elegir 6 variables de scorecard logistica WOE que maximicen AUC promedio de CV bajo restricciones duras.

Restricciones: cardinalidad exacta, maximo una variable por familia, sin pares incompatibles, IV mayor a 0,1, correlacion menor a 0,7, VIF menor a 5, y logistica convergente con signos coherentes.

El buscador combina poblacion genetica, intercambios locales simples y dobles, propuestas aleatorias y un sustituto que prioriza evaluaciones completas.

Decision clave: AUC manda para comparar optimizadores. KS, PR-AUC, calibracion, PSI y estabilidad quedan como diagnosticos. La regla operacional final va separada del objetivo de busqueda.

El protocolo separa eficiencia y generalizacion. En espacios chicos se enumera todo y se mide brecha al optimo contra ensayos. En espacios grandes se compara con igual presupuesto y se habla de mejor observado. Se exige validacion externa y pruebas temporales. Baselines: aleatorio, forward, stepwise adaptado, busqueda local, genetico basico, genetico con local y TPE.

Estado actual: sin implementacion ni resultados. Parametros iniciales por preregistrar. Figura del documento ilustrativa, no evidencia.

Riesgos principales: costo por candidato, filtro IV duro que descarta aporte condicional, sustituto fragil al inicio, y poca potencia en datasets chicos para diferencias finas de AUC.
