# Fase 3b: busqueda en el catalogo completo

## Contrato previo al piloto

Se usan las 35 columnas de `X_train.csv`, cinco folds y el evaluador comun
sin cambiar binning, WOE, IV, correlacion, VIF ni regresion. `X_test.csv`
y `y_test.csv` no se abren. Cada conjunto contiene seis variables y a
lo sumo una de cada familia de sustitucion. No hay incompatibilidades
conceptuales adicionales `E`: el diccionario no justifica prohibir pares
de familias distintas. Los filtros estadisticos siguen operando solo
en train de cada fold.

El borrador de ocho familias era demasiado amplio para este experimento.
Por decision del usuario se separan medidas distintas. Las 27 familias
congeladas estan en `src/sef_exp/familias.py`: hay seis con mas de una
variable y 21 individuales. `AM03` y `AT03`, por ejemplo, compiten como
conteos generales de satisfactorias; los conteos de 3, 6, 18 y 24 meses
pueden coexistir. `GO06`, `GO243` y `CVSSG990S` usan ventanas distintas y
pueden coexistir. Pagos, cambio de saldo, gasto, saldo maximo, numero y
recencia de cobranzas tampoco se bloquean entre si. Se conservan como
familias de sustitucion los dos scores, los conteos generales y de tres
meses de satisfactorias, los conteos de cuentas totales, las medidas de
antiguedad de cuentas y las consultas totales. `PR95` queda separado de
`PR09` porque especifica cobranzas impagas recientes.

El numero estructural es el coeficiente de grado seis de
`producto_g (1 + n_g x)`: **1.251.023** conjuntos. No se enumeraran ni se
llamara optimo al mejor J observado. La muestra aleatoria usa los
coeficientes de sufijo de ese producto para que cada conjunto
estructural tenga la misma probabilidad. Mutacion, cruce y vecinos
preservan cardinalidad y familias.

## Comparacion fijada

Metodos: aleatorio uniforme, local con reinicios, genetico local sin
sustituto y SEF completo. Cada metodo usa semillas 0 a 19 y presupuestos
de 100, 250, 500, 1000 y 2000 conjuntos unicos. Los duplicados consumen
propuestas, no trials. Un rechazo por IV, correlacion, VIF o ajuste
consume trial. Se registran J, mejor conjunto, rechazos por causa,
propuestas, tiempo de modelo, tiempo de buscador y uso de cache.

Local evalua vecinos simples del incumbente y, si no mejora, hasta 64
vecinos dobles muestreados; luego reinicia. Genetico y SEF comparten los
primeros 48 conjuntos aleatorios por semilla y 16 guiados por IV de los
train folds. Mantienen poblacion 64, ocho elites, lotes de 16 y bolsa de
96 hijos, 16 inmigrantes y hasta 16 vecinos. El sustituto ExtraTrees
de 32 arboles empieza tras 64 conjuntos factibles y se reentrena cada
cuatro lotes. Estas reglas siguen la fase 3; los vecinos se generan al
vuelo porque precalcular todos los pares no escala aqui.

El piloto ejecuta los cuatro metodos con 100 trials y una o dos semillas
para medir costo y comprobar trazas. No se elige un ganador ni se
ajustan parametros segun J del piloto. Si la corrida completa resulta
costosa, se informara el tiempo estimado antes de ejecutarla. El cache
de evaluaciones comparte resultados entre corridas, pero cada metodo
cuenta su propia consulta como trial y solo ve el resultado despues de
proponer el conjunto. `pared_modelo_s` suma los tiempos medidos cuando
se ajusto cada conjunto por primera vez y aproxima el costo aislado;
`pared_algoritmo_s` mide la busqueda sobre el cache. El precompute comun
se informa aparte.

La comparacion principal sera J de CV y su distribucion a presupuesto
igual, con diferencias pareadas por semilla, costo y estabilidad de
seleccion. El mejor J de todos los metodos sera una referencia
descriptiva posterior, no un oraculo ni una entrada al algoritmo. Las
20 semillas son repeticiones del optimizador sobre una sola tarea.
Ninguna ventaja aqui establece generalizacion; antes de abrir el test
se fijara por escrito el conjunto finalista y la regla de evaluacion
externa.

## Ejecucion

```sh
uv run python -m sef_exp.fase3b --piloto --trials 100 --semillas 0,1
uv run python -m sef_exp.fase3b --sin-graficos
uv run python -m sef_exp.fase3b --solo-graficos
uv run python -m sef_exp.fase3b --forward-ref
```

`work/fase3b_evaluaciones.csv` y las trazas por trial son checkpoints
desechables. `outputs/fase3b_resumen.csv`, `fase3b_metricas.csv`,
`fase3b_estabilidad.csv`, `fase3b_referencias.csv`,
`fase3b_curvas.png` y `fase3b_manifest.json` son los resultados
versionables. La consola informa el ultimo resultado, mejor J y S de
la corrida, mejor global observado, rechazos, propuestas y ETA.

## Referencia forward (agregada post-hoc, fuera de la comparacion)

El `forward` de fase 2 no es comparable: corrio en otro subespacio
(ocho familias, 1792 conjuntos). En su lugar se calcula un forward
greedy determinista sobre este catalogo con el mismo evaluador y
folds (`--forward-ref`, 187 ensayos). No tiene presupuesto por
trials ni semillas; es una referencia descriptiva con el mismo
estatus que el mejor observado. Su J y S quedan en
`outputs/fase3b_referencias.csv`.

`fase3b_curvas.png` se genera desde las trazas por trial (`--solo-graficos`):
panel izquierdo con la mediana del mejor J por trial en escala
logaritmica desde el trial 50, panel derecho con la fraccion a
0,001 del mejor observado en los cinco presupuestos. El trial 50 es
el primero con cobertura completa de factibles en las 20 semillas;
antes la mediana es ruido. La linea punteada horizontal es el
forward de referencia.
