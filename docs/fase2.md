# Fase 2: oraculo pequeno y baselines

## Especificacion congelada antes de medir

Fecha: 2026-09-22. Este experimento usa solo `X_train.csv` y `y_train.csv`.
`X_test.csv` y `y_test.csv` quedan reservados para la fase 4.

Se toman dos representantes semanticos de cada familia del borrador. La eleccion
busco cubrir ocho dominios con un espacio exhaustivo manejable. Tambien incluyo
deliberadamente las seis variables de un conjunto factible en la verificacion
manual de fase 1: `CVSC150, AT01, AT93, GO243, BC111, PR95`, cuyo J ya se
conocia. Por ello este subespacio es exploratorio y su seleccion no fue ciega
a resultados previos. Quedo fijo antes de enumerar el oraculo de fase 2.
No hay exclusiones por sensibilidad porque
las 35 columnas documentadas son caracteristicas de historial crediticio; no
hay edad, sexo, territorio ni otro atributo demografico en el catalogo actual.
Esa clasificacion se revisara si aparece metadato nuevo.

| Familia | Variables | Motivo |
|---|---|---|
| scores | CVSC100, CVSC150 | Dos scores CreditVision |
| satisfactorias | AM03, AM26 | Conteo general y antiguedad minima de 18 meses |
| totales | AT01, AT30 | Total de cuentas y cuentas con saldo cero |
| antiguedad | AT93, IN20 | Antiguedad global y de cuotas |
| consultas | GO06, GO243 | Ventanas de 6 y 24 meses |
| cv_uso | CVPRPAYMNT11, CVPRTRD | Pagos y numero de cuentas |
| revolvente | RE03, BC111 | Cuentas satisfactorias y saldo mayor |
| cobranzas | PR09, PR95 | Conteo general y cobranzas impagas recientes |

Cada solucion usa exactamente seis variables, a lo sumo una por familia.
El conjunto de incompatibilidades conceptuales `E` esta vacio: ninguna pareja
del diccionario justifica una prohibicion adicional a estas familias. Los
filtros estadisticos del evaluador comun siguen activos. El numero estructural
es `C(8, 6) * 2^6 = 1792`; el numero estadisticamente factible se conocera
solo tras enumerar el oraculo.

## Ejecucion y medicion

`uv run python -m sef_exp.fase2` calcula el oraculo, reanuda un CSV compatible
si se interrumpe y ejecuta los baselines. `--solo-oraculo` se detiene tras la
enumeracion. `--solo-figura` regenera metricas y PNG desde las tablas. La
exportacion PNG usa Kaleido y necesita Chrome local. La firma de configuracion incluye los datos train, el evaluador,
las familias y los umbrales. Un cambio invalida el checkpoint y exige
resguardar los archivos anteriores antes de ejecutar otra especificacion.

El oraculo escribe `outputs/fase2_oraculo.csv` con los 1792 candidatos,
estado, causa de rechazo, AUC por fold, J y tiempo de evaluacion. El evaluador
usa cinco folds estratificados fijos. Los WOE se calculan una vez por variable
y fold y se mantienen en memoria. Esta cache no cambia binning ni modelos.

Los baselines consultan la tabla del oraculo solo para el candidato que
proponen. No pueden inspeccionar filas no propuestas. Eso reproduce el valor
determinista del evaluador sin volver a ajustar cada modelo. `pared_modelo_s`
acumula los tiempos medidos por el oraculo para esas evaluaciones; no mide el
tiempo real del buscador y no debe interpretarse como tal.

- Aleatorio: permutacion sin reemplazo de los 1792 conjuntos por semilla.
- Forward: en cada tamano evalua todas las ampliaciones de una variable de
  familia aun libre y elige la de mayor AUC parcial factible. Si se bloquea,
  se informa. Es determinista y se ejecuta una vez. Cada modelo parcial cuenta
  como trial, aunque no entra en la curva de mejor conjunto completo.
- Local: inicia en un conjunto aleatorio. Explora intercambios simples y,
  cuando no mejora, dobles. Acepta el mejor vecino que mejore; si no existe,
  reinicia en un conjunto aun no visitado. Ejecuta 20 semillas.

Los presupuestos son 100, 250, 500, 1000 y 2000 trials unicos. El presupuesto
2000 excede el espacio: los metodos de conjuntos completos se censuran al
agotar 1792 candidatos. Las propuestas repetidas no consumen trial; el CSV
`work/fase2_trials.csv` guarda las consultas unicas. El resumen indica los
trials reales y propuestas totales alcanzadas. Comparar forward requiere advertir que un trial
parcial ajusta menos coeficientes que uno completo. La comparacion principal
se hace en conjuntos completos evaluados; los modelos parciales tambien se
cuentan para no conceder evaluaciones gratis a forward.

El resumen de presupuestos vive en `outputs/fase2_resumen.csv`. Para cada
metodo y semilla se mide el mejor J y la brecha frente al maximo factible del
oraculo. Si un metodo no encuentra solucion factible, se informa como fallo;
no se reemplaza por un J artificial. El test externo sigue cerrado.
`outputs/fase2_metricas.csv` contiene tasas de exito y medianas de llegada
para epsilon 0,005, 0,002 y 0,001. `outputs/fase2_curvas.png` muestra la
brecha y la tasa de llegada al optimo exacto. Los resultados e interpretacion
estan en `docs/fase2-resultados.md`.
