# Diccionario de variables de entrenamiento

Variables presentes en `X_train.csv` y `y_train.csv`. La columna sin nombre corresponde al índice del archivo y no se considera una variable.

| Variable | Descripción | Valor documentado | Clasificación / significado |
|---|---|---|---|
| `CVSC150` | Puntaje CreditVision para créditos automotrices subprime. | `0` | Score no informado por inferencia: el diccionario no define el código, pero en los datos coincide con `CVSC100=0`. |
| `AM133` | Número de cuentas satisfactorias con máximo dos atrasos de 30 días. | `0` | Default documentado; en este conteo indica ausencia de cuentas que cumplan la condición. |
| `BC58` | Antigüedad promedio de las tarjetas bancarias. | `0` | Default documentado; el diccionario no detalla si representa ausencia de tarjetas utilizables para calcular el promedio. |
| `AM24` | Número de cuentas satisfactorias con al menos 6 meses. | `0` | Default documentado; en este conteo indica ausencia de cuentas que cumplan la condición. |
| `CVPRPAYMNT11` | Número de pagos realizados en los últimos 12 meses. | `-7`, `-6`, `-2` | `-7`: sin hit en el buró; `-6`: sin cuentas en el archivo; `-2`: faltan datos para calcular el algoritmo. |
| `GO12` | Antigüedad promedio de las cuentas, en meses. | `0` | Default documentado; su significado específico no está detallado. |
| `AM23` | Número de cuentas satisfactorias con al menos 3 meses. | `0` | Default documentado; en este conteo indica ausencia de cuentas que cumplan la condición. |
| `AT23` | Número total de cuentas satisfactorias con al menos 3 meses. | `0` | Default documentado; en este conteo indica ausencia de cuentas que cumplan la condición. |
| `AM03` | Número de cuentas satisfactorias. | `0` | Default documentado; indica ausencia de cuentas satisfactorias. |
| `AT03` | Número total de cuentas satisfactorias. | `0` | Default documentado; indica ausencia de cuentas satisfactorias. |
| `IN20` | Meses desde la apertura del crédito en cuotas más antiguo. | `-1` | No existe una cuenta en cuotas utilizable para calcular la antigüedad. |
| `CVPRTRV10` | Meses con disminución de saldo no hipotecario durante el último año. | `-7`, `-6`, `-2`, `-1` | `-7`: sin hit; `-6`: sin cuentas; `-2`: datos insuficientes; `-1`: sin cuentas del tipo requerido. |
| `AT93` | Antigüedad de la cuenta abierta más antigua, en meses. | `0` | Default documentado; su significado específico no está detallado. |
| `BC111` | Saldo de la tarjeta bancaria que alcanzó el mayor saldo. | `0` | Default documentado; puede representar ausencia del dato o un saldo real igual a cero, por lo que debe conservarse un indicador. |
| `CVPRRVDEXQ4` | Proporción del gasto revolvente anual realizado en el cuarto trimestre. | `-9`, `-7`, `-6`, `-2`, `-1` | `-9`: denominador cero; `-7`: sin hit; `-6`: sin cuentas; `-2`: datos insuficientes; `-1`: sin cuentas del tipo requerido. |
| `CVPRNONMTTRD` | Número de cuentas no hipotecarias. | `-7`, `-6` | `-7`: sin hit en el buró; `-6`: sin cuentas en el archivo. |
| `AT01` | Número total de cuentas. | `0` | Default documentado; indica ausencia de cuentas. |
| `CVPRTRD` | Número de cuentas. | `-7` | Sin hit en el buró. El código está documentado, aunque no aparece en train ni test. |
| `AT30` | Número de cuentas con saldo igual a cero. | `0` | Default documentado; en este conteo indica ausencia de cuentas que cumplan la condición. |
| `AM01` | Número de cuentas. | `0` | Default documentado; indica ausencia de cuentas. |
| `AM20` | Meses desde la apertura de la cuenta más antigua. | `-1` | No hay cuentas de esta familia; coincide con `AM01=0`. |
| `AT20` | Meses desde la apertura de la cuenta más antigua. | `-1` | No hay cuentas de esta familia; coincide con `AT01=0`. |
| `GO06` | Número de consultas crediticias en los últimos 6 meses. | `0` | Default documentado; indica ausencia de consultas en el período. |
| `RE03` | Número de créditos revolventes satisfactorios. | `0` | Default documentado; indica ausencia de cuentas que cumplan la condición. |
| `GO158` | Número de consultas no automotrices sin duplicados. | `-6` | Default documentado sin significado global detallado; en train coincide con `GO04=0`. |
| `GO04` | Número total de consultas crediticias. | `0` | Default documentado; indica ausencia de consultas. |
| `PR95` | Cobranzas impagas de al menos USD 250 registradas en los últimos 36 meses. | `0` | Default documentado; indica ausencia de cobranzas que cumplan la condición. |
| `CVSSG990S` | Número de consultas sin duplicados en los últimos 12 meses. | `-4`, `-7` | `-4`: sin cuentas en el archivo; `-7`: sin hit en el buró. |
| `CVSC100` | Puntaje de riesgo CreditVision, entre 300 y 900. | `0` | Score no informado por inferencia: el código está fuera del rango válido, pero no está definido por el diccionario. |
| `CVSSG960S` | Número total de consultas sin duplicados. | `-4`, `-7` | `-4`: sin cuentas en el archivo; `-7`: sin hit en el buró. |
| `AM26` | Número de cuentas satisfactorias con al menos 18 meses. | `0` | Default documentado; en este conteo indica ausencia de cuentas que cumplan la condición. |
| `PR12` | Meses desde la cobranza más reciente. | `999` | No se detectó una cobranza previa cuya recencia calcular; coincide con `PR09=0`. |
| `AT27` | Número de cuentas satisfactorias con al menos 24 meses. | `0` | Default documentado; en este conteo indica ausencia de cuentas que cumplan la condición. |
| `GO243` | Número de consultas crediticias en los últimos 24 meses. | `0` | Default documentado; indica ausencia de consultas en el período. |
| `PR09` | Número de cobranzas. | `0` | Default documentado; indica ausencia de cobranzas. |
| `target` | Variable objetivo del modelo. | No aplica | No es una variable predictora. |

## Clasificación para el análisis

| Clasificación | Variables y valores | Inclusión en el gráfico |
|---|---|---|
| Código especial documentado | `CVPRPAYMNT11`: `-7`, `-6`, `-2`; `CVPRTRV10`: `-7`, `-6`, `-2`, `-1`; `CVPRRVDEXQ4`: `-9`, `-7`, `-6`, `-2`, `-1`; `CVPRNONMTTRD`: `-7`, `-6`; `CVPRTRD`: `-7`; `CVSSG990S` y `CVSSG960S`: `-4`, `-7`. | Sí. Son estados definidos explícitamente como especiales por CreditVision. |
| Default no regular distinguible | `IN20=-1`, `AM20=-1`, `AT20=-1`, `GO158=-6`, `PR12=999`. | Sí. No son códigos especiales de CreditVision, pero están fuera del rango regular y se identifican sin ambigüedad. |
| Default igual a cero | `AM133`, `BC58`, `AM24`, `GO12`, `AM23`, `AT23`, `AM03`, `AT03`, `AT93`, `BC111`, `AT01`, `AT30`, `AM01`, `GO06`, `RE03`, `GO04`, `PR95`, `AM26`, `AT27`, `GO243`, `PR09`. | No. En conteos, `0` es un valor válido; en otras variables no se puede distinguir un cero real del default. |
| Estado inferido, no documentado | `CVSC100=0`, `CVSC150=0`. | No. La evidencia sugiere score no informado, pero el código no está definido oficialmente como especial. |

## Tratamiento acordado para modelamiento

| Grupo | Variables | Tratamiento |
|---|---|---|
| Códigos especiales CreditVision | `CVPRPAYMNT11`, `CVPRTRV10`, `CVPRRVDEXQ4`, `CVPRNONMTTRD`, `CVPRTRD`, `CVSSG990S`, `CVSSG960S` | Convertir todos los códigos documentados a `null`. |
| Defaults no regulares tradicionales | `IN20`, `AM20`, `AT20`, `GO158`, `PR12` | Convertir el default documentado a `null`. |
| Defaults iguales a cero | Variables tradicionales indicadas en la clasificación anterior. | Mantener el valor `0` sin cambios. |

Los estados convertidos se tratarán como valores faltantes durante el modelamiento. Ya no se utiliza `-1` como marcador técnico y no se conserva una categoría separada por cada código.

## Notas sobre los códigos y defaults

- Los códigos y defaults provienen de `CreditVision Data Dictionary Oct 2021.xlsx` y `Credit_Report_Characteristics (Data Dictionaries).xlsx`.
- En variables de conteo, `0` es un valor válido con significado de ausencia y no un código especial. No se incluye en el cálculo ni en el gráfico de estados no regulares.
- En promedios, antigüedades o saldos cuyo default es `0`, el archivo no permite distinguir el default de un cero real; por prudencia tampoco se contabilizan como especiales.
- Los significados de `CVSC100=0` y `CVSC150=0` son inferencias basadas en el rango y la coincidencia observada entre ambos scores; no están definidos explícitamente en los archivos fuente.
