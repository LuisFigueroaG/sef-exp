# Datos, copia minima desde MRAP

Origen: `~/code/mrap/MRAP-score-originacion-financiamiento-automotriz/data/raw/`

Archivos copiados:

- `X_train.csv`, 23 MB, 241837 lineas con header
- `y_train.csv`, 2.5 MB, 241837 lineas con header, columna `target`
- `X_test.csv`, 9.9 MB, 103645 lineas con header
- `y_test.csv`, 1.1 MB, 103645 lineas con header

No se copia `unl_bf.csv` de 81 MB en esta etapa.

Predictoras: 35 columnas mas indice. Incluyen scores CreditVision, conteos de cuentas, antiguedades, consultas e impagos. Detalle en `diccionario_variables_train.md`.

Tratamiento de nulos acordado en MRAP: codigos especiales CreditVision a null, defaults no regulares tradicionales a null, defaults iguales a cero se mantienen. Ver diccionario copiado.

Estos CSV se versionan en este repo privado por decision explicita para reproducir sin depender de MRAP.
