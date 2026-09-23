# sef-exp

Experimento de seleccion de variables para scorecards con busqueda evolutiva y restricciones.

Estado: fases 1 a 3 completadas. Resultados en `docs/fase2-resultados.md`
y `docs/fase3-resultados.md`. El test reservado sigue cerrado. Este repo
es solo investigacion y experimentacion, no libreria.

## Objetivo

Probar si un buscador por familias e incompatibilidades alcanza AUC de CV cercano al optimo factible con menos evaluaciones que baselines, sin perder generalizacion ni estabilidad.

Unidad de busqueda: conjunto completo de variables. Unidad de evaluacion: entrenamiento y validacion de ese conjunto con procedimiento fijo.

## Datos

Solo datos MRAP de originacion automotriz, copia minima local en `data/`.

- `X_train.csv`, `y_train.csv`
- `X_test.csv`, `y_test.csv`
- Sin `unl_bf.csv` en esta etapa
- Diccionario y reglas de nulos en `data/diccionario_variables_train.md`

Ver `data/README.md` para origen, tamanos y tratamiento.

## Estructura

- `docs/` propuesta original y resumen
- `data/` datos versionados en este repo privado
- `src/sef_exp/` codigo del experimento desde cero
- `work/` pruebas desechables, no se versiona
- `outputs/` figuras y tablas versionadas

## Uso base

Requiere `uv`.

```sh
uv sync
uv run python -c "import sef_exp; print('ok')"
uv run python -m sef_exp.fase2 --solo-oraculo
uv run python -m sef_exp.fase2
uv run python -m sef_exp.fase3
uv run python -m sef_exp.fase3 --solo-graficos
```

## Roadmap

1. Contrato del problema y evaluador comun con folds y semillas fijas
2. Oraculo exhaustivo chico con aleatorio, forward e intercambios. Ver `docs/fase2.md`.
3. Genetico por familias, mejora local y sustituto, con ablaciones. Ver `docs/fase3.md`.
4. Generalizacion con test reservado; sin cortes temporales porque los CSV no tienen fecha

En el subespacio de fase 2, el local simple sigue siendo la referencia
mas fuerte a presupuesto bajo. El SEF no demostro superioridad general.
