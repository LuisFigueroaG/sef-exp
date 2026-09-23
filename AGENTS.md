# sef-exp, convenciones

Proyecto de investigacion desde cero. Desde MRAP solo vienen los datos. No heredar codigo ni estructura de MRAP.

- Gestor: `uv`. Tablas con Polars, graficos con Plotly.
- Codigo simple y lineal, sin abstracciones antes de tiempo.
- Todo numero experimental sale del evaluador comun con folds y semillas fijas.
- WOE, IV, correlacion y VIF se calculan solo en train de cada fold.
- No se optimiza binning, regularizacion y seleccion a la vez en el primer experimento.
- `work/` es desechable y no se versiona. `outputs/` guarda PNG con nombre estable.
- Documentar cada decision de familias e incompatibilidades antes de comparar resultados.
