# Piloto de fase 3b

Se ejecuto primero una prueba de humo con semillas 0 y 1, y despues un
piloto de costo con semillas 2 y 3 tras separar el tiempo de ajustes del
tiempo de busqueda. Cada metodo recibio 100 trials unicos por semilla.
Solo se leyo train. Los dos pilotos comparten el cache de evaluaciones;
ningun metodo consulto un resultado antes de proponer su conjunto.

En el piloto de costo, el precompute de binning y WOE para 35 variables
y cinco folds tomo 9,5 s. Las ocho corridas sumaron 800 trials nominales,
567 ajustes nuevos y 233 consultas al cache. Tardaron 62,1 s de pared
despues del precompute. El pico de memoria del proceso fue 737 MiB.
Estos tiempos dependen del orden del cache; `pared_modelo_s` en el CSV
suma el costo observado de cada propuesta y permite comparar el costo
estimado si cada metodo corriera aislado.

| Metodo | Mejor J, semilla 2 | Mejor J, semilla 3 | Rechazos de 100 | Propuestas medias |
|---|---:|---:|---:|---:|
| Aleatorio | 0,711692 | 0,708855 | 91, 90 | 100 |
| Local | 0,709247 | 0,711320 | 50, 51 | 100 |
| Genetico local | 0,711943 | 0,710119 | 77, 77 | 673 |
| SEF completo | 0,711943 | 0,710119 | 77, 77 | 673 |

La correlacion de WOE causo todos los rechazos de estas corridas. El
sustituto aun no se activa a 100 trials, por eso los dos geneticos son
identicos. Estas dos semillas no sirven para elegir metodo. El piloto
no cambio presupuestos, parametros ni reglas de seleccion. Si los
160.000 trials nominales de la corrida completa fueran todos ajustes
nuevos, este ritmo implicaria varias horas; el cache compartido reducira
ese tiempo en una cantidad que no se presupone aqui.

Los CSV detallados del piloto estan en `work/fase3b_piloto_resumen.csv`
y `work/fase3b_piloto_runs/`. Son desechables; el resumen de costo y las
decisiones metodologicas quedan en este documento.
