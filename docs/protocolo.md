# Protocolo adaptado SEF x MRAP

## Idea en una linea

Buscar conjuntos de 6 variables que maximicen AUC de CV con
restricciones duras, midiendo eficiencia contra oraculo y
generalizacion en test reservado.

## Matematica minima

z_j en {0,1}, suma z igual 6, suma en F_g menor o igual 1,
z_i mas z_j menor o igual 1 para pares en E.

N_k igual coef de x^k en producto (1 mas n_g por x).
Si todas tienen m, N_k igual C(G,k) por m^k.

AUC por fold con empates a medio punto, J(S) igual media
de AUC en K folds validos. Gini igual 2 por AUC menos 1,
mismo orden, no otro objetivo.

VIF_j igual 1 dividio (1 menos R2_j). Singularidad rechaza.

Curva mejor acumulado b(t) igual max hasta t de J.
Brecha r igual J estrella menos b(t).
Tiempo a epsilon T igual primer t con r menor o igual epsilon,
epsilon en 0.005, 0.002 y 0.001, censurado si no llega.
Brecha integrada R igual media de r en 1 a B.
Reportar tasa de exito y distribucion de T, no medias de
solo exitosas.

## Evaluador comun y eficiencia

Precompute por variable y fold con OptBinning fijo: bins,
WOE, IV, medias para correlacion. Cada candidato hace lookup
mas 5 logisticas de 6 columnas. Sin re-binning por candidato.
Cache por frozenset de S. Semillas fijas para folds y para
optimizadores por separado.

Costos a registrar: propuestas totales, unicos, descartes
previos por causa, evaluaciones completas, CPU, pared y memoria.
Precompute y oraculo fuera del presupuesto del buscador pero
reportados. Eje X en trials unicos, segundo panel en pared.

## Consola rica viva

Cada script de experimento muestra barra por presupuesto,
trial actual, J del trial, mejor acumulado y mejor S,
rechazos por causa y ETA. Al final guarda CSV de trials y
PNG de curva con nombre estable en outputs. Sin ruido
artificial ni suavizados que superen lo observado.

## Fases

Fase 1: evaluador mas precompute mas verificacion manual en
conjuntos chicos con numeros reproducibles.
Fase 2: oraculo en subespacio chico mas aleatorio restringido,
forward por AUC con registro de bloqueo, e intercambios
locales simples y dobles. Presupuestos 100, 250, 500, 1000
y 2000, 20 semillas, mismas tareas y semillas entre metodos.
Fase 3: genetico por familias, luego local y sustituto, con
ablaciones que quitan sustituto, local, dobles, reinicios e
inicializacion guiada. Si el sustituto no aporta, queda la
version simple. Si un clasico empata con menos recursos, esa
es la conclusion.
Fase 4: test reservado para finalistas. AUC externo, KS,
calibracion y frecuencia de seleccion. Sin temporal por falta
de fecha. Distinguir eficiencia en CV de mejora predictiva.

## Adaptaciones respecto a propuesta original

Signos libres, solo convergencia y finitos. Sin regla
operacional de cercania. Sin E inicial. Binning OptBinning
fijo precomputado por fold. Oraculo chico primero por costo
con 241k filas. work desechable, outputs PNG estables.
