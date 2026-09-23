# Familias borrador para k igual 6

Borrador mio para aprobacion. Disjuntas, maximo una por familia.
Sin E inicial. Los filtros IV, correlacion y VIF mandan despues.

## Catalogo propuesto, G igual 8

- F_scores, n igual 2: CVSC100, CVSC150.
  Sustitutos directos de score CreditVision.
- F_satisfactorias, n igual 8: AM133, AM24, AM23, AT23, AM03, AT03, AM26, AT27.
  Conteos de cuentas satisfactorias por antiguedad. Alta sustitucion interna.
- F_totales, n igual 3: AT01, AM01, AT30.
  Nivel de cuentas totales y saldo cero.
- F_antiguedad, n igual 6: BC58, GO12, AT93, IN20, AM20, AT20.
  Promedios y meses desde apertura. Incluye nulos a null por diccionario.
- F_consultas, n igual 6: GO06, GO04, GO158, GO243, CVSSG990S, CVSSG960S.
  Consultas por ventana y fuente. Sustitutos por ventana.
- F_cv_uso, n igual 5: CVPRPAYMNT11, CVPRTRV10, CVPRRVDEXQ4, CVPRNONMTTRD, CVPRTRD.
  Pagos, saldos y conteos CreditVision con codigos a null.
- F_revolvente, n igual 2: RE03, BC111.
  Revolventes satisfactorios y saldo mayor.
- F_cobranzas, n igual 3: PR95, PR12, PR09.
  Cobranzas numero y recencia. PR12 999 a null.

Suma 35. Revisar contra diccionario antes de congelar.

## Regla anti amplia

No crear F_morosidad unica que bloquee combinaciones utiles.
Nivel, recurrencia y tendencia quedan en familias distintas
si el negocio permite coexistencia. Este borrador ya separa
conteos, antiguedad, consultas y cobranzas.

## Incompatibilidades E

Vacia en fase 2. Se agregan pares solo con justificacion
de negocio y fecha de aprobacion. Correlacion y VIF no
reemplazan E conceptual.

## Tamano del espacio

N_k igual coeficiente de x^k en producto de (1 mas n_g por x).
Con G igual 8 y k igual 6 el espacio full es grande para
oraculo inicial. Por eso el oraculo chico fija 6 familias
chicas, ejemplo F_scores mas F_totales mas F_revolvente mas
F_cobranzas mas dos recortadas, meta cerca de 10k a 20k
combinaciones estructurales antes de filtros.

## Pendiente de aprobacion

Confirmar cortes de F_satisfactorias y F_antiguedad, politica
de sensibles, y subespacio chico exacto con su N_k calculado.
Sin esta aprobacion no se comparan resultados.
