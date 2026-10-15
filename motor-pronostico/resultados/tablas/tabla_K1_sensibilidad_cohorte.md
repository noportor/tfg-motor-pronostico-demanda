Tabla K1. Sensibilidad de la cohorte de evaluación: tamaño, cobertura y D valorizada a un paso en el bloque de prueba

| Cohorte | Series | Series con costo | Cobertura del valor de entrenamiento (%) | D motor (%) | D práctica vigente (%) | D promedio móvil de 12 meses (%) | D ingenuo (%) | D LightGBM (%) | Motor menos práctica vigente (pp) | Posición del motor |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 1.668 | 1.513 | 95,04 | 75,4 | 100,0 | 85,4 | 85,6 | 77,0 | −24,6 | 1 de 11 |
| A: cohorte al origen | 1.812 | 1.657 | 96,18 | 75,1 | 100,0 | 85,4 | 85,6 | 77,1 | −24,9 | 1 de 11 |
| B: cohorte al origen, sólo activas | 1.661 | 1.506 | 91,66 | 74,5 | 98,8 | 84,3 | 85,2 | 75,6 | −24,3 | 1 de 11 |

**Variante A por estado al origen y venta en el bloque de prueba**

| Estado al origen | Venta en prueba | Series | Series con costo | Cobertura (%) | D motor (%) | D práctica vigente (%) | Motor menos práctica vigente (pp) | Puntos de WMAPE en la A: motor | Puntos de WMAPE en la A: práctica vigente |
|---|---|---|---|---|---|---|---|---|---|
| todas | todas | 1.812 | 1.657 | 96,18 | 75,1 | 100,0 | −24,9 | 71,83 | 85,92 |
| todas | con venta en prueba | 1.668 | 1.513 | 95,04 | 75,1 | 100,0 | −24,9 | 71,78 | 85,91 |
| todas | sin venta en prueba | 144 | 144 | 1,14 | n/d | n/d | n/d | 0,05 | 0,02 |
| activa o dormida | todas | 1.777 | 1.622 | 95,49 | 74,8 | 99,7 | −25,0 | 71,38 | 85,49 |
| activa | todas | 1.661 | 1.506 | 91,66 | 74,2 | 98,8 | −24,6 | 69,99 | 84,02 |
| activa | con venta en prueba | 1.565 | 1.410 | 90,68 | 74,2 | 98,8 | −24,6 | 69,95 | 84,01 |
| activa | sin venta en prueba | 96 | 96 | 0,98 | n/d | n/d | n/d | 0,04 | 0,01 |
| dormida | todas | 116 | 116 | 3,83 | 111,3 | 159,0 | −47,6 | 1,39 | 1,46 |
| dormida | con venta en prueba | 68 | 68 | 3,67 | 111,3 | 159,0 | −47,6 | 1,38 | 1,46 |
| dormida | sin venta en prueba | 48 | 48 | 0,16 | n/d | n/d | n/d | 0,01 | 0,00 |
| descontinuada | todas | 35 | 35 | 0,69 | 172,9 | 178,2 | −5,2 | 0,45 | 0,44 |
| descontinuada | con venta en prueba | 35 | 35 | 0,69 | 172,9 | 178,2 | −5,2 | 0,45 | 0,44 |
| descontinuada | sin venta en prueba | 0 | 0 | 0,00 | n/d | n/d | n/d | 0,00 | 0,00 |

**Series de la variante A sin venta en el bloque de prueba: error valorizado por modelo**

| Modelo | Error valorizado de las series sin venta (Bs) | Puntos de WMAPE que agregan a la A | WMAPE original (%) | Sesgo original (%) | D original (%) | WMAPE en la A (%) | Sesgo en la A (%) | D en la A (%) |
|---|---|---|---|---|---|---|---|---|
| ma_12_gf | 21.315 | 0,02 | 83,83 | −16,14 | 99,97 | 85,92 | −14,04 | 99,97 |
| naive_m1_gf | 26.348 | 0,02 | 88,88 | −7,54 | 96,42 | 90,73 | −5,69 | 96,42 |
| naive_m1 | 32.471 | 0,03 | 81,93 | −3,64 | 85,57 | 83,46 | −2,10 | 85,57 |
| ma_2_gf | 41.855 | 0,03 | 87,45 | −10,59 | 98,04 | 90,91 | −7,13 | 98,04 |
| motor | 59.459 | 0,05 | 70,10 | −5,30 | 75,40 | 71,83 | −3,23 | 75,06 |
| ma_2 | 61.446 | 0,05 | 79,58 | −5,36 | 84,94 | 82,84 | −2,10 | 84,94 |
| ma_12 | 264.808 | 0,22 | 80,86 | −4,50 | 85,36 | 83,14 | −2,22 | 85,36 |
| exp_smooth_opt | 278.519 | 0,23 | 77,58 | −1,92 | 79,50 | 80,11 | 0,61 | 80,73 |
| lightgbm | 902.434 | 0,74 | 62,94 | −14,08 | 77,02 | 65,05 | −12,05 | 77,11 |
| croston | 1.387.735 | 1,13 | 84,10 | 2,03 | 86,13 | 87,91 | 5,84 | 93,75 |
| holt_winters | 1.624.428 | 1,33 | 77,78 | 5,72 | 83,50 | 82,30 | 10,24 | 92,54 |

**Quiebres registrados en el bloque de prueba (cruce por SKU y regional)**

- A, sin venta en prueba: 144 series; 141 con medición de existencias en el bloque; 131 con al menos un mes con semanas de existencia en cero; 117 con existencia en cero en todas las semanas medidas del bloque.
- A, con venta en prueba: 1.668 series; 1.507 con medición de existencias en el bloque; 1.097 con al menos un mes con semanas de existencia en cero; 59 con existencia en cero en todas las semanas medidas del bloque.

Fuente de los quiebres: archivo de quiebres de la captura (semanas con existencia en cero por SKU, regional y mes; sin canal). Sólo conteos agregados.


D valorizada = WMAPE valorizado + |sesgo valorizado agregado|, a un paso en el bloque de prueba, sobre las series con costo de cada cohorte. Cada D se compara con la de los otros modelos de la MISMA cohorte. Cobertura: valor de entrenamiento de la cohorte sobre el de la población objetivo. «Práctica vigente»: réplica de la fórmula de planificación con factor de crecimiento (ma_12_gf), sin los ajustes de consenso del plan final.

Variante A: sin el paso 6 de la cascada (presencia en prueba) y con la vida de cada serie extendida con ceros hasta el fin del período; agrega 144 series a la cohorte publicada, de las que 144 no tienen venta en el bloque de prueba. En la A las series de la cohorte publicada se evalúan en los doce meses del bloque (20.016 meses-serie, frente a 18.502 en la corrida publicada, que no evalúa los meses posteriores a la última venta).
Variante B: la A restringida a las series activas al origen (venta en los seis meses previos a abril de 2025, regla de clasificar_estados).
Estado al origen (columna estado_al_origen de fuera_de_muestra.csv de la A): activa, venta en los seis meses previos; dormida, de seis a once meses sin venta; descontinuada, doce o más, medido al cierre de validación sobre la actividad observada. En los grupos sin venta en prueba la D no existe (demanda valorizada nula): se informan los puntos de WMAPE que su error agrega a la D de la variante A (error valorizado del grupo sobre la demanda valorizada de toda la A).
Paso 5 en la A (presencia en validación sobre la vida observada, de la primera a la última venta del período): de las 35 series descontinuadas al origen, 0 tienen venta en validación y 35 tienen venta en prueba: pasan el paso 5 porque su vida observada se prolonga hasta el bloque de prueba, información posterior al origen. Por la misma razón 35 de ellas ya están en la cohorte publicada (su paso 5 mira las filas del bloque de validación, que existen porque la vida acotada llega hasta la última venta). La fila «activa o dormida» es la A sin ellas; la B, que exige actividad al origen, no las contiene.

Controles: en cada corrida, la cohorte de errores_por_serie, fuera_de_muestra y cohorte_flujo coincide; la D reconstruida desde los archivos por serie reproduce tabla_valorizada.csv (diferencia máxima original: 4.7e-08 pp, A: cohorte al origen: 5.7e-08 pp, B: cohorte al origen, sólo activas: 5.2e-08 pp); la A contiene la cohorte publicada; la B es la A restringida a las activas al origen; todas las series sin venta en prueba son series agregadas por la A.

Corridas: original = salidas (2026-10-14T23:33:28+00:00, commit 10bed4b, 2.278,9 s); A: cohorte al origen = salidas_sensibilidad_a (2026-10-15T00:16:13+00:00, commit 10bed4b, 2.553,1 s); B: cohorte al origen, sólo activas = salidas_sensibilidad_b (2026-10-15T01:56:55+00:00, commit fa64e3c, 2.150,4 s).
Archivo de quiebres: verificado contra datos.auxiliares de la corrida publicada.
Meses comunes: en los 18.502 meses-serie del bloque de prueba que evalúa la corrida original, 9 de los 11 modelos pronostican exactamente lo mismo en la A (croston, exp_smooth_opt, holt_winters, ma_12, ma_12_gf, ma_2, ma_2_gf, naive_m1, naive_m1_gf); restringida a esos meses, la D del motor en la A es 75,06 % y la de ma_12_gf, 99,97 %.
