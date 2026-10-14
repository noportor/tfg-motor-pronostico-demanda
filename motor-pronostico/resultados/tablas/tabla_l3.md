Tabla L3. Hiperparámetros del modelo global: valor elegido, rango explorado, función y lectura de la búsqueda

| Hiperparámetro | Valor elegido | Rango explorado (escala) | Función | Rango en la meseta (16 ensayos) | Importancia en la búsqueda (Optuna, Tweedie) | Correlación de rangos con la D interna (Tweedie) |
|---|---|---|---|---|---|---|
| Función de pérdida ('objective') | Tweedie | Absoluta, 8 ensayos (D interna de 72,6 a 81,3); cuadrática, 7 (75,3 a 95,6); Tweedie, 45 (59,7 a 71,4) (categórica) | Estructural: decide qué estadístico de la demanda aprende el modelo | Tweedie en los 16 | 0,84 (sobre los 60 ensayos) | No aplica |
| Potencia de la varianza Tweedie ('tweedie_variance_power') | 1,465 | 1,1 a 1,6 (lineal; sólo con pérdida Tweedie) | Forma de la pérdida: cuánto crece la varianza con el nivel | 1,39 a 1,59 | 0,20 | −0,30 |
| Tasa de aprendizaje ('learning_rate') | 0,053 | 0,02 a 0,12 (logarítmica) | Ritmo: contrae el aporte de cada árbol | 0,028 a 0,081 | 0,05 | +0,12 |
| Número de árboles (parada temprana) | 87 en el pliegue interno; 76 en la corrida principal | No se busca: parada a las 100 rondas sin mejora, con techo de 3.000 (de 33 a 226 en los ensayos Tweedie) | Ritmo: se compensa con la tasa | 52 a 144 | No aplica | −0,03 |
| Número máximo de hojas ('num_leaves') | 137 | 15 a 255 (logarítmica, entera) | Capacidad de cada árbol | 77 a 189 | 0,25 | −0,29 |
| Mínimo de observaciones por hoja ('min_data_in_leaf') | 61 | 10 a 200 (logarítmica, entera) | Capacidad: evidencia mínima de cada hoja | 38 a 113 | 0,05 | +0,19 |
| Fracción de variables por árbol ('feature_fraction') | 0,578 | 0,5 a 1 (lineal) | Regularización: submuestreo de columnas | 0,54 a 0,75 | 0,11 | +0,32 |
| Fracción de filas por iteración ('bagging_fraction') | 0,870 | 0,5 a 1 (lineal) | Regularización: submuestreo de filas | 0,73 a 0,88 | 0,13 | −0,02 |
| Penalización L2 ('lambda_l2') | 24,47 | 0 a 30 (lineal) | Regularización: encoge el valor de cada hoja | 22,2 a 29,9 | 0,21 | −0,49 |
| Valores fijos (no buscados) | Semilla del modelo 20260406; semilla de la búsqueda 20260408; un hilo; modo determinista; submuestreo de filas en cada iteración | No aplica | Reproducibilidad | No aplica | No aplica | No aplica |

Resumen por función de pérdida (apoyo de la nota y del texto; no se publica como tabla)

| Pérdida | Ensayos | Mejor ensayo | Mejor D | WMAPE del mejor | Sesgo del mejor | Mediana de D | Peor ensayo | Peor D | Árboles (mín. a máx.) | Segundos |
|---|---|---|---|---|---|---|---|---|---|---|
| Tweedie | 45 | 48 | 59,7 | 59,6 | +0,2 | 62,4 | 7 | 71,4 | 33 a 226 | 281,4 |
| absoluta | 8 | 39 | 72,6 | 59,9 | −12,7 | 73,8 | 8 | 81,3 | 82 a 1.053 | 132,8 |
| cuadrática | 7 | 4 | 75,3 | 67,3 | +8,0 | 79,7 | 9 | 95,6 | 48 a 184 | 47,1 |
