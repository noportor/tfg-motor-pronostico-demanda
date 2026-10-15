Tabla H3. Incertidumbre agrupada de las diferencias del contraste (propuesto menos referencia; negativo = el propuesto erra menos)

**Diferencia media por serie (unidades)**

| Comparación | Estimación | IC 95 % por SKU | Excluye el cero (SKU) | IC 95 % por bloques de 3 meses | Excluye el cero (bloques) | p publicado | r publicado |
|---|---|---|---|---|---|---|---|
| motor frente a promedio móvil de 12 meses | −32,77 | [−49,01; −19,11] | sí | [−101,85; −11,44] | sí | < 0,001 | −0,31 |
| motor frente a promedio móvil de 12 meses con factor de crecimiento | −42,52 | [−65,58; −24,01] | sí | [−74,15; −15,27] | sí | < 0,001 | −0,53 |
| motor frente a ingenuo | −3,77 | [−9,34; 0,99] | no | [−44,87; 9,55] | no | < 0,001 | −0,09 |
| LightGBM frente a promedio móvil de 12 meses | −32,21 | [−49,42; −18,00] | sí | [−102,25; −23,43] | sí | < 0,001 | −0,09 |
| LightGBM frente a ingenuo | −3,21 | [−10,64; 2,74] | no | [−65,49; 10,30] | no | 0,9998 | 0,09 |

**Diferencia mediana por serie (unidades)**

| Comparación | Estimación | IC 95 % por SKU | Excluye el cero (SKU) | IC 95 % por bloques de 3 meses | Excluye el cero (bloques) | p publicado | r publicado |
|---|---|---|---|---|---|---|---|
| motor frente a promedio móvil de 12 meses | −0,38 | [−0,56; −0,20] | sí | [−1,24; 0,00] | no | < 0,001 | −0,31 |
| motor frente a promedio móvil de 12 meses con factor de crecimiento | −2,02 | [−2,54; −1,60] | sí | [−2,76; −0,72] | sí | < 0,001 | −0,53 |
| motor frente a ingenuo | 0,00 | [0,00; 0,00] | no | [0,00; 0,00] | no | < 0,001 | −0,09 |
| LightGBM frente a promedio móvil de 12 meses | −0,07 | [−0,40; 0,24] | no | [−1,74; 0,12] | no | < 0,001 | −0,09 |
| LightGBM frente a ingenuo | 0,36 | [0,02; 0,71] | sí | [−1,95; 2,50] | no | 0,9998 | 0,09 |

**Diferencia de D valorizada del portafolio (puntos porcentuales)**

| Comparación | Estimación | IC 95 % por SKU | Excluye el cero (SKU) | IC 95 % por bloques de 3 meses | Excluye el cero (bloques) | p publicado | r publicado |
|---|---|---|---|---|---|---|---|
| motor frente a promedio móvil de 12 meses | −10,0 | [−14,0; −6,1] | sí | [−75,3; −1,3] | sí | < 0,001 | −0,31 |
| motor frente a promedio móvil de 12 meses con factor de crecimiento | −24,6 | [−32,0; −17,0] | sí | [−45,7; −10,2] | sí | < 0,001 | −0,53 |
| motor frente a ingenuo | −10,2 | [−14,7; −5,4] | sí | [−32,1; 0,9] | no | < 0,001 | −0,09 |
| LightGBM frente a promedio móvil de 12 meses | −8,3 | [−14,0; −2,9] | sí | [−84,6; 2,5] | no | < 0,001 | −0,09 |
| LightGBM frente a ingenuo | −8,6 | [−13,7; −3,3] | sí | [−30,0; 0,4] | no | 0,9998 | 0,09 |

Remuestreo: B = 2.000 réplicas, semilla 20260408, intervalos percentiles al 95 %. Conglomerados: 369 SKU que agrupan 1.668 series (cada SKU sorteado entra con todas sus series). Bloques: 12 meses del bloque de prueba en bloques móviles circulares de 3 meses. D valorizada sobre 1.513 series con costo. El p y el r son los del Wilcoxon unilateral publicado, que trata cada serie como independiente.

Controles: la D reconstruida reproduce la Tabla 8 valorizada con una diferencia máxima de 3,9 × 10⁻⁸ puntos; el panel mensual reproduce la métrica por serie con una diferencia máxima de 9,6 × 10⁻⁵ unidades, la D de cada modelo con 6,5 × 10⁻⁷ puntos y el WMAPE y el sesgo valorizados de cada mes de error_por_mes.csv (lightgbm, ma_12, motor, naive_m1) con 8,1 × 10⁻⁶ puntos (redondeo del archivo mensual). La estimación es la de la muestra completa, sin remuestrear, en ambas variantes.

Corrida: 2026-10-15T03:56:04+00:00 (commit 14e89a6aa2cbe1774771a8b83cbd6748094f4bf7).
