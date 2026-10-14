Tabla P2. Costo de cómputo y precisión por brazo en la corrida principal

| Brazo | Ajuste y pronóstico a un paso (s) | Con preparación de variables (s) | Por serie, amortizado (ms) | D a un paso (%) | D en el horizonte (%) | Rango medio de Friedman (Figura 10) | Series elegidas por el motor |
|---|---|---|---|---|---|---|---|
| Motor de selección automática | 11,29 | 40,29 | 24,15 | 75,4 | 91,2 | 4,67 | n. d. |
| LightGBM | 9,94 | 34,54 | 20,71 | 77,0 | 71,9 | 5,61 | 367 |
| Suavizado exponencial simple | 0,08 | 0,08 | 0,05 | 79,5 | 103,8 | 5,22 | 93 |
| Holt–Winters | 0,67 | 0,67 | 0,40 | 83,5 | 81,2 | 6,11 | 204 |
| Promedio móvil 2 meses | 0,08 | 0,08 | 0,05 | 84,9 | 116,6 | 4,63 | 232 |
| Promedio móvil 12 meses | 0,08 | 0,08 | 0,05 | 85,4 | 94,6 | 5,74 | 131 |
| Ingenuo | 0,01 | 0,01 | 0,01 | 85,6 | 120,5 | 4,96 | 339 |
| Croston | 0,08 | 0,08 | 0,05 | 86,1 | 93,5 | 6,78 | 96 |
| Ingenuo con factor de crecimiento | 0,07 | 0,07 | 0,04 | 96,4 | 158,0 | 7,49 | 38 |
| Prom. móvil 2 m con factor de crecimiento | 0,14 | 0,14 | 0,08 | 98,0 | 134,3 | 7,47 | 70 |
| Prom. móvil 12 m con factor de crecimiento | 0,14 | 0,14 | 0,08 | 100,0 | 113,2 | 7,32 | 98 |
| Ingenuo estacional (fuera del menú) | n. d. | n. d. | n. d. | 82,7 | 84,1 | n. d. | n. d. |
| SBA aproximado (fuera del menú) | n. d. | n. d. | n. d. | 88,5 | 99,1 | n. d. | n. d. |

Tiempos de la corrida generada el 2026-10-14T19:16:49+00:00 (commit f31684386648566a5f4584a44da763bedc632437); resultados de la corrida generada el 2026-10-14T19:16:49+00:00 (commit f31684386648566a5f4584a44da763bedc632437).

Tabla P3. Costo de cómputo y precisión de los pronosticadores de la extensión exploratoria en las dos gestiones de prueba

| Brazo | Gestión 2026: ajuste y pronóstico a un paso (s) | Gestión 2025: ajuste y pronóstico a un paso (s) | Gestión 2026: D a un paso (%) | Gestión 2025: D a un paso (%) | Gestión 2026: D en el horizonte (%) | Gestión 2025: D en el horizonte (%) | Calibración: s (ensayos) |
|---|---|---|---|---|---|---|---|
| Chronos-Bolt (fundacional) | 27,27 | 23,13 | 74,3 | 81,3 | 81,6 | 87,9 | Sin búsqueda propia |
| DLinear | 64,58 | 48,56 | 80,4 | 85,1 | 94,8 | 78,5 | Sin búsqueda propia |
| N-HiTS | 32,02 | 30,40 | 92,3 | 70,6 | 93,9 | 78,8 | 496,8 (25) |
| TFT | 147,76 | 146,17 | 106,1 | 83,2 | 114,1 | 110,9 | 1.949,1 (25) |
| LightGBM con proyección directa | 88,54 | 68,44 | 77,0 | 83,0 | 84,0 | 72,8 | Sin búsqueda propia |
| LightGBM | 8,98 | 6,86 | 77,0 | 83,0 | 71,9 | 70,1 | 461,3 (60) |
| Holt–Winters | 0,83 | 0,50 | 83,5 | 111,5 | 81,2 | 120,0 | Sin búsqueda propia |

Para la nota de la Tabla P3: ingenuo en el entorno de la extensión, 0,01 s, 85,6 % a un paso y 120,5 % en el horizonte (2026); 0,01 s, 78,7 % y 115,1 % (2025). Ensayos fallidos: lightgbm 0 de 60; nhits 3 de 25; tft 3 de 25.
