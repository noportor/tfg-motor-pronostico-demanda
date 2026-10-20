Tabla P2. Costo de cómputo y precisión por brazo en la corrida principal

| Brazo | Ajuste y pronóstico a un paso (s) | Con preparación de variables (s) | Por serie, amortizado (ms) | D a un paso (%) | D en el horizonte (%) | Rango medio de Friedman (Figura 10) | Series elegidas por el motor |
|---|---|---|---|---|---|---|---|
| Motor de selección automática | 12,64 | 45,48 | 27,27 | 75,4 | 91,2 | 4,67 | n. d. |
| LightGBM | 11,25 | 39,08 | 23,43 | 77,0 | 71,9 | 5,61 | 367 |
| Suavizado exponencial simple | 0,10 | 0,10 | 0,06 | 79,5 | 103,8 | 5,22 | 93 |
| Holt-Winters | 0,71 | 0,71 | 0,43 | 83,5 | 81,2 | 6,11 | 204 |
| Promedio móvil 2 meses | 0,07 | 0,07 | 0,04 | 84,9 | 116,6 | 4,63 | 232 |
| Promedio móvil 12 meses | 0,07 | 0,07 | 0,04 | 85,4 | 94,6 | 5,74 | 131 |
| Ingenuo | 0,01 | 0,01 | 0,01 | 85,6 | 120,5 | 4,96 | 339 |
| Croston | 0,09 | 0,09 | 0,05 | 86,1 | 93,5 | 6,78 | 96 |
| Ingenuo con factor de crecimiento | 0,07 | 0,07 | 0,04 | 96,4 | 158,0 | 7,49 | 38 |
| Prom. móvil 2 m con factor de crecimiento | 0,14 | 0,14 | 0,08 | 98,0 | 134,3 | 7,47 | 70 |
| Prom. móvil 12 m con factor de crecimiento | 0,13 | 0,13 | 0,08 | 100,0 | 113,2 | 7,32 | 98 |
| Ingenuo estacional (fuera del conjunto) | n. d. | n. d. | n. d. | 82,7 | 84,1 | n. d. | n. d. |
| SBA aproximado (fuera del conjunto) | n. d. | n. d. | n. d. | 88,5 | 99,1 | n. d. | n. d. |

Tiempos de la corrida generada el 2026-10-19T18:49:36+00:00 (commit 472b891eca2998e3ec55a8a49511032be388bf96); resultados de la corrida generada el 2026-10-19T18:49:36+00:00 (commit 472b891eca2998e3ec55a8a49511032be388bf96).

Tabla P3. Costo de cómputo y precisión de los pronosticadores de la extensión exploratoria en las dos gestiones de prueba

| Brazo | Gestión 2026: ajuste y pronóstico a un paso (s) | Gestión 2025: ajuste y pronóstico a un paso (s) | Gestión 2026: D a un paso (%) | Gestión 2025: D a un paso (%) | Gestión 2026: D en el horizonte (%) | Gestión 2025: D en el horizonte (%) | Calibración: s (ensayos) |
|---|---|---|---|---|---|---|---|
| Chronos-Bolt (fundacional) | 29,09 | 33,40 | 74,3 | 81,3 | 81,6 | 87,9 | Sin búsqueda propia |
| DLinear | 66,02 | 53,42 | 80,4 | 85,1 | 94,8 | 78,5 | Sin búsqueda propia |
| N-HiTS | 37,02 | 37,25 | 92,3 | 70,6 | 93,9 | 78,8 | 496,8 (25) |
| TFT | 146,07 | 200,06 | 106,1 | 83,2 | 114,1 | 110,9 | 1.949,1 (25) |
| LightGBM con proyección directa | 102,58 | 69,48 | 77,0 | 83,0 | 84,0 | 72,8 | Sin búsqueda propia |
| LightGBM | 10,62 | 7,26 | 77,0 | 83,0 | 71,9 | 70,1 | 461,3 (60) |
| Holt-Winters | 1,02 | 0,53 | 83,5 | 111,5 | 81,2 | 120,0 | Sin búsqueda propia |

Para la nota de la Tabla P3: ingenuo en el entorno de la extensión, 0,02 s, 85,6 % a un paso y 120,5 % en el horizonte (2026); 0,01 s, 78,7 % y 115,1 % (2025). Ensayos fallidos: lightgbm 0 de 60; nhits 3 de 25; tft 3 de 25.
