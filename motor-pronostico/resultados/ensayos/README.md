# Tablas de ensayos de la búsqueda de hiperparámetros

Una fila por ensayo de la búsqueda (TPE con semilla, pliegue interno de entrenamiento): la pérdida, los
hiperparámetros propuestos, la mejor iteración y las métricas internas (WMAPE, sesgo y D) con el tiempo
de cada ensayo. No contienen series, productos ni identificadores: se versionan para que la tabla de
hiperparámetros y la evaluación comparativa de la memoria sean reproducibles desde el repositorio.

- `lightgbm_tuning_ensayos.csv`: 60 ensayos del brazo global (corrida principal).
- `nhits_tuning_ensayos.csv` y `tft_tuning_ensayos.csv`: 25 ensayos de cada red de la extensión exploratoria.

Las copias de trabajo viven en `salidas_tuning*/`, que no se versiona; estas son las publicadas.
