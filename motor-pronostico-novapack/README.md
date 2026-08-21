# Motor de pronóstico de demanda — NOVAPACK S.A.

Artefacto de software del Trabajo Final de Grado *«Diseño de un motor de
pronóstico de demanda basado en series temporales para mejorar la precisión de
la planificación de la demanda en NOVAPACK S.A. para la gestión 2026»* —
UAGRM School of Engineering, Maestría en Ciencia de Datos e Inteligencia
Artificial.

Este repositorio contiene el código con el que se produjeron **todos los
resultados cuantitativos del documento**: cada tabla, figura y prueba
estadística proviene de una corrida reproducible de este pipeline. El Anexo G
del documento enlaza aquí; el Anexo H reproduce el contraste estadístico con
[`scripts/anexo_h_contraste.py`](scripts/anexo_h_contraste.py). La
especificación completa del experimento (requerimientos y reglas de negocio
RN-1 a RN-6) está en [`REQUERIMIENTOS.md`](REQUERIMIENTOS.md).

---

## Qué hace

Compara **once modelos** de pronóstico de demanda mensual sobre cada
combinación SKU–canal–regional del portafolio, y contrasta estadísticamente si
el motor de selección automática mejora la precisión frente a los métodos
vigentes de planificación.

| Brazo | Papel |
|---|---|
| `naive_m1`, `naive_m1_gf` | Línea base estándar y método vigente |
| `ma_2`, `ma_2_gf`, `ma_12`, `ma_12_gf` | Promedio móvil — método vigente |
| `exp_smooth_opt` | Clásico: suavizado exponencial simple |
| `holt_winters` | Clásico: nivel + tendencia + estacionalidad |
| `croston` | Clásico para demanda intermitente |
| `lightgbm` | Aprendizaje automático, modelo global |
| `motor` | **La propuesta**: elige por serie el mejor modelo en validación |

Los seis primeros brazos no son una elección del autor: replican los métodos
que el área de planificación aplica, verificados contra el sistema vigente. El
benchmark del estudio es el método real, no una línea base de conveniencia.

## Diseño experimental

**Período.** Nueve gestiones fiscales completas: abril-2017 → marzo-2026
(gestiones 2018 a 2026, nombradas por el año en que cierran).

**Variable objetivo.** Ventas efectivas mensuales por combinación
SKU–canal–regional.

**Partición.** Los cortes caen en frontera de gestión, para que ningún bloque
parta una campaña escolar por la mitad:

| Bloque | Gestiones | Meses | Para qué |
|---|---|---|---|
| Entrenamiento | 2018–2024 | abr-2017 … mar-2024 (84) | ajusta los modelos |
| Validación | 2025 | abr-2024 … mar-2025 (12) | el motor decide |
| Prueba | 2026 | abr-2025 … mar-2026 (12) | se reporta el desempeño |

**Evaluación.** Pronóstico a un mes, con origen rodante: cada mes se pronostica
usando la información disponible hasta el mes anterior. La métrica de decisión
es el error compuesto valorizado **D = WMAPE + |Bias|**, que pondera el error
de cada serie por el costo unitario del producto. La comparación formal entre
modelos usa pruebas no paramétricas pareadas (Wilcoxon; Friedman con post hoc
de Nemenyi).

## Correspondencia con el documento

| Artefacto del repositorio | Lugar en el documento |
|---|---|
| `salidas/tabla8_resultados.csv` | Tabla 8 — MAE, MAPE, RMSE y Bias por modelo |
| `salidas/pruebas_estadisticas.txt` | Constatación y validación: contraste estadístico |
| `salidas/matriz_contraste.csv` | Matriz serie × modelo sobre la que corren las pruebas |
| `salidas/seleccion_motor.csv` | Evidencia de la selección de modelo por serie |
| `salidas/errores_por_serie.csv` | Anexo con los errores por serie (SKU seudonimizados) |
| `salidas/figura01_*.png` … `figura13_*.png` | Catálogo de figuras del documento (F1–F13) |
| `salidas/inspeccion_datos.txt`, `cohorte_flujo.csv` | Criterios de inclusión y tamaño de la muestra |
| `salidas/manifiesto.json` | Trazabilidad de la corrida (datos, configuración, código) |
| `scripts/anexo_h_contraste.py` | Anexo H — reproduce el contraste estadístico |

## Reproducibilidad

- **Sin Python local**: todo corre en Docker con las versiones fijadas.
- **Configuración declarada**: cada corrida se define en `config/config.yaml`;
  cada variante experimental se registra con `--anular clave=valor` y queda
  asentada en el manifiesto, de modo que dos corridas distintas nunca pueden
  confundirse.
- **Manifiesto por corrida**: registra el SHA-256 del snapshot de datos, la
  configuración efectiva y la versión del código con que se obtuvo cada
  resultado.
- **Contraste sobre lo publicado**: las pruebas estadísticas se ejecutan sobre
  la misma `matriz_contraste.csv` que se publica como salida — lo reportado en
  el documento y lo reproducible son el mismo archivo.

## Cómo ejecutarlo

```bash
docker compose build

# 1) Congelar el snapshot de datos (requiere acceso a la fuente; ver Confidencialidad)
docker compose run --rm tfg python scripts/extraer_snapshot.py

# 2) Inspeccionar los datos y calibrar los criterios de inclusión
docker compose run --rm tfg python main.py inspeccionar

# 3) Ejecutar el experimento completo
docker compose run --rm tfg python main.py ejecutar

# Pruebas automatizadas (contrato de datos, pipeline y confidencialidad)
docker compose run --rm tfg python -m pytest -q
```

El **visor** permite explorar las corridas desde el navegador, en modo de solo
lectura: resultados por modelo, selección del motor, contraste estadístico y
comparación entre corridas.

```bash
docker compose up -d visor        # -> http://localhost:8501
```

## Confidencialidad

El histórico de ventas de NOVAPACK S.A. está cubierto por un acuerdo de
confidencialidad (Anexo B del documento), y con él el esquema interno del
sistema de origen. Por eso **los datos no están en este repositorio** y el
pipeline se ejecuta sobre un snapshot local que no se versiona. Tres barreras
lo garantizan:

1. **`.gitignore`** — `datos/crudo/`, `salidas*/` y la configuración de
   extracción local nunca se versionan.
2. **Seudonimización** — los códigos de producto se reemplazan al extraer
   (`SKU-0001`, …); la tabla de correspondencia queda fuera del repositorio.
3. **Control automático** — `tests/test_confidencialidad.py` revisa todos los
   archivos que git rastrea buscando nombres reales, identificadores del
   sistema de origen y credenciales; la suite falla si encuentra algo.

Sin el snapshot, el pipeline no puede ejecutarse fuera de la empresa; el
repositorio documenta el procedimiento completo, y los resultados publicados en
el documento llevan los hashes de su manifiesto para su verificación.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `main.py`, `src/` | El pipeline del experimento (etapas, modelos, reporte) |
| `scripts/` | Extracción del snapshot, ajuste de modelos y Anexo H |
| `tests/` | Contrato de datos, pipeline y control de confidencialidad |
| `visor/` | Visor de corridas (React + nginx, solo lectura) |
| `config/` | Configuración declarada del experimento |
| `REQUERIMIENTOS.md` | Especificación completa: requerimientos y reglas RN-1 a RN-6 |
