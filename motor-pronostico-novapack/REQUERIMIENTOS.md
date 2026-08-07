# Motor de pronóstico de demanda — NOVAPACK S.A.

Especificación del proyecto para desarrollo asistido con Claude Code.

> **Cómo cargar este archivo en Claude Code.** La forma directa es mencionarlo al
> abrir la sesión: `lee @REQUERIMIENTOS.md`. Si prefiere que se cargue solo cada
> vez, cree en la raíz de *este* repositorio un `CLAUDE.md` de una línea que diga
> `Ver @REQUERIMIENTOS.md`; al vivir dentro del repositorio, no afecta a sus otros
> proyectos. Otra opción es colocar el archivo en `.claude/rules/`, cuyos `.md` se
> cargan como memoria del proyecto.

---

## 1. Contexto

Este repositorio implementa el análisis cuantitativo de un Trabajo Final de Grado
de maestría (UAGRM School of Engineering, Ciencia de Datos e Inteligencia
Artificial). El objetivo de la tesis es diseñar un **motor de pronóstico de
demanda** que seleccione automáticamente el modelo más preciso para cada
combinación SKU–canal–regional de la empresa NOVAPACK S.A.

La hipótesis a contrastar es que ese motor **mejora significativamente** la
precisión del pronóstico frente a los métodos tradicionales que la empresa usa
hoy (promedio móvil) y frente a la línea base estándar (Naïve).

**Los resultados que produzca este código se publican en un documento académico
que será defendido ante un tribunal.** Esa condición gobierna todas las
decisiones de diseño que siguen.

### Estado de partida

- Existe un prototipo funcional de un solo archivo: `pipeline_tfg.py`. Sirve como
  referencia de la lógica, pero debe reescribirse modularmente.
- Los datos históricos de ventas son reales y están completos.
- El diagnóstico cualitativo de la tesis (entrevista y observación) ya fue
  realizado y es válido; **no forma parte de este repositorio**.

---

## 2. Requerimientos no negociables

Son restricciones de integridad y validez metodológica. Cualquier cambio que las
vulnere debe rechazarse aunque simplifique el código o mejore los números.

### RN-1 — Ningún resultado puede provenir de datos simulados

Los datos sintéticos se admiten **exclusivamente** en `tests/`. Está prohibido
generar valores de relleno, redondear "hacia lo esperado" o dejar constantes
hardcodeadas que sustituyan un cálculo. Si un número no se puede calcular, el
programa falla con un error explícito; no lo inventa.

### RN-2 — Partición en tres bloques

```
entrenamiento  ->  ajusta los modelos
validación     ->  el motor decide qué modelo gana en cada serie
prueba         ->  se reporta el desempeño final
```

La selección por serie que hace el motor **solo puede mirar validación**. Si esa
decisión se tomara con datos de prueba, el resultado reportado estaría
contaminado y la comparación sería inválida. Esta es la restricción más fácil de
romper por accidente y la más grave.

### RN-3 — Sin fuga temporal en las features

Toda variable derivada (rezagos, medias móviles, desviaciones) debe construirse
únicamente con información anterior al mes que se predice. Prohibido usar
estadísticos calculados sobre el conjunto completo (medias globales,
normalizaciones ajustadas con todo el histórico, codificaciones de destino sin
particionar).

### RN-4 — Evaluación homogénea a un paso

Los tres modelos se evalúan un mes hacia adelante usando el valor real del mes
anterior. Es la única comparación justa: el Promedio Móvil y el Naïve funcionan
así por construcción. Si en algún momento se agrega evaluación multihorizonte,
debe aplicarse por igual a todos los modelos y reportarse por separado.

### RN-5 — Reproducibilidad

Misma entrada y misma configuración deben producir exactamente la misma salida.
Semillas fijas, versiones de dependencias fijadas, sin dependencia del orden de
iteración de diccionarios ni de la hora del sistema.

### RN-6 — Trazabilidad

Cada número que llegue al documento debe poder rastrearse hasta el archivo de
salida y el commit que lo generó. Cada ejecución escribe un `manifiesto.json` con
timestamp, hash del archivo de datos, versión del código y configuración usada.

---

## 3. Datos

### Esquema de entrada esperado

Un archivo `.csv` o `.xlsx` con granularidad de transacción o de venta agregada:

| Campo | Tipo | Descripción |
|---|---|---|
| fecha | fecha | fecha de la venta |
| sku | texto | código de producto |
| canal | texto | mayorista, minorista, distribuidor, institucional |
| regional | texto | regional de destino |
| cantidad | numérico | unidades vendidas |

Los nombres reales de las columnas se declaran en configuración, no se asumen.

### Construcción de las series

1. Agregar a nivel **mensual** por combinación SKU–canal–regional.
2. Completar los meses sin registro con cero, **solo entre la primera y la última
   venta de cada serie**. No extrapolar hacia atrás: un mes anterior al
   lanzamiento del producto no es demanda cero, es ausencia de producto.
3. Las cantidades negativas (devoluciones) se tratan según se decida en
   configuración; por defecto se descartan y se reporta cuántas eran.

### Criterios de inclusión y exclusión

Parametrizables. Valores de partida sugeridos, a calibrar con el resultado de la
inspección:

- historial mínimo: 36 meses
- proporción máxima de meses en cero: 0,70
- volumen acumulado mínimo: 50 unidades

El módulo debe reportar **cuántas series excluyó cada criterio por separado**.
Ese desglose va literalmente al apartado «Criterios de inclusión y exclusión» de
la tesis, y el N resultante es el tamaño de la muestra.

---

## 4. Requerimientos funcionales

Cada módulo lleva sus criterios de aceptación. Un módulo no está terminado hasta
que sus pruebas pasan.

### RF-1 — Carga (`src/carga.py`)

- Lee CSV y Excel; el formato se infiere de la extensión.
- Mapea nombres de columnas desde configuración.
- Falla con un mensaje que enumere las columnas encontradas si falta alguna.
- Convierte tipos y reporta cuántos valores no pudo convertir.

**Aceptación:** con un archivo cuyas columnas no coinciden, el error indica qué
falta y qué hay.

### RF-2 — Inspección (`src/inspeccion.py`)

Corresponde a la fase de comprensión de datos de CRISP-DM. Produce un informe
con: rango de fechas, conteo de SKU, canales y regionales, registros duplicados,
nulos, negativos, ceros, distribución de registros por combinación y volumen por
año.

**Aceptación:** el informe permite decidir los criterios de inclusión sin abrir
los datos manualmente.

### RF-3 — Construcción de series (`src/series.py`)

**Aceptación:** una serie con ventas en enero y marzo produce tres filas
(enero, febrero=0, marzo). Una serie que empieza en marzo no genera filas de
enero ni febrero.

### RF-4 — Partición (`src/particion.py`)

Corta por fecha en los tres bloques. Configurable.

**Aceptación:** existe una prueba que verifica que ninguna fecha de prueba es
anterior a una de validación, y ninguna de validación es anterior a una de
entrenamiento.

### RF-5 — Features (`src/features.py`)

Rezagos 1, 2, 3, 6 y 12; medias y desviaciones móviles de 3, 6 y 12 meses
calculadas sobre valores desplazados; mes, año y tendencia; SKU, canal y regional
como categóricas.

**Aceptación:** existe una prueba que altera un valor futuro de la serie y
verifica que **ninguna** feature del mes actual cambia. Esta prueba es la
garantía de la RN-3 y debe escribirse antes que el módulo.

### RF-6 — Modelos (`src/modelos/`)

Interfaz común para todos:

```python
class Modelo(Protocol):
    nombre: str
    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None: ...
    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame: ...
```

Once brazos (ver §11, decisión cerrada el 2026-08-06):

**Métodos que la empresa usa hoy** — confirmados contra el sistema en producción.
No son una elección del autor; §10 exige que el benchmark sea el método real.

- **`naive_m1`**: el valor del mes anterior. Es además la línea base estándar.
- **`naive_m1_gf`**: idem más el factor de crecimiento del año móvil.
- **`ma_2`**, **`ma_2_gf`**: promedio móvil bimestral, con y sin crecimiento.
- **`ma_12`**, **`ma_12_gf`**: promedio móvil anual, con y sin crecimiento.
  `ma_12` es el **benchmark declarado** en el contraste confirmatorio.

**Clásicos estadísticos** — incorporados por recomendación de la dirección del
trabajo, para que la comparación no quede expuesta a la crítica de que LightGBM
solo se midió contra métodos triviales.

- **`exp_smooth_opt`**: suavizado exponencial simple, α elegido por serie.
- **`holt_winters`**: aditivo, nivel + tendencia + estacionalidad (m = 12).
- **`croston`**: para las series de demanda intermitente.

**Los dos brazos del estudio**

- **`lightgbm`**: modelo global sobre todas las series, con early stopping contra
  validación. Objetivo `regression_l1`, coherente con que la métrica principal
  es el MAE.
- **`motor`**: elige por serie el modelo con menor MAE **en validación** y lo
  aplica en prueba.

Los parámetros de los modelos estadísticos se estiman **solo con
entrenamiento**, minimizando el error absoluto —no el cuadrático— para no
optimizar una cosa y reportar otra. Se implementan en NumPy y no se importan de
una biblioteca de series temporales: los optimizadores de máxima verosimilitud
pueden converger a óptimos distintos entre versiones y la RN-5 dejaría de
cumplirse.

Todas las predicciones se truncan en cero: la demanda no es negativa.

Todos los modelos deben cubrir **exactamente los mismos meses**. Cuando un modelo
no puede pronosticar un mes (el promedio móvil de doce no existe antes del
duodécimo mes de vida) se aplica un respaldo declarado —Naïve y, si tampoco
existe, la media de entrenamiento— y **se cuenta**. Sin esto la comparación
pareada dejaría de ser pareada.

**Aceptación:** cada modelo tiene una prueba con una serie construida a mano cuyo
resultado esperado se conoce de antemano.

### RF-7 — Métricas (`src/metricas.py`)

Por serie, sobre el bloque de prueba: MAE, RMSE, MAPE, Bias y MASE.

Reglas que no pueden alterarse:

- **MAPE**: indefinido cuando la demanda real es cero. Se calcula sobre los meses
  con demanda positiva y se reporta qué porcentaje de observaciones quedó fuera.
- **RMSE**: depende de la escala. Se calcula por serie y luego se promedia; nunca
  sobre el conjunto agrupado.
- **Bias**: porcentual, `(media(pred) - media(real)) / media(real) * 100`.
- **MASE**: escalado con el error del naive en entrenamiento. Es la métrica
  robusta a ceros y la que conviene reportar cuando el MAPE excluye demasiado.

Se reportan **media y mediana** de cada métrica.

> **Por qué la mediana importa.** En las pruebas del prototipo apareció un caso
> donde la media del MAE era casi idéntica entre dos modelos mientras la mediana
> difería en más de la mitad y Wilcoxon daba altamente significativo. Unas pocas
> series de gran volumen dominan el promedio. Reportar solo la media lleva a
> conclusiones equivocadas.

**Aceptación:** con predicción perfecta, MAE, RMSE y Bias son cero y MAPE es
cero. Con una serie que tiene meses en cero, MAPE no devuelve infinito ni NaN
silencioso: devuelve el valor sobre los meses válidos y el conteo de excluidos.

### RF-8 — Contraste estadístico (`src/pruebas.py`)

- Shapiro–Wilk sobre las diferencias de error, para justificar el uso de pruebas
  no paramétricas.
- Wilcoxon de rangos con signo, pareado y unilateral, para cada comparación
  contra el promedio móvil y contra Naïve.
- Friedman sobre los cuatro modelos, con post hoc de Nemenyi.
- Tamaño del efecto `r = Z / raíz(N)` en cada comparación.
- Porcentaje de series en que cada modelo gana.
- Tasa de acierto del motor: cuántas veces eligió en validación el modelo que
  efectivamente resultó mejor en prueba.

Referencia metodológica: Demšar (2006) recomienda exactamente este esquema —
Wilcoxon para dos modelos, Friedman con post hoc para varios.

**Aceptación:** con dos vectores idénticos, ninguna prueba reporta significancia.

### RF-9 — Figuras (`src/figuras.py`)

Regenera las figuras del documento: barras del error compuesto por modelo y
dispersión de MAE contra Bias por serie. Salida en PNG a 200 dpi o superior,
escala de grises legible en impresión, sin dependencia de fuentes del sistema.

### RF-10 — Reporte (`src/reporte.py`)

Consolida las salidas listadas en la sección 6 y escribe el `manifiesto.json` de
la RN-6.

---

## 5. Estructura del repositorio

```
motor-pronostico-novapack/
├── REQUERIMIENTOS.md
├── README.md
├── requirements.txt            # pipeline (sin dependencias de red)
├── requirements-extraccion.txt # solo el extractor
├── Dockerfile                  # entorno reproducible con versiones fijadas
├── docker-compose.yml
├── .gitignore                  # datos/crudo/ NUNCA se versiona
├── main.py                     # CLI: inspeccionar | ejecutar
├── config/
│   ├── config.yaml
│   ├── extraccion.ejemplo.yaml # plantilla del esquema de origen
│   └── extraccion.local.yaml   # el esquema REAL — NUNCA se versiona
├── datos/
│   ├── crudo/                  # inmutable
│   └── procesado/
├── scripts/
│   └── extraer_snapshot.py     # habla con la BD UNA vez y congela el CSV
├── src/
│   ├── config.py
│   ├── carga.py
│   ├── inspeccion.py
│   ├── series.py
│   ├── particion.py            # partición + criterios de inclusión
│   ├── features.py
│   ├── modelos/
│   │   ├── base.py
│   │   ├── naive.py
│   │   ├── promedio_movil.py
│   │   ├── suavizado.py        # suavizado exponencial y Holt-Winters
│   │   ├── croston.py
│   │   ├── lightgbm_modelo.py
│   │   └── motor.py
│   ├── metricas.py
│   ├── pruebas.py
│   ├── figuras.py
│   └── reporte.py
├── tests/
├── notebooks/
└── salidas/
```

Tres desviaciones respecto del boceto inicial, todas deliberadas:

- `suavizado.py` y `croston.py` aparecen al incorporar los clásicos estadísticos.
- Los criterios de inclusión viven en `particion.py` porque solo tienen sentido
  definidos respecto del bloque de entrenamiento; separarlos invitaría a medirlos
  sobre la historia completa, que es la forma silenciosa de contaminar la muestra.
- `scripts/extraer_snapshot.py` separa la extracción —que habla con la base de
  datos, una sola vez— del pipeline, que no lo hace nunca (§9).

**Sobre los datos y el control de versiones.** El histórico de NOVAPACK está
cubierto por un acuerdo de confidencialidad (Anexo B de la tesis). `datos/crudo/`
va en `.gitignore` desde el primer commit.

El repositorio **se publica**: el Anexo G enlaza el código. Eso extiende la
obligación más allá de los datos, al esquema interno del sistema de origen
—nombres de base, de esquema, de tabla y de columnas—, que por eso vive en
`config/extraccion.local.yaml`, fuera del control de versiones.

«Revisar antes de publicar» no es un control: basta un descuido una vez. La
revisión está automatizada en `tests/test_confidencialidad.py`, que inspecciona
**los archivos que git rastrea** —exactamente los que se publicarían— buscando
identificadores de la empresa, credenciales, direcciones y etiquetas reales de
regional, y hace fallar la suite si encuentra algo. Los falsos positivos se
declaran uno por uno con su motivo escrito.

---

## 6. Salidas requeridas

Cada una alimenta un punto concreto del documento de tesis:

| Archivo | Destino en la tesis |
|---|---|
| `inspeccion_datos.txt` | Criterios de inclusión y exclusión; N de la muestra |
| `errores_por_serie.csv` | Insumo de las pruebas; se adjunta como anexo |
| `tabla8_resultados.csv` | Tabla 8 — MAE, MAPE, RMSE, Bias por modelo |
| `pruebas_estadisticas.txt` | Apartado «Contraste estadístico de la hipótesis» |
| `seleccion_motor.csv` | Evidencia de la selección por serie |
| `figura2_error.png` | Figura 2 |
| `figura3_dispersion.png` | Figura 3 |
| `manifiesto.json` | Trazabilidad y reproducibilidad |

---

## 7. Plan de trabajo sugerido

Cada fase termina con pruebas en verde y un commit.

1. **Andamiaje**: estructura, configuración, dependencias, `.gitignore`.
2. **Carga e inspección**. Ejecutar contra los datos reales y **detenerse a leer
   el informe** antes de seguir. Aquí se calibran los criterios de inclusión.
3. **Series y partición**, con la prueba de orden temporal.
4. **Features**, escribiendo primero la prueba de no-fuga (RF-5).
5. **Modelos base** (Naïve y promedio móvil): rápidos y verificables a mano.
6. **LightGBM**.
7. **Motor de selección**.
8. **Métricas**.
9. **Pruebas estadísticas**.
10. **Figuras y reporte**.

El orden importa: cada fase se valida con datos reales antes de construir la
siguiente. Un error en la partición descubierto en la fase 9 obliga a rehacer
todo lo posterior.

---

## 8. Entorno

Versiones verificadas como funcionales:

```
pandas 3.0.2
numpy 2.4.4
scipy 1.17.1
scikit-learn 1.8.0
matplotlib 3.10.8
lightgbm 4.7.0
scikit-posthocs 0.14.0
openpyxl 3.1.5
pytest
pyyaml
```

Fijar versiones en `requirements.txt`. Un cambio de versión de LightGBM puede
alterar los resultados; si ocurre después de haber escrito los números en la
tesis, hay que volver a ejecutar todo y actualizar el documento.

---

## 9. Convenciones

- Código y comentarios en español, coherente con el documento académico.
- Nombres de funciones y variables descriptivos: `calcular_mape_sin_ceros`, no
  `calc_m`.
- Cada función que implementa una decisión metodológica lleva en su docstring la
  justificación y, cuando corresponda, la referencia bibliográfica.
- Sin efectos secundarios ocultos: las funciones no escriben archivos salvo las
  de la capa de reporte.
- `pytest` para las pruebas. Sin dependencias de red en tiempo de ejecución.

---

## 10. Qué no hacer

- Ajustar parámetros mirando el conjunto de prueba.
- Eliminar series porque «empeoran el resultado». Los criterios de exclusión se
  definen por razones de calidad de datos, se fijan antes de ver los resultados y
  se documentan.
- Sustituir el promedio móvil por una versión más débil que la que usa la empresa
  para agrandar la mejora. El benchmark debe ser el método real.
- Reportar solo la métrica que favorece la conclusión.
- Rellenar un valor faltante con una estimación plausible.
- Silenciar advertencias sin entender qué las produce.

---

## 11. Decisiones cerradas (2026-08-06)

Las decisiones que estaban abiertas quedaron resueltas con el autor. Se dejan
registradas aquí porque cada una condiciona los números y hay que poder
justificarlas ante el tribunal.

| Decisión | Resolución | Motivo |
|---|---|---|
| **Fuente de datos** | Extracción congelada desde la base corporativa, con SHA-256, a `datos/crudo/` | Reproducibilidad: la base se recarga a diario y leerla en vivo haría imposible la RN-5 |
| **Variable objetivo** | Ventas efectivas mensuales | «Demanda total» incluye venta perdida, que no se registró antes de 2020: la variable cambiaría de definición a mitad de la serie |
| **Ventana temporal** | abril-2017 → marzo-2026 · 9 **gestiones fiscales** completas (2018–2026) | La empresa cierra en marzo. Cortar por año calendario partiría cada campaña escolar entre dos bloques |
| **Cortes** | Train gestiones 2018–2024 (84 m) · Validación 2025 (12 m) · Prueba 2026 (12 m) | Máxima historia de entrenamiento respetando el mínimo de 12 + 12 |
| **Modelos de la empresa** | `naive_m1`, `naive_m1_gf`, `ma_2`, `ma_2_gf`, `ma_12`, `ma_12_gf` | Son los que Planificación aplica hoy; §10 exige el método real como benchmark |
| **Clásicos añadidos** | `exp_smooth_opt`, `holt_winters`, `croston` | Recomendación de la dirección del trabajo |
| **Benchmark declarado** | `ma_12` frente a `naive_m1` | Fijado ANTES de ver resultados (RF-8) |
| **Devoluciones** | Se descartan y se cuentan | Parametrizable en `config.yaml`; el conteo va al informe |
| **Criterios de inclusión** | Umbrales de partida del §3, a recalibrar tras leer el informe de inspección | Se miden **solo sobre entrenamiento** |
| **Período declarado en la tesis** | Se amplía de 2017–2024 a 2017–2026 | El histórico llega más lejos; hay que actualizar la delimitación temporal del documento |

### Pendientes de corregir en el documento

Detectados al construir el análisis; no son decisiones de código sino del texto:

1. **La Tabla 8 actual no lleva fila para el motor.** Compara Naïve, promedio
   móvil y LightGBM, pero el objeto de la tesis es el motor de selección: tiene
   que aparecer como brazo con su propio resultado.
2. **«MAE Promedio (%)» no es un MAE.** El MAE está en unidades; un error
   promedio expresado en porcentaje es un MAPE. Y el «error compuesto
   (MAE + |Bias|)» suma una magnitud en unidades con otra en porcentaje, que no
   es una operación válida. El programa emite las dos versiones dimensionalmente
   coherentes —`error_compuesto_pct` = MAPE + |Bias| y `error_compuesto_unidades`
   = MAE + |Δmedias|— y el documento debe declarar cuál usa.
3. **El texto dice cuatro canales** (mayorista, minorista, distribuidor,
   institucional) y **cinco regionales**. El maestro de datos tiene dos canales.
   Hay que reconciliar el texto con la fuente antes de la defensa.
4. **«Los datos fueron depurados para eliminar valores atípicos»** (§3.2.1 del
   documento). No se eliminan valores atípicos: eliminarlos sin un criterio
   fijado de antemano es exactamente lo que prohíbe el §10 de este requerimiento.
   Hay que reescribir esa frase para que describa lo que el código hace.

---

## 12. Nota final sobre los resultados

Es posible que el motor gane por menos margen del esperado, o que en parte de las
series no gane. **Ese resultado también es válido y se reporta tal cual.** Una
mejora modesta, contrastada estadísticamente y bien explicada, es más defendible
ante un tribunal que una mejora espectacular que no resiste preguntas sobre cómo
se obtuvo.

Si algún contraste no resulta significativo, lo que cambia es la redacción de las
conclusiones, no los números.
