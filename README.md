# Motor de pronóstico de demanda — NOVAPACK S.A.

Análisis cuantitativo del Trabajo Final de Grado *«Diseño de un motor de
pronóstico de demanda basado en series temporales para mejorar la precisión de
la planificación de la demanda en NOVAPACK S.A. para la gestión 2026»*
— UAGRM School of Engineering, Maestría en Ciencia de Datos e Inteligencia
Artificial.

La especificación completa está en [`REQUERIMIENTOS.md`](REQUERIMIENTOS.md). Este
archivo explica cómo correrlo y qué produce.

---

## Qué hace

Compara **once modelos** de pronóstico de demanda mensual sobre cada combinación
SKU–canal–regional, y contrasta estadísticamente si el motor de selección
automática mejora la precisión frente a los métodos que la empresa usa hoy.

| Brazo | Papel |
|---|---|
| `naive_m1`, `naive_m1_gf` | Línea base estándar y método actual |
| `ma_2`, `ma_2_gf`, `ma_12`, `ma_12_gf` | Promedio móvil — método actual |
| `exp_smooth_opt` | Clásico: suavizado exponencial simple |
| `holt_winters` | Clásico: nivel + tendencia + estacionalidad |
| `croston` | Clásico para demanda intermitente |
| `lightgbm` | Aprendizaje automático, modelo global |
| `motor` | **La propuesta**: elige por serie el mejor en validación |

Los seis primeros no son una elección del autor: son los modelos que
Planificación aplica hoy, verificados contra el sistema en producción. §10 del
requerimiento lo exige — el benchmark tiene que ser el método real.

---

## Diseño experimental

**Período.** Nueve gestiones fiscales completas: **abril-2017 → marzo-2026**
(gestiones 2018 a 2026). NOVAPACK cierra en marzo; una gestión se nombra por el
año en que cierra.

**Variable objetivo.** Ventas efectivas mensuales. *No* «demanda total»: la venta
perdida no se registró antes de 2020, de modo que esa variable cambia de
definición a mitad de la serie.

**Partición (RN-2).** Los cortes caen en frontera de gestión, no de año
calendario, para que ningún bloque parta una campaña escolar por la mitad:

| Bloque | Gestiones | Meses | Para qué |
|---|---|---|---|
| Entrenamiento | 2018–2024 | abr-2017 … mar-2024 (84) | ajusta los modelos |
| Validación | 2025 | abr-2024 … mar-2025 (12) | el motor decide |
| Prueba | 2026 | abr-2025 … mar-2026 (12) | se reporta el desempeño |

**Evaluación (RN-4).** Un mes hacia adelante, usando el valor real del mes
anterior. Es la única comparación justa: el promedio móvil y el Naïve funcionan
así por construcción.

---

## Cómo correrlo

La máquina de desarrollo no necesita Python: todo corre en Docker con las
versiones fijadas.

```bash
docker compose build
```

### 1 · Congelar el snapshot de datos (una sola vez)

El esquema del sistema de origen —nombres de base, esquema, tabla y columnas— no
está en el repositorio: es información de la empresa, cubierta por el mismo
acuerdo de confidencialidad que los datos. Se declara en un archivo local que no
se versiona.

```bash
cp config/extraccion.ejemplo.yaml config/extraccion.local.yaml
# completar config/extraccion.local.yaml con el esquema real

export TFG_DB_HOST=... TFG_DB_PORT=... 
export TFG_DB_NAME=... TFG_DB_USER=... TFG_DB_PASSWORD=...
docker compose run --rm tfg python scripts/extraer_snapshot.py
```

Produce `datos/crudo/ventas_novapack.csv` + `manifiesto_extraccion.json` con el
SHA-256. Los códigos de SKU se seudonimizan por defecto (`SKU-0001`, …) porque
`errores_por_serie.csv` se adjunta como anexo del documento; la tabla de
correspondencia queda en `datos/crudo/mapeo_sku.csv`, que no se versiona.

A partir de ese momento el pipeline **no vuelve a consultar la base de datos**.

### 2 · Inspeccionar y calibrar los criterios de inclusión

```bash
docker compose run --rm tfg python main.py inspeccionar
```

Escribe `salidas/inspeccion_datos.txt`. **Detenerse a leerlo** antes de seguir:
ahí se deciden los umbrales de `inclusion` en `config/config.yaml`, y el N
resultante es el tamaño de la muestra que se declara en la tesis.

### 3 · Ejecutar el experimento

```bash
docker compose run --rm tfg python main.py ejecutar
```

### 4 · Ablaciones — cambiar una decisión y volver a medir

`--anular` cambia una clave de la configuración para una corrida, sin duplicar el
archivo. La anulación queda registrada en el manifiesto, así que dos corridas con
ablaciones distintas nunca pueden confundirse (sus hashes de configuración
difieren). Solo se pueden anular claves **existentes**: una errata falla en vez de
correr en silencio con el valor viejo.

```bash
# ¿Qué pasa si el motor elige con la regla nativa del sistema en producción?
docker compose run --rm tfg python main.py ejecutar \
  --anular modelos.motor_regla=mae_mas_bias \
  --anular salidas.directorio=salidas_ablacion_mae_mas_bias

# ¿Y si se exige más historia para entrar en la muestra?
docker compose run --rm tfg python main.py ejecutar \
  --anular inclusion.historial_minimo_meses=48 \
  --anular salidas.directorio=salidas_historial_48
```

### Pruebas

```bash
docker compose run --rm tfg python -m pytest -q
```

---

## Salidas

Todo va a `salidas/`. Cada archivo alimenta un punto concreto del documento:

| Archivo | Destino en la tesis |
|---|---|
| `inspeccion_datos.txt` | Criterios de inclusión y exclusión; N de la muestra |
| `cohorte_flujo.csv` | Diagrama de flujo de la muestra, criterio por criterio |
| `errores_por_serie.csv` | Insumo de las pruebas; se adjunta como anexo |
| `tabla8_resultados.csv` | **Tabla 8** — MAE, MAPE, RMSE, Bias por modelo |
| `resumen_metricas.csv` | Tabla 8 ampliada, con MASE y medianas |
| `pruebas_estadisticas.txt` | Apartado «Contraste estadístico de la hipótesis» |
| `seleccion_motor.csv` | Evidencia de la selección por serie |
| `victorias_por_modelo.csv`, `nemenyi.csv`, `rangos_friedman.csv` | Anexos del contraste |
| `parametros_por_serie.csv` | α de cada serie en los modelos con parámetro |
| `lightgbm_importancias.csv` | Importancia de las variables |
| `figura2_error.png` | **Figura 2** — error compuesto por modelo |
| `figura3_dispersion.png` | **Figura 3** — MAE contra Bias por serie |
| `figura4_diferencia_critica.png` | Diagrama de diferencia crítica (complementario) |
| `manifiesto.json` | Trazabilidad y reproducibilidad (RN-6) |

---

## Confidencialidad

El histórico de NOVAPACK está cubierto por un acuerdo de confidencialidad
(Anexo B de la tesis), y con él el esquema interno del sistema de origen. Este
repositorio se publica: el Anexo G enlaza el código.

Tres barreras, en orden de fiabilidad decreciente:

1. **`.gitignore`** — `datos/crudo/`, `salidas*/` y `config/extraccion.local.yaml`
   nunca se versionan.
2. **Seudonimización** — los códigos de producto se reemplazan al extraer, porque
   `errores_por_serie.csv` se adjunta como anexo. La correspondencia queda fuera
   del repositorio.
3. **Control automático** — `tests/test_confidencialidad.py` revisa **los
   archivos que git rastrea** buscando nombres de la empresa, de bases, de
   esquemas, direcciones IP, credenciales y etiquetas reales de regional. Falla
   la suite si encuentra algo. Revisar esto a mano antes de cada publicación no
   funciona: basta un descuido una vez.

Si el control marca un falso positivo, se agrega a `EXCEPCIONES` **con el motivo
escrito**. Una excepción sin justificación es una excepción que dentro de seis
meses nadie recuerda por qué está.

---

## Estructura

```
src/
  config.py      configuración validada — todas las decisiones metodológicas
  carga.py       RF-1 · lectura de CSV/Excel y conversión de tipos
  inspeccion.py  RF-2 · informe descriptivo (CRISP-DM: comprensión de datos)
  series.py      RF-3 · panel mensual por SKU–canal–regional
  particion.py   RF-4 · tres bloques + criterios de inclusión
  features.py    RF-5 · variables derivadas, sin fuga temporal
  modelos/       RF-6 · los once brazos
  metricas.py    RF-7 · MAE, RMSE, MAPE, Bias, MASE
  pruebas.py     RF-8 · Shapiro, Wilcoxon, Friedman + Nemenyi
  figuras.py     RF-9 · figuras del documento
  reporte.py     RF-10 · salidas y manifiesto
scripts/
  extraer_snapshot.py   extracción y congelado del histórico
tests/           pytest — único lugar donde se admiten datos sintéticos (RN-1)
```

### Desviaciones respecto de la estructura sugerida en §5

Se documentan aquí para que el lector pueda seguir la correspondencia:

- `src/modelos/suavizado.py` y `src/modelos/croston.py` no estaban en la
  estructura original: aparecen al incorporar los tres modelos estadísticos
  clásicos que pidió la dirección del trabajo.
- Los criterios de inclusión viven en `particion.py` y no en un módulo aparte,
  porque solo tienen sentido definidos respecto del bloque de entrenamiento.
- `scripts/extraer_snapshot.py` y `main.py` se agregan para separar la
  extracción (que habla con la base de datos, una vez) del pipeline (que nunca
  lo hace).
