# El sistema de pronóstico en producción de NOVAPACK — hallazgos verificados en su código

**Para qué existe este documento.** Cuando toque redactar la memoria, estas son
las justificaciones que dan cuerpo a las decisiones del experimento. Cada
hallazgo se verificó **leyendo el código fuente** del sistema de pronóstico que
la empresa opera hoy (revisión interna, agosto 2026), no de oídas. Se registra
acá, anonimizado, para tenerlo a mano; ninguna referencia identifica a la
empresa real ni a sus sistemas.

Regla de alcance acordada: los planes de evolución del sistema en producción
**no entran en el TFG documentado**. Este archivo solo registra lo que el
sistema ES hoy, que es lo que el experimento necesita citar.

---

## 1. El criterio de selección vigente es MAE + |Bias|

El motor de selección del sistema en producción elige modelos con la métrica
`combined`, definida en su código como **MAE + |Bias|** — exactamente la regla
compuesta que este experimento usa para el motor (`motor_regla:
mae_mas_bias`) y el fundamento de la métrica de decisión **D = WMAPE_val +
|Bias_val|**.

**Dónde se usa en la memoria:** la elección de D no es una preferencia del
investigador; es el criterio de selección **documentado y operativo** del área
de Planificación, llevado a su forma valorizada para que la agregación entre
SKUs sea dimensionalmente válida (§ métrica de decisión). El experimento no
inventa la vara: la hereda y la corrige donde no era agregable.

## 2. El backtest vigente ya es multihorizonte — no a un paso

Para seleccionar modelos, el sistema en producción trunca la historia 12 meses
antes del final, ajusta una sola vez en ese punto, proyecta esos 12 meses **sin
re-engancharse a la realidad** y compara contra lo observado. Es decir: su
ventana de selección evalúa horizontes h=1…12 desde un origen fijo.

**Dónde se usa:** justifica la etapa `evaluacion_multihorizonte` del pipeline.
El protocolo a un paso del experimento (RN-4) es *más benévolo* con los modelos
de proyección plana que el criterio de la propia empresa; la evaluación
multihorizonte cierra esa brecha y se reporta por separado, como exige la RN-4.

## 3. La operación es rolling mensual con horizonte de 18 meses

El sistema corre **cada mes** y proyecta **18 meses** hacia adelante (horizonte
por defecto de la corrida mensual). Cada corrida re-entrena y re-proyecta todo.

**Dónde se usa:** define el diseño de orígenes rodantes de la etapa
multihorizonte (cada mes de prueba es un origen) y el re-entrenamiento por
origen del brazo global. El objetivo operativo de 18 meses se declara; el
histórico disponible permite medir hasta h=12 (el bloque de prueba cierra el
período), y la curva de degradación D(h) es la evidencia de qué modelos
sostienen el horizonte largo.

## 4. La proyección multihorizonte es recursiva

El promedio móvil del sistema en producción se **realimenta con sus propios
pronósticos**: el pronóstico del mes h entra a la ventana móvil del mes h+1
(por eso converge asintóticamente). El brazo de aprendizaje automático hace lo
mismo: predice mes a mes y anexa cada predicción a la historia para construir
los rezagos del mes siguiente.

**Dónde se usa:** `src/multihorizonte.py` replica exactamente esa mecánica
(proyección recursiva realimentada, truncada en cero antes de realimentar).
Las variantes con factor de crecimiento suman el GF sobre la historia
extendida, igual que en producción. Croston queda constante (sus estados solo
se actualizan con demanda observada) y las pruebas verifican que la recursión
reproduce las formas cerradas del suavizado exponencial (línea plana) y de
Holt-Winters (ℓ + h·b + s).

## 5. El modelo global vigente: features y dos defectos que no se heredan

El brazo de aprendizaje automático en producción es un **modelo global**
(uno solo para todas las series, que predice por serie) con estas features:
rezagos 1/2/3/6/12, medias móviles de 3 y 12, desviación de 12, **precio
promedio**, **quiebres de stock** (semanas con inventario en cero, agregadas
del inventario semanal), **tipo de cambio** (nivel y variación), calendario
(mes, trimestre) y categóricas (categoría, regional, canal).

Dos defectos verificados en su código, que el experimento evita a propósito:

- **Desalineación entrenamiento/proyección (train/serve skew).** Al proyectar
  el futuro, el sistema fija precio = 0, quiebres = 0 y congela el tipo de
  cambio en el último valor: entrena con precios reales y predice con ceros.
  El experimento usa solo features calculables honestamente en el futuro
  (rezagos, móviles, calendario, categóricas); si alguna vez se incorporan las
  demás, deberán proyectarse sin ese salto.
- **Asimetría de protocolo.** El brazo de aprendizaje automático lleva la marca
  `skip_backtest`: NO compite en el backtest de selección en igualdad de
  condiciones con los demás modelos (usa su validación interna). En el
  experimento, TODOS los brazos se evalúan bajo el mismo protocolo (RN-4) —
  es una corrección metodológica declarable como aporte.

**Dónde se usa:** (a) fundamenta que el LightGBM del experimento es un **piso**
declarado — las features actuales son la base mínima, y el techo conocido del
sistema vigente ya incluye señales de precio, quiebre y tipo de cambio; (b) da
la lista concreta de features candidatas para el trabajo futuro (dentro del
marco de ablaciones del pipeline, `--anular`).

## 6. Cómo trata el sistema vigente a las series difíciles

El sistema no recorta la muestra: **clasifica** cada serie por estado
(activa / producto nuevo / reciente / dormida / descontinuada / insuficiente)
y por régimen de demanda con la matriz ADI–CV² de Syntetos–Boylan (umbral
ADI 1,32 y CV² 0,49 — los mismos del informe de inspección del experimento), y
asigna a cada clase los modelos elegibles y un modelo de respaldo.

Además, su cálculo de MAE **excluye los pares (real=0, pronóstico=0)** para no
premiar aciertos triviales en series intermitentes, y marca como descontinuada
una serie con 12 meses sin venta.

**Dónde se usa:** (a) molde para la decisión de muestra del experimento
(clasificar y declarar, no descartar en silencio — incluye el destino de las
181 series apagadas por el confinamiento); (b) la exclusión de pares 0-0 es una
**divergencia declarada**: el experimento evalúa todos los meses del bloque; si
se adoptara la exclusión, sería por ablación, nunca por defecto silencioso.

## 7. Global vs por serie — precisión para la memoria

En el experimento (igual que en producción) **todo se evalúa por serie**
SKU–canal–regional. La diferencia entre brazos está solo en el *ajuste*: los
estadísticos ajustan un modelo por serie (cada una con sus parámetros); el
brazo de aprendizaje automático ajusta **un** modelo global que aprende de
todas las series a la vez y **predice por serie**. Ningún número global se
compara jamás contra un número por serie: los errores se calculan serie por
serie y se agregan valorizados.

---

*Los nombres de modelos citados (ma_2, ma_12, variantes _gf, etc.) son los del
sistema en producción y coinciden con los brazos del experimento. El sistema
vigente incluye además variantes con ajuste estacional (p. ej. ma_24_sa) y
modelos compuestos que NO forman parte de los seis métodos declarados por
Planificación como los de uso corriente; el experimento usa exactamente esos
seis más los clásicos recomendados por la dirección del trabajo.*
