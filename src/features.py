"""RF-5 — Construcción de variables derivadas, sin fuga temporal.

Regla única que gobierna este módulo (RN-3): **toda feature de un mes se
construye únicamente con información anterior a ese mes**. En la práctica eso
significa que cada estadístico móvil se calcula sobre la serie ya desplazada un
período, y que no existe ningún estadístico ajustado sobre el conjunto completo
(medias globales, normalizaciones, codificación de destino).

Consecuencia deliberada: los primeros meses de cada serie tienen features en
``NaN``. No se imputan (§10: «Rellenar un valor faltante con una estimación
plausible» está prohibido). LightGBM trata el ``NaN`` como una categoría de
partición propia, que es el comportamiento correcto: «no hay historia
suficiente» es información, no un dato que falte.

La prueba que garantiza todo esto vive en ``tests/test_features.py`` y se
escribió antes que este archivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

# Columnas que identifican la fila; no son features ni objetivo.
COLUMNAS_CONTEXTO = ("serie", "sku", "canal", "regional", "periodo", "bloque")


def nombres_de_features(cfg: Config) -> list[str]:
    """Lista ordenada y estable de las columnas de entrada del modelo.

    El orden es fijo y no depende de la iteración de ningún diccionario: sin
    esto, LightGBM podría recibir las columnas en orden distinto entre corridas
    y la RN-5 dejaría de cumplirse.
    """
    nombres: list[str] = []
    nombres += [f"rezago_{k}" for k in cfg.features.rezagos]
    nombres += [f"media_movil_{w}" for w in cfg.features.ventanas_moviles]
    nombres += [f"desv_movil_{w}" for w in cfg.features.ventanas_moviles]
    if cfg.features.incluir_mes:
        nombres += ["mes"]
    if cfg.features.incluir_anio:
        nombres += ["anio"]
    if cfg.features.incluir_tendencia:
        nombres += ["tendencia"]
    nombres += list(cfg.features.categoricas)
    return nombres


def construir_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Panel largo -> tabla de entrenamiento con features y objetivo.

    Args:
        panel: columnas ``serie``, ``sku``, ``canal``, ``regional``,
            ``periodo`` (``pd.Period`` mensual) e ``y``.

    Returns:
        Una fila por (serie, mes) con las columnas de contexto, el objetivo
        ``y`` y las features listadas por :func:`nombres_de_features`.
    """
    faltantes = {"serie", "periodo", "y"} - set(panel.columns)
    if faltantes:
        raise ValueError(f"El panel no tiene las columnas {sorted(faltantes)}.")

    tabla = panel.sort_values(["serie", "periodo"], kind="stable").reset_index(drop=True)
    agrupado = tabla.groupby("serie", observed=True, sort=False)["y"]

    # --- Rezagos ------------------------------------------------------------
    for k in cfg.features.rezagos:
        tabla[f"rezago_{k}"] = agrupado.shift(k)

    # --- Estadísticos móviles SOBRE LA SERIE YA DESPLAZADA ------------------
    # `shift(1)` primero y `rolling` después. Al revés —rolling y luego shift—
    # daría el mismo número aquí, pero el orden elegido hace la propiedad
    # evidente al leer: la ventana nunca toca el mes que se predice.
    desplazada = agrupado.shift(1)
    por_serie_desplazada = desplazada.groupby(tabla["serie"], observed=True, sort=False)
    for w in cfg.features.ventanas_moviles:
        tabla[f"media_movil_{w}"] = por_serie_desplazada.transform(
            lambda s, w=w: s.rolling(window=w, min_periods=w).mean()
        )
        # ddof=1: desviación muestral, coherente con la práctica habitual en
        # series de demanda y con el CV² de la clasificación Syntetos–Boylan.
        tabla[f"desv_movil_{w}"] = por_serie_desplazada.transform(
            lambda s, w=w: s.rolling(window=w, min_periods=w).std(ddof=1)
        )

    # --- Calendario ---------------------------------------------------------
    periodos = pd.PeriodIndex(tabla["periodo"])
    if cfg.features.incluir_mes:
        # Mes como entero 1..12, tal como pide la RF-5. No se añade codificación
        # cíclica seno/coseno: un árbol puede partir el eje 1..12 por cualquier
        # punto y recuperar la estacionalidad sin ella, de modo que solo sumaría
        # dos columnas y una decisión más que justificar.
        tabla["mes"] = periodos.month.to_numpy()
    if cfg.features.incluir_anio:
        tabla["anio"] = periodos.year.to_numpy()
    if cfg.features.incluir_tendencia:
        # Antigüedad de la serie en meses, empezando en 0. Depende solo del mes
        # de nacimiento, nunca de los valores observados.
        tabla["tendencia"] = tabla.groupby("serie", observed=True, sort=False).cumcount()

    # --- Categóricas --------------------------------------------------------
    # Se dejan como `category` de pandas: LightGBM las consume nativamente sin
    # necesidad de one-hot ni de codificación de destino. Se evita a propósito
    # cualquier target encoding: ajustarlo sobre el panel completo sería fuga
    # (RN-3) y ajustarlo por bloque agregaría un parámetro más que justificar.
    for columna in cfg.features.categoricas:
        if columna not in tabla.columns:
            raise ValueError(
                f"La categórica '{columna}' no está en el panel. "
                f"Columnas disponibles: {sorted(tabla.columns)}"
            )
        tabla[columna] = tabla[columna].astype("category")

    # Las categóricas son a la vez columnas de contexto y features, así que la
    # lista final se deduplica conservando el orden. Sin esto la tabla saldría
    # con 'sku', 'canal' y 'regional' repetidas y `matriz_de_entrada` fallaría.
    columnas: list[str] = []
    for columna in (*COLUMNAS_CONTEXTO, "y", *nombres_de_features(cfg)):
        if columna in tabla.columns and columna not in columnas:
            columnas.append(columna)
    return tabla[columnas]


def matriz_de_entrada(tabla: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sub-tabla con solo las columnas que entran al modelo, en orden fijo."""
    return tabla[nombres_de_features(cfg)]
