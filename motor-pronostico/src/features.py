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
    y la RN-5 dejaría de cumplirse. Los grupos v2 entran al final, cada uno con
    su orden interno fijo; con su flag apagado (o su fuente ausente) las
    columnas no existen (o son NaN), y el contrato no cambia de forma.
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

    v2 = cfg.features.v2
    if v2.get("derivadas"):
        nombres += ["meses_desde_venta", "prop_ceros_movil_12",
                    "media_mismo_mes", "razon_interanual", "momentum_3_12",
                    "min_movil_12", "max_movil_12"]
    if v2.get("precio"):
        nombres += ["precio_rezago_1", "precio_delta_pct"]
    if v2.get("perdidas"):
        nombres += ["perdidas_rezago_1", "perdidas_movil_12"]
    if v2.get("quiebres"):
        nombres += ["quiebre_pct_rezago_1", "quiebre_pct_movil_3"]
    if v2.get("tipo_cambio"):
        nombres += ["tc_rezago_1", "tc_delta_pct"]
    if v2.get("categoria"):
        nombres += ["categoria"]
    # V6: calendario comercial declarado — al final, para no mover el orden
    # de las columnas existentes entre generaciones (RN-5).
    if cfg.features.calendario:
        nombres += ["temporada_alta", "meses_a_clases"]
    return nombres


def categoricas_de(cfg: Config) -> list[str]:
    """Las columnas categóricas que consume LightGBM, v2 incluida."""
    categoricas = list(cfg.features.categoricas)
    if cfg.features.v2.get("categoria"):
        categoricas.append("categoria")
    return categoricas


def construir_features(
    panel: pd.DataFrame, cfg: Config, exogenas=None
) -> pd.DataFrame:
    """Panel largo -> tabla de entrenamiento con features y objetivo.

    Args:
        panel: columnas ``serie``, ``sku``, ``canal``, ``regional``,
            ``periodo`` (``pd.Period`` mensual) e ``y``.
        exogenas: el paquete de :func:`src.exogenas.cargar_exogenas`. Con
            ``None`` (o con una fuente ausente), las features exógenas salen
            como columnas NaN: «sin dato» declarado, jamás un cero inventado —
            y el contrato de columnas no cambia de forma.

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

    # --- Features v2: derivadas causales del propio panel -------------------
    # Cada una usa EXCLUSIVAMENTE información anterior al mes de su fila (los
    # shift van antes que cualquier ventana); la prueba de no-fuga las barre
    # junto con todas las demás.
    v2 = cfg.features.v2
    por_serie = tabla.groupby("serie", observed=True, sort=False)

    if v2.get("derivadas"):
        # Recencia: meses transcurridos desde la última venta positiva ANTERIOR
        # al mes de la fila (NaN antes de la primera venta).
        indice = por_serie.cumcount()
        marcador = indice.where(tabla["y"] > 0)
        ultimo_previo = marcador.groupby(
            tabla["serie"], observed=True
        ).transform(lambda s: s.shift(1).ffill())
        tabla["meses_desde_venta"] = indice - ultimo_previo

        ceros = (tabla["y"] <= 0).astype(float)
        tabla["prop_ceros_movil_12"] = ceros.groupby(
            tabla["serie"], observed=True
        ).transform(lambda s: s.shift(1).rolling(12, min_periods=12).mean())

        # Media histórica del MISMO mes calendario (expandiente, sin el año en
        # curso): la forma estacional propia de la serie, sin mirar el presente.
        mes_calendario = pd.PeriodIndex(tabla["periodo"]).month
        tabla["media_mismo_mes"] = tabla.groupby(
            [tabla["serie"], mes_calendario], observed=True
        )["y"].transform(lambda s: s.shift(1).expanding().mean())

        rezago_1 = agrupado.shift(1)
        rezago_13 = agrupado.shift(13)
        tabla["razon_interanual"] = (rezago_1 + 1.0) / (rezago_13 + 1.0)

        mm3 = por_serie_desplazada.transform(
            lambda s: s.rolling(3, min_periods=3).mean()
        )
        mm12 = por_serie_desplazada.transform(
            lambda s: s.rolling(12, min_periods=12).mean()
        )
        tabla["momentum_3_12"] = np.where(mm12 > 0, mm3 / mm12, np.nan)
        tabla["min_movil_12"] = por_serie_desplazada.transform(
            lambda s: s.rolling(12, min_periods=12).min()
        )
        tabla["max_movil_12"] = por_serie_desplazada.transform(
            lambda s: s.rolling(12, min_periods=12).max()
        )

    # --- Features v2: exógenas (rezagadas SIEMPRE) --------------------------
    def _alinear(frame, claves, columna) -> pd.Series:
        """Trae la columna exógena alineada a la tabla (NaN donde no hay)."""
        if frame is None:
            return pd.Series(np.nan, index=tabla.index)
        union = tabla[claves].merge(
            frame[[*claves, columna]], on=claves, how="left"
        )
        return union[columna].set_axis(tabla.index)

    def _rezagar(valores: pd.Series, meses: int = 1) -> pd.Series:
        return valores.groupby(tabla["serie"], observed=True).shift(meses)

    if v2.get("precio"):
        precio = _alinear(getattr(exogenas, "precio", None),
                          ["serie", "periodo"], "precio")
        p1, p2 = _rezagar(precio, 1), _rezagar(precio, 2)
        tabla["precio_rezago_1"] = p1
        tabla["precio_delta_pct"] = np.where(p2 > 0, (p1 - p2) / p2, np.nan)

    if v2.get("perdidas"):
        perdidas = _alinear(getattr(exogenas, "perdidas", None),
                            ["serie", "periodo"], "perdidas")
        tabla["perdidas_rezago_1"] = _rezagar(perdidas, 1)
        tabla["perdidas_movil_12"] = perdidas.groupby(
            tabla["serie"], observed=True
        ).transform(lambda s: s.shift(1).rolling(12, min_periods=12).sum())

    if v2.get("quiebres"):
        quiebres = getattr(exogenas, "quiebres", None)
        if quiebres is not None:
            quiebres = quiebres.assign(
                quiebre_pct=lambda q: np.where(
                    q["semanas_medidas"] > 0,
                    q["semanas_en_cero"] / q["semanas_medidas"], np.nan,
                )
            )
        pct = _alinear(quiebres, ["sku", "regional", "periodo"], "quiebre_pct")
        tabla["quiebre_pct_rezago_1"] = _rezagar(pct, 1)
        tabla["quiebre_pct_movil_3"] = pct.groupby(
            tabla["serie"], observed=True
        ).transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())

    if v2.get("tipo_cambio"):
        tc = getattr(exogenas, "tipo_cambio", None)
        if tc is None:
            tabla["tc_rezago_1"] = np.nan
            tabla["tc_delta_pct"] = np.nan
        else:
            # El TC es global por mes: el rezago se resuelve por calendario,
            # sin agrupar por serie.
            mapa = tc.to_dict()
            t1 = pd.Series(
                [mapa.get(p - 1, np.nan) for p in tabla["periodo"]],
                index=tabla.index, dtype=float,
            )
            t2 = pd.Series(
                [mapa.get(p - 2, np.nan) for p in tabla["periodo"]],
                index=tabla.index, dtype=float,
            )
            tabla["tc_rezago_1"] = t1
            tabla["tc_delta_pct"] = np.where(t2 > 0, (t1 - t2) / t2, np.nan)

    if v2.get("categoria"):
        mapa_categoria = getattr(exogenas, "categoria_por_serie", None)
        if mapa_categoria is None:
            tabla["categoria"] = pd.Series(
                pd.array([None] * len(tabla), dtype="string")
            ).astype("category")
        else:
            tabla["categoria"] = (
                tabla["serie"].map(mapa_categoria).astype("category")
            )

    # --- Features V6: calendario comercial declarado ------------------------
    # Deterministas del calendario y de la categoría: no usan NINGUNA
    # observación (RN-3 trivial) y son conocidas a cualquier horizonte — en el
    # multihorizonte se computan para los meses futuros sin persistencia ni
    # supuesto alguno. El calendario es una decisión de DOMINIO declarada en
    # la configuración (informante del área) y verificada contra el bloque de
    # entrenamiento, no minada de los datos.
    calendario = cfg.features.calendario or {}
    if calendario:
        mes_calendario_v6 = pd.Series(
            pd.PeriodIndex(tabla["periodo"]).month, index=tabla.index
        )
        inicio_clases = int(calendario.get("inicio_clases_mes", 2))
        # Cuenta regresiva circular al inicio de clases: 0 en el mes de inicio,
        # 1 el mes anterior, ... 11 el mes siguiente.
        tabla["meses_a_clases"] = (
            (inicio_clases - mes_calendario_v6) % 12
        ).astype(float)

        temporadas = calendario.get("temporadas", {}) or {}
        mapa_categoria = getattr(exogenas, "categoria_por_serie", None)
        if mapa_categoria is None:
            # Sin el maestro de categorías no se puede saber la temporada:
            # «sin dato» declarado, nunca un cero inventado.
            tabla["temporada_alta"] = np.nan
        else:
            categoria_serie = tabla["serie"].map(mapa_categoria)
            temporada = pd.Series(0.0, index=tabla.index)
            for categoria_t, meses_t in temporadas.items():
                meses_set = {int(m) for m in meses_t}
                temporada[
                    (categoria_serie == categoria_t)
                    & mes_calendario_v6.isin(meses_set)
                ] = 1.0
            temporada[categoria_serie.isna()] = np.nan
            tabla["temporada_alta"] = temporada

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
