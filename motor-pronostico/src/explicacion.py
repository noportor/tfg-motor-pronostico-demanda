"""Explicación del modelo global y concentración del valor del portafolio.

Dos análisis pensados para la constatación de la propuesta (y su defensa):

- **Contribuciones TreeSHAP** del modelo global sobre el bloque de prueba,
  agregadas para todo el bloque y por separado en los meses de pico estacional.
  LightGBM implementa TreeSHAP de forma nativa (``pred_contrib=True``), así que
  el análisis no agrega ninguna dependencia al entorno que produce los números.
  Responde con especificidad *qué variables influyen* cuando el modelo anticipa
  el pico de demanda, en lugar de mostrar solo la importancia global.

- **Pareto de la demanda valorizada por SKU**: qué fracción del portafolio
  concentra el valor. Es el fundamento empírico de la ponderación por costo de
  la métrica de decisión D — si el valor está concentrado, promediar errores
  sin ponderar daría el mismo peso a un producto marginal que a uno crítico.

Ambas funciones son puras: reciben datos, devuelven tablas. El pipeline las
publica como CSV y las dibuja en el catálogo de figuras.
"""

from __future__ import annotations

import pandas as pd

#: Un mes es "de pico" si su participación en la demanda valorizada anual del
#: entrenamiento supera en este factor a la participación uniforme (1/12).
#: Con 1,25 la regla aísla la campaña escolar sin capturar meses apenas por
#: encima del promedio. La decisión queda declarada en el manifiesto.
UMBRAL_FACTOR_PICO = 1.25


def meses_pico(
    panel: pd.DataFrame,
    costo_por_serie: pd.Series,
    factor: float = UMBRAL_FACTOR_PICO,
) -> list[int]:
    """Meses calendario de pico según la demanda valorizada del panel dado.

    La regla es la misma participación mensual que dibuja la figura del perfil
    estacional: valor del mes / valor total. Se declara "pico" al mes cuya
    participación supera ``factor / 12``. Si ningún mes supera el umbral (una
    demanda plana), devuelve el mes de mayor participación para que el análisis
    dirigido nunca quede vacío.
    """
    costos = costo_por_serie.reindex(panel.columns).fillna(0.0)
    valor = panel.clip(lower=0).mul(costos, axis=1).sum(axis=1)
    por_mes = valor.groupby(valor.index.month).sum()
    participacion = por_mes / por_mes.sum()
    umbral = factor / 12.0
    meses = sorted(int(m) for m in participacion[participacion > umbral].index)
    if not meses:
        meses = [int(participacion.idxmax())]
    return meses


def agregar_contribuciones(
    contribuciones: pd.DataFrame,
    columnas: list[str],
    meses: list[int],
) -> pd.DataFrame:
    """Agrega las contribuciones TreeSHAP por variable: global, pico y resto.

    Args:
        contribuciones: una fila por predicción, con las columnas de features
            (la contribución de cada una, en unidades de la predicción), más
            ``serie`` y ``periodo``.
        columnas: nombres de las features (el orden del modelo).
        meses: meses calendario considerados de pico.

    Returns:
        Una fila por feature, ordenada por contribución absoluta media en el
        bloque completo, con la participación porcentual de cada feature sobre
        el total de contribución absoluta y las medias absoluta y con signo
        dentro y fuera del pico. La media CON SIGNO en el pico dice hacia dónde
        empuja la variable cuando el pico ocurre; la absoluta dice cuánto pesa.
    """
    es_pico = contribuciones["periodo"].dt.month.isin(meses)
    pico = contribuciones.loc[es_pico, columnas]
    resto = contribuciones.loc[~es_pico, columnas]

    tabla = pd.DataFrame({
        "contrib_abs_media": contribuciones[columnas].abs().mean(),
        "contrib_abs_media_pico": pico.abs().mean(),
        "contrib_media_pico": pico.mean(),
        "contrib_abs_media_resto": resto.abs().mean(),
        "contrib_media_resto": resto.mean(),
    })
    total = float(tabla["contrib_abs_media"].sum())
    tabla.insert(
        1, "participacion_pct",
        100.0 * tabla["contrib_abs_media"] / total if total > 0 else 0.0,
    )
    return (
        tabla.sort_values("contrib_abs_media", ascending=False)
        .rename_axis("feature")
        .reset_index()
    )


def error_por_mes(
    real: pd.DataFrame,
    predicciones: dict[str, pd.DataFrame],
    costo_por_serie: pd.Series,
    modelos: list[str],
) -> pd.DataFrame:
    """WMAPE y Bias valorizados por MES del bloque, para cada modelo.

    Es el corte que la lectura por estrato no da: cómo le va a cada brazo en
    cada mes calendario de la prueba — dentro del pico estacional, y sobre
    todo a la salida, donde un método que arrastra el nivel de campaña queda
    expuesto. Solo series con costo en el maestro, la misma vara que la suite
    valorizada.
    """
    costos = costo_por_serie.reindex(real.columns)
    con_costo = costos.notna()
    real_v = real.loc[:, con_costo].mul(costos[con_costo], axis=1)
    demanda = real_v.clip(lower=0).sum(axis=1)

    filas = []
    for nombre in modelos:
        pred = predicciones[nombre].reindex(index=real.index, columns=real.columns)
        pred_v = pred.loc[:, con_costo].mul(costos[con_costo], axis=1)
        delta = pred_v.sub(real_v)
        error = delta.abs().sum(axis=1)
        sesgo = delta.sum(axis=1)
        for periodo in real.index:
            d = float(demanda[periodo])
            filas.append({
                "periodo": str(periodo),
                "modelo": nombre,
                "demanda_valorizada": d,
                "wmape_val": 100.0 * float(error[periodo]) / d if d > 0 else float("nan"),
                "bias_val": 100.0 * float(sesgo[periodo]) / d if d > 0 else float("nan"),
            })
    return pd.DataFrame(filas)


def error_por_temporada(
    real: pd.DataFrame,
    predicciones: dict[str, pd.DataFrame],
    costo_por_serie: pd.Series,
    meses: list[int],
    modelos: list[str],
) -> pd.DataFrame:
    """La misma suite valorizada, partida en meses de pico y resto del año.

    El WMAPE agregado de una temporada se calcula desde las sumas (error total
    sobre demanda total de esos meses), no promediando los ratios mensuales:
    promediar ratios daría el mismo peso a un mes chico que a uno de campaña.
    """
    costos = costo_por_serie.reindex(real.columns)
    con_costo = costos.notna()
    real_v = real.loc[:, con_costo].mul(costos[con_costo], axis=1)
    es_pico = real.index.month.isin(meses)

    filas = []
    for nombre in modelos:
        pred = predicciones[nombre].reindex(index=real.index, columns=real.columns)
        pred_v = pred.loc[:, con_costo].mul(costos[con_costo], axis=1)
        delta = pred_v.sub(real_v)
        for etiqueta, mascara in (("pico", es_pico), ("resto", ~es_pico)):
            demanda = float(real_v.loc[mascara].clip(lower=0).sum().sum())
            error = float(delta.loc[mascara].abs().sum().sum())
            sesgo = float(delta.loc[mascara].sum().sum())
            filas.append({
                "modelo": nombre,
                "temporada": etiqueta,
                "meses": int(mascara.sum()),
                "demanda_valorizada": demanda,
                "wmape_val": 100.0 * error / demanda if demanda > 0 else float("nan"),
                "bias_val": 100.0 * sesgo / demanda if demanda > 0 else float("nan"),
            })
    return pd.DataFrame(filas)


def pareto_valorizado(
    panel: pd.DataFrame,
    costo_por_serie: pd.Series,
) -> pd.DataFrame:
    """Curva de concentración de la demanda valorizada, por SKU.

    Valoriza la demanda de todo el panel (|unidades| × costo unitario), agrega
    las combinaciones canal–regional de cada SKU y ordena de mayor a menor.
    Las series sin costo en el maestro quedan fuera (aportarían valor cero y
    aplanarían la curva sin significar nada).

    Returns:
        Una fila por SKU con su demanda valorizada, participación porcentual,
        participación acumulada y la fracción acumulada del padrón de SKUs —
        las coordenadas de la curva de Pareto.
    """
    costos = costo_por_serie.reindex(panel.columns)
    con_costo = costos.notna()
    valor_por_serie = (
        panel.loc[:, con_costo].clip(lower=0)
        .mul(costos[con_costo], axis=1)
        .sum(axis=0)
    )
    sku = valor_por_serie.index.str.split("|").str[0]
    por_sku = valor_por_serie.groupby(sku).sum().sort_values(ascending=False)

    total = float(por_sku.sum())
    tabla = por_sku.rename("demanda_valorizada").rename_axis("sku").reset_index()
    tabla["participacion_pct"] = 100.0 * tabla["demanda_valorizada"] / total
    tabla["acumulada_pct"] = tabla["participacion_pct"].cumsum()
    tabla["fraccion_skus_pct"] = 100.0 * (tabla.index + 1) / len(tabla)
    return tabla
