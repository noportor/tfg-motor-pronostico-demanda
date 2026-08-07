"""Modelos y optimización — qué parámetros eligió cada modelo y a qué costo.

La 'optimización' del experimento es esto: rejillas explícitas por serie (α de
SES y Croston; α, β, γ de Holt-Winters) y parada temprana de LightGBM. Todo
sale de `parametros_por_serie.csv`, `lightgbm_importancias.csv` y de la etapa
`modelos` del flujo.
"""

import altair as alt
import pandas as pd
import streamlit as st

from tablero import estilo, lectores
from tablero.comunes import corrida_activa, titulo

corrida = corrida_activa()
titulo(
    "Modelos y optimización",
    "Los parámetros se estiman SOLO con entrenamiento; validación la usa "
    "únicamente la parada temprana de LightGBM (RN-2).",
)

etapa_modelos = next(
    (e for e in lectores.etapas_de(corrida) if e["id"] == "modelos"), {}
)
conteos = etapa_modelos.get("conteos", {})

# --- Costo y respaldos -------------------------------------------------------
izquierda, derecha = st.columns(2)

with izquierda:
    st.markdown("**Tiempo de ajuste por modelo** (s)")
    tiempos = conteos.get("duracion_por_modelo_s", {})
    if tiempos:
        tabla = pd.DataFrame(
            {"modelo": list(tiempos), "valor": list(tiempos.values())}
        )
        grafico = estilo.barras_por_modelo(
            tabla, "valor", "segundos",
            benchmarks=corrida.benchmarks(), formato=",.1f", ascendente=False,
        )
        st.altair_chart(estilo.aplicar(grafico), use_container_width=True)

with derecha:
    st.markdown("**Respaldos en validación + prueba**")
    st.caption(
        "Meses evaluados donde el modelo no pudo pronosticar y se aplicó la "
        "cascada declarada (Naïve → media de entrenamiento). Debería ser ~0: "
        "si no lo es, hay que decirlo en el documento."
    )
    respaldos = conteos.get("respaldos_en_validacion_y_prueba", {})
    if respaldos:
        st.dataframe(
            pd.DataFrame(
                {"modelo": list(respaldos), "respaldos": list(respaldos.values())}
            ),
            use_container_width=True, hide_index=True,
        )

st.divider()

# --- Parámetros elegidos por serie (fragmento) -------------------------------
@st.fragment
def parametros_por_serie() -> None:
    st.markdown("**Qué parámetros ganaron, serie por serie**")
    parametros = corrida.tabla("parametros_por_serie.csv")
    if parametros is None or parametros.empty:
        st.info("Esta corrida no registró parámetros por serie.")
        return

    combinaciones = (
        parametros[["modelo", "parametro"]].drop_duplicates().itertuples(index=False)
    )
    opciones = [f"{c.modelo} · {c.parametro}" for c in combinaciones]
    eleccion = st.selectbox("Modelo · parámetro", opciones)
    modelo_sel, parametro_sel = [parte.strip() for parte in eleccion.split("·")]

    seleccion = parametros.loc[
        (parametros["modelo"] == modelo_sel)
        & (parametros["parametro"] == parametro_sel)
    ]
    # La rejilla es discreta: contar series por valor dice más que un
    # histograma continuo (y al navegador viaja el CONTEO, no las series).
    # Un solo tono: es una única serie.
    conteo = (
        seleccion.groupby("valor").size().rename("series").reset_index()
    )
    grafico = (
        alt.Chart(conteo)
        .mark_bar(color=estilo.ACENTO, cornerRadiusEnd=3, width={"band": 0.6})
        .encode(
            x=alt.X("valor:O", title=f"{parametro_sel} elegido"),
            y=alt.Y("series:Q", title="Series"),
            tooltip=["valor:O", alt.Tooltip("series:Q", format=",.0f")],
        )
        .properties(height=240)
    )
    st.altair_chart(estilo.aplicar(grafico), use_container_width=True)
    st.caption(
        "Un α alto = la serie pide reaccionar rápido; α bajo = suavizar. La "
        "forma de esta distribución es un retrato del portafolio."
    )


parametros_por_serie()

st.divider()

# --- LightGBM ----------------------------------------------------------------
st.markdown("**LightGBM** — el brazo de aprendizaje automático")
importancias = corrida.tabla("lightgbm_importancias.csv")
mejor_iteracion = (corrida.manifiesto() or {}).get("resultados", {}).get(
    "lightgbm_mejor_iteracion"
)
if mejor_iteracion:
    st.metric("Árboles (elegidos por parada temprana contra validación)",
              f"{mejor_iteracion:,}")

if importancias is not None and not importancias.empty:
    primeras = importancias.nlargest(15, "ganancia").copy()
    grafico = (
        alt.Chart(primeras)
        .mark_bar(color="#eb6834", cornerRadiusEnd=3, height={"band": 0.62})
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("ganancia:Q", title="Ganancia acumulada", axis=alt.Axis(format="~s")),
            tooltip=["feature:N", alt.Tooltip("ganancia:Q", format=",.0f"),
                     alt.Tooltip("divisiones:Q", format=",.0f")],
        )
        .properties(height=26 * len(primeras) + 40)
    )
    st.altair_chart(estilo.aplicar(grafico), use_container_width=True)
    st.caption(
        "Importancia por ganancia (reducción de error acumulada). Que dominen "
        "los rezagos y medias móviles es lo esperable en demanda mensual."
    )
