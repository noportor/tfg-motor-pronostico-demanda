"""Datos — la inspección (RF-2) en versión interactiva.

Todo sale de ``inspeccion.json``; el informe textual completo queda accesible
abajo. Es la página con la que se calibraron los criterios de inclusión.
"""

import altair as alt
import pandas as pd
import streamlit as st

from tablero import estilo
from tablero.comunes import corrida_activa, titulo

corrida = corrida_activa()
datos = corrida.inspeccion()

titulo(
    "Inspección de los datos",
    "Comprensión de datos (CRISP-DM). De aquí salen los umbrales de inclusión "
    "y el N de la muestra.",
)

if not datos:
    st.warning("Esta corrida no tiene `inspeccion.json`; regenerála.")
    st.stop()

# --- Dimensiones -------------------------------------------------------------
rango = datos.get("rango_fechas", ["n/d", "n/d"])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("SKU", f"{datos.get('n_sku', 0):,}")
c2.metric("Canales", f"{datos.get('n_canales', 0):,}")
c3.metric("Regionales", f"{datos.get('n_regionales', 0):,}")
c4.metric("Combinaciones", f"{datos.get('n_combinaciones', 0):,}")
c5.metric("Rango", f"{rango[0]} → {rango[1]}")

intermitentes = datos.get("series_intermitentes_adi_1_32")
if intermitentes and datos.get("panel_series"):
    proporcion = 100 * intermitentes / datos["panel_series"]
    st.caption(
        f"{intermitentes:,} de {datos['panel_series']:,} series "
        f"({proporcion:.1f} %) son demanda intermitente (ADI ≥ 1,32, "
        f"Syntetos–Boylan) — la razón de incluir Croston entre los modelos."
    )

st.divider()

izquierda, derecha = st.columns(2)

# --- Estacionalidad ----------------------------------------------------------
with izquierda:
    st.markdown("**Estacionalidad agregada** — participación de cada mes (%)")
    estacionalidad = pd.DataFrame(datos.get("estacionalidad_mensual", []))
    if not estacionalidad.empty:
        base = alt.Chart(estacionalidad).encode(
            x=alt.X("mes:O", title="Mes"),
            y=alt.Y("porcentaje:Q", title="Participación (%)"),
            tooltip=[alt.Tooltip("mes:O"), alt.Tooltip("porcentaje:Q", format=".2f")],
        )
        barras = base.mark_bar(color=estilo.ACENTO, cornerRadiusEnd=3, width={"band": 0.6})
        # Referencia: reparto plano = 100/12. La estacionalidad ES la distancia
        # a esta línea.
        referencia = (
            alt.Chart(pd.DataFrame({"y": [100 / 12]}))
            .mark_rule(color=estilo.TINTA_MUTED, strokeDash=[4, 3])
            .encode(y="y:Q")
        )
        st.altair_chart(
            estilo.aplicar((barras + referencia).properties(height=260)),
            use_container_width=True,
        )
        st.caption("La línea punteada es el reparto plano (8,33 %).")

# --- Volumen por gestión ------------------------------------------------------
with derecha:
    st.markdown("**Volumen por gestión fiscal** (unidades)")
    gestiones = pd.DataFrame(datos.get("volumen_por_gestion", []))
    if not gestiones.empty:
        grafico = (
            alt.Chart(gestiones)
            .mark_bar(color=estilo.ACENTO, cornerRadiusEnd=3, width={"band": 0.6})
            .encode(
                x=alt.X("gestion:O", title="Gestión (cierra en marzo)"),
                y=alt.Y("unidades:Q", title="Unidades", axis=alt.Axis(format="~s")),
                tooltip=[
                    alt.Tooltip("gestion:O", title="Gestión"),
                    alt.Tooltip("unidades:Q", format=",.0f"),
                    alt.Tooltip("registros:Q", format=",.0f"),
                    alt.Tooltip("combinaciones:Q", format=",.0f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(estilo.aplicar(grafico), use_container_width=True)

st.divider()

# --- Supervivencia a los umbrales --------------------------------------------
st.markdown("**Cuántas combinaciones sobreviven a cada umbral** — la evidencia "
            "detrás de los criterios de inclusión")
col_a, col_b, col_c = st.columns(3)

def _tabla_supervivencia(columna, clave, titulo_col):
    with columna:
        st.caption(titulo_col)
        tabla = pd.DataFrame(datos.get(clave, []))
        if not tabla.empty:
            st.dataframe(tabla, use_container_width=True, hide_index=True)

_tabla_supervivencia(col_a, "supervivencia_historial", "por historial mínimo (meses)")
with col_b:
    st.caption("por proporción de meses en cero")
    ceros = datos.get("proporcion_ceros", {})
    if ceros:
        st.dataframe(
            pd.DataFrame({"percentil": list(ceros), "proporción": list(ceros.values())}),
            use_container_width=True, hide_index=True,
        )
with col_c:
    st.caption("por volumen acumulado (unidades)")
    volumen = datos.get("volumen_por_combinacion", {})
    if volumen:
        st.dataframe(
            pd.DataFrame({"percentil": list(volumen), "unidades": list(volumen.values())}),
            use_container_width=True, hide_index=True,
        )

# --- Reparto e informe completo ----------------------------------------------
with st.expander("Reparto por regional y canal"):
    reparto = pd.DataFrame(datos.get("reparto_canal_regional", []))
    if not reparto.empty:
        st.dataframe(reparto, use_container_width=True, hide_index=True)

with st.expander("Informe de inspección completo (texto)"):
    ruta = corrida.ruta / "inspeccion_datos.txt"
    if ruta.exists():
        st.text(ruta.read_text(encoding="utf-8"))
