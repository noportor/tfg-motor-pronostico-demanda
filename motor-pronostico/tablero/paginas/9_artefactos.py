"""Artefactos — todo lo que la corrida produjo, sin vista a medida.

Esta página es el mecanismo de aditividad «cero código»: cualquier CSV, PNG o
TXT nuevo que el pipeline escriba en salidas*/ aparece acá sin tocar el
tablero. Cuando un artefacto merezca una vista propia, se le escribe una página
— mientras tanto, ya es visible.
"""

import streamlit as st

from tablero import lectores
from tablero.comunes import corrida_activa, titulo

corrida = corrida_activa()
titulo(
    "Artefactos de la corrida",
    f"Todo lo que hay en `{corrida.nombre}/`. Lo nuevo aparece solo.",
)

declarados = lectores.artefactos_declarados(corrida)

# --- Tablas ------------------------------------------------------------------
csvs = corrida.csvs()
if csvs:
    st.markdown("**Tablas**")
    nuevos = [c for c in csvs if c not in declarados]
    if nuevos:
        st.caption(
            "Sin etapa que los declare (probablemente recién agregados): "
            + " · ".join(f"`{n}`" for n in nuevos)
        )
    eleccion = st.selectbox("Archivo", csvs)
    tabla = corrida.tabla(eleccion)
    if tabla is not None:
        st.caption(f"{len(tabla):,} filas × {len(tabla.columns)} columnas")
        st.dataframe(tabla.head(2000), use_container_width=True, hide_index=True)
        if len(tabla) > 2000:
            st.caption("Se muestran las primeras 2.000 filas.")

# --- Figuras -----------------------------------------------------------------
figuras = corrida.figuras()
if figuras:
    st.markdown("**Figuras**")
    pestanas = st.tabs([f.stem for f in figuras])
    for pestana, figura in zip(pestanas, figuras):
        pestana.image(str(figura), use_container_width=True)

# --- Textos ------------------------------------------------------------------
textos = corrida.textos()
if textos:
    st.markdown("**Informes de texto**")
    for nombre in textos:
        with st.expander(nombre):
            st.text((corrida.ruta / nombre).read_text(encoding="utf-8"))

# --- JSON --------------------------------------------------------------------
st.markdown("**Documentos JSON**")
for nombre in ("flujo.json", "inspeccion.json", "manifiesto.json"):
    if (corrida.ruta / nombre).exists():
        with st.expander(nombre):
            st.json(lectores.cargar_json(corrida.ruta / nombre), expanded=False)
