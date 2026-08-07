"""Corrida — la ficha de trazabilidad (RN-6) en versión legible.

Responde la pregunta que abre cualquier revisión: ¿QUÉ produjo estos números?
Datos (hash), configuración (hash + anulaciones), código (commit) y versiones.
"""

import pandas as pd
import streamlit as st

from tablero.comunes import corrida_activa, titulo

corrida = corrida_activa()
manifiesto = corrida.manifiesto() or {}

titulo(
    "Corrida",
    "Cada número del documento se rastrea hasta esta ficha (RN-6).",
)

datos = manifiesto.get("datos", {})
codigo = manifiesto.get("codigo", {})
configuracion = manifiesto.get("configuracion", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Duración de la corrida", f"{manifiesto.get('duracion_segundos', 0):,.0f} s")
c2.metric("Datos (SHA-256)", (datos.get("sha256") or "n/d")[:12])
c3.metric("Configuración (SHA-256)", (configuracion.get("sha256") or "n/d")[:12])
c4.metric("Commit", (codigo.get("commit") or "n/d")[:12])

if codigo.get("arbol_limpio") is False:
    st.warning(
        "El árbol de trabajo tenía cambios sin commitear al ejecutar. Los "
        "números de esta corrida no son citables hasta regenerarla desde un "
        "commit limpio."
    )
if corrida.anulaciones():
    st.info(
        "Esta corrida es una **ablación**: " + "; ".join(corrida.anulaciones())
        + ". El hash de configuración la distingue de la principal."
    )

st.divider()

izquierda, derecha = st.columns(2)

with izquierda:
    st.markdown("**Versiones cargadas en la corrida**")
    dependencias = manifiesto.get("dependencias", {})
    st.dataframe(
        pd.DataFrame(
            {"paquete": list(dependencias), "versión": list(dependencias.values())}
        ),
        use_container_width=True, hide_index=True,
    )

with derecha:
    st.markdown("**Salidas producidas** (con su hash)")
    salidas = manifiesto.get("salidas", [])
    if salidas:
        tabla = pd.DataFrame(salidas)
        tabla["sha256"] = tabla["sha256"].str[:16] + "…"
        st.dataframe(tabla, use_container_width=True, hide_index=True)

with st.expander("Configuración completa de la corrida"):
    st.json(configuracion.get("contenido", {}))
