"""Motor — la propuesta de la tesis: qué eligió, cuánto acertó.

Fuentes: la etapa `motor` del flujo, las notas del manifiesto y
`seleccion_motor.csv` (la evidencia por serie).
"""

import pandas as pd
import streamlit as st

from tablero import estilo
from tablero.comunes import corrida_activa, titulo

corrida = corrida_activa()
manifiesto = corrida.manifiesto() or {}
motor = (manifiesto.get("resultados") or {}).get("motor") or {}

titulo(
    "Motor de selección",
    "Elige por serie el modelo con menor error EN VALIDACIÓN y lo aplica en "
    "prueba (RN-2). Nunca mira el bloque de prueba para decidir.",
)

if not motor:
    st.warning("Esta corrida no tiene notas del motor en el manifiesto.")
    st.stop()

acierto = motor.get("acierto") or {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regla de selección", motor.get("regla", "n/d"))
c2.metric("Empates en validación", f"{motor.get('empates', 0):,}",
          help="Se desempata alfabéticamente — declarado, no silencioso.")
c3.metric(
    "Tasa de acierto en prueba",
    f"{100 * acierto.get('tasa_acierto', 0):.1f} %",
    delta=f"azar: {100 * acierto.get('azar_esperado', 0):.0f} %",
    delta_color="off",
)
c4.metric(
    "Exceso de MAE mediano",
    f"{acierto.get('exceso_mae_mediano', 0):,.3f}",
    help="Cuánto MAE de más se paga por no haber elegido el óptimo de prueba "
         "(el costo del winner's curse).",
)

st.caption(
    "La brecha entre la tasa de acierto y el 100 % es el *winner's curse*: "
    "elegir el mínimo entre varios candidatos sobre 12 meses sobreestima al "
    "ganador. Es un resultado en sí mismo y se reporta tal cual (§12)."
)

st.divider()

# --- Reparto de elegidos -----------------------------------------------------
st.markdown("**Qué modelo ganó la validación, y en cuántas series**")
reparto = motor.get("reparto") or {}
if reparto:
    tabla = pd.DataFrame(
        {"modelo": list(reparto), "valor": [float(v) for v in reparto.values()]}
    )
    grafico = estilo.barras_por_modelo(
        tabla, "valor", "series en que fue elegido",
        benchmarks=corrida.benchmarks(), formato=",.0f", ascendente=False,
    )
    st.altair_chart(estilo.aplicar(grafico), use_container_width=True)

# --- Evidencia por serie (fragmento) -----------------------------------------
@st.fragment
def evidencia_por_serie() -> None:
    st.markdown("**La selección, serie por serie** — `seleccion_motor.csv`")
    seleccion = corrida.tabla("seleccion_motor.csv")
    if seleccion is None or seleccion.empty:
        return
    filtro = st.multiselect(
        "Filtrar por modelo elegido",
        sorted(seleccion["modelo_elegido"].dropna().unique()),
    )
    visible = (
        seleccion.loc[seleccion["modelo_elegido"].isin(filtro)]
        if filtro else seleccion
    )
    st.caption(
        f"{len(visible):,} series. Las columnas de modelo son el MAE de cada "
        "candidato EN VALIDACIÓN; el motor eligió el mínimo."
    )
    st.dataframe(visible.head(500).round(3), use_container_width=True, hide_index=True)
    if len(visible) > 500:
        st.caption("Se muestran las primeras 500; el CSV completo está en salidas/.")


evidencia_por_serie()
