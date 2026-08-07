"""Comparar corridas — ablaciones lado a lado.

Para qué existe: cambiar UNA decisión (`--anular ...`) y ver qué movió. El hash
de configuración distingue las corridas; las anulaciones dicen en qué difieren.
"""

import altair as alt
import pandas as pd
import streamlit as st

from tablero import estilo, lectores
from tablero.comunes import titulo

titulo(
    "Comparar corridas",
    "Cada directorio salidas*/ es una corrida. Las ablaciones se lanzan por "
    "CLI con --anular y aparecen acá solas.",
)

corridas = lectores.descubrir_corridas()
if len(corridas) < 2:
    st.info(
        "Hace falta más de una corrida para comparar. Lanzá una ablación:\n\n"
        "```\ndocker compose run --rm tfg python main.py ejecutar \\\n"
        "  --anular modelos.motor_regla=mae_mas_bias \\\n"
        "  --anular salidas.directorio=salidas_ablacion_mae_mas_bias\n```"
    )
    st.stop()

elegidas = st.multiselect(
    "Corridas a comparar",
    corridas,
    default=corridas[:2],
    format_func=lambda c: c.etiqueta(),
)
if len(elegidas) < 2:
    st.stop()

# --- En qué difieren ----------------------------------------------------------
st.markdown("**En qué difiere cada corrida**")
filas = []
for corrida in elegidas:
    manifiesto = corrida.manifiesto() or {}
    filas.append({
        "corrida": corrida.nombre,
        "anulaciones": "; ".join(corrida.anulaciones()) or "— (configuración base)",
        "config_sha": (manifiesto.get("configuracion", {}).get("sha256") or "")[:12],
        "datos_sha": (manifiesto.get("datos", {}).get("sha256") or "")[:12],
        "commit": ((manifiesto.get("codigo") or {}).get("commit") or "")[:12],
    })
diferencias = pd.DataFrame(filas)
st.dataframe(diferencias, use_container_width=True, hide_index=True)
if diferencias["datos_sha"].nunique() > 1:
    st.warning(
        "Las corridas usan archivos de datos DISTINTOS: la comparación mezcla "
        "el efecto de la decisión con el de los datos."
    )

st.divider()

# --- Métrica lado a lado (fragmento: el clic solo re-ejecuta esto) ------------
@st.fragment
def metrica_lado_a_lado() -> None:
    st.markdown("**Una métrica, todas las corridas**")
    seleccion_izq, seleccion_der, _ = st.columns([1, 1, 2])
    METRICAS = {
        "MAE (unidades)": ("mae", ",.1f"),
        "MAPE (%)": ("mape", ",.1f"),
        "RMSE (unidades)": ("rmse", ",.1f"),
        "MASE": ("mase", ",.3f"),
        "Bias (%)": ("bias", "+,.1f"),
    }
    metrica_titulo = seleccion_izq.selectbox("Métrica", list(METRICAS), index=3)
    agregado = seleccion_der.selectbox("Resumen", ["mediana", "media"])
    metrica, formato = METRICAS[metrica_titulo]
    columna = f"{metrica}_{agregado}"

    marcos = []
    for corrida in elegidas:
        resumen = corrida.tabla("resumen_metricas.csv")
        if resumen is not None and columna in resumen.columns:
            marcos.append(
                resumen[["modelo", columna]]
                .rename(columns={columna: "valor"})
                .assign(corrida=corrida.nombre)
            )
    if not marcos:
        st.warning("Ninguna de las corridas elegidas tiene `resumen_metricas.csv`.")
        return

    junto = pd.concat(marcos, ignore_index=True)

    # Pivote con Δ contra la primera corrida elegida: el número que responde
    # «¿la decisión mejoró o empeoró?»
    pivote = junto.pivot(index="modelo", columns="corrida", values="valor")
    base = elegidas[0].nombre
    if base in pivote.columns:
        for otra in [c.nombre for c in elegidas[1:] if c.nombre in pivote.columns]:
            pivote[f"Δ {otra}"] = pivote[otra] - pivote[base]
    st.dataframe(
        pivote.round(3).reset_index().sort_values(base),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Δ = corrida − {base} (negativo = menos error que la base).")

    # Barras agrupadas SOLO para los protagonistas: con once modelos × N
    # corridas el gráfico agrupado se vuelve ilegible; la tabla de arriba es la
    # vista completa y esto es el resumen visual.
    protagonistas = [m for m in ("motor", "lightgbm") if m in pivote.index]
    visible = junto.loc[junto["modelo"].isin(protagonistas)]
    if not visible.empty:
        grafico = (
            alt.Chart(visible)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y("corrida:N", title=None),
                x=alt.X("valor:Q", title=f"{metrica_titulo} — {agregado}",
                        axis=alt.Axis(format=formato)),
                yOffset=alt.YOffset("modelo:N"),
                color=alt.Color(
                    "modelo:N",
                    scale=alt.Scale(domain=["motor", "lightgbm"],
                                    range=[estilo.COLOR_ROL["Motor (propuesta)"],
                                           estilo.COLOR_ROL["LightGBM"]]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["corrida:N", "modelo:N",
                         alt.Tooltip("valor:Q", format=formato)],
            )
            .properties(height=64 * len(elegidas) + 40)
        )
        st.altair_chart(estilo.aplicar(grafico), use_container_width=True)


metrica_lado_a_lado()
