"""Resultados — el dashboard: Tabla 8, comparación por métrica y explorador.

Regla de la página: TODO número mostrado sale de un archivo de la corrida
(`resumen_metricas.csv`, `errores_por_serie.csv`, manifiesto). El tablero
reagrega para visualizar, nunca calcula resultados nuevos.
"""

import altair as alt
import pandas as pd
import streamlit as st

from tablero import estilo
from tablero.comunes import corrida_activa, numero, titulo

corrida = corrida_activa()
resumen = corrida.tabla("resumen_metricas.csv")
tabla8 = corrida.tabla("tabla8_resultados.csv")
manifiesto = corrida.manifiesto() or {}
benchmarks = corrida.benchmarks()

titulo("Resultados", "Bloque de prueba: la última gestión cerrada. Menor es mejor.")

if resumen is None or resumen.empty:
    st.warning("Esta corrida no tiene `resumen_metricas.csv`.")
    st.stop()

indexado = resumen.set_index("modelo")
motor_notas = (manifiesto.get("resultados") or {}).get("motor") or {}
acierto = motor_notas.get("acierto") or {}

# --- KPIs --------------------------------------------------------------------
benchmark = benchmarks["promedio_movil"]
mejora = None
if "motor" in indexado.index and benchmark in indexado.index:
    mae_motor = indexado.loc["motor", "mae_mediana"]
    mae_benchmark = indexado.loc[benchmark, "mae_mediana"]
    if mae_benchmark:
        mejora = 100 * (mae_benchmark - mae_motor) / mae_benchmark

c1, c2, c3, c4 = st.columns(4)
c1.metric("Series evaluadas (N)", f"{int(indexado['series'].iloc[0]):,}")
c2.metric(
    "Mejor MASE mediano",
    indexado["mase_mediana"].idxmin(),
    delta=f"{indexado['mase_mediana'].min():.3f}",
    delta_color="off",
)
c3.metric(
    f"Motor vs {benchmark} (MAE mediano)",
    f"−{mejora:.1f} %" if mejora is not None and mejora >= 0
    else (f"+{-mejora:.1f} %" if mejora is not None else "n/d"),
    help="Reducción del MAE mediano del motor frente al benchmark declarado.",
)
c4.metric(
    "Acierto del motor",
    f"{100 * acierto['tasa_acierto']:.1f} %" if acierto else "n/d",
    delta=f"azar: {100 * acierto['azar_esperado']:.0f} %" if acierto else None,
    delta_color="off",
    help="Cuántas veces el elegido en validación fue el mejor en prueba.",
)

st.divider()

# --- Comparación por métrica -------------------------------------------------
st.markdown("**Comparación por métrica**")
seleccion_izq, seleccion_der, _ = st.columns([1, 1, 2])
METRICAS = {
    "MAE (unidades)": ("mae", ",.1f"),
    "MAPE (%)": ("mape", ",.1f"),
    "RMSE (unidades)": ("rmse", ",.1f"),
    "MASE": ("mase", ",.3f"),
    "Bias (%)": ("bias", "+,.1f"),
}
metrica_titulo = seleccion_izq.selectbox("Métrica", list(METRICAS))
agregado = seleccion_der.selectbox("Resumen", ["mediana", "media"])
metrica, formato = METRICAS[metrica_titulo]
columna = f"{metrica}_{agregado}"

grafico = estilo.barras_por_modelo(
    resumen[["modelo", columna]].rename(columns={columna: "valor"}),
    "valor",
    f"{metrica_titulo} — {agregado}",
    benchmarks=benchmarks,
    formato=formato,
)
st.altair_chart(estilo.aplicar(grafico), use_container_width=True)
if metrica == "mape":
    excluido = resumen["pct_excluido_del_mape_medio"].iloc[0]
    st.caption(
        f"El MAPE excluye los meses con demanda real cero "
        f"(~{excluido:.0f} % de las observaciones); con muchos ceros, leer MASE."
    )

# --- Tabla 8 -----------------------------------------------------------------
st.markdown("**Tabla 8 — como va al documento**")
if tabla8 is not None:
    st.dataframe(
        tabla8.style.format(precision=2, thousands=","),
        use_container_width=True, hide_index=True,
    )

st.divider()

# --- Explorador por serie ----------------------------------------------------
st.markdown("**Distribución del error por serie** — lo que el promedio esconde")
errores = corrida.tabla("errores_por_serie.csv")
if errores is not None:
    fila = st.columns([1, 1, 2])
    metrica_explorar = fila[0].selectbox(
        "Métrica por serie", ["mae", "mape", "rmse", "mase", "bias"]
    )
    modelos_disponibles = sorted(errores["modelo"].unique())
    protagonistas = [m for m in ("motor", "lightgbm", benchmark)
                     if m in modelos_disponibles]
    elegidos = fila[1].multiselect(
        "Modelos", modelos_disponibles, default=protagonistas
    )
    if elegidos:
        seleccion = errores.loc[
            errores["modelo"].isin(elegidos), ["modelo", metrica_explorar]
        ].dropna()
        # Percentiles y no histograma: con colas de miles de unidades el
        # histograma se vuelve una sola barra; la tabla de percentiles es la
        # lectura honesta. (Reagregación para visualizar, no un resultado nuevo.)
        percentiles = (
            seleccion.groupby("modelo")[metrica_explorar]
            .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            .unstack()
        )
        percentiles.columns = [f"p{int(100 * c)}" for c in percentiles.columns]
        st.dataframe(
            percentiles.round(2).reset_index(), use_container_width=True, hide_index=True
        )

        recorte = seleccion[metrica_explorar].quantile(0.95)
        visibles = seleccion.loc[seleccion[metrica_explorar] <= recorte]
        grafico = (
            alt.Chart(estilo.con_rol(visibles, benchmarks=benchmarks))
            .mark_boxplot(size=22, color=estilo.ACENTO, outliers={"size": 8})
            .encode(
                y=alt.Y("modelo:N", title=None),
                x=alt.X(f"{metrica_explorar}:Q", title=metrica_explorar.upper()),
                color=estilo.color_por_rol(leyenda=False),
            )
            .properties(height=32 * len(elegidos) + 40)
        )
        st.altair_chart(estilo.aplicar(grafico), use_container_width=True)
        st.caption(
            f"Eje recortado al percentil 95 conjunto "
            f"({numero(recorte)}): quedan fuera "
            f"{int((seleccion[metrica_explorar] > recorte).sum()):,} series. "
            "La identidad la da la etiqueta del eje; el color marca el rol."
        )

st.divider()

# --- Figuras del documento ---------------------------------------------------
st.markdown("**Figuras del documento** — tal como van a la tesis")
figuras = corrida.figuras()
if figuras:
    pestanas = st.tabs([f.stem for f in figuras])
    for pestana, figura in zip(pestanas, figuras):
        pestana.image(str(figura), use_container_width=True)
