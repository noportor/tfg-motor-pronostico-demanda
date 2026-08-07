"""Resultados — el dashboard: Tabla 8, comparación por métrica y explorador.

Regla de la página: TODO número mostrado sale de un archivo de la corrida
(`resumen_metricas.csv`, `errores_por_serie.csv`, manifiesto). El tablero
reagrega para visualizar, nunca calcula resultados nuevos.

Fluidez: las secciones con controles son ``st.fragment`` — cambiar la métrica
re-ejecuta SOLO esa sección, no la página entera. Y al navegador viajan
agregados (once filas, percentiles), nunca las miles de filas por serie: además
de lento, ese volumen pasa por la capa nativa de serialización, que es donde el
proceso puede morir sin dejar traceback.
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

METRICAS = {
    "MAE (unidades)": ("mae", ",.1f"),
    "MAPE (%)": ("mape", ",.1f"),
    "RMSE (unidades)": ("rmse", ",.1f"),
    "MASE": ("mase", ",.3f"),
    "Bias (%)": ("bias", "+,.1f"),
}

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


# --- Comparación por métrica (fragmento: el clic solo re-ejecuta esto) -------
@st.fragment
def comparacion_por_metrica() -> None:
    st.markdown("**Comparación por métrica**")
    izquierda, derecha, _ = st.columns([1, 1, 2])
    metrica_titulo = izquierda.selectbox("Métrica", list(METRICAS))
    agregado = derecha.selectbox("Resumen", ["mediana", "media"])
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


comparacion_por_metrica()

# --- Tabla 8 -----------------------------------------------------------------
st.markdown("**Tabla 8 — como va al documento**")
if tabla8 is not None:
    # Redondeo simple, sin Styler: el Styler pasa por otra ruta de
    # serialización nativa y no aporta nada que un round no dé.
    st.dataframe(tabla8.round(2), use_container_width=True, hide_index=True)

st.divider()


# --- Explorador por serie (fragmento) ----------------------------------------
@st.fragment
def explorador_por_serie() -> None:
    st.markdown("**Distribución del error por serie** — lo que el promedio esconde")
    errores = corrida.tabla("errores_por_serie.csv")
    if errores is None:
        return

    fila = st.columns([1, 1, 2])
    metrica = fila[0].selectbox(
        "Métrica por serie", ["mae", "mape", "rmse", "mase", "bias"]
    )
    modelos_disponibles = sorted(errores["modelo"].unique())
    protagonistas = [m for m in ("motor", "lightgbm", benchmark)
                     if m in modelos_disponibles]
    elegidos = fila[1].multiselect(
        "Modelos", modelos_disponibles, default=protagonistas
    )
    if not elegidos:
        return

    seleccion = errores.loc[
        errores["modelo"].isin(elegidos), ["modelo", metrica]
    ].dropna()

    # Percentiles precalculados AQUÍ: al navegador viaja una fila por modelo,
    # no las miles de series. (Reagregación para visualizar, no un resultado
    # nuevo: el detalle está en errores_por_serie.csv.)
    percentiles = (
        seleccion.groupby("modelo")[metrica]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
    )
    percentiles.columns = ["p10", "p25", "p50", "p75", "p90"]
    percentiles = percentiles.reset_index()

    st.dataframe(percentiles.round(2), use_container_width=True, hide_index=True)

    datos = estilo.con_rol(percentiles, benchmarks=benchmarks)
    base = alt.Chart(datos).encode(
        y=alt.Y("modelo:N", title=None,
                sort=alt.EncodingSortField("p50", order="ascending")),
    )
    # Diagrama de rango medio: línea fina p10–p90, banda p25–p75, marca en la
    # mediana. Misma lectura que un boxplot, con once filas de datos.
    rango = base.mark_rule(strokeWidth=2, color=estilo.EJE).encode(
        x=alt.X("p10:Q", title=metrica.upper()), x2="p90:Q",
    )
    banda = base.mark_bar(height={"band": 0.45}, cornerRadius=3).encode(
        x="p25:Q", x2="p75:Q", color=estilo.color_por_rol(),
        tooltip=["modelo:N"] + [
            alt.Tooltip(f"{p}:Q", format=",.2f")
            for p in ("p10", "p25", "p50", "p75", "p90")
        ],
    )
    mediana = base.mark_tick(thickness=2.5, size=22, color=estilo.TINTA).encode(
        x="p50:Q"
    )
    st.altair_chart(
        estilo.aplicar((rango + banda + mediana).properties(
            height=30 * len(percentiles) + 40
        )),
        use_container_width=True,
    )
    st.caption(
        "Línea: p10–p90 · banda: p25–p75 · marca: mediana. "
        "La identidad la da la etiqueta del eje; el color marca el rol."
    )


explorador_por_serie()

st.divider()

# --- Figuras del documento ---------------------------------------------------
st.markdown("**Figuras del documento** — tal como van a la tesis")
figuras = corrida.figuras()
if figuras:
    pestanas = st.tabs([f.stem for f in figuras])
    for pestana, figura in zip(pestanas, figuras):
        pestana.image(str(figura), use_container_width=True)
