"""Contraste — Wilcoxon, Friedman y Nemenyi, con su guía de lectura.

Fuentes: las notas del manifiesto (donde el pipeline serializó cada prueba),
`rangos_friedman.csv` y `nemenyi.csv`. El informe redactado completo está al
final, tal como va al documento.
"""

import altair as alt
import pandas as pd
import streamlit as st

from tablero import estilo
from tablero.comunes import corrida_activa, titulo

corrida = corrida_activa()
manifiesto = corrida.manifiesto() or {}
resultados = manifiesto.get("resultados") or {}
config = corrida.configuracion()
alfa = float(config.get("pruebas", {}).get("alfa", 0.05))

titulo(
    "Contraste estadístico",
    "Con miles de series cualquier diferencia sale significativa: la lectura "
    "válida es el TAMAÑO DEL EFECTO primero, el p-valor después.",
)

# --- Wilcoxon ----------------------------------------------------------------
st.markdown("**Wilcoxon de rangos con signo** — pareado, unilateral "
            "(Demšar 2006)")
wilcoxon = resultados.get("wilcoxon") or []
if wilcoxon:
    tabla = pd.DataFrame(wilcoxon)
    tabla["% series a favor"] = (
        100 * tabla["gana_propuesto"] / tabla["n_pares"]
    ).round(1)
    tabla["r"] = tabla["r"].round(3)
    tabla["p"] = tabla["p"].map(lambda p: f"{p:.2e}")
    tabla["veredicto"] = tabla["significativo"].map(
        lambda s: "significativo" if s else "no significativo"
    )
    st.dataframe(
        tabla[["propuesto", "referencia", "r", "% series a favor", "p", "veredicto"]],
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "r = Z/√N (convención de Cohen: 0,1 pequeño · 0,3 mediano · 0,5 grande). "
        "El signo negativo indica MENOS error del modelo propuesto."
    )

st.divider()

# --- Friedman ----------------------------------------------------------------
friedman = resultados.get("friedman") or {}
st.markdown("**Friedman + post hoc de Nemenyi** — los once modelos a la vez")
if friedman:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chi² de Friedman", f"{friedman.get('chi2', 0):,.0f}")
    c2.metric("p", f"{friedman.get('p', 1):.2e}")
    c3.metric("W de Kendall", f"{friedman.get('kendall_w', 0):.3f}",
              help="Concordancia del ranking entre series. No comparable entre "
                   "análisis con distinto número de modelos.")
    c4.metric("Diferencia crítica (CD)", f"{friedman.get('diferencia_critica', 0):.3f}",
              help="Dos modelos con rangos medios a menos de esta distancia no "
                   "son distinguibles al nivel declarado.")

    rangos = friedman.get("rangos_medios") or {}
    if rangos:
        tabla_rangos = pd.DataFrame(
            {"modelo": list(rangos), "rango": [float(v) for v in rangos.values()]}
        )
        datos = estilo.con_rol(tabla_rangos, benchmarks=corrida.benchmarks())
        # Puntos, no barras: el rango medio es una POSICIÓN (1 = mejor), no una
        # magnitud que se acumule desde cero.
        grafico = (
            alt.Chart(datos)
            .mark_circle(size=110)
            .encode(
                y=alt.Y("modelo:N",
                        sort=alt.EncodingSortField("rango", order="ascending"),
                        title=None),
                x=alt.X("rango:Q", title="Rango medio de Friedman (1 = mejor)",
                        scale=alt.Scale(zero=False)),
                color=estilo.color_por_rol(),
                tooltip=["modelo:N", alt.Tooltip("rango:Q", format=".2f"), "rol:N"],
            )
            .properties(height=26 * len(datos) + 40)
        )
        st.altair_chart(estilo.aplicar(grafico), use_container_width=True)

# --- Nemenyi -----------------------------------------------------------------
nemenyi = corrida.tabla("nemenyi.csv")
if nemenyi is not None and not nemenyi.empty:
    st.markdown("**Matriz de Nemenyi** — qué pares son distinguibles")
    largo = nemenyi.melt(id_vars="modelo", var_name="contra", value_name="p")
    largo["distinguible"] = largo["p"] < alfa
    # Dos clases, no un gradiente: la pregunta del post hoc es binaria
    # (¿distinguibles al nivel α?); el p exacto va en el tooltip.
    grafico = (
        alt.Chart(largo)
        .mark_rect(stroke="#ffffff", strokeWidth=2)
        .encode(
            x=alt.X("contra:N", title=None),
            y=alt.Y("modelo:N", title=None),
            color=alt.Color(
                "distinguible:N",
                scale=alt.Scale(domain=[True, False],
                                range=[estilo.ACENTO, estilo.NEUTRO_CLARO]),
                legend=alt.Legend(title=f"p < {alfa:g}", orient="top"),
            ),
            tooltip=["modelo:N", "contra:N", alt.Tooltip("p:Q", format=".4f")],
        )
        .properties(height=380)
    )
    st.altair_chart(estilo.aplicar(grafico), use_container_width=True)
    st.caption(
        "El rango studentizado ya controla el error por familia: no se aplica "
        "otra corrección encima."
    )

with st.expander("Informe completo del contraste (como va al documento)"):
    ruta = corrida.ruta / "pruebas_estadisticas.txt"
    if ruta.exists():
        st.text(ruta.read_text(encoding="utf-8"))
