"""Estilo de los gráficos del tablero.

Regla de color única en todo el tablero: **el color codifica el ROL del modelo
en el experimento, no su identidad**. La identidad la lleva siempre la etiqueta
del eje o la leyenda — nunca el color solo.

- ``motor`` (la propuesta) y ``lightgbm`` (el brazo ML) son los protagonistas y
  llevan los dos acentos. El par azul/naranja está validado con el verificador
  de la guía de visualización (CVD ΔE 24,7; visión normal 33,6; ambos ≥3:1
  sobre superficie blanca — todos los pares).
- Los benchmarks declarados y el resto son CONTEXTO y van en grises neutros.
  El gris claro queda por debajo de 3:1 a propósito (es contexto); la regla de
  alivio se cumple porque cada gráfico lleva etiquetas visibles y toda cifra
  tiene su tabla al lado.

El tema del tablero está fijado en claro (.streamlit/config.toml): estos valores
están validados contra superficie blanca y un modo oscuro automático los dejaría
sin validar.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# -- paleta por rol ----------------------------------------------------------

COLOR_ROL = {
    "Motor (propuesta)": "#2a78d6",
    "LightGBM": "#eb6834",
    "Benchmark declarado": "#52514e",
    "Otros métodos": "#898781",
}
ORDEN_ROLES = list(COLOR_ROL)

# Tinta y cromo, tomados de la instancia de referencia de la guía.
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_MUTED = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
ACENTO = "#2a78d6"
NEUTRO_CLARO = "#eeedea"

# Secuencial (magnitud): un solo tono, claro → oscuro.
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def rol_de(modelo: str, benchmarks: dict[str, str] | None = None) -> str:
    benchmarks = benchmarks or {}
    if modelo == "motor":
        return "Motor (propuesta)"
    if modelo == "lightgbm":
        return "LightGBM"
    if modelo in set(benchmarks.values()):
        return "Benchmark declarado"
    return "Otros métodos"


def con_rol(df: pd.DataFrame, columna_modelo: str = "modelo",
            benchmarks: dict[str, str] | None = None) -> pd.DataFrame:
    salida = df.copy()
    salida["rol"] = salida[columna_modelo].map(lambda m: rol_de(m, benchmarks))
    return salida


def escala_rol() -> alt.Scale:
    # Dominio en orden FIJO: el color sigue al rol aunque un filtro cambie
    # cuántos modelos quedan en pantalla.
    return alt.Scale(domain=ORDEN_ROLES, range=[COLOR_ROL[r] for r in ORDEN_ROLES])


def color_por_rol(leyenda: bool = True) -> alt.Color:
    return alt.Color(
        "rol:N", scale=escala_rol(),
        legend=alt.Legend(title=None, orient="top") if leyenda else None,
    )


def aplicar(grafico: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    """Cromo recesivo: la grilla y los ejes se ven menos que los datos."""
    return (
        grafico
        .configure_view(strokeOpacity=0)
        .configure_axis(
            gridColor=GRILLA, domainColor=EJE, tickColor=EJE,
            labelColor=TINTA_SECUNDARIA, titleColor=TINTA_SECUNDARIA,
            labelFontSize=12, titleFontSize=12,
        )
        .configure_legend(labelColor=TINTA_SECUNDARIA, labelFontSize=12)
    )


def barras_por_modelo(
    df: pd.DataFrame,
    columna_valor: str,
    titulo_valor: str,
    benchmarks: dict[str, str] | None = None,
    formato: str = ",.1f",
    ascendente: bool = True,
) -> alt.LayerChart:
    """Barras horizontales por modelo, coloreadas por rol y con valor al final.

    Es LA vista del tablero: comparar una magnitud entre los once modelos. Las
    barras van ordenadas por valor (menor error primero) y cada una lleva su
    cifra —la etiqueta directa es la vía de identidad, no el color.
    """
    datos = con_rol(df, benchmarks=benchmarks)
    orden = alt.EncodingSortField(
        field=columna_valor, order="ascending" if ascendente else "descending"
    )
    base = alt.Chart(datos).encode(
        y=alt.Y("modelo:N", sort=orden, title=None),
        x=alt.X(f"{columna_valor}:Q", title=titulo_valor,
                axis=alt.Axis(format=formato)),
        tooltip=[
            alt.Tooltip("modelo:N", title="Modelo"),
            alt.Tooltip(f"{columna_valor}:Q", title=titulo_valor, format=formato),
            alt.Tooltip("rol:N", title="Rol"),
        ],
    )
    barras = base.mark_bar(height={"band": 0.62}, cornerRadiusEnd=3).encode(
        color=color_por_rol()
    )
    etiquetas = base.mark_text(align="left", dx=4, color=TINTA_SECUNDARIA).encode(
        text=alt.Text(f"{columna_valor}:Q", format=formato)
    )
    return (barras + etiquetas).properties(height=26 * len(datos) + 40)
