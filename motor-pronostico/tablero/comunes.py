"""Utilidades compartidas por las páginas del tablero."""

from __future__ import annotations

import streamlit as st

from tablero import lectores


def corrida_activa() -> lectores.Corrida:
    """La corrida elegida en la barra lateral. Detiene la página si no hay."""
    corrida = st.session_state.get("corrida")
    if corrida is None:
        st.warning(
            "No hay ninguna corrida seleccionada. Generá una con:\n\n"
            "```\ndocker compose run --rm tfg python main.py ejecutar\n```"
        )
        st.stop()
    return corrida


def titulo(texto: str, subtitulo: str | None = None) -> None:
    st.markdown(f"### {texto}")
    if subtitulo:
        st.caption(subtitulo)


def render_conteos(conteos: dict) -> None:
    """Renderizador GENÉRICO de los conteos de una etapa del flujo.

    Es la pieza que hace aditivo el tablero: una etapa nueva con conteos de
    cualquier forma razonable (escalares, dicts anidados, listas de dicts) se
    muestra sin escribir una vista a medida.
    """
    import pandas as pd

    escalares = {}
    for clave, valor in conteos.items():
        if isinstance(valor, list) and valor and isinstance(valor[0], dict):
            st.caption(clave.replace("_", " "))
            st.dataframe(pd.DataFrame(valor), use_container_width=True, hide_index=True)
        elif isinstance(valor, dict):
            st.caption(clave.replace("_", " "))
            st.dataframe(
                pd.DataFrame(
                    {"clave": list(valor), "valor": [valor[k] for k in valor]}
                ),
                use_container_width=True, hide_index=True,
            )
        else:
            escalares[clave] = valor
    if escalares:
        st.dataframe(
            pd.DataFrame(
                {"conteo": list(escalares),
                 "valor": [escalares[k] for k in escalares]}
            ),
            use_container_width=True, hide_index=True,
        )


def numero(valor: float | int | None, formato: str = ",.2f") -> str:
    if valor is None:
        return "n/d"
    try:
        return format(float(valor), formato)
    except (TypeError, ValueError):
        return str(valor)
