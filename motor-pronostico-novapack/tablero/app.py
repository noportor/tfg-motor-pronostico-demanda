"""Punto de entrada del tablero.

    docker compose up tablero      ->  http://localhost:8501

Las páginas se DESCUBREN de ``tablero/paginas/`` por convención de nombre
(``<orden>_<titulo>.py``): agregar una página nueva es soltar un archivo ahí —
no hay ningún registro que actualizar. El orden lo da el prefijo numérico.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROYECTO = Path(__file__).resolve().parent.parent
if str(PROYECTO) not in sys.path:
    sys.path.insert(0, str(PROYECTO))

from tablero import lectores  # noqa: E402

st.set_page_config(
    page_title="Motor de pronóstico — NOVAPACK",
    page_icon="📈",
    layout="wide",
)

# --- Selector de corrida (común a todas las páginas) ------------------------

corridas = lectores.descubrir_corridas()

with st.sidebar:
    st.markdown("**Motor de pronóstico de demanda**")
    st.caption("NOVAPACK S.A. — análisis del TFG")

    if not corridas:
        st.warning(
            "No se encontró ninguna corrida.\n\n"
            "Generá una desde la carpeta del proyecto:\n"
            "```\ndocker compose run --rm tfg \\\n  python main.py ejecutar\n```"
        )
        st.stop()

    seleccion = st.selectbox(
        "Corrida",
        corridas,
        format_func=lambda c: c.etiqueta(),
        help="Todo directorio salidas*/ con manifiesto.json es una corrida; "
             "las ablaciones aparecen solas.",
    )
    st.session_state["corrida"] = seleccion

    manifiesto = seleccion.manifiesto() or {}
    codigo = manifiesto.get("codigo", {})
    st.caption(
        f"Generada: {manifiesto.get('generado_en', 'n/d')}\n\n"
        f"Commit: `{(codigo.get('commit') or 'n/d')[:12]}`"
        + ("" if codigo.get("arbol_limpio") else "  ⚠ árbol con cambios")
    )
    if seleccion.anulaciones():
        st.info("Ablación: " + "; ".join(seleccion.anulaciones()))

# --- Páginas descubiertas por convención ------------------------------------

def _titulo_de(archivo: Path) -> str:
    # "4_resultados.py" -> "Resultados"
    nombre = archivo.stem.split("_", 1)[-1].replace("_", " ")
    return nombre[:1].upper() + nombre[1:]


paginas = [
    st.Page(str(ruta), title=_titulo_de(ruta))
    for ruta in sorted((Path(__file__).parent / "paginas").glob("[0-9]*_*.py"))
]
if not paginas:
    st.error("No hay páginas en tablero/paginas/.")
    st.stop()

st.navigation(paginas).run()
