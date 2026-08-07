"""Flujo — el pipeline de punta a punta, con sus conteos y decisiones.

El diagrama y los detalles salen ÍNTEGROS de ``flujo.json``: esta página no
sabe cuántas etapas hay ni cómo se llaman. Una etapa nueva registrada con
``reporte.etapa(...)`` aparece aquí sin tocar este archivo — ese es el contrato.
"""

import streamlit as st

from tablero import lectores
from tablero.comunes import corrida_activa, render_conteos, titulo

corrida = corrida_activa()
etapas = lectores.etapas_de(corrida)

titulo(
    "Flujo del experimento",
    "De los datos crudos al contraste estadístico. Cada caja declara qué "
    "entra, qué sale y qué decidió.",
)

if not etapas:
    st.warning(
        "Esta corrida no tiene `flujo.json` (es anterior al contrato de "
        "etapas). Regenerála con `python main.py ejecutar`."
    )
    st.stop()


def _resumen_magnitud(dic: dict) -> str:
    """Primer valor numérico de entrada/salida, para la etiqueta del nodo."""
    for clave, valor in dic.items():
        if isinstance(valor, (int, float)):
            return f"{clave.replace('_', ' ')}: {valor:,.0f}"
    return ""


# --- Diagrama ---------------------------------------------------------------
# DOT crudo: streamlit lo renderiza sin dependencias extra. Colores del cromo
# de la guía; la información viva del nodo es su magnitud de salida.
nodos, aristas = [], []
for indice, etapa in enumerate(etapas):
    salida = _resumen_magnitud(etapa.get("salida") or {}) or _resumen_magnitud(
        etapa.get("entrada") or {}
    )
    etiqueta = etapa["titulo"].replace('"', "'")
    sub = f"\\n{salida}" if salida else ""
    rf = f"  ·  {etapa['rf']}" if etapa.get("rf") else ""
    nodos.append(
        f'  e{indice} [label="{etiqueta}{rf}{sub}" shape=box style="rounded,filled" '
        f'fillcolor="#f6f5f2" color="#c3c2b7" fontcolor="#0b0b0b" fontsize=11];'
    )
    if indice:
        aristas.append(f"  e{indice - 1} -> e{indice} [color=\"#898781\"];")

st.graphviz_chart(
    "digraph {\n  rankdir=TB;\n  bgcolor=transparent;\n  "
    'node [fontname="Helvetica"];\n'
    + "\n".join(nodos) + "\n" + "\n".join(aristas) + "\n}",
    use_container_width=True,
)

st.divider()

# --- Detalle genérico por etapa ---------------------------------------------
for etapa in etapas:
    rf = f"{etapa['rf']} · " if etapa.get("rf") else ""
    duracion = (
        f"  ·  {etapa['duracion_s']:.1f} s" if etapa.get("duracion_s") else ""
    )
    with st.expander(f"{rf}{etapa['titulo']}{duracion}"):
        entrada, salida = etapa.get("entrada") or {}, etapa.get("salida") or {}
        if entrada or salida:
            columnas = st.columns(max(len(entrada) + len(salida), 1))
            for posicion, (clave, valor) in enumerate(
                [*entrada.items(), *salida.items()]
            ):
                etiqueta = clave.replace("_", " ")
                valor_texto = (
                    f"{valor:,.0f}" if isinstance(valor, (int, float)) else str(valor)
                )
                columnas[posicion].metric(etiqueta, valor_texto)

        if etapa.get("decisiones"):
            st.markdown("**Decisiones**")
            st.json(etapa["decisiones"], expanded=False)

        if etapa.get("conteos"):
            st.markdown("**Conteos**")
            render_conteos(etapa["conteos"])

        if etapa.get("artefactos"):
            st.markdown(
                "**Artefactos**: " + " · ".join(f"`{a}`" for a in etapa["artefactos"])
            )
        for nota in etapa.get("notas", []):
            st.caption(f"↳ {nota}")
