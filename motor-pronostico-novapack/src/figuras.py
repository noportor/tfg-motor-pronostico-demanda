"""RF-9 — Figuras del documento.

Restricciones de forma, todas por la misma razón: la tesis se imprime.

- **Escala de grises.** Nada de codificar información solo con color: una figura
  que solo se entiende a color deja de entenderse fotocopiada. Se usan además
  tramas y marcadores distintos para que cada serie sea identificable sin color.
- **200 dpi o más.**
- **Sin dependencia de fuentes del sistema.** Se fija DejaVu Sans, que viene con
  matplotlib, de modo que la figura sale idéntica en cualquier máquina (RN-5).
- Los ejes se recortan por percentiles declarados cuando hay valores extremos, y
  la figura anota cuántos puntos quedaron fuera. Recortar sin decirlo sería
  ocultar datos.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


def _es(valor: float, decimales: int = 1, signo: bool = False) -> str:
    """Números en convención castellana: miles con punto, decimales con coma,
    signo menos tipográfico. Todas las anotaciones pasan por acá — mezclar
    '1,668' (miles anglosajones) con '8,3 %' (coma decimal) en la misma tesis
    haría ambiguo cuál coma es cuál.
    """
    if valor is None or not np.isfinite(valor):
        return "—"
    plantilla = f"{{:{'+' if signo else ''},.{decimales}f}}"
    texto = plantilla.format(valor)
    if decimales:
        entero, _, decimal = texto.rpartition(".")
        texto = entero.replace(",", ".") + "," + decimal
    else:
        texto = texto.replace(",", ".")
    return texto.replace("-", "−")


def _es_entero(valor) -> str:
    return _es(float(valor), 0)


def _estilo(dpi: int) -> None:
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.85",
        "grid.linewidth": 0.5,
        "axes.edgecolor": "0.2",
        "axes.labelcolor": "0.1",
        "text.color": "0.1",
        "xtick.color": "0.2",
        "ytick.color": "0.2",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


TRAMAS = ["", "///", "...", "\\\\\\", "xxx", "+++", "ooo", "***", "|||", "---", "OOO"]
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]


def _guardar(figura, destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    return destino


def figura_perfil_categorias(
    valor_mensual: pd.DataFrame,
    estacionales: tuple[str, ...],
    destino: Path,
    dpi: int = 300,
) -> Path:
    """F1 — Perfil mensual del valor por categoría (solo entrenamiento).

    El trabajo de esta figura: justificar la población objetivo mostrando los
    dos regímenes del negocio — recurrentes casi planas contra estacionales de
    campaña. Las estacionales van en trazos oscuros con marcador; las
    recurrentes, en grises. La línea de reparto uniforme (100/12) es la vara.

    Args:
        valor_mensual: columnas ``categoria``, ``mes`` (1..12) y ``pct``
            (participación del mes en el valor anual de SU categoría).
    """
    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 4.2))

    estilos_estacionales = [
        dict(color="black", linestyle="-", marker="o", markersize=4, linewidth=1.8),
        dict(color="0.25", linestyle="--", marker="s", markersize=4, linewidth=1.6),
    ]
    # Grises medios, no claros: la recurrente más tenue seguía legible en
    # pantalla pero se perdía en la fotocopia.
    estilos_recurrentes = [
        dict(color="0.40", linestyle="-.", marker="^", markersize=3.5, linewidth=1.3),
        dict(color="0.50", linestyle=":", marker="D", markersize=3.5, linewidth=1.3),
        dict(color="0.58", linestyle="-", marker="v", markersize=3.5, linewidth=1.2),
    ]

    i_est = i_rec = 0
    for categoria in sorted(valor_mensual["categoria"].unique()):
        datos = valor_mensual.loc[valor_mensual["categoria"] == categoria]
        datos = datos.sort_values("mes")
        if categoria in estacionales:
            estilo = estilos_estacionales[i_est % len(estilos_estacionales)]
            i_est += 1
        else:
            estilo = estilos_recurrentes[i_rec % len(estilos_recurrentes)]
            i_rec += 1
        eje.plot(datos["mes"], datos["pct"], label=categoria, **estilo)

    eje.axhline(100 / 12, color="0.7", linewidth=0.8, linestyle="--")
    # Anotación ENTERA por encima de la línea de referencia (va="bottom"): con
    # va="center" la línea discontinua pasaba entre las dos líneas del texto y
    # el paréntesis rozaba el marcador del mes 12.
    eje.text(12.4, 100 / 12 + 0.25, "8,3 %\n(uniforme)", fontsize=7,
             va="bottom", color="0.45")
    eje.set_xticks(range(1, 13))
    eje.set_xlabel("Mes calendario")
    eje.set_ylabel("Participación en el valor anual de la categoría (%)")
    # Sin título incrustado: el epígrafe se escribe en el documento y acá se
    # duplicaría (regla de TODO el catálogo).
    eje.legend(frameon=False, fontsize=8)
    eje.set_xlim(0.6, 14.3)
    return _guardar(figura, destino)


def figura_volumen_gestiones(
    valor_por_gestion: pd.DataFrame,
    gestiones_evento: tuple[int, ...],
    gestion_validacion: int,
    gestion_prueba: int,
    destino: Path,
    dpi: int = 300,
) -> Path:
    """F2 — Valor por gestión fiscal, con el contexto marcado.

    Valorizado a propósito: sumar unidades entre SKUs no tiene sentido físico
    (la regla de todo el estudio). El confinamiento va con trama; validación y
    prueba, en tonos distintos — el lector ve de un vistazo que el año de
    validación es atípico y que nada de eso se ocultó.
    """
    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 3.8))

    datos = valor_por_gestion.sort_values("gestion").reset_index(drop=True)
    for i, fila in datos.iterrows():
        gestion = int(fila["gestion"])
        color, trama = "0.55", ""
        if gestion in gestiones_evento:
            color, trama = "0.80", "///"
        if gestion == gestion_validacion:
            color = "0.35"
        if gestion == gestion_prueba:
            color = "black"
        eje.bar(i, fila["valor_mm"], color=color, hatch=trama,
                edgecolor="black", linewidth=0.6, width=0.66)

    eje.set_xticks(range(len(datos)))
    eje.set_xticklabels([f"g{int(g)}" for g in datos["gestion"]])
    eje.set_ylabel("Demanda valorizada (millones de Bs)")
    # El significado de cada tono/trama iba en el título; sin título (los
    # epígrafes se escriben en el documento) la decodificación tiene que
    # vivir en una leyenda. FUERA del área de datos: las barras llegan casi
    # al tope del panel en ambos extremos y cualquier esquina interna choca.
    eje.legend(handles=[
        Patch(facecolor="0.55", edgecolor="black", label="entrenamiento"),
        Patch(facecolor="0.80", edgecolor="black", hatch="///",
              label="confinamiento"),
        Patch(facecolor="0.35", edgecolor="black", label="validación"),
        Patch(facecolor="black", edgecolor="black", label="prueba"),
    ], frameon=False, fontsize=8, ncols=4,
        loc="lower left", bbox_to_anchor=(0, 1.01))
    eje.grid(axis="x", visible=False)
    return _guardar(figura, destino)


def figura_unidad_analisis(
    tabla_sku: pd.DataFrame, destino: Path, dpi: int = 300
) -> Path:
    """F3 — La independencia dentro del SKU, en dos paneles.

    Izquierda: en cuántas combinaciones canal-regional vive cada SKU (la barra
    de los single-combo va atenuada: para ellos la pregunta no existe).
    Derecha: correlación mediana entre las combinaciones del mismo SKU — la
    masa a la izquierda de 0,5 es el argumento.
    """
    _estilo(dpi)
    figura, (izq, der) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    conteo = tabla_sku["n_combos"].clip(upper=10).value_counts().sort_index()
    colores = ["0.85" if k == 1 else "0.45" for k in conteo.index]
    izq.bar(range(len(conteo)), conteo.to_numpy(), color=colores,
            edgecolor="black", linewidth=0.6)
    izq.set_xticks(range(len(conteo)))
    izq.set_xticklabels([("10+" if k == 10 else str(int(k))) for k in conteo.index])
    izq.set_xlabel("Combinaciones canal-regional")
    izq.set_ylabel("SKUs")
    izq.set_title("¿En cuántas combinaciones\nvive cada SKU?", fontsize=9)
    # La barra atenuada era la única codificación sin decodificar en la
    # figura. Arriba a la IZQUIERDA: la moda de la distribución está en 7-9
    # combinaciones y la esquina derecha cae sobre esas barras.
    izq.legend(handles=[
        Patch(facecolor="0.85", edgecolor="black",
              label="1 combinación (no aplica)"),
        Patch(facecolor="0.45", edgecolor="black",
              label="2 o más combinaciones"),
    ], frameon=False, fontsize=6.5, loc="upper left")

    correlaciones = tabla_sku["correlacion_mediana"].dropna()
    der.hist(correlaciones, bins=np.arange(-1.0, 1.01, 0.2), color="0.45",
             edgecolor="black", linewidth=0.6)
    der.axvline(0.5, color="black", linewidth=0.9, linestyle="--")
    der.text(0.57, der.get_ylim()[1] * 0.92,
             f"{100 * (correlaciones < 0.5).mean():.0f} % < 0,5",
             fontsize=8)
    der.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _es(v, 1)))
    der.set_xlabel("Correlación mediana intra-SKU (niveles)\n"
                   f"({_es_entero(len(correlaciones))} SKUs con correlación "
                   "calculable)")
    der.set_ylabel("SKUs")
    der.set_title("¿Se mueven juntas las\ncombinaciones de un SKU?", fontsize=9)

    figura.tight_layout()
    return _guardar(figura, destino)


def figura_muestra_sensibilidad(
    tabla_flujo: pd.DataFrame,
    sensibilidad: pd.DataFrame,
    umbrales: dict,
    destino: Path,
    dpi: int = 300,
) -> Path:
    """F4 — La muestra: cascada de exclusiones + sensibilidad del umbral clave.

    Izquierda: cada criterio y cuántas series dejó (el N no aparece de la
    nada). Derecha: cómo se moverían N y cobertura valorizada si el umbral de
    ceros —el único con margen real— fuera otro; el punto marca la decisión.
    """
    _estilo(dpi)
    figura, (izq, der) = plt.subplots(
        1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    pasos = tabla_flujo.reset_index(drop=True)
    posiciones = np.arange(len(pasos))[::-1]
    izq.barh(posiciones, pasos["series"], color="0.55",
             edgecolor="black", linewidth=0.6, height=0.62)
    for pos, (_, fila) in zip(posiciones, pasos.iterrows()):
        etiqueta = _es_entero(fila["series"])
        if fila["excluidas"]:
            etiqueta += f"  (−{_es_entero(fila['excluidas'])})"
        izq.text(fila["series"], pos, "  " + etiqueta, va="center", fontsize=7.5)
    izq.set_yticks(posiciones)
    # ≥/≤ tipográficos y coma decimal también en los NOMBRES de los pasos
    # (llegan del pipeline con "<=" y "0.70").
    izq.set_yticklabels(
        [re.sub(r"(\d)\.(\d)", r"\1,\2",
                str(p).split(". ", 1)[-1]
                .replace(">=", "≥").replace("<=", "≤"))
         for p in pasos["paso"]], fontsize=7.5
    )
    izq.set_xlabel("Series")
    izq.set_title("La cascada de la muestra", fontsize=9)
    izq.grid(axis="y", visible=False)
    # Margen amplio: la etiqueta de la barra más larga tiene que caber ENTERA
    # dentro del panel (con 0.28 el spine derecho tachaba los números).
    izq.margins(x=0.48)

    base_hist = umbrales["historial_minimo_meses"]
    base_vol = umbrales["volumen_minimo_unidades"]
    base_ceros = umbrales["proporcion_maxima_ceros"]
    corte = sensibilidad.loc[
        (sensibilidad["historial_minimo_meses"] == base_hist)
        & (sensibilidad["volumen_minimo_unidades"] == base_vol)
    ].sort_values("proporcion_maxima_ceros")

    der.plot(corte["proporcion_maxima_ceros"], corte["n_series"],
             color="black", marker="o", markersize=4, linewidth=1.4,
             label="N de la muestra")
    der.set_xlabel("Umbral de proporción de ceros")
    der.set_ylabel("Series")
    der.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _es(v, 0)))
    if "pct_demanda_train_valorizada" in corte.columns:
        der2 = der.twinx()
        # Marcador hueco: en los extremos de la curva el cuadrado caía sobre
        # el círculo de N y lo tapaba por completo.
        der2.plot(corte["proporcion_maxima_ceros"],
                  corte["pct_demanda_train_valorizada"],
                  color="0.55", marker="s", markersize=4, linewidth=1.2,
                  linestyle="--", markerfacecolor="white",
                  label="Cobertura valorizada")
        der2.set_ylabel("Cobertura del valor (%)", color="0.4")
        der2.tick_params(axis="y", colors="0.4")
        der2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _es(v, 1)))
        der2.grid(visible=False)
    seleccionado = corte.loc[
        corte["proporcion_maxima_ceros"] == base_ceros
    ]
    if len(seleccionado):
        der.scatter(seleccionado["proporcion_maxima_ceros"],
                    seleccionado["n_series"], s=90, facecolors="none",
                    edgecolors="black", linewidths=1.4, zorder=5)
    der.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _es(v, 2)))
    # Título de UNA línea: la segunda línea parentética quedaba en la misma
    # base que el título del panel izquierdo y se leían como una sola frase.
    der.set_title("Sensibilidad del umbral de ceros", fontsize=9)
    # Tres líneas cortas: en dos líneas, la primera se extendía hasta cruzarse
    # con el tramo inicial de la curva de N.
    der.text(0.97, 0.05,
             f"historial ≥ {base_hist} m\nvolumen ≥ {base_vol:.0f} u\n"
             "el círculo es la decisión",
             transform=der.transAxes, ha="right", va="bottom", fontsize=7,
             color="0.3")

    figura.tight_layout()
    return _guardar(figura, destino)


def figura_linea_de_tiempo(cfg, destino: Path, dpi: int = 300) -> Path:
    """F5 — El diseño temporal completo en una sola barra.

    Diagrama, no datos: gestiones fiscales, los tres bloques, la ventana del
    evento (trama), el fold interno del ajuste de hiperparámetros y los cortes.
    Es la figura que responde de un vistazo «¿y acá no hay fuga?».
    """
    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.6, 2.6))

    inicio = cfg.periodo.inicio.ordinal
    fin_ent = cfg.particion.fin_entrenamiento.ordinal
    fin_val = cfg.particion.fin_validacion.ordinal
    fin = cfg.periodo.fin.ordinal

    def _tramo(desde, hasta, y, alto, color, trama="", etiqueta=None, tam=8.5):
        eje.barh(y, hasta - desde + 1, left=desde, height=alto, color=color,
                 hatch=trama, edgecolor="black", linewidth=0.7)
        if etiqueta:
            eje.text((desde + hasta) / 2, y, etiqueta, ha="center",
                     va="center", fontsize=tam,
                     color="white" if color in ("black", "0.35") else "black")

    _tramo(inicio, fin_ent, 0.62, 0.34, "0.72",
           etiqueta="ENTRENAMIENTO · gestiones 2018–2024 (84 meses)")
    # Cuerpo menor en las barras de un año: a 8,5 pt «VALIDACIÓN» desbordaba
    # su barra e invadía la de entrenamiento con texto blanco sobre gris claro.
    _tramo(fin_ent + 1, fin_val, 0.62, 0.34, "0.35",
           etiqueta="VALIDACIÓN\n g2025", tam=7)
    _tramo(fin_val + 1, fin, 0.62, 0.34, "black", etiqueta="PRUEBA\n g2026",
           tam=7)

    if cfg.evento is not None:
        ventana = cfg.evento.ventana()
        _tramo(ventana[0].ordinal, ventana[-1].ordinal, 0.20, 0.16, "0.88",
               trama="///")
        eje.text((ventana[0].ordinal + ventana[-1].ordinal) / 2, 0.06,
                 "evento exógeno (corregido solo en el ajuste)",
                 ha="center", va="top", fontsize=7, color="0.3")

    ajuste = cfg.crudo.get("lightgbm_tuning") or {}
    if ajuste:
        desde = pd.Period(str(ajuste["fold_desde"]), freq="M").ordinal
        hasta = pd.Period(str(ajuste["fold_hasta"]), freq="M").ordinal
        _tramo(desde, hasta, 1.02, 0.16, "0.55", trama="...")
        eje.text(desde - 2, 1.02, "fold interno del ajuste\n"
                 "de hiperparámetros (g2024)", ha="right", va="center",
                 fontsize=7, color="0.3")

    # La decisión del motor, sin flechas que crucen otras marcas: una nota
    # corta bajo el bloque de validación.
    eje.text((fin_ent + 1 + fin_val) / 2, 0.34,
             "el motor se selecciona aquí", ha="center", va="center",
             fontsize=7, color="0.25", style="italic")

    marcas = [pd.Period(f"{anio}-04", freq="M") for anio in range(2017, 2027)]
    eje.set_xticks([m.ordinal for m in marcas])
    eje.set_xticklabels([f"abr\n{m.year}" for m in marcas], fontsize=7)
    eje.set_xlim(inicio - 1.5, fin + 2.5)
    eje.set_ylim(-0.30, 1.45)
    eje.set_yticks([])
    eje.grid(visible=False)
    return _guardar(figura, destino)


def figura_origenes_rodantes(cfg, destino: Path, dpi: int = 300) -> Path:
    """F6 — El protocolo multihorizonte: orígenes rodantes sobre la prueba.

    Cada fila es un origen (lo que el planificador sabe ese día, en gris) y su
    proyección hacia adelante (línea con marcadores, h=1…H). La figura explica
    sin palabras por qué h=1 agrega 12 orígenes y h=12 uno solo.
    """
    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 3.6))

    mh = cfg.multihorizonte
    inicio_origenes = cfg.particion.fin_validacion
    fin = cfg.periodo.fin
    n_origenes = (fin - 1 - inicio_origenes).n + 1

    for fila in range(n_origenes):
        origen = inicio_origenes + fila
        horizonte = min(mh.horizonte_maximo, (fin - origen).n)
        y = n_origenes - fila
        eje.plot([inicio_origenes.ordinal - 3.5, origen.ordinal], [y, y],
                 color="0.8", linewidth=3.2, solid_capstyle="butt")
        meses = [origen.ordinal + h for h in range(1, horizonte + 1)]
        eje.plot(meses, [y] * len(meses), color="black", linewidth=1.0,
                 marker="o", markersize=2.6)
        eje.scatter([origen.ordinal], [y], marker="|", s=90, color="black",
                    linewidths=1.6, zorder=5)

    primero = n_origenes
    # «historia conocida» centrada SOBRE las barras grises que describe: a la
    # izquierda del lienzo quedaba desanclada y el lector no la vinculaba.
    eje.text(inicio_origenes.ordinal - 1.75, primero + 1.0,
             "historia conocida", fontsize=7.5, color="0.35", ha="center")
    eje.annotate("origen", xy=(inicio_origenes.ordinal, primero + 0.25),
                 xytext=(inicio_origenes.ordinal + 1.2, primero + 1.0),
                 fontsize=7.5, ha="center",
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    eje.text(inicio_origenes.ordinal + 8.5, primero + 1.0,
             f"proyección h=1…{mh.horizonte_maximo} "
             "(se trunca al fin de la prueba)",
             fontsize=7.5, color="0.1", ha="center")

    meses_eje = [inicio_origenes + k for k in range(0, (fin - inicio_origenes).n + 1, 3)]
    eje.set_xticks([m.ordinal for m in meses_eje])
    eje.set_xticklabels([str(m) for m in meses_eje], fontsize=7)
    eje.set_ylabel("Origen (uno por mes de prueba)")
    eje.set_yticks([])
    eje.set_ylim(0.2, n_origenes + 1.9)
    eje.grid(axis="y", visible=False)
    return _guardar(figura, destino)


def figura_casos(
    panel_real: pd.DataFrame,
    predicciones: dict[str, pd.DataFrame],
    casos: list[tuple[str, str]],
    inicio_validacion,
    inicio_prueba,
    destino: Path,
    dpi: int = 300,
    desde=None,
) -> Path:
    """F13 — Casos ilustrativos: la realidad con los pronósticos encima.

    Un panel por caso. Los casos NO se eligen a dedo: cada uno es la serie de
    mayor demanda valorizada de su régimen (la regla va en la etiqueta y en la
    memoria) — representativos por regla declarada, no por conveniencia. Los
    pronósticos se dibujan SOLO desde validación: dentro de entrenamiento
    serían el ajuste in-sample y confundirían la lectura.
    """
    _estilo(dpi)
    n = len(casos)
    filas = (n + 1) // 2
    figura, ejes = plt.subplots(
        filas, 2, figsize=(7.4, 2.6 * filas), squeeze=False, sharex=True
    )

    estilos = {
        "motor": dict(color="black", linestyle="--", linewidth=1.3,
                      marker="o", markersize=2.6),
        "lightgbm": dict(color="0.35", linestyle="-.", linewidth=1.2,
                         marker="s", markersize=2.4),
    }
    # El benchmark en gris medio, la realidad más clara y gruesa: eran tonos
    # casi iguales y en la zona densa de pronósticos no se separaban.
    estilo_otro = dict(color="0.5", linestyle=":", linewidth=1.1,
                       marker="^", markersize=2.4)

    for indice, (serie, etiqueta) in enumerate(casos):
        eje = ejes[indice // 2][indice % 2]
        real = panel_real[serie]
        if desde is not None:
            real = real.loc[real.index >= desde]
        meses = real.index
        eje.plot([m.ordinal for m in meses], real.to_numpy(), color="0.78",
                 linewidth=1.9, label="real")

        for nombre, panel_pred in predicciones.items():
            pred = panel_pred[serie].loc[
                panel_pred.index >= inicio_validacion
            ]
            eje.plot([m.ordinal for m in pred.index], pred.to_numpy(),
                     label=nombre, **estilos.get(nombre, estilo_otro))

        eje.axvline(inicio_validacion.ordinal - 0.5, color="0.3",
                    linewidth=0.8, linestyle="--")
        eje.axvspan(inicio_prueba.ordinal - 0.5, meses[-1].ordinal + 0.5,
                    color="0.93", zorder=0)
        # Sin margen automático a la derecha: dejaba una franja blanca DESPUÉS
        # del sombreado de prueba que sugería datos posteriores inexistentes.
        eje.set_xlim(meses[0].ordinal - 0.5, meses[-1].ordinal + 0.5)
        eje.set_title(etiqueta, fontsize=8.5)
        if indice % 2 == 0:
            eje.set_ylabel("Demanda (unidades)", fontsize=8)
        marcas = [m for m in meses if m.month == 4][::2]
        eje.set_xticks([m.ordinal for m in marcas])
        eje.set_xticklabels([str(m.year) for m in marcas], fontsize=7)

    for sobrante in range(n, filas * 2):
        ejes[sobrante // 2][sobrante % 2].axis("off")

    manijas, nombres = ejes[0][0].get_legend_handles_labels()
    # El sombreado (= prueba) y la línea discontinua (= inicio de validación)
    # los explicaba el título; sin título van como entradas de la leyenda.
    manijas.append(Line2D([0], [0], color="0.3", linewidth=0.8,
                          linestyle="--"))
    nombres.append("inicio de validación")
    manijas.append(Patch(facecolor="0.93", edgecolor="0.6", linewidth=0.5))
    nombres.append("prueba (sombreado)")
    figura.legend(manijas, nombres, frameon=False, fontsize=7.5, ncols=3,
                  loc="lower center", bbox_to_anchor=(0.5, -0.02))
    figura.tight_layout(rect=(0, 0.055, 1, 1))
    return _guardar(figura, destino)


def figura_busqueda_hiperparametros(
    ensayos: pd.DataFrame, destino: Path, dpi: int = 300
) -> Path:
    """F10 — La búsqueda de hiperparámetros, completa: un punto por ensayo.

    D interna contra el número de ensayo, con un marcador por OBJETIVO de
    entrenamiento. Si la hipótesis del sesgo es cierta, se ven tres nubes
    separadas — y el lector ve además el camino del TPE, no solo al ganador.
    """
    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 3.8))

    estilos = {
        "tweedie": dict(marker="o", color="black", s=26),
        "regression_l1": dict(marker="s", color="0.45", s=24),
        "regression_l2": dict(marker="^", color="0.70", s=26),
    }
    for objetivo, grupo in ensayos.groupby("objective", sort=True):
        estilo = estilos.get(str(objetivo),
                             dict(marker="x", color="0.3", s=24))
        eje.scatter(grupo["ensayo"], grupo["D_interna"], label=str(objetivo),
                    edgecolors="black", linewidths=0.5, **estilo)

    mejor = ensayos.loc[ensayos["D_interna"].idxmin()]
    eje.scatter([mejor["ensayo"]], [mejor["D_interna"]], s=170,
                facecolors="none", edgecolors="black", linewidths=1.5, zorder=5)
    # Anotación DEBAJO del punto circulado, sin flecha: por debajo del mínimo
    # no hay puntos por definición (la flecha anterior atravesaba dos ensayos
    # vecinos de la nube tweedie).
    eje.set_ylim(bottom=eje.get_ylim()[0] - 3.5)
    eje.annotate(
        f"mejor: D = {_es(mejor['D_interna'], 1)}",
        xy=(mejor["ensayo"], mejor["D_interna"]),
        xytext=(0, -16), textcoords="offset points",
        ha="center", va="top", fontsize=8,
    )

    eje.set_xlabel("Ensayo (orden del TPE, semilla fija)")
    eje.set_ylabel("D interna valorizada (%)")
    eje.legend(frameon=False, fontsize=8, title="objective",
               title_fontsize=8)
    return _guardar(figura, destino)


def figura_d_descompuesta(
    suite_val: pd.DataFrame, destino: Path, dpi: int = 300
) -> Path:
    """F7 — LA figura principal: D descompuesta en sus dos términos.

    Barras apiladas: el segmento oscuro es el WMAPE valorizado (cuánto se
    erra), el claro con trama es |Bias| (cuánto de ese error es sistemático).
    La descomposición cuenta sola la historia del brazo de aprendizaje
    automático: su primer segmento siempre fue el más corto — lo hundía el
    segundo. El signo del sesgo va anotado (la barra usa el valor absoluto).
    """
    _estilo(dpi)
    datos = suite_val.sort_values("D", ascending=False).reset_index(drop=True)
    n = len(datos)
    posiciones = np.arange(n)

    figura, eje = plt.subplots(figsize=(7.0, max(3.4, 0.42 * n + 1.4)))

    eje.barh(posiciones, datos["wmape_val"], height=0.6, color="0.35",
             edgecolor="black", linewidth=0.6, label="WMAPE valorizado")
    eje.barh(posiciones, datos["bias_val"].abs(), left=datos["wmape_val"],
             height=0.6, color="0.85", hatch="///", edgecolor="black",
             linewidth=0.6, label="|Bias| valorizado")

    for i, fila in datos.iterrows():
        eje.text(fila["D"], i,
                 f"  D = {_es(fila['D'], 1)}  "
                 f"(bias {_es(fila['bias_val'], 1, signo=True)})",
                 va="center", ha="left", fontsize=7.5)

    eje.set_yticks(posiciones)
    eje.set_yticklabels(datos["modelo"])
    eje.set_xlabel("D = WMAPE valorizado + |Bias| valorizado (%)")
    # Leyenda FUERA del área de datos: la barra más larga ocupa todo el ancho
    # y cualquier esquina interna choca con una barra o con su etiqueta.
    eje.legend(frameon=False, fontsize=8, ncols=2,
               loc="lower right", bbox_to_anchor=(1.0, 1.005))
    eje.grid(axis="y", visible=False)
    eje.set_xlim(left=0)
    # Margen para que la anotación de la barra más larga quepa ENTERA dentro
    # del panel (con 0.30 el spine derecho tachaba ocho de doce anotaciones).
    eje.margins(x=0.46)
    return _guardar(figura, destino)


def figura3_dispersion(
    errores: dict[str, pd.DataFrame],
    destino: Path,
    dpi: int = 300,
    percentil_recorte: float = 95.0,
) -> Path:
    """Figura 3 — Dispersión de MAE contra Bias, una serie por punto.

    Un panel por modelo, con ejes compartidos para que la comparación visual sea
    legítima. La nube ideal se pega al fondo del panel (MAE bajo) y a la línea de
    Bias cero: sin error y sin sesgo sistemático.

    Dos decisiones de escala, ambas obligadas por la forma real de los datos:

    - **El MAE va en escala simétrica logarítmica.** Va de fracciones de unidad a
      varios miles: en escala lineal el 95 % de las series se apelmaza contra el
      eje y la figura no muestra nada. El tramo entre 0 y 1 se mantiene lineal
      (``linthresh=1``) para que las series con error casi nulo —incluido el cero
      exacto— sigan siendo visibles.
    - **El eje de Bias se recorta por percentil y el recorte se declara** en el
      pie, con el número de puntos que quedan fuera. Recortar sin decirlo sería
      esconder datos.
    """
    _estilo(dpi)
    nombres = list(errores)
    n = len(nombres)
    columnas = min(4, n)
    filas = math.ceil(n / columnas)

    # Límites comunes, recortados por percentil y declarados en el pie.
    todos_mae = np.concatenate([
        errores[m]["mae"].dropna().to_numpy() for m in nombres
    ])
    todos_bias = np.concatenate([
        errores[m]["bias"].dropna().to_numpy() for m in nombres
    ])
    limite_mae = float(np.percentile(todos_mae, 99.5)) if todos_mae.size else 1.0
    limite_bias = float(np.percentile(np.abs(todos_bias), percentil_recorte)) if todos_bias.size else 1.0
    limite_mae = max(limite_mae, 1.0)
    limite_bias = max(limite_bias, 1e-6)

    figura, ejes = plt.subplots(
        filas, columnas, figsize=(3.0 * columnas, 2.7 * filas),
        sharex=True, sharey=True, squeeze=False,
    )

    fuera_total = 0
    for indice, nombre in enumerate(nombres):
        eje = ejes[indice // columnas][indice % columnas]
        tabla = errores[nombre][["mae", "bias"]].dropna()
        x = tabla["bias"].to_numpy()
        y = tabla["mae"].to_numpy()

        dentro = (np.abs(x) <= limite_bias) & (y <= limite_mae)
        fuera_total += int((~dentro).sum())

        eje.scatter(
            x[dentro], y[dentro], s=5, marker=MARCADORES[indice % len(MARCADORES)],
            facecolors="none", edgecolors="0.25", linewidths=0.4, alpha=0.55,
        )
        eje.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        eje.scatter(
            [np.median(x)], [np.median(y)], s=60, marker="+",
            color="black", linewidths=1.4, zorder=5,
        )
        eje.set_title(
            f"{nombre}\nmediana: MAE {_es(np.median(y), 1)} · "
            f"Bias {_es(np.median(x), 1, signo=True)} %",
            fontsize=8.5,
        )

    for sobrante in range(n, filas * columnas):
        ejes[sobrante // columnas][sobrante % columnas].axis("off")

    # Con sharex, las columnas cuya celda inferior quedó apagada perdían las
    # etiquetas del eje x: se reactivan en el último panel visible de cada
    # columna.
    for columna in range(columnas):
        for fila_i in range(filas - 1, -1, -1):
            eje = ejes[fila_i][columna]
            if eje.axison:
                eje.tick_params(axis="x", labelbottom=True)
                break

    # Un 5 % de aire alrededor de los límites declarados: los puntos que caen
    # exactamente en el recorte se dibujaban partidos por el marco del panel.
    ejes[0][0].set_xlim(-limite_bias * 1.05, limite_bias * 1.05)
    ejes[0][0].set_yscale("symlog", linthresh=1.0)
    ejes[0][0].set_ylim(-0.4, limite_mae * 1.06)

    total_puntos = sum(len(errores[m]) for m in nombres)
    figura.supxlabel("Bias (%) — a la derecha, sobrestima", fontsize=9)
    figura.supylabel("MAE (unidades, escala logarítmica)", fontsize=9)
    figura.text(
        0.01, -0.01,
        f"Eje horizontal recortado al percentil {_es(percentil_recorte, 0)} "
        f"de |Bias| (±{_es(limite_bias, 0)} %); eje vertical al percentil "
        f"99,5 del MAE ({_es(limite_mae, 0)} unidades), en escala logarítmica "
        "simétrica con tramo lineal por debajo de 1.\n"
        f"Quedan fuera del recuadro {_es_entero(fuera_total)} puntos de "
        f"{_es_entero(total_puntos)} "
        f"({_es(100 * fuera_total / max(total_puntos, 1), 1)} %). La cruz "
        "marca la mediana de cada modelo, calculada sobre TODAS las series.\n"
        "La columna de puntos en Bias = −100 % son series cuyo pronóstico "
        "colapsa a cero en el bloque de prueba.",
        fontsize=8, ha="left", va="top",
    )
    figura.tight_layout()

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    return destino


def figura5_horizonte(
    resumen_horizonte: pd.DataFrame,
    destino: Path,
    benchmarks: tuple[str, str] = ("ma_12", "naive_m1"),
    dpi: int = 300,
) -> Path:
    """Figura 5 — Curva de degradación: D valorizada contra el horizonte.

    Es la figura del protocolo multihorizonte (RN-4, segunda parte): a un paso
    los modelos que proyectan línea recta nunca quedan expuestos; esta curva
    muestra quién sostiene el horizonte de planificación.

    Se destacan cuatro brazos (motor, LightGBM y los dos benchmarks de la
    empresa) y el resto queda en gris claro con una sola entrada de leyenda:
    once líneas igualmente marcadas serían ilegibles en escala de grises.
    El pie declara cuántos orígenes agrega cada horizonte — el extremo derecho
    de la curva se apoya en un solo origen y se lee con esa cautela.

    El eje vertical se recorta como en la Figura 3: por percentil declarado.
    Las variantes con factor de crecimiento divergen al componer su pendiente
    mes a mes (llegan a varios cientos por ciento) y sin recorte aplastarían
    la banda donde se decide la comparación; qué líneas se salen y hasta dónde
    llegan queda anotado en el pie, nunca oculto.
    """
    _estilo(dpi)
    tabla = resumen_horizonte.pivot(index="horizonte", columns="modelo", values="D")
    tabla = tabla.sort_index()

    valores_d = tabla.to_numpy(dtype=float)
    valores_d = valores_d[~np.isnan(valores_d)]
    limite = float(np.percentile(valores_d, 90.0)) if valores_d.size else 100.0
    # Los brazos destacados tienen que verse completos: el recorte jamás puede
    # cortar la línea del motor o de un benchmark declarado.
    visibles = [m for m in ("motor", "lightgbm", *benchmarks) if m in tabla.columns]
    if visibles:
        limite = max(limite, 1.1 * float(tabla[visibles].max().max()))

    # Borde blanco en los marcadores: donde dos curvas se cruzan (pasa en
    # h=1 y h=11) los marcadores caen en el mismo punto y sin el anillo el de
    # abajo desaparece.
    destacados = {
        "motor": dict(color="black", linewidth=1.9, linestyle="-",
                      marker="o", markersize=4.5,
                      markeredgecolor="white", markeredgewidth=0.6),
        "lightgbm": dict(color="0.25", linewidth=1.5, linestyle="--",
                         marker="s", markersize=4,
                         markeredgecolor="white", markeredgewidth=0.6),
        benchmarks[0]: dict(color="0.45", linewidth=1.5, linestyle="-.",
                            marker="^", markersize=4,
                            markeredgecolor="white", markeredgewidth=0.6),
        benchmarks[1]: dict(color="0.45", linewidth=1.3, linestyle=":",
                            marker="D", markersize=3.5,
                            markeredgecolor="white", markeredgewidth=0.6),
    }

    figura, eje = plt.subplots(figsize=(7.0, 4.4))

    otros_etiquetados = False
    for modelo in sorted(tabla.columns):
        if modelo in destacados:
            continue
        eje.plot(
            tabla.index, tabla[modelo], color="0.82", linewidth=0.9,
            label=None if otros_etiquetados else "otros brazos", zorder=1,
        )
        otros_etiquetados = True
    for modelo, estilo in destacados.items():
        if modelo not in tabla.columns:
            continue
        eje.plot(tabla.index, tabla[modelo], label=modelo, zorder=3, **estilo)

    eje.set_xlabel("Horizonte h (meses desde el origen)")
    eje.set_ylabel("D = WMAPE val. + |Bias val.| (%)")
    eje.set_xticks(list(tabla.index))
    eje.set_ylim(0, limite)
    eje.legend(frameon=False, fontsize=8, ncols=2)

    fuera = [
        (modelo, float(tabla[modelo].max()))
        for modelo in tabla.columns if float(tabla[modelo].max()) > limite
    ]
    nota_recorte = ""
    if fuera:
        nota_recorte = (
            f" Eje vertical recortado en {_es(limite, 0)} % (percentil 90 de "
            "D); se salen del recuadro: "
            + ", ".join(f"{m} (llega a {_es(v, 0)} %)"
                        for m, v in sorted(fuera))
            + "."
        )

    origenes_por_h = (
        resumen_horizonte.groupby("horizonte")["n_origenes"].max().sort_index()
    )
    figura.text(
        0.01, -0.02,
        "Cada punto agrega todos los orígenes que alcanzan ese horizonte — "
        + ", ".join(f"h={h} ({int(n)})" for h, n in origenes_por_h.items())
        + " orígenes.\nLos horizontes largos se apoyan en pocos orígenes y se "
        "leen con esa cautela." + nota_recorte,
        fontsize=7.5, ha="left", va="top", wrap=True,
    )

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    return destino


def figura4_diferencia_critica(
    rangos_medios: pd.Series,
    diferencia_critica: float,
    n_bloques: int,
    destino: Path,
    dpi: int = 300,
    alfa: float | None = None,
) -> Path:
    """Diagrama de diferencia crítica (Demšar 2006). Complementa al post hoc.

    Los modelos se ordenan por rango medio y se unen con una barra los que NO
    son distinguibles al nivel declarado. Es la lectura visual de la tabla de
    Nemenyi, y evita el error frecuente de leer una tabla de p-valores como si
    fuera un ranking. Solo se dibujan los grupos MAXIMALES: una barra contenida
    en otra no agrega información y sugiere más grupos de los que hay.
    """
    _estilo(dpi)
    ordenados = rangos_medios.sort_values()
    nombres = list(ordenados.index)
    valores = ordenados.to_numpy(dtype=float)
    n = len(nombres)

    ancho_figura = 8.0
    figura, eje = plt.subplots(figsize=(ancho_figura, 1.3 + 0.30 * n))

    # El margen derecho se calcula a partir de la etiqueta más larga: con un
    # margen fijo, el nombre del último modelo se sale del recuadro en cuanto
    # alguien agrega un brazo con nombre largo.
    etiquetas = [f"{nombre}  ({_es(valor, 2)})"
                 for nombre, valor in zip(nombres, valores)]
    rango = float(valores.max() - valores.min()) or 1.0
    ancho_ejes_pulgadas = ancho_figura * 0.86
    ancho_etiqueta_pulgadas = max(len(e) for e in etiquetas) * 0.64 * 8.5 / 72.0
    fraccion = min(0.55, ancho_etiqueta_pulgadas / ancho_ejes_pulgadas)
    margen_izquierdo = 0.08 * rango
    amplitud = (rango + margen_izquierdo) / max(1e-6, 1.0 - fraccion)
    margen_derecho = amplitud - rango - margen_izquierdo
    eje.set_xlim(valores.min() - margen_izquierdo, valores.max() + margen_derecho)

    for i, (etiqueta, valor) in enumerate(zip(etiquetas, valores)):
        eje.plot([valor], [i], marker="o", color="black", markersize=4)
        eje.text(valor + 0.02 * rango, i, etiqueta, va="center", fontsize=8.5)

    # La regla del CD, arriba a la izquierda: sin ella el lector no puede
    # comparar visualmente la longitud de las barras contra el umbral. El
    # rótulo va DESPLAZADO en vertical respecto de la línea de la regla: a la
    # misma altura, la línea parecía tacharlo.
    x_regla = valores.min() - margen_izquierdo * 0.6
    for x_cap in (x_regla, x_regla + diferencia_critica):
        eje.plot([x_cap, x_cap], [-0.50, -0.34], color="black", linewidth=1.0)
    eje.plot([x_regla, x_regla + diferencia_critica], [-0.42, -0.42],
             color="black", linewidth=1.6)
    eje.text(x_regla + diferencia_critica + 0.025 * rango, -0.72,
             f"CD = {_es(diferencia_critica, 3)}", va="center", fontsize=8)

    # Grupos no distinguibles MAXIMALES: como los valores están ordenados,
    # j(i) es no decreciente y basta saltar los grupos contenidos en el
    # anterior.
    nivel = n + 0.4
    j_maximo = -1
    for i in range(n):
        j = i
        while j + 1 < n and (valores[j + 1] - valores[i]) <= diferencia_critica:
            j += 1
        if j > i and j > j_maximo:
            eje.plot([valores[i], valores[j]], [nivel, nivel], color="black",
                     linewidth=2.4)
            nivel += 0.32
        j_maximo = max(j_maximo, j)

    # Los límites se fijan DESPUÉS de dibujar las barras de agrupación: cuántas
    # hay depende de los datos, y con un límite fijo las últimas quedarían
    # fuera. El tope en −1,05 deja aire para la regla del CD y su rótulo.
    eje.set_ylim(-1.05, nivel + 0.3)
    eje.invert_yaxis()
    eje.set_yticks([])
    eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _es(v, 1)))
    eje.set_xlabel("Rango medio de Friedman (1 = mejor)")
    eje.grid(axis="y", visible=False)
    # La CD y el N iban en el título; sin título (el epígrafe se escribe en el
    # documento) son datos de la figura y se declaran en el pie.
    nota_alfa = f"α = {_es(alfa, 2)}; " if alfa is not None else ""
    figura.text(
        0.01, -0.06,
        f"{nota_alfa}{_es_entero(n_bloques)} series, {n} modelos. Los modelos "
        "unidos por una barra horizontal no son distinguibles entre sí; la "
        "regla superior muestra la longitud de la diferencia crítica (CD).",
        fontsize=7.5, ha="left", va="top",
    )

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    return destino
