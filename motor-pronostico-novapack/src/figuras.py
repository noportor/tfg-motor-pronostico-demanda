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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


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


def figura2_error_compuesto(
    resumen: pd.DataFrame,
    destino: Path,
    dpi: int = 300,
    columna: str = "error_compuesto_pct_media",
    columna_mediana: str = "error_compuesto_pct_mediana",
    etiqueta: str = "Error compuesto (MAPE + |Bias|), %",
) -> Path:
    """Figura 2 — Error compuesto promedio por modelo.

    Se dibujan **media y mediana** una junto a otra a propósito. Unas pocas
    series de gran volumen pueden dominar el promedio; si las dos barras
    discrepan mucho, la figura lo enseña en lugar de esconderlo.
    """
    _estilo(dpi)
    datos = resumen.sort_values(columna).reset_index(drop=True)

    n = len(datos)
    posiciones = np.arange(n)
    ancho = 0.38

    alto = max(3.2, 0.42 * n + 1.6)
    figura, eje = plt.subplots(figsize=(7.0, alto))

    eje.barh(
        posiciones + ancho / 2, datos[columna], height=ancho,
        color="0.35", edgecolor="black", linewidth=0.6, label="Media",
    )
    eje.barh(
        posiciones - ancho / 2, datos[columna_mediana], height=ancho,
        color="0.80", edgecolor="black", linewidth=0.6, hatch="///", label="Mediana",
    )

    for i, fila in datos.iterrows():
        eje.text(fila[columna], i + ancho / 2, f"  {fila[columna]:,.1f}",
                 va="center", ha="left", fontsize=8)
        eje.text(fila[columna_mediana], i - ancho / 2, f"  {fila[columna_mediana]:,.1f}",
                 va="center", ha="left", fontsize=8, color="0.35")

    eje.set_yticks(posiciones)
    eje.set_yticklabels(datos["modelo"])
    eje.set_xlabel(etiqueta)
    eje.set_title("Figura 2. Error compuesto por modelo\n"
                  "(bloque de prueba; menor es mejor)", loc="left", fontsize=10)
    eje.legend(frameon=False, loc="lower right")
    eje.grid(axis="y", visible=False)
    eje.set_xlim(left=0)
    eje.margins(x=0.16)

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    return destino


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
            f"{nombre}\nmediana: MAE {np.median(y):,.1f} · Bias {np.median(x):+.1f} %",
            fontsize=8.5,
        )

    for sobrante in range(n, filas * columnas):
        ejes[sobrante // columnas][sobrante % columnas].axis("off")

    ejes[0][0].set_xlim(-limite_bias, limite_bias)
    ejes[0][0].set_yscale("symlog", linthresh=1.0)
    ejes[0][0].set_ylim(0, limite_mae)

    figura.supxlabel("Bias (%) — a la derecha, sobrestima", fontsize=9)
    figura.supylabel("MAE (unidades, escala logarítmica)", fontsize=9)
    figura.suptitle(
        "Figura 3. Dispersión de MAE contra Bias por serie (bloque de prueba)",
        fontsize=10, x=0.01, ha="left",
    )
    figura.text(
        0.01, -0.01,
        f"Eje horizontal recortado al percentil {percentil_recorte:g} de |Bias| "
        f"(±{limite_bias:,.0f} %); eje vertical al percentil 99,5 del MAE "
        f"({limite_mae:,.0f} unidades), en escala logarítmica simétrica con tramo "
        f"lineal por debajo de 1. Quedan fuera del recuadro {fuera_total:,} puntos "
        f"de {sum(len(errores[m]) for m in nombres):,} "
        f"({100 * fuera_total / max(sum(len(errores[m]) for m in nombres), 1):.1f} %). "
        "La cruz marca la mediana de cada modelo, calculada sobre TODAS las series.",
        fontsize=7, ha="left", va="top",
    )
    figura.tight_layout()

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
) -> Path:
    """Diagrama de diferencia crítica (Demšar 2006). Complementa al post hoc.

    Los modelos se ordenan por rango medio y se unen con una barra los que NO
    son distinguibles al nivel declarado. Es la lectura visual de la tabla de
    Nemenyi, y evita el error frecuente de leer una tabla de p-valores como si
    fuera un ranking.
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
    etiquetas = [f"{nombre}  ({valor:.2f})" for nombre, valor in zip(nombres, valores)]
    rango = float(valores.max() - valores.min()) or 1.0
    ancho_ejes_pulgadas = ancho_figura * 0.86
    ancho_etiqueta_pulgadas = max(len(e) for e in etiquetas) * 0.60 * 8.5 / 72.0
    fraccion = min(0.55, ancho_etiqueta_pulgadas / ancho_ejes_pulgadas)
    margen_izquierdo = 0.08 * rango
    amplitud = (rango + margen_izquierdo) / max(1e-6, 1.0 - fraccion)
    margen_derecho = amplitud - rango - margen_izquierdo
    eje.set_xlim(valores.min() - margen_izquierdo, valores.max() + margen_derecho)

    for i, (etiqueta, valor) in enumerate(zip(etiquetas, valores)):
        eje.plot([valor], [i], marker="o", color="black", markersize=4)
        eje.text(valor + 0.02 * rango, i, etiqueta, va="center", fontsize=8.5)

    # Grupos no distinguibles: cadenas de modelos consecutivos dentro de la CD.
    nivel = n + 0.4
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (valores[j + 1] - valores[i]) <= diferencia_critica:
            j += 1
        if j > i:
            eje.plot([valores[i], valores[j]], [nivel, nivel], color="black", linewidth=2.4)
            nivel += 0.32
        i += 1

    # Los límites se fijan DESPUÉS de dibujar las barras de agrupación: cuántas
    # hay depende de los datos, y con un límite fijo las últimas quedarían fuera.
    eje.set_ylim(-0.6, nivel + 0.3)
    eje.invert_yaxis()
    eje.set_yticks([])
    eje.set_xlabel("Rango medio de Friedman (1 = mejor)")
    eje.set_title(
        f"Diagrama de diferencia crítica · CD = {diferencia_critica:.3f} · "
        f"{n_bloques:,} series · {n} modelos",
        loc="left", fontsize=10,
    )
    eje.grid(axis="y", visible=False)
    figura.text(
        0.01, -0.06,
        "Los modelos unidos por una barra horizontal no son distinguibles entre sí "
        "al nivel de significación declarado.",
        fontsize=7, ha="left", va="top",
    )

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)
    return destino
