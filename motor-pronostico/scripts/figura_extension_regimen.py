"""Figura de la extensión exploratoria — sensibilidad al régimen (Tabla 12 en imagen).

Dibuja, para los sistemas y pronosticadores clave de la extensión, la D
valorizada a un paso en las dos gestiones de prueba medidas: la gestión 2026
(año normal, ``salidas_ml2_v9``) y la gestión 2025 (inicio de la crisis
cambiaria, ``salidas_ml2_v9_blindaje``). Cada fila une ambos valores con un
segmento: la longitud del segmento ES la sensibilidad al régimen, y el orden
de las filas (por peor caso, ascendente) deja arriba al sistema conmutado.

El insumo son las ``tabla_valorizada.csv`` de ambas corridas: la figura no
recalcula nada, solo dibuja lo que la Tabla 12 y el texto del apartado 4.4
ya reportan. Uso:

    docker compose run --rm tfg python scripts/figura_extension_regimen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.figuras import _es, _estilo, _guardar  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# (clave en tabla_valorizada.csv, rótulo del documento)
SISTEMAS = [
    ("mezcla_conmutada", "Sistema conmutado"),
    ("mezcla_pond", "Menú curado (ponderado)"),
    ("mezcla_prom", "Menú curado (promedio)"),
    ("motor", "Motor (menú ampliado)"),
    ("chronos", "Fundacional calibrado"),
    ("lightgbm", "LightGBM (global)"),
    ("nhits", "N-HiTS"),
    ("ma_12", "Promedio móvil 12 m"),
]


def _d_por_modelo(ruta: Path) -> dict[str, float]:
    tabla = pd.read_csv(ruta)
    return dict(zip(tabla["Modelo"], tabla["D = WMAPE + |Bias| (%)"]))


def figura_extension_regimen(destino: Path, dpi: int = 300) -> Path:
    normal = _d_por_modelo(RAIZ / "salidas_ml2_v9" / "tabla_valorizada.csv")
    quiebre = _d_por_modelo(RAIZ / "salidas_ml2_v9_blindaje" / "tabla_valorizada.csv")

    filas = [
        (rotulo, normal[clave], quiebre[clave])
        for clave, rotulo in SISTEMAS
    ]
    # Peor caso ascendente: el sistema que acota ambos regímenes queda arriba.
    filas.sort(key=lambda f: max(f[1], f[2]))

    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 4.0))

    for i, (rotulo, d_normal, d_quiebre) in enumerate(reversed(filas)):
        y = i
        eje.plot([d_normal, d_quiebre], [y, y], color="0.65", linewidth=1.4,
                 zorder=1)
        eje.plot(d_normal, y, marker="o", markersize=7, color="white",
                 markeredgecolor="black", markeredgewidth=1.3, zorder=3)
        eje.plot(d_quiebre, y, marker="o", markersize=7, color="black",
                 zorder=3)
        # Valores directos, cada uno del lado de afuera de su marcador.
        izquierda = d_normal <= d_quiebre
        eje.annotate(_es(d_normal, 1), (d_normal, y), fontsize=7.5,
                     ha="right" if izquierda else "left", va="center",
                     xytext=(-7 if izquierda else 7, 0),
                     textcoords="offset points", color="0.35")
        eje.annotate(_es(d_quiebre, 1), (d_quiebre, y), fontsize=7.5,
                     ha="left" if izquierda else "right", va="center",
                     xytext=(7 if izquierda else -7, 0),
                     textcoords="offset points")

    eje.set_yticks(range(len(filas)))
    eje.set_yticklabels([f[0] for f in reversed(filas)])
    eje.set_xlabel("D = WMAPE + |Bias| valorizados, a un paso (%)")
    eje.legend(handles=[
        Line2D([], [], marker="o", color="white", markeredgecolor="black",
               markeredgewidth=1.3, markersize=7, linestyle="none",
               label="gestión 2026 (año normal)"),
        Line2D([], [], marker="o", color="black", markersize=7,
               linestyle="none", label="gestión 2025 (quiebre)"),
    ], frameon=False, fontsize=8, ncols=2, loc="lower left",
        bbox_to_anchor=(0, 1.01))
    eje.grid(axis="y", visible=False)
    eje.set_xlim(60, 120)
    return _guardar(figura, destino)


if __name__ == "__main__":
    ruta = figura_extension_regimen(
        RAIZ / "salidas_ml2_v9" / "figura_extension_regimen.png"
    )
    print(f"Figura escrita en {ruta}")
