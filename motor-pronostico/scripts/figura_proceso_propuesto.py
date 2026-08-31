"""Figura del proceso propuesto (TO-BE), espejo del esquema actual del diagnóstico.

El diagrama del flujo actual (Figura 5 del documento) muestra cinco etapas
secuenciales sin retroalimentación. Esta figura presenta el proceso que el
motor habilita con la misma gramática visual (cajas y flechas, cinco etapas)
para que la comparación sea directa, y agrega el lazo que el diagnóstico
constató ausente: la retroalimentación del error observado hacia la corrida
siguiente. La reunión de consenso se conserva: el motor reemplaza el armado
manual, no la decisión.

Es un diagrama de diseño: no lee datos de ninguna corrida. Uso:

    docker compose run --rm tfg python scripts/figura_proceso_propuesto.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.figuras import _estilo, _guardar  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ETAPAS = [
    "Snapshot mensual\ndel histórico y los\nregistros auxiliares",
    "Ejecución del motor:\npronóstico y selección\nautomática por serie",
    "Métricas de error\ny manifiesto\nde trazabilidad",
    "Reunión de\nplanificación: revisión\ny consenso",
    "Plan de demanda\nregistrado",
]
RETRO = ("Retroalimentación: el error del plan frente a la venta real\n"
         "se mide en la corrida del mes siguiente")


def figura_proceso_propuesto(destino: Path, dpi: int = 300) -> Path:
    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 1.9))
    eje.set_xlim(0, 100)
    eje.set_ylim(0, 26)
    eje.axis("off")

    ancho, alto, y0 = 16.4, 12.0, 9.5
    paso = (100 - 5 * ancho - 2 * 1.0) / 4 + ancho
    xs = [1.0 + i * paso for i in range(5)]
    for x, texto in zip(xs, ETAPAS):
        eje.add_patch(Rectangle((x, y0), ancho, alto, facecolor="white",
                                edgecolor="black", linewidth=1.1))
        eje.text(x + ancho / 2, y0 + alto / 2, texto, ha="center",
                 va="center", fontsize=6.9)
    for x in xs[:-1]:
        eje.add_patch(FancyArrowPatch(
            (x + ancho, y0 + alto / 2), (x + paso, y0 + alto / 2),
            arrowstyle="-|>", mutation_scale=11, color="black",
            linewidth=1.1, shrinkA=0, shrinkB=0))

    # Lazo de retroalimentación: del plan registrado al snapshot siguiente.
    x_fin = xs[4] + ancho / 2
    x_ini = xs[0] + ancho / 2
    y_bajo = 3.2
    eje.plot([x_fin, x_fin, x_ini], [y0, y_bajo, y_bajo], color="0.45",
             linewidth=1.0, linestyle=(0, (4, 2)))
    eje.add_patch(FancyArrowPatch(
        (x_ini, y_bajo), (x_ini, y0), arrowstyle="-|>", mutation_scale=10,
        color="0.45", linewidth=1.0, linestyle=(0, (4, 2)),
        shrinkA=0, shrinkB=0))
    eje.text((x_ini + x_fin) / 2, y_bajo - 0.6, RETRO, ha="center", va="top",
             fontsize=6.4, color="0.30")

    figura.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return _guardar(figura, destino)


if __name__ == "__main__":
    ruta = figura_proceso_propuesto(RAIZ / "salidas" / "figura_proceso_propuesto.png")
    print(f"Figura escrita en {ruta}")
