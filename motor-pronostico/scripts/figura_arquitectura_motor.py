"""Figura de la arquitectura funcional del motor (los cinco módulos en imagen).

Dibuja lo que el apartado 4.2.2 del documento describe en prosa: la entrada
(el snapshot congelado con huella SHA-256), los cinco módulos funcionales en
cadena dentro del entorno reproducible (la imagen de contenedor), la
configuración única que parametriza la corrida, y las salidas (pronósticos,
tablas y figuras, manifiesto de trazabilidad y visor analítico local).

Es un diagrama de diseño: no lee datos de ninguna corrida. Uso:

    docker compose run --rm tfg python scripts/figura_arquitectura_motor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.figuras import _estilo, _guardar  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

MODULOS = [
    "1. Ingesta y\nvalidación\nde datos",
    "2. Prepro-\ncesamiento",
    "3. Modelado\n(diez modelos\ncandidatos)",
    "4. Evaluación\ny selección\n(métrica D)",
    "5. Salida de\nresultados",
]
ENTRADA = ("Snapshot congelado\n(huella SHA-256):\nventas, costos,\n"
           "quiebres de inventario\ny tipo de cambio")
CONFIG = "Configuración única de la corrida (parámetros y semillas)"
CONTENEDOR = ("Imagen de contenedor: entorno reproducible con versiones "
              "fijadas y suite de pruebas automatizadas")
SALIDAS = [
    "Pronósticos\nconsolidados",
    "Tablas y figuras\ndel estudio",
    "Manifiesto de\ntrazabilidad",
    "Visor analítico\nlocal",
]


def figura_arquitectura_motor(destino: Path, dpi: int = 300) -> Path:
    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 2.6))
    eje.set_xlim(0, 100)
    eje.set_ylim(0, 36)
    eje.axis("off")

    # Entrada, a la izquierda.
    ex, ey, eancho, ealto = 1.0, 10.5, 15.5, 15.0
    eje.add_patch(Rectangle((ex, ey), eancho, ealto, facecolor="0.94",
                            edgecolor="black", linewidth=1.0))
    eje.text(ex + eancho / 2, ey + ealto / 2, ENTRADA, ha="center",
             va="center", fontsize=6.3)

    # Marco punteado del entorno reproducible, con los cinco módulos adentro.
    mx0, my0, mancho, malto = 19.5, 6.5, 59.5, 23.0
    eje.add_patch(Rectangle((mx0, my0), mancho, malto, facecolor="none",
                            edgecolor="0.45", linewidth=0.9,
                            linestyle=(0, (4, 2))))
    eje.text(mx0 + mancho / 2, my0 - 0.9, CONTENEDOR, ha="center", va="top",
             fontsize=6.0, color="0.30")

    ancho, alto, y0 = 10.4, 13.0, 10.5
    paso = (mancho - 5 * ancho - 2 * 1.2) / 4 + ancho
    xs = [mx0 + 1.2 + i * paso for i in range(5)]
    for x, texto in zip(xs, MODULOS):
        eje.add_patch(Rectangle((x, y0), ancho, alto, facecolor="white",
                                edgecolor="black", linewidth=1.1))
        eje.text(x + ancho / 2, y0 + alto / 2, texto, ha="center",
                 va="center", fontsize=6.2)
    flechas = [((ex + eancho, ey + ealto / 2), (xs[0], y0 + alto / 2))]
    flechas += [((x + ancho, y0 + alto / 2), (x + paso, y0 + alto / 2))
                for x in xs[:-1]]
    for desde, hasta in flechas:
        eje.add_patch(FancyArrowPatch(desde, hasta, arrowstyle="-|>",
                                      mutation_scale=10, color="black",
                                      linewidth=1.0, shrinkA=0, shrinkB=0))

    # Configuración única, arriba del marco, alimenta la cadena.
    cy = 32.6
    eje.text(mx0 + mancho / 2, cy + 0.4, CONFIG, ha="center", va="bottom",
             fontsize=6.3)
    eje.add_patch(FancyArrowPatch(
        (mx0 + mancho / 2, cy), (mx0 + mancho / 2, my0 + malto),
        arrowstyle="-|>", mutation_scale=9, color="0.45", linewidth=0.9,
        linestyle=(0, (4, 2)), shrinkA=0, shrinkB=0))

    # Salidas, a la derecha, en pila.
    sx, sancho, salto, sep = 84.5, 14.5, 7.4, 1.4
    total = 4 * salto + 3 * sep
    sy0 = 18.0 - total / 2 + (4 - 1) * (salto + sep)
    for i, texto in enumerate(SALIDAS):
        y = sy0 - i * (salto + sep)
        eje.add_patch(Rectangle((sx, y), sancho, salto, facecolor="0.94",
                                edgecolor="black", linewidth=1.0))
        eje.text(sx + sancho / 2, y + salto / 2, texto, ha="center",
                 va="center", fontsize=6.2)
        eje.add_patch(FancyArrowPatch(
            (xs[4] + ancho, y0 + alto / 2), (sx, y + salto / 2),
            arrowstyle="-|>", mutation_scale=8, color="black",
            linewidth=0.8, shrinkA=0, shrinkB=0))

    figura.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return _guardar(figura, destino)


if __name__ == "__main__":
    ruta = figura_arquitectura_motor(RAIZ / "salidas" / "figura_arquitectura_motor.png")
    print(f"Figura escrita en {ruta}")
