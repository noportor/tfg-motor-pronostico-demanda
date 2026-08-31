"""Figura del reparto de la selección del motor: qué modelo eligió y cuántas veces.

Lee el conteo ``reparto_de_elegidos`` de la etapa del motor en
``salidas/flujo.json`` (la misma corrida que sustenta la Tabla 10 del
documento) y lo dibuja como barras horizontales: una por modelo, ordenadas
por número de series asignadas. El único brazo de aprendizaje automático
(LightGBM) va en barra oscura; los métodos estadísticos, en barra clara: la
figura muestra que el motor es genuinamente híbrido y que ningún modelo
domina el portafolio. Uso:

    docker compose run --rm tfg python scripts/figura_reparto_seleccion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.figuras import _es, _es_entero, _estilo, _guardar  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# (clave en flujo.json, rótulo de la Tabla 10 del documento, es aprendizaje automático)
BRAZOS = {
    "lightgbm": ("LightGBM", True),
    "naive_m1": ("Ingenuo (Naïve)", False),
    "ma_2": ("Promedio móvil 2 meses", False),
    "holt_winters": ("Holt–Winters", False),
    "ma_12": ("Promedio móvil 12 meses", False),
    "ma_12_gf": ("Prom. móvil 12 m con factor de crecimiento", False),
    "croston": ("Croston", False),
    "exp_smooth_opt": ("Suavizado exponencial simple", False),
    "ma_2_gf": ("Prom. móvil 2 m con factor de crecimiento", False),
    "naive_m1_gf": ("Ingenuo con factor de crecimiento", False),
}


def figura_reparto_seleccion(destino: Path, dpi: int = 300) -> Path:
    flujo = json.loads((RAIZ / "salidas" / "flujo.json").read_text("utf-8"))
    etapa = next(e for e in flujo["etapas"] if e["id"] == "motor")
    reparto = etapa["conteos"]["reparto_de_elegidos"]
    total = sum(reparto.values())
    assert set(reparto) == set(BRAZOS), "brazos distintos a los esperados"

    filas = sorted(reparto.items(), key=lambda kv: kv[1])
    ml = sum(v for k, v in reparto.items() if BRAZOS[k][1])

    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 3.4))
    ys = range(len(filas))
    for y, (clave, n) in zip(ys, filas):
        rotulo, es_ml = BRAZOS[clave]
        eje.barh(y, n, height=0.62, color="0.15" if es_ml else "0.62",
                 edgecolor="black", linewidth=0.6)
        eje.annotate(f"{_es_entero(n)}  ({_es(100.0 * n / total, 1)} %)",
                     (n, y), xytext=(5, 0), textcoords="offset points",
                     va="center", fontsize=7.5)
    eje.set_yticks(list(ys))
    eje.set_yticklabels([BRAZOS[clave][0] for clave, _ in filas], fontsize=8)
    eje.set_xlabel(f"Series asignadas por el motor (de {_es_entero(total)})")
    eje.set_xlim(0, max(v for _, v in filas) * 1.22)
    eje.grid(axis="y", visible=False)
    eje.legend(handles=[
        Patch(facecolor="0.62", edgecolor="black", linewidth=0.6,
              label=(f"métodos estadísticos: {_es_entero(total - ml)} series "
                     f"({_es(100.0 * (total - ml) / total, 1)} %)")),
        Patch(facecolor="0.15", edgecolor="black", linewidth=0.6,
              label=(f"aprendizaje automático: {_es_entero(ml)} series "
                     f"({_es(100.0 * ml / total, 1)} %)")),
    ], frameon=False, fontsize=8, loc="lower right")
    return _guardar(figura, destino)


if __name__ == "__main__":
    ruta = figura_reparto_seleccion(RAIZ / "salidas" / "figura_reparto_seleccion.png")
    print(f"Figura escrita en {ruta}")
