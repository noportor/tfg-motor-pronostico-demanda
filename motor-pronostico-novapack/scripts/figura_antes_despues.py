"""F9 del catálogo — Antes/después del feature engineering, en una figura.

Compara DOS corridas completas del pipeline: la base (features v2 + tweedie) y
la ablación PISO (features v1 + hiperparámetros originales), para el brazo de
aprendizaje automático y el motor, en los dos protocolos (un paso y
multihorizonte global).

Es una figura de SÍNTESIS entre corridas, por eso no vive dentro del pipeline:
este script lee las tablas de ambas corridas, escribe la figura en
``salidas_figuras/`` y deja un mini-manifiesto que cita el manifiesto (y su
hash de configuración) de cada corrida — la trazabilidad de la RN-6 no se
negocia ni para las figuras de síntesis.

Uso::

    docker compose run --rm tfg python scripts/figura_antes_despues.py \
        --antes salidas_ablacion_piso_v1 --despues salidas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.figuras import _es, _estilo, _guardar  # noqa: E402

MODELOS = ("lightgbm", "motor")


def _leer(corrida: Path) -> dict:
    valorizada = pd.read_csv(corrida / "tabla_valorizada.csv")
    valorizada = valorizada.set_index("Modelo")["D = WMAPE + |Bias| (%)"]
    manifiesto = json.loads(
        (corrida / "manifiesto.json").read_text(encoding="utf-8")
    )
    mh = manifiesto["resultados"].get("multihorizonte", {}) or {}
    d_global = mh.get("D_global_pct", {}) or {}
    return {
        "un_paso": {m: float(valorizada.get(m)) for m in MODELOS},
        "multihorizonte": {m: float(d_global.get(m, float("nan")))
                           for m in MODELOS},
        "configuracion_sha256": manifiesto["configuracion"]["sha256"],
        "datos_sha256": manifiesto["datos"]["sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antes", default="salidas_ablacion_piso_v1",
                        help="Corrida PISO (features v1 + params originales)")
    parser.add_argument("--despues", default="salidas",
                        help="Corrida base (features v2 + tweedie)")
    parser.add_argument("--destino", default="salidas_figuras")
    args = parser.parse_args(argv)

    antes = _leer(RAIZ / args.antes)
    despues = _leer(RAIZ / args.despues)
    if antes["datos_sha256"] != despues["datos_sha256"]:
        raise SystemExit(
            "Las dos corridas usan snapshots DISTINTOS: la comparación no es "
            "pareada. Re-corré la que esté vieja."
        )

    _estilo(300)
    protocolos = (("un_paso", "Un paso (D, %)"),
                  ("multihorizonte", "Multihorizonte global (D(h), %)"))
    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)

    # Escala vertical COMÚN: con límites por panel, barras de valores
    # parecidos quedaban a alturas distintas y la comparación a ojo mentía.
    tope = 1.30 * max(
        corrida[clave][m]
        for corrida in (antes, despues) for clave, _ in protocolos
        for m in MODELOS if np.isfinite(corrida[clave][m])
    )

    for eje, (clave, titulo) in zip(ejes, protocolos):
        posiciones = np.arange(len(MODELOS))
        ancho = 0.36
        v_antes = [antes[clave][m] for m in MODELOS]
        v_despues = [despues[clave][m] for m in MODELOS]
        eje.bar(posiciones - ancho / 2, v_antes, ancho, color="0.8",
                hatch="///", edgecolor="black", linewidth=0.7,
                label="antes (piso v1)")
        eje.bar(posiciones + ancho / 2, v_despues, ancho, color="0.3",
                edgecolor="black", linewidth=0.7,
                label="después (features v2 + búsqueda)")
        for x, (va, vd) in zip(posiciones, zip(v_antes, v_despues)):
            eje.text(x - ancho / 2, va, _es(va, 1), ha="center",
                     va="bottom", fontsize=8)
            eje.text(x + ancho / 2, vd, _es(vd, 1), ha="center",
                     va="bottom", fontsize=8)
            # El delta lleva unidad (puntos porcentuales): sin ella el número
            # en negrita quedaba sin explicación en la figura.
            eje.annotate(f"{_es(vd - va, 1, signo=True)} pp",
                         xy=(x, max(va, vd)),
                         xytext=(x, max(va, vd) + tope * 0.055), ha="center",
                         fontsize=8.5, fontweight="bold")
        eje.set_xticks(posiciones)
        eje.set_xticklabels(MODELOS)
        eje.set_title(titulo, fontsize=9)
        eje.set_ylim(0, tope)
        eje.grid(axis="x", visible=False)

    ejes[0].set_ylabel("D valorizada (%)")
    # Leyenda ARRIBA de los paneles, en el espacio que dejó el título: dentro
    # del panel izquierdo tapaba la anotación del delta de lightgbm.
    manijas, nombres = ejes[0].get_legend_handles_labels()
    figura.legend(manijas, nombres, frameon=False, fontsize=8, ncols=2,
                  loc="lower center", bbox_to_anchor=(0.5, 0.97))
    figura.tight_layout(rect=(0, 0, 1, 0.96))

    destino = RAIZ / args.destino
    ruta = _guardar(figura, destino / "figura09_antes_despues.png")

    (destino / "figura09_manifiesto.json").write_text(json.dumps({
        "figura": ruta.name,
        "corrida_antes": {"directorio": args.antes,
                          "configuracion_sha256": antes["configuracion_sha256"]},
        "corrida_despues": {"directorio": args.despues,
                            "configuracion_sha256": despues["configuracion_sha256"]},
        "datos_sha256": despues["datos_sha256"],
        "valores": {"antes": antes["un_paso"] | {"multihorizonte": antes["multihorizonte"]},
                    "despues": despues["un_paso"] | {"multihorizonte": despues["multihorizonte"]}},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Figura  : {ruta}")
    print(f"Antes   : {args.antes} (config {antes['configuracion_sha256'][:12]}…)")
    print(f"Después : {args.despues} (config {despues['configuracion_sha256'][:12]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
