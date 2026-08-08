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

from src.figuras import _estilo, _guardar  # noqa: E402

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
            eje.text(x - ancho / 2, va, f"{va:.1f}", ha="center",
                     va="bottom", fontsize=8)
            eje.text(x + ancho / 2, vd, f"{vd:.1f}", ha="center",
                     va="bottom", fontsize=8)
            eje.annotate(f"{vd - va:+.1f}", xy=(x, max(va, vd)),
                         xytext=(x, max(va, vd) * 1.12), ha="center",
                         fontsize=8.5, fontweight="bold")
        eje.set_xticks(posiciones)
        eje.set_xticklabels(MODELOS)
        eje.set_title(titulo, fontsize=9)
        eje.set_ylim(0, max(*v_antes, *v_despues) * 1.28)
        eje.grid(axis="x", visible=False)

    ejes[0].set_ylabel("D valorizada (%)")
    ejes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    figura.suptitle(
        "Figura 9. El feature engineering y la función de pérdida, aislados: "
        "misma muestra, mismos protocolos", fontsize=10, x=0.01, ha="left",
    )
    figura.tight_layout()

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
