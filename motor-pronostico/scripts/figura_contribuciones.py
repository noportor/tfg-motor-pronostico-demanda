"""Figura de las contribuciones TreeSHAP (Figura 6 del documento) desde la corrida publicada.

Vuelve a dibujar la figura sin reentrenar nada: lee el agregado por variable
``salidas/contribuciones_shap.csv`` y los meses de pico declarados en la etapa
``explicacion`` de ``salidas/flujo.json`` (la misma corrida que sustenta el
texto), y llama a ``src.figuras.figura_contribuciones``, la misma función que
usa ``main.py``. El eje está en la escala logarítmica del modelo (objetivo
Tweedie): cada contribución φ es un factor multiplicativo exp(φ) sobre la
predicción. Además imprime las cifras que cita el texto (participaciones,
contribuciones de pico frente a resto y sus factores). Uso:

    docker compose run --rm tfg python scripts/figura_contribuciones.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd  # noqa: E402

from src.figuras import figura_contribuciones  # noqa: E402

SALIDAS = RAIZ / "salidas"
DESTINO = RAIZ / "resultados" / "figuras" / "figura06_contribuciones.png"

# Variables que el texto cita, en el orden en que las cita.
CITADAS = ["rezago_1", "media_mismo_mes", "rezago_12", "mes"]
EXOGENAS = [
    "precio_rezago_1", "precio_delta_pct", "quiebre_pct_rezago_1",
    "quiebre_pct_movil_3", "perdidas_rezago_1", "perdidas_movil_12",
    "tc_rezago_1", "tc_delta_pct",
]


def meses_de_pico() -> list[int]:
    flujo = json.loads((SALIDAS / "flujo.json").read_text("utf-8"))
    etapa = next(e for e in flujo["etapas"] if e["id"] == "explicacion")
    return [int(m) for m in etapa["decisiones"]["meses_pico"]]


def cifras(tabla: pd.DataFrame) -> None:
    datos = tabla.set_index("feature")
    print(f"{'variable':<18}{'part. %':>9}{'|φ| pico':>10}{'|φ| resto':>10}"
          f"{'φ pico':>9}{'exp(φ)':>8}")
    for clave in CITADAS:
        f = datos.loc[clave]
        print(f"{clave:<18}{f['participacion_pct']:>9.2f}"
              f"{f['contrib_abs_media_pico']:>10.3f}{f['contrib_abs_media_resto']:>10.3f}"
              f"{f['contrib_media_pico']:>+9.3f}{math.exp(f['contrib_media_pico']):>8.3f}")
    estacionales = datos.loc[["media_mismo_mes", "rezago_12", "mes"], "participacion_pct"].sum()
    exogenas = datos.loc[EXOGENAS, "participacion_pct"].sum()
    print(f"perfil estacional (media mismo mes + rezago 12 + mes): {estacionales:.2f} %")
    print(f"exógenas (precio, quiebres, pérdidas, tipo de cambio): {exogenas:.2f} %")
    print(f"suma de participaciones: {datos['participacion_pct'].sum():.2f} %")


def main() -> None:
    tabla = pd.read_csv(SALIDAS / "contribuciones_shap.csv")
    meses = meses_de_pico()
    cifras(tabla)
    ruta = figura_contribuciones(tabla, meses, DESTINO)
    print(f"figura: {ruta.relative_to(RAIZ)} (meses de pico {meses})")


if __name__ == "__main__":
    main()
