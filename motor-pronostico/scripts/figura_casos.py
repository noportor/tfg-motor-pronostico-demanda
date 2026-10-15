"""Figura de los casos representativos por régimen (Figura 13 del documento) desde la corrida publicada.

Vuelve a dibujar la figura sin reentrenar nada: lee los insumos que la corrida
principal persistió en ``salidas/insumos_figuras.pkl`` (el panel de ventas
observadas de los cuatro casos, los pronósticos de los brazos dibujados, los
casos elegidos por regla y los inicios de validación y de prueba) y llama a
``src.figuras.figura_casos``, la misma función que usa ``main.py``. La serie
observada se rotula «Ventas observadas»: es la venta registrada, no la demanda
real, que no se observa cuando hay quiebres de inventario. Antes de dibujar,
verifica el SHA-256 de los insumos contra el ``manifiesto.json`` de la misma
corrida. Uso:

    docker compose run --rm tfg python scripts/figura_casos.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.figuras import figura_casos  # noqa: E402

SALIDAS = RAIZ / "salidas"
INSUMOS = SALIDAS / "insumos_figuras.pkl"
DESTINO = RAIZ / "resultados" / "figuras" / "figura13_casos.png"


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def verificar_insumos() -> str:
    """Comprueba que el pkl sea el que selló el manifiesto de la corrida."""
    manifiesto = json.loads((SALIDAS / "manifiesto.json").read_text("utf-8"))
    sellos = [a for a in manifiesto.get("salidas", [])
              if isinstance(a, dict) and Path(str(a.get("archivo", ""))).name == INSUMOS.name]
    actual = sha256(INSUMOS)
    if not sellos:
        raise SystemExit(f"El manifiesto de {SALIDAS.name}/ no sella {INSUMOS.name}.")
    esperado = sellos[0].get("sha256")
    if esperado != actual:
        raise SystemExit(f"{INSUMOS.name} no coincide con el manifiesto: {actual} frente a {esperado}.")
    return actual


def main() -> None:
    huella = verificar_insumos()
    with INSUMOS.open("rb") as archivo:
        insumos = pickle.load(archivo)
    ruta = figura_casos(
        insumos["panel_real_casos"], insumos["predicciones_casos"],
        insumos["casos"], insumos["inicio_validacion"], insumos["inicio_prueba"],
        DESTINO, dpi=int(insumos.get("dpi", 300)), desde=insumos["desde_casos"],
    )
    print(f"insumos: {INSUMOS.relative_to(RAIZ)} (SHA-256 {huella[:12]}, verificado contra el manifiesto)")
    for serie, etiqueta in insumos["casos"]:
        print(f"  caso: {etiqueta.splitlines()[0]}")
    print(f"figura: {ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
