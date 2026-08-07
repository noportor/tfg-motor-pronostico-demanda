"""Prueba de humo del tablero: ejecuta cada página de verdad.

    docker compose run --rm tablero python tablero/humo.py

Corre en la imagen del TABLERO (la del pipeline no tiene streamlit, a
propósito). Usa el framework de testing de Streamlit para ejecutar ``app.py`` y
cada página contra las corridas reales de ``salidas*/``, y falla si alguna
levanta una excepción. Es lo que la compilación no puede ver: un KeyError sobre
el manifiesto, un encoding de Altair mal armado, una columna que ya no existe.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROYECTO = Path(__file__).resolve().parent.parent
if str(PROYECTO) not in sys.path:
    sys.path.insert(0, str(PROYECTO))

from streamlit.testing.v1 import AppTest  # noqa: E402

from tablero import lectores  # noqa: E402


def main() -> int:
    corridas = lectores.descubrir_corridas()
    if not corridas:
        print("No hay corridas en salidas*/ — nada que probar.")
        return 1

    fallas = 0

    # --- app.py (descubrimiento + navegación) -------------------------------
    # Se ejecuta DOS veces: con ruta absoluta y con la ruta relativa con la que
    # el servidor real lo lanza (`streamlit run tablero/app.py`). No es
    # redundancia: con ruta relativa __file__ queda relativo al CWD y la
    # resolución de páginas cambia — así se escapó un bug que el humo absoluto
    # no veía.
    for etiqueta, ruta_app in (
        ("app.py (absoluta)", str(PROYECTO / "tablero" / "app.py")),
        ("app.py (relativa, como el servidor)", "tablero/app.py"),
    ):
        aplicacion = AppTest.from_file(ruta_app)
        aplicacion.run(timeout=60)
        if aplicacion.exception:
            fallas += 1
            print(f"FALLA  {etiqueta}")
            for excepcion in aplicacion.exception:
                print("      ", excepcion.value)
        else:
            print(f"OK     {etiqueta}")

    # --- cada página, con la corrida principal en el estado -----------------
    paginas = sorted((PROYECTO / "tablero" / "paginas").glob("[0-9]*_*.py"))
    for pagina in paginas:
        prueba = AppTest.from_file(str(pagina))
        prueba.session_state["corrida"] = corridas[0]
        prueba.run(timeout=60)
        if prueba.exception:
            fallas += 1
            print(f"FALLA  {pagina.name}")
            for excepcion in prueba.exception:
                print("      ", excepcion.value)
        else:
            print(f"OK     {pagina.name}")

    total = len(paginas) + 2  # las páginas + las dos ejecuciones de app.py
    print(f"\n{total - fallas} de {total} sin excepciones.")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
