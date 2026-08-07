"""Control de confidencialidad sobre los archivos versionados.

Este repositorio se publica: el Anexo G de la tesis enlaza el código. El
histórico de ventas está cubierto por un acuerdo de confidencialidad (Anexo B), y
con él el esquema interno del sistema de origen: nombres de base, de esquema, de
tabla, direcciones e identificadores de la empresa.

Revisar eso a mano antes de cada publicación no funciona —basta un descuido una
vez—, así que se automatiza. Esta prueba falla si algo de eso aparece en un
archivo versionado.

Cómo agregar un patrón
----------------------
Si aparece un identificador interno nuevo, se agrega a ``PATRONES_PROHIBIDOS``.
Si un hallazgo es un falso positivo, se agrega a ``EXCEPCIONES`` con el motivo
escrito: una excepción sin justificación es una excepción que nadie recuerda por
qué está.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent

# Cada entrada es (nombre legible, expresión regular).
PATRONES_PROHIBIDOS: list[tuple[str, str]] = [
    ("nombre real de la empresa", r"(?i)\bmadepa\b"),
    ("nombre del grupo empresarial", r"(?i)\bla\s+papelera\b|\bpapelera\b"),
    ("nombre del sistema de pronóstico interno", r"(?i)\bthales\b"),
    ("nombre de la base de datos corporativa", r"(?i)\bbixdb\b|\bbix_v2\b|\bdsxdbodoo\b"),
    ("esquema interno", r"(?i)\bhub_thales\b|\bhub_core\b|\bbi_analytics\b"),
    ("dirección IP", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("dominio corporativo", r"(?i)[\w.-]+@[\w.-]*madepa[\w.-]*"),
    ("usuario de base de datos", r"(?i)\bnicolas_oporto\b|\bpapelera_ro\b|\bmadepa_bi\b"),
    ("regional real", r"(?i)\bSANTA\s+CRUZ\b|\bCOCHABAMBA\b|\bLA\s+PAZ\b|\bTARIJA\b"),
    ("credencial embebida", r"(?i)(password|contrase[nñ]a|secret|token)\s*[:=]\s*[\"'][^\"'{$]{6,}"),
]

# Falsos positivos justificados. Clave: (archivo, patrón) -> motivo.
EXCEPCIONES: dict[tuple[str, str], str] = {
    ("tests/test_confidencialidad.py", "nombre real de la empresa"):
        "Este archivo ES la lista de patrones; contiene los términos por definición.",
    ("tests/test_confidencialidad.py", "nombre del grupo empresarial"): "Ídem.",
    ("tests/test_confidencialidad.py", "nombre del sistema de pronóstico interno"): "Ídem.",
    ("tests/test_confidencialidad.py", "nombre de la base de datos corporativa"): "Ídem.",
    ("tests/test_confidencialidad.py", "esquema interno"): "Ídem.",
    ("tests/test_confidencialidad.py", "dirección IP"): "Ídem.",
    ("tests/test_confidencialidad.py", "dominio corporativo"): "Ídem.",
    ("tests/test_confidencialidad.py", "usuario de base de datos"): "Ídem.",
    ("tests/test_confidencialidad.py", "regional real"): "Ídem.",
    ("tests/test_confidencialidad.py", "credencial embebida"): "Ídem.",
    ("REQUERIMIENTOS.md", "regional real"):
        "La delimitación espacial del estudio es pública: la tesis declara "
        "Santa Cruz de la Sierra en su portada.",
}

# Rutas versionadas que no se revisan.
EXTENSIONES_BINARIAS = {".png", ".pdf", ".xlsx", ".ico", ".jpg", ".jpeg"}


def _archivos_versionados() -> list[Path]:
    """Lo que git realmente rastrea. Es lo único que se publicaría."""
    salida = subprocess.run(
        ["git", "-C", str(RAIZ), "ls-files"],
        capture_output=True, text=True, timeout=30,
    )
    if salida.returncode != 0:
        pytest.skip("No es un repositorio git: no hay nada que publicar todavía.")
    rutas = [RAIZ / linea for linea in salida.stdout.splitlines() if linea.strip()]
    return [
        r for r in rutas
        if r.is_file() and r.suffix.lower() not in EXTENSIONES_BINARIAS
    ]


@pytest.fixture(scope="module")
def hallazgos() -> list[tuple[str, str, int, str]]:
    """(archivo, patrón, línea, texto) de cada coincidencia no exceptuada."""
    encontrados = []
    compilados = [(nombre, re.compile(patron)) for nombre, patron in PATRONES_PROHIBIDOS]

    for ruta in _archivos_versionados():
        relativa = ruta.relative_to(RAIZ).as_posix()
        try:
            texto = ruta.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for numero, linea in enumerate(texto.splitlines(), start=1):
            for nombre, patron in compilados:
                if patron.search(linea) and (relativa, nombre) not in EXCEPCIONES:
                    encontrados.append((relativa, nombre, numero, linea.strip()[:110]))
    return encontrados


def test_ningun_identificador_interno_en_archivos_versionados(hallazgos):
    if hallazgos:
        detalle = "\n".join(
            f"  {archivo}:{numero}  [{patron}]\n      {texto}"
            for archivo, patron, numero, texto in hallazgos
        )
        pytest.fail(
            f"Se encontraron {len(hallazgos)} identificadores internos en archivos "
            f"versionados. El repositorio se publica: hay que sacarlos, o "
            f"justificarlos en EXCEPCIONES.\n\n{detalle}"
        )


def test_no_se_versiona_ningun_archivo_de_datos():
    """`datos/crudo/` NUNCA se versiona (Anexo B)."""
    versionados = [p.relative_to(RAIZ).as_posix() for p in _archivos_versionados()]
    filtrados = [
        v for v in versionados
        if (v.startswith("datos/") or v.startswith("salidas"))
        and not v.endswith(".gitkeep")
    ]
    assert not filtrados, (
        f"Hay archivos de datos o de salidas versionados: {filtrados}"
    )


def test_no_se_versiona_la_configuracion_de_extraccion():
    """El esquema de origen vive fuera del repositorio."""
    versionados = [p.relative_to(RAIZ).as_posix() for p in _archivos_versionados()]
    assert "config/extraccion.local.yaml" not in versionados, (
        "config/extraccion.local.yaml contiene el esquema interno y no puede "
        "versionarse. Está en .gitignore: si aparece aquí es porque se forzó "
        "con `git add -f`."
    )
    assert "config/extraccion.ejemplo.yaml" in versionados, (
        "Falta la plantilla: sin ella nadie puede reproducir la extracción."
    )
