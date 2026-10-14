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
Los patrones GENÉRICOS (direcciones, credenciales, claves) viven en
``PATRONES_PROHIBIDOS``. Los identificadores CONCRETOS de la empresa (su nombre
real, el de su sistema de origen, bases, esquemas, usuarios, dominio) no pueden
escribirse en este archivo, porque el repositorio se publica y la lista los
revelaría: viven en ``config/confidencialidad.local.txt``, fuera del control de
versiones, una expresión regular por línea (``nombre legible<TAB>expresión``).
La prueba los carga cuando el archivo existe y avisa cuando no.
Si un hallazgo es un falso positivo, se agrega a ``EXCEPCIONES`` con el motivo
escrito: una excepción sin justificación es una excepción que nadie recuerda por
qué está.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

PROYECTO = Path(__file__).resolve().parent.parent


def _raiz_del_repositorio() -> Path:
    """El repositorio de la tesis, que contiene este proyecto y el documento.

    El control tiene que mirar TODO lo versionado, no solo la carpeta del
    código: el documento, los anexos y cualquier archivo que alguien deje en la
    raíz se publican igual.
    """
    for candidato in [PROYECTO, *PROYECTO.parents]:
        if (candidato / ".git").exists():
            return candidato
    return PROYECTO


RAIZ = _raiz_del_repositorio()
# Prefijo de la carpeta del proyecto dentro del repositorio, para escribir las
# excepciones sin atarlas al nombre de la carpeta.
SUBCARPETA = (
    f"{PROYECTO.relative_to(RAIZ).as_posix()}/" if PROYECTO != RAIZ else ""
)

# Cada entrada es (nombre legible, expresión regular). Sólo patrones genéricos:
# los identificadores concretos de la empresa están en la lista local.
PATRONES_PROHIBIDOS: list[tuple[str, str]] = [
    # 0.0.0.0 (escucha universal) y 127.* (loopback) no revelan nada: son las
    # direcciones de cualquier Dockerfile del mundo. El patrón caza las demás.
    ("dirección IP", r"\b(?!0\.0\.0\.0\b)(?!127\.)(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("regional real", r"(?i)\bSANTA\s+CRUZ\b|\bCOCHABAMBA\b|\bLA\s+PAZ\b|\bTARIJA\b"),
    ("credencial embebida", r"(?i)(password|contrase[nñ]a|secret|token)\s*[:=]\s*[\"'][^\"'{$]{6,}"),
    # Un token suelto en un archivo no tiene la forma `token: "..."`, así que el
    # patrón de arriba no lo ve. Estos buscan la CREDENCIAL en sí, esté donde
    # esté. Se agregaron después de que un token clásico de GitHub terminara
    # commiteado en la raíz del repositorio.
    ("token de GitHub", r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    ("clave de acceso AWS", r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("clave privada", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("cadena de conexión con contraseña", r"(?i)\b\w+://[^\s:@/]+:[^\s:@/]+@"),
]

# Identificadores concretos de la empresa: lista local, fuera del repositorio.
LISTA_LOCAL = PROYECTO / "config" / "confidencialidad.local.txt"


def _patrones_locales() -> list[tuple[str, str]]:
    """(nombre legible, expresión) por línea de la lista local; vacía si no existe."""
    if not LISTA_LOCAL.exists():
        return []
    patrones = []
    for linea in LISTA_LOCAL.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        nombre, _, patron = linea.partition("\t")
        if nombre.strip() and patron.strip():
            patrones.append((nombre.strip(), patron.strip()))
    return patrones


PATRONES_LOCALES = _patrones_locales()

# Archivos que por su NOMBRE parecen guardar una credencial. Ninguno de estos
# debería estar versionado jamás, tenga dentro lo que tenga.
NOMBRES_SOSPECHOSOS = re.compile(
    r"(?i)(^|/)(token[^/]*|[^/]*\.(token|pem|key|p12|pfx)|[^/]*_secret[^/]*"
    r"|credentials?[^/]*|\.env(\..*)?)$"
)

# Falsos positivos justificados. Clave: (archivo, patrón) -> motivo.
# Este archivo contiene todos los términos por definición: es la lista.
_YO = f"{SUBCARPETA}tests/test_confidencialidad.py"
EXCEPCIONES: dict[tuple[str, str], str] = {
    (_YO, nombre): "Este archivo ES la lista de patrones; los contiene por definición."
    for nombre, _ in PATRONES_PROHIBIDOS
}
_UBICACION_PUBLICA = (
    "La delimitación espacial del estudio es pública: figura en la portada de la "
    "tesis y es la sede de la universidad. No identifica a la empresa."
)
EXCEPCIONES.update({
    ("README.md", "regional real"): _UBICACION_PUBLICA,
    (f"{SUBCARPETA}REQUERIMIENTOS.md", "regional real"): _UBICACION_PUBLICA,
    (f"{SUBCARPETA}README.md", "regional real"): _UBICACION_PUBLICA,
})

# Rutas versionadas que no se revisan. Los documentos ofimáticos son binarios
# comprimidos: el control NO los inspecciona, y esa limitación se declara en el
# README. Por eso .docx, .xlsx, .pdf, .pptx y .zip no pueden versionarse (lo
# exige test_no_se_versiona_ningun_documento_ofimatico_ni_comprimido); las
# imágenes sí, y lo que entre en `documento/` hay que revisarlo a ojo.
EXTENSIONES_BINARIAS = {
    ".png", ".pdf", ".xlsx", ".ico", ".jpg", ".jpeg",
    ".docx", ".doc", ".pptx", ".ppt", ".odt", ".zip",
}


def _archivos_versionados() -> list[Path]:
    """Lo que git realmente rastrea EN TODO EL REPOSITORIO.

    Es lo único que se publicaría, y por eso se consulta desde la raíz del
    repositorio y no desde la carpeta del proyecto: el documento y los anexos
    viven fuera de ella.
    """
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
    compilados = [
        (nombre, re.compile(patron))
        for nombre, patron in PATRONES_PROHIBIDOS + PATRONES_LOCALES
    ]

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


def test_la_lista_local_de_identificadores_esta_cargada():
    """Sin la lista local, el control de identificadores concretos no corre.

    No es una falla: en un clon ajeno la lista no existe por diseño. Se marca
    como omitida para que quede a la vista en la salida de pytest.
    """
    if not PATRONES_LOCALES:
        pytest.skip(
            "config/confidencialidad.local.txt no está presente: sólo corrieron los "
            "patrones genéricos."
        )
    assert len(PATRONES_LOCALES) >= 5


def test_ningun_archivo_versionado_parece_una_credencial():
    """Por el NOMBRE, antes de mirar el contenido.

    Se agregó después de que un `token.txt` con un token clásico de GitHub
    terminara commiteado en la raíz del repositorio: el control de contenido de
    entonces buscaba `token: "..."` como asignación y no veía un archivo suelto.
    """
    sospechosos = [
        p.relative_to(RAIZ).as_posix()
        for p in _archivos_versionados()
        if NOMBRES_SOSPECHOSOS.search(p.relative_to(RAIZ).as_posix())
    ]
    assert not sospechosos, (
        f"Hay archivos versionados cuyo nombre indica que guardan una credencial: "
        f"{sospechosos}. Sacalos del repositorio Y considerá la credencial quemada: "
        f"queda en los objetos de git aunque se borre el archivo."
    )


def test_no_se_versiona_ningun_archivo_de_datos():
    """`datos/crudo/` NUNCA se versiona (Anexo B)."""
    versionados = [p.relative_to(RAIZ).as_posix() for p in _archivos_versionados()]
    filtrados = [
        v for v in versionados
        if (v.startswith(f"{SUBCARPETA}datos/") or v.startswith(f"{SUBCARPETA}salidas"))
        and not v.endswith(".gitkeep")
    ]
    assert not filtrados, (
        f"Hay archivos de datos o de salidas versionados: {filtrados}"
    )


def test_no_se_versiona_la_configuracion_de_extraccion():
    """El esquema de origen vive fuera del repositorio."""
    versionados = [p.relative_to(RAIZ).as_posix() for p in _archivos_versionados()]
    assert f"{SUBCARPETA}config/extraccion.local.yaml" not in versionados, (
        "config/extraccion.local.yaml contiene el esquema interno y no puede "
        "versionarse. Está en .gitignore: si aparece aquí es porque se forzó "
        "con `git add -f`."
    )
    assert f"{SUBCARPETA}config/extraccion.ejemplo.yaml" in versionados, (
        "Falta la plantilla: sin ella nadie puede reproducir la extracción."
    )


#: Documentos ofimáticos y comprimidos: el control de contenido no puede mirar
#: adentro (son binarios), así que no se versionan nunca. La tesis y las
#: observaciones del tribunal viven en carpetas ignoradas.
EXTENSIONES_PROHIBIDAS = {".docx", ".xlsx", ".pdf", ".pptx", ".zip"}


def test_no_se_versiona_ningun_documento_ofimatico_ni_comprimido():
    """Ningún .docx, .xlsx, .pdf, .pptx ni .zip versionado en el repositorio.

    El control de identificadores lee texto: un binario de Office o un
    comprimido podría llevar datos o nombres de la empresa sin que ninguna
    prueba lo vea. En lugar de declarar la limitación, se prohíbe versionarlos.
    """
    salida = subprocess.run(
        ["git", "-C", str(RAIZ), "ls-files"],
        capture_output=True, text=True, timeout=30,
    )
    if salida.returncode != 0:
        pytest.skip("No es un repositorio git: no hay nada que publicar todavía.")
    prohibidos = [
        linea for linea in salida.stdout.splitlines()
        if Path(linea).suffix.lower() in EXTENSIONES_PROHIBIDAS
    ]
    assert not prohibidos, (
        f"Hay documentos ofimáticos o comprimidos versionados: {prohibidos}. "
        f"El control de confidencialidad no puede revisar su contenido; "
        f"sacalos del repositorio (git rm --cached) y dejalos en una carpeta "
        f"ignorada."
    )
