"""RF-1 — Carga del archivo crudo.

Lee CSV o Excel, traduce los nombres reales de columna a los nombres canónicos
del programa y convierte los tipos, reportando cuántos valores no pudo
convertir.

Ninguna conversión fallida se rellena con un valor plausible (RN-1): las filas
inconvertibles se descartan y se cuentan, y ese conteo viaja hasta el informe de
inspección.

Las categorías de producto se seudonimizan AL CARGAR, no al extraer: la captura
conserva su nombre real (y con él el SHA-256 que sellan las corridas) y una
tabla local, ``mapeo_categoria.csv`` junto al archivo de ventas y fuera del
repositorio como el mapeo de SKU, traduce cada categoría a su seudónimo
(``Categoría A``, …) antes de cualquier otro uso. Si la tabla existe y la
captura trae una categoría sin fila, la carga se detiene: ningún nombre real
sigue vivo aguas abajo. Sin la tabla (las pruebas, un clon del repositorio
público) la columna pasa tal cual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config

COLUMNAS_CANONICAS = ("fecha", "sku", "canal", "regional", "cantidad")

EXTENSIONES_CSV = {".csv", ".txt", ".tsv"}
EXTENSIONES_EXCEL = {".xlsx", ".xlsm", ".xls"}

#: Tabla local de correspondencia de categorías, junto al archivo de ventas.
NOMBRE_MAPEO_CATEGORIA = "mapeo_categoria.csv"
COLUMNAS_MAPEO_CATEGORIA = ("categoria_real", "categoria_seudonimo")


class ErrorDeCarga(Exception):
    """El archivo de entrada no puede leerse tal como está declarado."""


def ruta_mapeo_categoria(ruta_datos: str | Path) -> Path:
    """Dónde se busca la tabla de correspondencia: junto al archivo de ventas."""
    return Path(ruta_datos).parent / NOMBRE_MAPEO_CATEGORIA


def leer_mapeo_categoria(ruta: str | Path) -> dict[str, str] | None:
    """Categoría real -> seudónimo, o ``None`` si la tabla no existe.

    La tabla se valida entera antes de usarla: una fila con un valor vacío,
    una categoría real repetida o dos categorías con el mismo seudónimo
    (que las fundiría en una sola serie de categoría) detienen la carga.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    tabla = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8")
    faltantes = [c for c in COLUMNAS_MAPEO_CATEGORIA if c not in tabla.columns]
    if faltantes:
        raise ErrorDeCarga(
            f"La tabla de correspondencia de categorías {ruta} no tiene las "
            f"columnas {faltantes}; se esperan {list(COLUMNAS_MAPEO_CATEGORIA)}."
        )
    reales = tabla["categoria_real"].str.strip()
    seudonimos = tabla["categoria_seudonimo"].str.strip()
    if (reales == "").any() or (seudonimos == "").any():
        raise ErrorDeCarga(
            f"La tabla de correspondencia de categorías {ruta} tiene filas con "
            "un valor vacío."
        )
    if reales.duplicated().any() or seudonimos.duplicated().any():
        raise ErrorDeCarga(
            f"La tabla de correspondencia de categorías {ruta} repite una "
            "categoría o un seudónimo: cada categoría tiene que tener un "
            "seudónimo propio."
        )
    mapeo = dict(zip(reales, seudonimos))
    if not conserva_el_orden(mapeo):
        raise ErrorDeCarga(
            f"La tabla de correspondencia de categorías {ruta} no conserva el "
            "orden alfabético de los nombres reales: ordenados por nombre real, "
            "los seudónimos también tienen que quedar ordenados. Si no, los "
            "códigos de la columna categórica cambian y la corrida deja de "
            "reproducir a la publicada."
        )
    return mapeo


def conserva_el_orden(mapeo: dict[str, str]) -> bool:
    """¿Ordenar por nombre real y por seudónimo da la misma permutación?

    La columna ``categoria`` pasa a ``category`` de pandas (features v2 y la
    red neuronal), cuyos códigos siguen el orden de los valores. Si el
    seudónimo respeta el orden de los nombres reales, cada categoría conserva
    su código y el modelo ve exactamente los mismos números.
    """
    por_nombre_real = [mapeo[real] for real in sorted(mapeo)]
    return por_nombre_real == sorted(mapeo.values())


def seudonimizar_categorias(
    categorias: pd.Series, mapeo: dict[str, str] | None, origen: str | Path = ""
) -> pd.Series:
    """Reemplaza cada categoría por su seudónimo; identidad sin tabla.

    Los valores nulos siguen nulos. Una categoría sin fila en la tabla es un
    error: dejarla pasar mantendría vivo un nombre real.
    """
    if mapeo is None:
        return categorias
    observadas = set(categorias.dropna().unique())
    sin_fila = sorted(str(c) for c in observadas - set(mapeo))
    if sin_fila:
        raise ErrorDeCarga(
            f"La captura trae {len(sin_fila)} categoría(s) sin fila en la tabla "
            f"de correspondencia {origen}: {sin_fila}. Agregalas a la tabla "
            "con su seudónimo antes de correr."
        )
    return categorias.map(mapeo, na_action="ignore").astype("string")


@dataclass
class InformeCarga:
    """Todo lo que hay que poder contar sobre la lectura del archivo."""

    ruta: Path
    sha256: str
    filas_leidas: int = 0
    filas_validas: int = 0
    fechas_no_convertidas: int = 0
    cantidades_no_convertidas: int = 0
    filas_con_clave_nula: int = 0
    columnas_originales: list[str] = field(default_factory=list)
    mapeo_aplicado: dict[str, str] = field(default_factory=dict)
    # Tabla de correspondencia de categorías aplicada (ruta y SHA-256, nunca
    # su contenido); None si no había tabla y la columna pasó tal cual.
    mapeo_categoria: Path | None = None
    mapeo_categoria_sha256: str | None = None

    @property
    def filas_descartadas(self) -> int:
        return self.filas_leidas - self.filas_validas

    def lineas(self) -> list[str]:
        lineas = [
            f"Archivo                        : {self.ruta}",
            f"SHA-256                        : {self.sha256}",
            f"Columnas en el archivo         : {', '.join(self.columnas_originales)}",
            f"Mapeo aplicado                 : "
            + ", ".join(f"{o} -> {c}" for c, o in self.mapeo_aplicado.items()),
        ]
        if self.mapeo_categoria is not None:
            lineas.append(
                f"Categorías seudonimizadas con  : {self.mapeo_categoria.name} "
                f"(SHA-256 {self.mapeo_categoria_sha256})"
            )
        return lineas + [
            f"Filas leídas                   : {self.filas_leidas:,}",
            f"Fechas no convertibles         : {self.fechas_no_convertidas:,}",
            f"Cantidades no convertibles     : {self.cantidades_no_convertidas:,}",
            f"Filas con sku/canal/regional nulo: {self.filas_con_clave_nula:,}",
            f"Filas descartadas              : {self.filas_descartadas:,}",
            f"Filas válidas                  : {self.filas_validas:,}",
        ]


@dataclass
class InformePoblacion:
    """La población objetivo aplicada, con todo lo excluido contado."""

    categorias_incluidas: list[str] = field(default_factory=list)
    filas_antes: int = 0
    filas_en_poblacion: int = 0
    series_fuera: int = 0
    skus_fuera: int = 0
    filas_por_categoria_excluida: dict[str, int] = field(default_factory=dict)

    @property
    def filas_fuera(self) -> int:
        return self.filas_antes - self.filas_en_poblacion

    def lineas(self) -> list[str]:
        if not self.categorias_incluidas:
            return [
                "Población objetivo: PORTAFOLIO COMPLETO (sin acotar por "
                "categoría — modo ablación).",
            ]
        salida = [
            "Población objetivo: " + ", ".join(self.categorias_incluidas),
            f"  Filas en población       : {self.filas_en_poblacion:,} de "
            f"{self.filas_antes:,}",
            f"  Fuera de población       : {self.filas_fuera:,} filas · "
            f"{self.series_fuera:,} series · {self.skus_fuera:,} SKUs",
        ]
        for categoria, filas in sorted(
            self.filas_por_categoria_excluida.items(), key=lambda kv: -kv[1]
        ):
            salida.append(f"    {categoria:<28} {filas:>9,} filas")
        return salida


def aplicar_poblacion(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, InformePoblacion]:
    """Acota el crudo a la población objetivo declarada, contando lo excluido.

    La población es una decisión de DISEÑO (qué estudia la tesis), no un
    criterio de inclusión: se aplica antes de construir las series y su
    justificación vive en config.yaml. Con la lista vacía (ablación) el filtro
    es un passthrough declarado.
    """
    informe = InformePoblacion(filas_antes=len(df))
    if cfg.poblacion is None or not cfg.poblacion.categorias:
        informe.filas_en_poblacion = len(df)
        return df, informe

    if "categoria" not in df.columns:
        raise ErrorDeCarga(
            "La población objetivo está declarada pero el archivo no trae la "
            "columna de categoría: re-extraé el snapshot con columna_categoria."
        )

    informe.categorias_incluidas = list(cfg.poblacion.categorias)
    observadas = set(df["categoria"].dropna().unique())
    inexistentes = [c for c in cfg.poblacion.categorias if c not in observadas]
    if inexistentes:
        # Un error tipográfico en una categoría descartaría en silencio una
        # parte de la población. Se falla con la lista real a la vista.
        raise ErrorDeCarga(
            f"Categorías declaradas que NO existen en el archivo: {inexistentes}. "
            f"Disponibles: {sorted(observadas)}"
        )

    dentro = df["categoria"].isin(cfg.poblacion.categorias)
    fuera = df.loc[~dentro]
    informe.filas_en_poblacion = int(dentro.sum())
    informe.series_fuera = int(
        fuera[["sku", "canal", "regional"]].drop_duplicates().shape[0]
    )
    informe.skus_fuera = int(fuera["sku"].nunique())
    informe.filas_por_categoria_excluida = {
        str(categoria): int(cuantas)
        for categoria, cuantas in fuera["categoria"].value_counts().items()
    }

    return df.loc[dentro].reset_index(drop=True), informe


def _sha256_de(ruta: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _leer_crudo(ruta: Path, cfg: Config) -> pd.DataFrame:
    """Lee el archivo sin interpretar nada: todo como texto.

    Se lee como texto a propósito. Si se dejara que el lector infiriera tipos,
    un valor inconvertible se volvería NaN en silencio y no habría forma de
    contarlo, que es justamente lo que la RF-1 exige reportar.
    """
    sufijo = ruta.suffix.lower()
    if sufijo in EXTENSIONES_CSV:
        separador = "\t" if sufijo == ".tsv" else cfg.datos.separador
        return pd.read_csv(
            ruta,
            sep=separador,
            encoding=cfg.datos.codificacion,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
        )
    if sufijo in EXTENSIONES_EXCEL:
        return pd.read_excel(ruta, dtype=str)
    raise ErrorDeCarga(
        f"Extensión '{sufijo}' no reconocida. Se admiten "
        f"{sorted(EXTENSIONES_CSV | EXTENSIONES_EXCEL)}."
    )


def cargar(cfg: Config, ruta: str | Path | None = None) -> tuple[pd.DataFrame, InformeCarga]:
    """Lee el archivo crudo y devuelve el DataFrame canónico y su informe.

    El DataFrame resultante tiene exactamente las columnas
    ``fecha`` (datetime64), ``sku``, ``canal``, ``regional`` (texto) y
    ``cantidad`` (float).
    """
    ruta = Path(ruta) if ruta is not None else cfg.ruta_datos
    if not ruta.is_absolute():
        ruta = cfg.raiz / ruta
    if not ruta.exists():
        raise ErrorDeCarga(
            f"No existe el archivo de datos: {ruta}\n"
            "Ejecutá primero la extracción:\n"
            "    python scripts/extraer_snapshot.py"
        )

    bruto = _leer_crudo(ruta, cfg)
    informe = InformeCarga(
        ruta=ruta,
        sha256=_sha256_de(ruta),
        filas_leidas=len(bruto),
        columnas_originales=list(bruto.columns),
        mapeo_aplicado=dict(cfg.datos.columnas),
    )

    # --- El mensaje de error que exige la aceptación de la RF-1 --------------
    faltantes = {
        canonica: origen
        for canonica, origen in cfg.datos.columnas.items()
        if origen not in bruto.columns
    }
    if faltantes:
        detalle = "\n".join(
            f"    - '{origen}'  (esperada para la columna canónica '{canonica}')"
            for canonica, origen in sorted(faltantes.items())
        )
        raise ErrorDeCarga(
            f"El archivo {ruta.name} no tiene todas las columnas declaradas en "
            f"config.yaml.\n"
            f"  Faltan:\n{detalle}\n"
            f"  Columnas encontradas en el archivo ({len(bruto.columns)}):\n"
            f"    {', '.join(map(str, bruto.columns))}\n"
            f"  Corregí datos.columnas en {cfg.ruta_config} o el archivo de entrada."
        )

    df = pd.DataFrame({
        canonica: bruto[cfg.datos.columnas[canonica]]
        for canonica in COLUMNAS_CANONICAS
    })
    if "categoria" in cfg.datos.columnas:
        # Seudónimo ANTES de cualquier otro uso: la población, las exógenas,
        # el calendario y las figuras sólo ven ``Categoría A``, …
        ruta_mapeo = ruta_mapeo_categoria(ruta)
        mapeo = leer_mapeo_categoria(ruta_mapeo)
        df["categoria"] = seudonimizar_categorias(
            bruto[cfg.datos.columnas["categoria"]].astype("string").str.strip(),
            mapeo, ruta_mapeo,
        )
        if mapeo is not None:
            informe.mapeo_categoria = ruta_mapeo
            informe.mapeo_categoria_sha256 = _sha256_de(ruta_mapeo)
    # Exógenas de la misma tabla (features v2). NaN = sin dato, nunca 0.
    for opcional in ("precio", "perdidas"):
        if opcional in cfg.datos.columnas:
            df[opcional] = pd.to_numeric(
                bruto[cfg.datos.columnas[opcional]], errors="coerce"
            )

    # --- Conversión de tipos, contando lo que no se pudo convertir -----------
    # Primero se intenta con formato homogéneo, que es lo rápido para cientos de
    # miles de filas. Si el archivo mezcla formatos —el análisis documental de la
    # tesis reporta nomenclaturas distintas en los años más antiguos— se reintenta
    # elemento a elemento antes de dar una fecha por perdida.
    fecha = pd.to_datetime(df["fecha"], errors="coerce")
    nulos_originales = int(df["fecha"].isna().sum())
    if int(fecha.isna().sum()) > nulos_originales:
        fecha = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
    informe.fechas_no_convertidas = int(fecha.isna().sum()) - nulos_originales

    cantidad = pd.to_numeric(df["cantidad"], errors="coerce")
    informe.cantidades_no_convertidas = int(
        cantidad.isna().sum() - df["cantidad"].isna().sum()
    )

    for columna in ("sku", "canal", "regional"):
        df[columna] = df[columna].astype("string").str.strip()

    clave_nula = df[["sku", "canal", "regional"]].isna().any(axis=1)
    informe.filas_con_clave_nula = int(clave_nula.sum())

    df["fecha"] = fecha
    df["cantidad"] = cantidad

    valido = fecha.notna() & cantidad.notna() & ~clave_nula
    df = df.loc[valido].reset_index(drop=True)
    informe.filas_validas = len(df)

    if informe.filas_validas == 0:
        raise ErrorDeCarga(
            f"Ninguna de las {informe.filas_leidas:,} filas de {ruta.name} sobrevivió "
            "a la conversión de tipos. Revisá el formato de fecha y el separador "
            "decimal declarados en config.yaml."
        )

    return df, informe
