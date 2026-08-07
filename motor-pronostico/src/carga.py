"""RF-1 — Carga del archivo crudo.

Lee CSV o Excel, traduce los nombres reales de columna a los nombres canónicos
del programa y convierte los tipos, reportando cuántos valores no pudo
convertir.

Ninguna conversión fallida se rellena con un valor plausible (RN-1): las filas
inconvertibles se descartan y se cuentan, y ese conteo viaja hasta el informe de
inspección.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config

COLUMNAS_CANONICAS = ("fecha", "sku", "canal", "regional", "cantidad")

EXTENSIONES_CSV = {".csv", ".txt", ".tsv"}
EXTENSIONES_EXCEL = {".xlsx", ".xlsm", ".xls"}


class ErrorDeCarga(Exception):
    """El archivo de entrada no puede leerse tal como está declarado."""


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

    @property
    def filas_descartadas(self) -> int:
        return self.filas_leidas - self.filas_validas

    def lineas(self) -> list[str]:
        return [
            f"Archivo                        : {self.ruta}",
            f"SHA-256                        : {self.sha256}",
            f"Columnas en el archivo         : {', '.join(self.columnas_originales)}",
            f"Mapeo aplicado                 : "
            + ", ".join(f"{o} -> {c}" for c, o in self.mapeo_aplicado.items()),
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
        df["categoria"] = (
            bruto[cfg.datos.columnas["categoria"]].astype("string").str.strip()
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
