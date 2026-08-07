"""Fuentes exógenas de las features v2 del brazo de aprendizaje automático.

Cuatro señales que los modelos clásicos no pueden ver, cada una con su fuente y
su cobertura DECLARADA (ninguna ausencia es silenciosa):

- **Precio mensual** y **venta perdida**: vienen en el propio snapshot (mismas
  filas que la venta). La venta perdida es 0 antes de 2020 por definición de la
  fuente — se declara y queda como está, nunca se inventa.
- **Quiebres de stock**: ``datos/crudo/quiebres.csv`` (semanas con existencia
  cero por sku-regional-mes, mensualizadas en la extracción). La fuente empieza
  en 2022-04: los meses anteriores son NaN — «no hay historia» es información.
- **Tipo de cambio**: ``datos/crudo/tipo_cambio.csv`` (mensual, régimen
  variable). Los meses anteriores al primer dato se completan con el régimen
  FIJO documentado (``features.tc_regimen_fijo``): no es imputación, es el tipo
  de cambio oficial constante de esa era.

El módulo solo CARGA y alinea; la conversión a features causales (rezagos,
móviles) vive en ``src/features.py``, donde la prueba de no-fuga la vigila.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Config


class ErrorDeExogenas(Exception):
    """Una fuente declarada en la configuración no está disponible."""


@dataclass
class InformeExogenas:
    series_con_precio: int = 0
    filas_con_perdidas: int = 0
    filas_quiebres: int = 0
    meses_quiebres: tuple[str, str] | None = None
    meses_tc_fuente: int = 0
    meses_tc_regimen_fijo: int = 0
    meses_tc_cola_arrastrada: int = 0
    lineas_salida: list[str] = field(default_factory=list)

    def lineas(self) -> list[str]:
        return self.lineas_salida or ["Sin fuentes exógenas activas."]


@dataclass
class Exogenas:
    """Las fuentes alineadas al panel. ``None`` = grupo de features apagado."""

    precio: pd.DataFrame | None          # serie, periodo, precio
    perdidas: pd.DataFrame | None        # serie, periodo, perdidas
    quiebres: pd.DataFrame | None        # sku, regional, periodo, semanas_*
    tipo_cambio: pd.Series | None        # index periodo -> tc (período COMPLETO)
    categoria_por_serie: pd.Series | None
    informe: InformeExogenas = field(default_factory=InformeExogenas)


def _serie_de(df: pd.DataFrame) -> pd.Series:
    return (df["sku"].astype(str) + "|" + df["canal"].astype(str) + "|"
            + df["regional"].astype(str))


def cargar_exogenas(cfg: Config, df: pd.DataFrame) -> Exogenas:
    """Carga las fuentes que los flags de ``features.v2`` piden.

    Args:
        df: el crudo canónico YA acotado a la población objetivo (las exógenas
            de series fuera de la población no le sirven a nadie).

    Un flag encendido cuya fuente falta es un ERROR explícito: apagarlo por
    ausencia silenciosa dejaría al modelo sin una feature declarada y al
    investigador sin enterarse.
    """
    v2 = cfg.features.v2
    informe = InformeExogenas()
    salida = Exogenas(precio=None, perdidas=None, quiebres=None,
                      tipo_cambio=None, categoria_por_serie=None,
                      informe=informe)

    trabajo = df.copy()
    trabajo["periodo"] = trabajo["fecha"].dt.to_period("M")
    trabajo["serie"] = _serie_de(trabajo)

    if v2.get("categoria"):
        if "categoria" not in trabajo.columns:
            raise ErrorDeExogenas(
                "features.v2.categoria está encendida pero el snapshot no trae "
                "la columna 'categoria' (re-extraer con columna_categoria)."
            )
        # Atributo ESTÁTICO as-of-snapshot (declarado): si una serie cambiara
        # de categoría en el tiempo, vale la última observada — determinista
        # por construcción (orden por período), nunca por orden del archivo.
        salida.categoria_por_serie = (
            trabajo.sort_values(["serie", "periodo"], kind="stable")
            .drop_duplicates("serie", keep="last")
            .set_index("serie")["categoria"]
        )

    for clave in ("precio", "perdidas"):
        if not v2.get(clave):
            continue
        if clave not in trabajo.columns:
            raise ErrorDeExogenas(
                f"features.v2.{clave} está encendida pero el snapshot no trae "
                f"la columna '{clave}' (re-extraer con la columna declarada)."
            )
        # El origen trae una fila por serie-mes; el promedio es inocuo si
        # hubiera duplicados y determinista siempre.
        tabla = (
            trabajo.groupby(["serie", "periodo"], observed=True)[clave]
            .mean().reset_index()
        )
        setattr(salida, clave, tabla)

    if salida.precio is not None:
        informe.series_con_precio = int(
            salida.precio.loc[salida.precio["precio"] > 0, "serie"].nunique()
        )
        informe.lineas_salida.append(
            f"Precio mensual                : {informe.series_con_precio:,} series "
            f"con precio positivo"
        )
    if salida.perdidas is not None:
        informe.filas_con_perdidas = int((salida.perdidas["perdidas"] > 0).sum())
        informe.lineas_salida.append(
            f"Venta perdida                 : {informe.filas_con_perdidas:,} "
            f"celdas serie-mes con pérdida (0 antes de 2020 por definición)"
        )

    if v2.get("quiebres"):
        if not cfg.features.archivo_quiebres:
            raise ErrorDeExogenas(
                "features.v2.quiebres está encendida sin features.archivo_quiebres."
            )
        ruta = cfg.raiz / cfg.features.archivo_quiebres
        if not ruta.exists():
            raise ErrorDeExogenas(
                f"No existe {ruta}. Re-extraé el snapshot con la sección "
                f"'inventario' en extraccion.local.yaml."
            )
        quiebres = pd.read_csv(ruta)
        quiebres["periodo"] = pd.to_datetime(quiebres["fecha"]).dt.to_period("M")
        quiebres = quiebres.drop(columns="fecha")
        # Claves normalizadas al MISMO dtype que el panel canónico: un dtype
        # inferido distinto haría fallar el merge en silencio (todo NaN) y la
        # feature moriría sin que nadie lo note.
        for clave in ("sku", "regional"):
            quiebres[clave] = quiebres[clave].astype("string").str.strip()
        for columna in ("semanas_medidas", "semanas_en_cero"):
            quiebres[columna] = pd.to_numeric(quiebres[columna], errors="raise")
        duplicados = quiebres.duplicated(["sku", "regional", "periodo"]).sum()
        if duplicados:
            raise ErrorDeExogenas(
                f"quiebres.csv tiene {duplicados} claves (sku, regional, mes) "
                "duplicadas: un merge con duplicados multiplicaría filas del "
                "panel. Revisá la extracción."
            )
        salida.quiebres = quiebres
        informe.filas_quiebres = len(quiebres)
        informe.meses_quiebres = (
            str(quiebres["periodo"].min()), str(quiebres["periodo"].max())
        )
        informe.lineas_salida.append(
            f"Quiebres de stock             : {len(quiebres):,} filas "
            f"sku-regional-mes ({informe.meses_quiebres[0]} .. "
            f"{informe.meses_quiebres[1]}; antes: NaN declarado)"
        )

    if v2.get("tipo_cambio"):
        if not cfg.features.archivo_tipo_cambio:
            raise ErrorDeExogenas(
                "features.v2.tipo_cambio está encendida sin "
                "features.archivo_tipo_cambio."
            )
        ruta = cfg.raiz / cfg.features.archivo_tipo_cambio
        if not ruta.exists():
            raise ErrorDeExogenas(
                f"No existe {ruta}. Re-extraé el snapshot con la sección "
                f"'tipo_cambio' en extraccion.local.yaml."
            )
        fuente = pd.read_csv(ruta)
        fuente["periodo"] = pd.to_datetime(fuente["fecha"]).dt.to_period("M")
        if fuente["periodo"].duplicated().any():
            raise ErrorDeExogenas(
                "tipo_cambio.csv tiene meses duplicados — revisá la extracción."
            )
        fuente = fuente.set_index("periodo")["tc"].astype(float).sort_index()

        # Serie COMPLETA del período: régimen fijo documentado antes del primer
        # dato; huecos intermedios y meses de COLA se arrastran del último
        # conocido — CONTADOS y declarados, jamás en silencio (una cola
        # arrastrada en plena era variable diría «sin variación cambiaria»
        # y eso el lector tiene derecho a verlo).
        todos = pd.period_range(cfg.periodo.inicio, cfg.periodo.fin, freq="M")
        tc = fuente.reindex(todos)
        previos = int(tc.loc[tc.index < fuente.index.min()].isna().sum())
        cola = int(tc.loc[tc.index > fuente.index.max()].isna().sum())
        tc.loc[tc.index < fuente.index.min()] = cfg.features.tc_regimen_fijo
        tc = tc.ffill()
        if tc.isna().any():
            raise ErrorDeExogenas(
                "El tipo de cambio quedó con huecos al inicio del período que "
                "el régimen fijo no cubre — revisá la fuente."
            )
        salida.tipo_cambio = tc
        informe.meses_tc_fuente = len(fuente)
        informe.meses_tc_regimen_fijo = previos
        informe.meses_tc_cola_arrastrada = cola
        informe.lineas_salida.append(
            f"Tipo de cambio                : {len(fuente)} meses de fuente + "
            f"{previos} meses al régimen fijo {cfg.features.tc_regimen_fijo}"
            + (f" + {cola} meses de COLA arrastrados del último dato (¡mirar!)"
               if cola else "")
        )

    return salida
