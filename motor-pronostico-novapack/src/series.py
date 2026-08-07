"""RF-3 — Construcción de las series mensuales por SKU–canal–regional.

Tres decisiones metodológicas viven aquí, y las tres se declaran en la memoria:

1. **Agregación mensual.** El grano de análisis es la combinación
   SKU–canal–regional, que es la unidad sobre la que NOVAPACK planifica.

2. **Relleno de ceros acotado a la vida de la serie.** Los meses sin registro se
   completan con cero SOLO entre la primera y la última venta. No se extrapola
   hacia atrás: un mes anterior al lanzamiento del producto no es demanda cero,
   es ausencia de producto, y tratarlo como cero inventaría demanda que nunca
   existió y sesgaría a la baja todas las medias móviles del arranque.

3. **Nacimiento y muerte de la serie = primer y último mes con venta positiva.**
   Un mes con solo devoluciones no marca el lanzamiento de un producto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config

CLAVE_SERIE = ("sku", "canal", "regional")


@dataclass
class InformeSeries:
    """Conteos de la construcción, para el informe de inspección."""

    filas_entrada: int = 0
    filas_negativas: int = 0
    unidades_negativas: float = 0.0
    tratamiento_devoluciones: str = ""
    filas_fuera_de_periodo: int = 0
    combinaciones_totales: int = 0
    combinaciones_sin_venta_positiva: int = 0
    series_construidas: int = 0
    filas_panel: int = 0
    meses_rellenados_con_cero: int = 0
    rango: tuple[str, str] = ("", "")
    detalle: dict = field(default_factory=dict)

    def lineas(self) -> list[str]:
        return [
            f"Filas de entrada                 : {self.filas_entrada:,}",
            f"Tratamiento de devoluciones      : {self.tratamiento_devoluciones}",
            f"Filas con cantidad negativa      : {self.filas_negativas:,} "
            f"({self.unidades_negativas:,.0f} unidades)",
            f"Filas fuera del período declarado: {self.filas_fuera_de_periodo:,}",
            f"Combinaciones SKU–canal–regional : {self.combinaciones_totales:,}",
            f"  sin ninguna venta positiva     : {self.combinaciones_sin_venta_positiva:,} "
            f"(descartadas: no tienen fecha de nacimiento)",
            f"Series construidas               : {self.series_construidas:,}",
            f"Filas del panel mensual          : {self.filas_panel:,}",
            f"  de ellas, ceros intercalados   : {self.meses_rellenados_con_cero:,}",
            f"Rango del panel                  : {self.rango[0]} .. {self.rango[1]}",
        ]


def _indice_mensual(periodos: pd.PeriodIndex | pd.Series) -> np.ndarray:
    """Period('2017-04') -> 24208. Entero monótono para construir rangos."""
    valores = pd.PeriodIndex(periodos)
    return valores.year.to_numpy() * 12 + (valores.month.to_numpy() - 1)


def _periodo_desde_indice(indices: np.ndarray) -> pd.PeriodIndex:
    """24208 -> Period('2017-04'). Vectorizado: el panel tiene cientos de miles
    de filas y construirlo elemento a elemento sería inaceptablemente lento."""
    fechas = pd.to_datetime({
        "year": indices // 12,
        "month": indices % 12 + 1,
        "day": np.ones(len(indices), dtype=np.int64),
    })
    return pd.DatetimeIndex(fechas).to_period("M")


def construir_series(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, InformeSeries]:
    """Devuelve el panel largo mensual y el informe de construcción.

    El panel largo tiene las columnas ``serie``, ``sku``, ``canal``,
    ``regional``, ``periodo`` (``pd.Period`` mensual) y ``y`` (float).
    """
    informe = InformeSeries(
        filas_entrada=len(df),
        tratamiento_devoluciones=cfg.series.devoluciones,
    )

    trabajo = df.copy()
    trabajo["periodo"] = trabajo["fecha"].dt.to_period("M")

    # --- Devoluciones -------------------------------------------------------
    negativas = trabajo["cantidad"] < 0
    informe.filas_negativas = int(negativas.sum())
    informe.unidades_negativas = float(trabajo.loc[negativas, "cantidad"].sum())
    if cfg.series.devoluciones == "descartar":
        trabajo = trabajo.loc[~negativas]
    # con 'netear' las filas negativas se conservan y se suman algebraicamente
    # en la agregación mensual que viene a continuación.

    # --- Recorte al período declarado --------------------------------------
    dentro = (trabajo["periodo"] >= cfg.periodo.inicio) & (
        trabajo["periodo"] <= cfg.periodo.fin
    )
    informe.filas_fuera_de_periodo = int((~dentro).sum())
    trabajo = trabajo.loc[dentro]

    if trabajo.empty:
        raise ValueError(
            f"No queda ninguna fila dentro del período declarado "
            f"({cfg.periodo.inicio} .. {cfg.periodo.fin}). Revisá periodo en config.yaml."
        )

    # --- Agregación mensual -------------------------------------------------
    agregado = (
        trabajo.groupby([*CLAVE_SERIE, "periodo"], observed=True, dropna=False)["cantidad"]
        .sum()
        .reset_index()
        .rename(columns={"cantidad": "y"})
    )
    agregado["serie"] = (
        agregado["sku"].astype(str) + "|"
        + agregado["canal"].astype(str) + "|"
        + agregado["regional"].astype(str)
    )
    informe.combinaciones_totales = int(agregado["serie"].nunique())

    # --- Nacimiento y muerte: primer y último mes con venta POSITIVA --------
    positivas = agregado.loc[agregado["y"] > 0]
    if positivas.empty:
        raise ValueError(
            "Ninguna combinación tiene una venta positiva; no hay series que construir."
        )

    idx_positivas = _indice_mensual(positivas["periodo"])
    limites = (
        pd.DataFrame({"serie": positivas["serie"].to_numpy(), "idx": idx_positivas})
        .groupby("serie", observed=True)["idx"]
        .agg(["min", "max"])
    )
    informe.combinaciones_sin_venta_positiva = (
        informe.combinaciones_totales - len(limites)
    )

    # --- Rejilla mensual, acotada a la vida de cada serie --------------------
    if cfg.series.rellenar_ceros_intermedios:
        largos = (limites["max"] - limites["min"] + 1).to_numpy()
        series_repetidas = np.repeat(limites.index.to_numpy(), largos)
        desplazamientos = np.concatenate(
            [np.arange(n, dtype=np.int64) for n in largos]
        ) if len(largos) else np.array([], dtype=np.int64)
        idx_rejilla = np.repeat(limites["min"].to_numpy(), largos) + desplazamientos

        rejilla = pd.DataFrame({"serie": series_repetidas, "idx": idx_rejilla})
    else:
        # Sin relleno: el panel son exactamente los meses observados.
        rejilla = pd.DataFrame({
            "serie": agregado["serie"].to_numpy(),
            "idx": _indice_mensual(agregado["periodo"]),
        }).drop_duplicates()

    observado = pd.DataFrame({
        "serie": agregado["serie"].to_numpy(),
        "idx": _indice_mensual(agregado["periodo"]),
        "y": agregado["y"].to_numpy(dtype=float),
    })

    panel = rejilla.merge(observado, on=["serie", "idx"], how="left")
    informe.meses_rellenados_con_cero = int(panel["y"].isna().sum())
    panel["y"] = panel["y"].fillna(0.0)

    # Etiquetas de la serie: se recuperan del agregado, no se re-parsean del
    # identificador (un SKU podría contener el separador).
    etiquetas = (
        agregado[["serie", *CLAVE_SERIE]]
        .drop_duplicates(subset="serie")
        .set_index("serie")
    )
    panel = panel.join(etiquetas, on="serie")
    panel["periodo"] = _periodo_desde_indice(panel["idx"].to_numpy())
    panel = (
        panel[["serie", *CLAVE_SERIE, "periodo", "y"]]
        .sort_values(["serie", "periodo"], kind="stable")
        .reset_index(drop=True)
    )

    informe.series_construidas = int(panel["serie"].nunique())
    informe.filas_panel = len(panel)
    informe.rango = (str(panel["periodo"].min()), str(panel["periodo"].max()))

    return panel, informe


def a_panel_ancho(largo: pd.DataFrame) -> pd.DataFrame:
    """Panel largo -> matriz (índice = mes, columnas = serie).

    Los meses anteriores al nacimiento y posteriores a la última venta quedan
    como ``NaN``, NO como cero: la diferencia importa y es la razón de ser de la
    RF-3. Cualquier modelo que consuma esta matriz debe tratar ``NaN`` como
    «la serie no existía», nunca como «vendió cero».
    """
    ancho = largo.pivot(index="periodo", columns="serie", values="y")
    indice = pd.period_range(ancho.index.min(), ancho.index.max(), freq="M")
    ancho = ancho.reindex(indice)
    # Orden de columnas fijado: sin esto, el desempate del motor y el orden de
    # las filas de LightGBM dependerían del orden de un diccionario (RN-5).
    return ancho.reindex(sorted(ancho.columns), axis=1)
