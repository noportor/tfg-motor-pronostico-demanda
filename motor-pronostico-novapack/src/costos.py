"""Maestro de costos unitarios por serie — la entrada de las métricas valorizadas.

Por qué existe
--------------
Las unidades de medida difieren ENTRE SKUs (cada producto se cuenta en la suya),
de modo que ninguna suma de unidades entre productos tiene sentido físico. La
única agregación legítima entre SKUs es en dinero: ``|error| × costo unitario``.
Sobre eso se construye la suite valorizada y la métrica de decisión
``D = WMAPE_val + |Bias_val|``.

El costo es CONSTANTE por serie en la fuente (un valor puntual del último corte,
no una serie histórica de costos). Ventaja metodológica: no mezcla error de
pronóstico con deriva de precios. Limitación a declarar: valoriza todo el
período con el costo del corte.

El archivo ``datos/crudo/costos.csv`` lo escribe la extracción (mismo mapa de
seudónimos que las ventas) y NUNCA se versiona: el costo es información sensible
de la empresa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config


@dataclass
class MaestroCostos:
    """Costo unitario por serie, con su cobertura declarada."""

    costo_por_serie: pd.Series          # índice = serie ("SKU|CANAL|REGIONAL")
    ruta: Path | None

    @property
    def disponible(self) -> bool:
        return self.ruta is not None and not self.costo_por_serie.empty

    def cobertura_sobre(self, series: pd.Index) -> float:
        """Proporción de las series dadas que tienen costo."""
        if not self.disponible or len(series) == 0:
            return 0.0
        return float(self.costo_por_serie.reindex(series).notna().mean())


def cargar_costos(cfg: Config) -> MaestroCostos:
    """Lee el maestro si existe. Su ausencia NO detiene el experimento: las
    métricas en unidades y el contraste pareado no lo necesitan; solo la suite
    valorizada, que en ese caso se omite y se declara."""
    ruta = cfg.ruta_datos.parent / "costos.csv"
    if not ruta.exists():
        return MaestroCostos(costo_por_serie=pd.Series(dtype=float), ruta=None)

    tabla = pd.read_csv(ruta)
    esperadas = {"sku", "canal", "regional", "costo_unitario"}
    faltantes = esperadas - set(tabla.columns)
    if faltantes:
        raise ValueError(
            f"{ruta.name} no tiene las columnas {sorted(faltantes)}; "
            f"regenerá la extracción."
        )

    serie = (
        tabla["sku"].astype(str) + "|"
        + tabla["canal"].astype(str) + "|"
        + tabla["regional"].astype(str)
    )
    costos = pd.Series(tabla["costo_unitario"].to_numpy(dtype=float), index=serie)
    # Un costo no positivo no puede ponderar un error: se descarta y se cuenta
    # vía cobertura (outliers visibles, nunca silenciosos).
    costos = costos[costos > 0]
    return MaestroCostos(costo_por_serie=costos, ruta=ruta)
