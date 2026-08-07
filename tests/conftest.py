"""Utilidades compartidas por las pruebas.

Los datos sintéticos se admiten EXCLUSIVAMENTE aquí (RN-1). Ningún número que
llegue al documento académico puede provenir de este archivo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.config import cargar_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """Configuración real del experimento (no una copia sintética).

    Se usa la de verdad a propósito: si alguien cambia un corte en config.yaml y
    rompe el orden de los bloques, las pruebas tienen que enterarse.
    """
    return cargar_config()


def panel_de(
    valores_por_serie: dict[str, list[float]],
    inicio: str = "2017-04",
    canal: str = "X",
    regional: str = "SANTA CRUZ",
) -> pd.DataFrame:
    """Construye un panel largo a partir de listas de valores mensuales.

    Todas las series arrancan en el mismo mes, que es lo cómodo para las pruebas
    que no están comprobando el tratamiento del nacimiento.
    """
    filas = []
    for serie, valores in valores_por_serie.items():
        meses = pd.period_range(inicio, periods=len(valores), freq="M")
        for periodo, valor in zip(meses, valores):
            filas.append({
                "serie": f"{serie}|{canal}|{regional}",
                "sku": serie,
                "canal": canal,
                "regional": regional,
                "periodo": periodo,
                "y": float(valor),
            })
    return pd.DataFrame(filas)


def serie_estacional(n_meses: int, base: float = 100.0, semilla: int = 7) -> list[float]:
    """Serie con nivel, tendencia suave y estacionalidad anual, sin ruido salvaje."""
    rng = np.random.default_rng(semilla)
    t = np.arange(n_meses)
    estacional = 30 * np.sin(2 * np.pi * (t % 12) / 12)
    tendencia = 0.5 * t
    ruido = rng.normal(0, 5, n_meses)
    return list(np.clip(base + tendencia + estacional + ruido, 0, None).round(2))
