"""Modelos Naïve.

``naive_m1`` es la línea base estándar de la literatura de pronóstico (Hyndman &
Athanasopoulos, 2021) y es además el método que NOVAPACK aplica de hecho en la
mayor parte del portafolio: en el sistema en producción 2.847 de las series
tienen asignado exactamente este modelo.

``naive_m1_gf`` es la variante que la empresa usa cuando quiere corregir por
tendencia: al último valor observado le suma el factor de crecimiento del año
móvil.
"""

from __future__ import annotations

import pandas as pd

from .base import factor_de_crecimiento


class Naive:
    """ŷ(t) = y(t−1). El valor del mes anterior."""

    nombre = "naive_m1"

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        """No tiene parámetros que estimar."""
        return None

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        return datos.shift(1)


class NaiveConCrecimiento:
    """ŷ(t) = y(t−1) + GF(t), con GF el factor de crecimiento del año móvil."""

    nombre = "naive_m1_gf"

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        return None

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        return datos.shift(1) + factor_de_crecimiento(datos)
