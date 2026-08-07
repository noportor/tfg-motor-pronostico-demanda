"""Promedio móvil — el método tradicional de NOVAPACK.

§10 de los requerimientos prohíbe expresamente sustituir el promedio móvil por
una versión más débil que la que usa la empresa para agrandar la mejora. Por eso
las ventanas no son arbitrarias: son las que Planificación aplica hoy, tal como
están implementadas en el sistema en producción.

- ``ma_2``  = (y(t−1) + y(t−2)) / 2 — el bimestre móvil.
- ``ma_12`` = media de los doce meses previos — el año móvil.
- las variantes ``_gf`` suman el factor de crecimiento del año móvil, que es
  como el área corrige por tendencia.

Una precisión importante para la memoria: el promedio móvil del sistema en
producción es **recursivo** (se realimenta con sus propios pronósticos y por eso
converge de forma asintótica). Este módulo implementa la fórmula a un paso,
porque la RN-4 fija así la evaluación principal; a un mes de horizonte las dos
formulaciones coinciden exactamente. El caso recursivo con horizonte mayor que
uno SÍ forma parte del experimento, pero vive en ``src/multihorizonte.py``: la
proyección multihorizonte extiende el panel y realimenta estas mismas fórmulas
mes a mes, que es literalmente la recursión del sistema en producción.
"""

from __future__ import annotations

import pandas as pd

from .base import factor_de_crecimiento


class PromedioMovil:
    """Media de los ``k`` meses previos."""

    def __init__(self, ventana: int, con_crecimiento: bool = False):
        if ventana < 1:
            raise ValueError(f"La ventana del promedio móvil debe ser >= 1; llegó {ventana}.")
        self.ventana = ventana
        self.con_crecimiento = con_crecimiento
        self.nombre = f"ma_{ventana}" + ("_gf" if con_crecimiento else "")

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        """No tiene parámetros que estimar: la ventana la fija la empresa."""
        return None

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        # shift(1) primero: la ventana nunca toca el mes que se predice.
        # min_periods = ventana: con menos historia no se emite un promedio
        # calculado sobre menos meses de los declarados; se emite NaN y el
        # respaldo lo resuelve explícitamente (§10: no inventar valores).
        base = (
            datos.shift(1)
            .rolling(window=self.ventana, min_periods=self.ventana)
            .mean()
        )
        if self.con_crecimiento:
            base = base + factor_de_crecimiento(datos)
        return base
