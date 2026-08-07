"""Método de Croston para demanda intermitente.

Se incorpora porque una parte apreciable del portafolio de NOVAPACK es demanda
esporádica —productos de baja rotación con muchos meses en cero— y el informe de
inspección cuantifica exactamente cuántas series caen en ese régimen. Sobre ese
tipo de series el promedio móvil y el Naïve tienen un problema estructural bien
documentado: predicen cero justo después de un mes sin venta y sobre-reaccionan
justo después de un pedido grande.

Croston (1972) separa el problema en dos: cuánto se pide cuando se pide, y cada
cuánto se pide. Suaviza ambas magnitudes por separado y pronostica el cociente.

    z(t) = α·y(t) + (1−α)·z(t−1)      tamaño medio del pedido
    p(t) = α·q    + (1−α)·p(t−1)      intervalo medio entre pedidos
    ŷ    = z / p

donde ``q`` es el número de meses transcurridos desde el pedido anterior. Los
estados solo se actualizan en los meses con demanda positiva; en los meses en
cero el pronóstico se mantiene constante, que es precisamente la propiedad que
lo hace estable en este régimen.

α se elige por serie minimizando el error absoluto en entrenamiento, igual que
en los demás modelos con parámetro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import como_panel, error_absoluto_por_serie, valores


def _recursion_croston(matriz: np.ndarray, alfa: float) -> np.ndarray:
    """Recursión de Croston, vectorizada sobre las series."""
    n_filas, n_series = matriz.shape
    prediccion = np.full((n_filas, n_series), np.nan)

    tamano = np.full(n_series, np.nan)      # z
    intervalo = np.full(n_series, np.nan)   # p
    espera = np.ones(n_series)              # q: meses desde el pedido anterior

    for t in range(n_filas):
        prediccion[t] = tamano / intervalo

        y = matriz[t]
        vivo = ~np.isnan(y)
        positivo = vivo & (y > 0)
        primera = positivo & np.isnan(tamano)
        posterior = positivo & ~np.isnan(tamano)

        tamano = np.where(primera, y, tamano)
        intervalo = np.where(primera, espera, intervalo)
        tamano = np.where(posterior, alfa * y + (1.0 - alfa) * tamano, tamano)
        intervalo = np.where(
            posterior, alfa * espera + (1.0 - alfa) * intervalo, intervalo
        )

        # El contador solo avanza mientras la serie está viva.
        espera = np.where(positivo, 1.0, np.where(vivo, espera + 1.0, espera))

    return prediccion


class Croston:
    """Croston clásico con α elegido por serie en entrenamiento."""

    nombre = "croston"

    def __init__(self, alfas: list[float] | None = None):
        self.alfas = np.asarray(
            sorted(set(float(a) for a in (alfas or [0.05, 0.1, 0.2, 0.3]))), dtype=float
        )
        self.alfa_por_serie: pd.Series | None = None

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        matriz = valores(entrenamiento)

        mejor_error = np.full(matriz.shape[1], np.inf)
        mejor_alfa = np.full(matriz.shape[1], np.nan)

        for alfa in self.alfas:
            prediccion = _recursion_croston(matriz, float(alfa))
            error = error_absoluto_por_serie(prediccion, matriz)
            mejora = error < mejor_error
            mejor_error = np.where(mejora, error, mejor_error)
            mejor_alfa = np.where(mejora, alfa, mejor_alfa)

        self.alfa_por_serie = pd.Series(
            mejor_alfa, index=entrenamiento.columns, name="alfa"
        )

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        if self.alfa_por_serie is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")

        matriz = valores(datos)
        alfas = self.alfa_por_serie.reindex(datos.columns).to_numpy(dtype=float)
        prediccion = np.full_like(matriz, np.nan)

        for alfa in np.unique(alfas[~np.isnan(alfas)]):
            columnas = np.where(alfas == alfa)[0]
            prediccion[:, columnas] = _recursion_croston(matriz[:, columnas], float(alfa))

        return como_panel(prediccion, datos)
