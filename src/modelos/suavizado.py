"""Suavizado exponencial simple y Holt-Winters.

Ambos se incorporan al experimento por recomendación de la dirección del trabajo:
son los modelos estadísticos clásicos de referencia en la literatura de
pronóstico (Hyndman & Athanasopoulos, 2021) y su ausencia dejaría la comparación
expuesta a la crítica de que LightGBM solo se midió contra métodos triviales.

Se implementan aquí, en NumPy, en lugar de importarlos de una biblioteca de
series temporales. Tres razones, todas metodológicas:

1. **Reproducibilidad exacta (RN-5).** Los optimizadores de máxima verosimilitud
   de las bibliotecas dependen de la versión de la rutina numérica subyacente y
   pueden converger a óptimos distintos entre versiones. Una rejilla explícita
   de parámetros da siempre el mismo resultado.
2. **Protocolo homogéneo (RN-4).** Todos los brazos tienen que pronosticar a un
   paso con el valor real del mes anterior. Fijar los parámetros en
   entrenamiento y luego avanzar el estado con observaciones reales es directo
   si uno controla la recursión, y torpe de forzar desde fuera de una biblioteca.
3. **Auditabilidad.** Cada línea de la recursión es verificable a mano contra la
   fórmula del libro, que es lo que un tribunal puede pedir.

Los parámetros se eligen minimizando el **error absoluto** en el bloque de
entrenamiento, no el error cuadrático. Es coherente con que la métrica principal
del estudio sea el MAE y con el objetivo ``regression_l1`` de LightGBM: elegir
por SSE y reportar MAE optimizaría una cosa y mediría otra.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import como_panel, error_absoluto_por_serie, valores


def _primer_indice_observado(matriz: np.ndarray) -> np.ndarray:
    """Índice de la primera fila no-NaN de cada columna (nacimiento de la serie)."""
    observado = ~np.isnan(matriz)
    tiene = observado.any(axis=0)
    primero = observado.argmax(axis=0).astype(float)
    primero[~tiene] = np.nan
    return primero


def _error_absoluto(
    predicciones: np.ndarray, reales: np.ndarray, filas: np.ndarray
) -> np.ndarray:
    """Envoltorio que recorta a las filas indicadas antes de sumar el error."""
    return error_absoluto_por_serie(predicciones[filas], reales[filas])


# ---------------------------------------------------------------------------
# Suavizado exponencial simple
# ---------------------------------------------------------------------------

def _recursion_ses(matriz: np.ndarray, alfa: float) -> np.ndarray:
    """Recursión del suavizado exponencial simple, vectorizada sobre las series.

    ``ŷ(t) = ℓ(t−1)``  con  ``ℓ(t) = α·y(t) + (1−α)·ℓ(t−1)``

    El nivel se inicializa con la primera observación de cada serie, de modo que
    el primer mes de vida no tiene pronóstico (``NaN``) y el segundo pronostica
    exactamente el primero.
    """
    n_filas, n_series = matriz.shape
    prediccion = np.full((n_filas, n_series), np.nan)
    nivel = np.full(n_series, np.nan)

    for t in range(n_filas):
        prediccion[t] = nivel
        y = matriz[t]
        observado = ~np.isnan(y)
        sin_iniciar = observado & np.isnan(nivel)
        actualiza = observado & ~np.isnan(nivel)
        nivel = np.where(sin_iniciar, y, nivel)
        nivel = np.where(actualiza, alfa * y + (1.0 - alfa) * nivel, nivel)

    return prediccion


class SuavizadoExponencial:
    """Suavizado exponencial simple con α elegido por serie en entrenamiento."""

    nombre = "exp_smooth_opt"

    def __init__(self, alfas: np.ndarray | list[float]):
        self.alfas = np.asarray(sorted(set(float(a) for a in alfas)), dtype=float)
        if self.alfas.size == 0:
            raise ValueError("La rejilla de alfa está vacía.")
        self.alfa_por_serie: pd.Series | None = None

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        matriz = valores(entrenamiento)
        filas = np.arange(matriz.shape[0])

        mejor_error = np.full(matriz.shape[1], np.inf)
        mejor_alfa = np.full(matriz.shape[1], np.nan)

        for alfa in self.alfas:
            prediccion = _recursion_ses(matriz, alfa)
            error = _error_absoluto(prediccion, matriz, filas)
            # Comparación estricta: ante empate gana el primer alfa de la
            # rejilla, que está ordenada. Sin esto el resultado dependería del
            # orden de recorrido y la RN-5 se rompería.
            mejora = error < mejor_error
            mejor_error = np.where(mejora, error, mejor_error)
            mejor_alfa = np.where(mejora, alfa, mejor_alfa)

        self.alfa_por_serie = pd.Series(mejor_alfa, index=entrenamiento.columns,
                                        name="alfa")

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        if self.alfa_por_serie is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")

        matriz = valores(datos)
        alfas = self.alfa_por_serie.reindex(datos.columns).to_numpy(dtype=float)
        prediccion = np.full_like(matriz, np.nan)

        # Se agrupa por valor de α para reutilizar la recursión: los distintos
        # α son pocos (la rejilla) y las series muchas.
        for alfa in np.unique(alfas[~np.isnan(alfas)]):
            columnas = np.where(alfas == alfa)[0]
            prediccion[:, columnas] = _recursion_ses(matriz[:, columnas], float(alfa))

        return como_panel(prediccion, datos)


# ---------------------------------------------------------------------------
# Holt-Winters aditivo
# ---------------------------------------------------------------------------

def _recursion_holt_winters(
    matriz: np.ndarray,
    inicio: np.ndarray,
    nivel0: np.ndarray,
    pendiente0: np.ndarray,
    estacional0: np.ndarray,
    alfa: float,
    beta: float,
    gamma: float,
    m: int,
) -> np.ndarray:
    """Holt-Winters aditivo, vectorizado sobre las series.

    ``ŷ(t) = ℓ(t−1) + b(t−1) + s(t−m)``

    ::

        ℓ(t) = α·[y(t) − s(t−m)] + (1−α)·[ℓ(t−1) + b(t−1)]
        b(t) = β·[ℓ(t) − ℓ(t−1)] + (1−β)·b(t−1)
        s(t) = γ·[y(t) − ℓ(t)]   + (1−γ)·s(t−m)

    La recursión arranca cuando la serie acumula ``2m`` meses de vida, que son
    los que consume la inicialización. Ninguna predicción utiliza información
    posterior al mes que predice.
    """
    n_filas, n_series = matriz.shape
    prediccion = np.full((n_filas, n_series), np.nan)

    nivel = nivel0.copy()
    pendiente = pendiente0.copy()
    estacional = estacional0.copy()          # (m, n_series), indexada por t % m

    arranque = inicio + 2 * m                # primer mes con pronóstico

    for t in range(n_filas):
        ranura = t % m
        activo = t >= arranque
        prediccion[t] = np.where(activo, nivel + pendiente + estacional[ranura], np.nan)

        y = matriz[t]
        actualiza = activo & ~np.isnan(y)
        if not actualiza.any():
            continue

        nivel_anterior = nivel
        nivel_nuevo = alfa * (y - estacional[ranura]) + (1.0 - alfa) * (nivel + pendiente)
        pendiente_nueva = beta * (nivel_nuevo - nivel_anterior) + (1.0 - beta) * pendiente
        estacional_nueva = gamma * (y - nivel_nuevo) + (1.0 - gamma) * estacional[ranura]

        nivel = np.where(actualiza, nivel_nuevo, nivel)
        pendiente = np.where(actualiza, pendiente_nueva, pendiente)
        estacional[ranura] = np.where(actualiza, estacional_nueva, estacional[ranura])

    return prediccion


def _inicializacion_holt_winters(
    matriz: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estado inicial clásico a partir de los primeros ``2m`` meses de cada serie.

    ``ℓ0`` = media del primer año · ``b0`` = (media del segundo año − ℓ0) / m ·
    ``s0(j)`` = y(j) − ℓ0 para el primer año.

    Devuelve además la máscara de series con vida suficiente: las que no llegan a
    ``2m`` meses no pueden inicializarse y quedan sin pronóstico, lo que el
    respaldo resuelve explícitamente en lugar de inventar un valor.
    """
    n_filas, n_series = matriz.shape
    primero = _primer_indice_observado(matriz)
    vida = np.where(np.isnan(primero), 0, n_filas - np.nan_to_num(primero, nan=0.0))
    suficiente = (~np.isnan(primero)) & (vida >= 2 * m)

    inicio = np.where(suficiente, np.nan_to_num(primero, nan=0.0), n_filas + 1).astype(int)

    nivel0 = np.full(n_series, np.nan)
    pendiente0 = np.full(n_series, np.nan)
    estacional0 = np.full((m, n_series), np.nan)

    columnas = np.where(suficiente)[0]
    if columnas.size:
        desplazamientos = np.arange(2 * m)[:, None]
        filas = inicio[columnas][None, :] + desplazamientos        # (2m, k)
        ventana = matriz[filas, columnas[None, :]]                 # (2m, k)

        with warnings.catch_warnings():
            # Serie que muere dentro de su propia ventana de inicialización: el
            # promedio sobre vacío avisa. Queda en NaN y más abajo se neutraliza
            # a cero, pero esa serie nunca llega a predecir porque su `inicio`
            # queda fuera de rango.
            warnings.simplefilter("ignore", RuntimeWarning)
            primer_anio = np.nanmean(ventana[:m], axis=0)
            segundo_anio = np.nanmean(ventana[m:], axis=0)
        nivel0[columnas] = primer_anio
        pendiente0[columnas] = (segundo_anio - primer_anio) / m

        # La ranura del mes calendario `inicio + j` en un buffer de tamaño m.
        ranuras = (inicio[columnas][None, :] + np.arange(m)[:, None]) % m
        estacional0[ranuras, columnas[None, :]] = ventana[:m] - primer_anio[None, :]

    # Series sin estacionalidad estimable: seguridad numérica, no imputación —
    # sin vida suficiente `inicio` queda fuera de rango y nunca se predice.
    estacional0 = np.nan_to_num(estacional0, nan=0.0)
    nivel0 = np.nan_to_num(nivel0, nan=0.0)
    pendiente0 = np.nan_to_num(pendiente0, nan=0.0)

    return inicio, nivel0, pendiente0, estacional0, suficiente


class HoltWinters:
    """Holt-Winters aditivo con (α, β, γ) elegidos por serie en entrenamiento."""

    nombre = "holt_winters"

    def __init__(
        self,
        periodo_estacional: int = 12,
        alfas: list[float] | None = None,
        betas: list[float] | None = None,
        gammas: list[float] | None = None,
    ):
        self.m = int(periodo_estacional)
        self.alfas = sorted(set(alfas or [0.1, 0.3, 0.5, 0.7, 0.9]))
        self.betas = sorted(set(betas or [0.0, 0.1, 0.2]))
        self.gammas = sorted(set(gammas or [0.0, 0.1, 0.2]))
        self.parametros_por_serie: pd.DataFrame | None = None

    @property
    def combinaciones(self) -> int:
        return len(self.alfas) * len(self.betas) * len(self.gammas)

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        matriz = valores(entrenamiento)
        filas = np.arange(matriz.shape[0])
        inicio, nivel0, pendiente0, estacional0, _ = _inicializacion_holt_winters(
            matriz, self.m
        )

        n_series = matriz.shape[1]
        mejor_error = np.full(n_series, np.inf)
        mejor = np.full((3, n_series), np.nan)

        for alfa in self.alfas:
            for beta in self.betas:
                for gamma in self.gammas:
                    prediccion = _recursion_holt_winters(
                        matriz, inicio, nivel0, pendiente0, estacional0.copy(),
                        alfa, beta, gamma, self.m,
                    )
                    error = _error_absoluto(prediccion, matriz, filas)
                    mejora = error < mejor_error
                    mejor_error = np.where(mejora, error, mejor_error)
                    mejor[0] = np.where(mejora, alfa, mejor[0])
                    mejor[1] = np.where(mejora, beta, mejor[1])
                    mejor[2] = np.where(mejora, gamma, mejor[2])

        self.parametros_por_serie = pd.DataFrame(
            {"alfa": mejor[0], "beta": mejor[1], "gamma": mejor[2]},
            index=entrenamiento.columns,
        )

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        if self.parametros_por_serie is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")

        matriz = valores(datos)
        parametros = self.parametros_por_serie.reindex(datos.columns)
        inicio, nivel0, pendiente0, estacional0, _ = _inicializacion_holt_winters(
            matriz, self.m
        )
        prediccion = np.full_like(matriz, np.nan)

        # Se agrupa por terna de parámetros: son a lo sumo las de la rejilla, de
        # modo que la recursión corre unas pocas decenas de veces y no una por
        # serie. La clave es textual para que el agrupamiento sea determinista.
        redondeados = parametros.round(6)
        claves = (
            redondeados["alfa"].astype(str) + "|"
            + redondeados["beta"].astype(str) + "|"
            + redondeados["gamma"].astype(str)
        )
        posiciones = pd.Series(np.arange(len(datos.columns)), index=datos.columns)
        for clave, grupo in claves.groupby(claves, sort=True):
            alfa, beta, gamma = (float(v) for v in clave.split("|"))
            if any(np.isnan(v) for v in (alfa, beta, gamma)):
                continue
            columnas = posiciones.reindex(grupo.index).to_numpy()
            if not columnas.size:
                continue
            prediccion[:, columnas] = _recursion_holt_winters(
                matriz[:, columnas],
                inicio[columnas], nivel0[columnas], pendiente0[columnas],
                estacional0[:, columnas].copy(),
                float(alfa), float(beta), float(gamma), self.m,
            )

        return como_panel(prediccion, datos)
