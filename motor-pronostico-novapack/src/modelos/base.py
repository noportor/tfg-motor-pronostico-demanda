"""RF-6 — Interfaz común de los modelos y utilidades compartidas.

Protocolo de evaluación (RN-4)
------------------------------
Los modelos se evalúan **un mes hacia adelante usando el valor real del mes
anterior**. Es la única comparación justa: el promedio móvil y el Naïve
funcionan así por construcción, y evaluarlos contra un LightGBM que pronostica
doce meses de corrido compararía dos cosas distintas.

En términos operativos, todo modelo de este paquete implementa la misma función:

    dado el historial REAL de una serie hasta el mes t−1, ¿cuánto se venderá
    en el mes t?

Los parámetros que cada modelo necesite (el alfa del suavizado exponencial, los
árboles de LightGBM) se estiman **solo con el bloque de entrenamiento**. Lo que
sí avanza mes a mes es el *estado* alimentado con valores reales observados, que
es exactamente lo que hace un planificador cada mes cuando cierra el período.

Representación de los datos
---------------------------
Panel **ancho**: índice = mes (``pd.Period``), columnas = serie, valores = la
demanda observada. Fuera de la vida de la serie el valor es ``NaN``, nunca cero
(ver ``series.py``). Todos los modelos estadísticos operan sobre esta matriz con
recursiones vectorizadas: el bucle recorre los ~108 meses y cada paso opera
sobre las miles de series a la vez.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Modelo(Protocol):
    """Interfaz común. La cumplen los once brazos del experimento."""

    nombre: str

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        """Estima los parámetros. Solo puede mirar ``entrenamiento``.

        ``validacion`` se recibe porque LightGBM la necesita para la parada
        temprana. Ningún otro modelo puede usarla: hacerlo convertiría la
        validación en parte del entrenamiento y el motor estaría eligiendo sobre
        datos ya vistos.
        """

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Pronóstico a un paso para cada (mes, serie) del panel recibido."""


# ---------------------------------------------------------------------------
# Utilidades sobre el panel ancho
# ---------------------------------------------------------------------------

def valores(panel: pd.DataFrame) -> np.ndarray:
    """Matriz T×S de floats, con NaN fuera de la vida de cada serie."""
    return panel.to_numpy(dtype=float, copy=True)


def como_panel(matriz: np.ndarray, referencia: pd.DataFrame) -> pd.DataFrame:
    """Envuelve una matriz T×S con el índice y las columnas del panel."""
    return pd.DataFrame(matriz, index=referencia.index, columns=referencia.columns)


def truncar_en_cero(panel: pd.DataFrame) -> pd.DataFrame:
    """La demanda no es negativa: toda predicción se trunca en cero (RF-6).

    Los ``NaN`` se conservan como ``NaN``: significan «este modelo no puede
    pronosticar aquí», que es información distinta de «pronostica cero».
    """
    return panel.clip(lower=0.0)


def enmascarar_fuera_de_vida(
    predicciones: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Anula las predicciones de meses en que la serie no existía.

    Sin esta máscara, una media móvil seguiría emitiendo números durante años
    después de la última venta de un producto descontinuado, y esos meses
    fantasma entrarían en las métricas.
    """
    return predicciones.where(panel.notna())


def desplazar(matriz: np.ndarray, k: int) -> np.ndarray:
    """Desplaza la matriz k filas hacia abajo rellenando con NaN.

    Equivale a ``panel.shift(k)`` pero sobre numpy, que es donde viven las
    recursiones.
    """
    if k <= 0:
        raise ValueError("El desplazamiento debe ser >= 1: un rezago 0 sería fuga.")
    salida = np.full_like(matriz, np.nan)
    if k < matriz.shape[0]:
        salida[k:] = matriz[:-k]
    return salida


def error_absoluto_por_serie(
    predicciones: np.ndarray, reales: np.ndarray
) -> np.ndarray:
    """Suma de errores absolutos por serie. Criterio de ajuste de los modelos
    con parámetro.

    Se optimiza el error ABSOLUTO y no el cuadrático porque la métrica principal
    del estudio es el MAE y el objetivo de LightGBM es ``regression_l1``. Elegir
    los parámetros minimizando el error cuadrático y después reportar el MAE
    sería optimizar una cosa y medir otra.

    Devuelve ``inf`` en las series sin ninguna observación comparable, para que
    nunca resulten elegidas por ser el mínimo de un conjunto vacío.
    """
    valido = ~np.isnan(predicciones) & ~np.isnan(reales)
    suma = np.where(valido, np.abs(predicciones - reales), 0.0).sum(axis=0)
    cuenta = valido.sum(axis=0)
    return np.where(cuenta > 0, suma, np.inf)


def factor_de_crecimiento(panel: pd.DataFrame) -> pd.DataFrame:
    """Factor de crecimiento mensual, replicando la fórmula que usa la empresa.

    ``GF(t) = [ Σ(y de los 12 meses previos) − Σ(y de los 12 anteriores a esos) ] / 12``

    Es la variación media mensual entre los dos últimos años móviles. Los
    modelos con sufijo ``_gf`` le suman este término a su nivel base, que es
    como Planificación corrige por tendencia hoy.

    Requiere 24 meses de historia; antes de eso devuelve ``NaN``.
    """
    desplazado = panel.shift(1)
    suma_ultimo_anio = desplazado.rolling(window=12, min_periods=12).sum()
    return (suma_ultimo_anio - suma_ultimo_anio.shift(12)) / 12.0


def aplicar_respaldo(
    predicciones: pd.DataFrame,
    panel: pd.DataFrame,
    entrenamiento: pd.DataFrame,
    meses_evaluados: pd.Index | None = None,
) -> tuple[pd.DataFrame, int]:
    """Rellena los huecos de pronóstico y devuelve cuántos hubo.

    Por qué hace falta
    ------------------
    Un modelo puede no tener pronóstico para un mes concreto: el promedio móvil
    de doce meses no existe antes del duodécimo mes de vida de la serie, y las
    variantes con factor de crecimiento necesitan veinticuatro. Si cada modelo
    se evaluara solo sobre los meses que puede pronosticar, cada uno se estaría
    midiendo sobre un conjunto distinto de observaciones y la comparación
    pareada —que es la base de todo el contraste estadístico— dejaría de ser
    pareada.

    Por eso todos los modelos se completan hasta cubrir exactamente los mismos
    meses, con una cascada declarada: primero el valor del mes anterior (el
    Naïve, que es lo que haría un planificador sin más información) y, si
    tampoco existe, la media de la serie en entrenamiento.

    El conteo de respaldos se reporta. Interesa el de los meses **evaluados**
    (validación y prueba): los huecos del arranque de cada serie en el bloque de
    entrenamiento son inevitables y no afectan a ninguna métrica. En la cohorte
    del estudio el conteo evaluado debería ser cero o casi, porque los criterios
    de inclusión exigen 36 meses de historia en entrenamiento y al llegar a
    validación toda serie tiene historia de sobra. Si no lo fuera, hay que
    decirlo.

    Args:
        meses_evaluados: meses sobre los que se cuenta. Si es ``None`` se cuenta
            sobre el panel completo.
    """
    completado = predicciones.copy()
    faltantes = completado.isna() & panel.notna()
    if meses_evaluados is not None:
        faltantes = faltantes.loc[faltantes.index.isin(meses_evaluados)]
    n_faltantes = int(faltantes.to_numpy().sum())
    if not completado.isna().to_numpy().any():
        return completado, n_faltantes

    completado = completado.fillna(panel.shift(1))
    media_entrenamiento = entrenamiento.mean(axis=0, skipna=True)
    completado = completado.fillna(
        pd.DataFrame(
            np.repeat(media_entrenamiento.to_numpy()[None, :], len(completado), axis=0),
            index=completado.index, columns=completado.columns,
        )
    )
    return completado, n_faltantes


def mae_naive_entrenamiento(entrenamiento: pd.DataFrame) -> pd.Series:
    """Denominador del MASE: error medio del naive dentro del entrenamiento.

    ``MAE_naive = media(|y(t) − y(t−1)|)`` sobre el bloque de entrenamiento.

    Escalar por este valor hace comparables series de volúmenes muy distintos,
    que es imprescindible cuando el portafolio va de productos que se venden por
    unidad a otros que se venden por caja. Es la métrica que conviene reportar
    cuando el MAPE excluye demasiadas observaciones por demanda cero.
    """
    diferencias = (entrenamiento - entrenamiento.shift(1)).abs()
    return diferencias.mean(axis=0, skipna=True)
