"""Explicación del modelo y Pareto del valor — criterios de aceptación.

«Los meses de pico salen de la demanda valorizada de entrenamiento con una
regla declarada, nunca a dedo; la agregación de contribuciones conserva las
magnitudes por bloque; y la curva de Pareto es monótona, cierra en 100 % y
excluye las series sin costo en lugar de aplanarse con ceros.»

Datos 100 % sintéticos (RN-1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import explicacion


def _panel(valores: dict[str, list[float]], inicio: str = "2020-01") -> pd.DataFrame:
    largo = len(next(iter(valores.values())))
    indice = pd.period_range(inicio, periods=largo, freq="M")
    return pd.DataFrame(valores, index=indice, dtype=float)


# --- meses_pico -------------------------------------------------------------

def test_meses_pico_detecta_el_mes_dominante():
    # Dos años exactos: cada febrero vale 12 veces un mes normal.
    valores = [12.0 if mes == 2 else 1.0
               for anio in range(2) for mes in range(1, 13)]
    panel = _panel({"SKU-1|Z|LP": valores})
    costos = pd.Series({"SKU-1|Z|LP": 1.0})

    assert explicacion.meses_pico(panel, costos) == [2]


def test_meses_pico_con_demanda_plana_devuelve_un_mes():
    panel = _panel({"SKU-1|Z|LP": [5.0] * 24})
    costos = pd.Series({"SKU-1|Z|LP": 1.0})

    meses = explicacion.meses_pico(panel, costos)
    assert len(meses) == 1


def test_meses_pico_pondera_por_costo():
    # En unidades A y B empatan; el costo de B es 100 veces mayor, así que el
    # pico valorizado es el de B (junio), no el de A (febrero).
    a = [10.0 if mes == 2 else 1.0 for mes in range(1, 13)] * 2
    b = [10.0 if mes == 6 else 1.0 for mes in range(1, 13)] * 2
    panel = _panel({"A|Z|LP": a, "B|Z|LP": b})
    costos = pd.Series({"A|Z|LP": 1.0, "B|Z|LP": 100.0})

    assert explicacion.meses_pico(panel, costos) == [6]


# --- agregar_contribuciones -------------------------------------------------

def _contribuciones_sinteticas() -> tuple[pd.DataFrame, list[str]]:
    periodos = pd.period_range("2025-01", periods=4, freq="M")  # ene..abr
    tabla = pd.DataFrame({
        "f_grande": [10.0, -10.0, 2.0, -2.0],
        "f_chica": [1.0, 1.0, 1.0, 1.0],
        "serie": ["S"] * 4,
        "periodo": periodos,
    })
    return tabla, ["f_grande", "f_chica"]


def test_agregar_contribuciones_ordena_y_reparte():
    tabla, columnas = _contribuciones_sinteticas()
    resultado = explicacion.agregar_contribuciones(tabla, columnas, meses=[1, 2])

    assert list(resultado["feature"]) == ["f_grande", "f_chica"]
    assert resultado["participacion_pct"].sum() == pytest.approx(100.0)
    fila = resultado.set_index("feature").loc["f_grande"]
    # En el pico (ene, feb): |10| y |-10| -> media absoluta 10, con signo 0.
    assert fila["contrib_abs_media_pico"] == pytest.approx(10.0)
    assert fila["contrib_media_pico"] == pytest.approx(0.0)
    # Fuera del pico (mar, abr): |2| y |-2| -> media absoluta 2.
    assert fila["contrib_abs_media_resto"] == pytest.approx(2.0)


# --- error_por_mes / error_por_temporada ------------------------------------

def test_error_por_mes_valoriza_y_normaliza():
    real = _panel({"S|Z|LP": [10.0, 10.0]}, inicio="2026-01")
    predicciones = {"m": _panel({"S|Z|LP": [12.0, 8.0]}, inicio="2026-01")}
    costos = pd.Series({"S|Z|LP": 2.0})

    tabla = explicacion.error_por_mes(real, predicciones, costos, ["m"])

    assert len(tabla) == 2
    # |error| mensual = 2 unidades x costo 2 = 4; demanda = 20 -> 20 %.
    assert tabla["wmape_val"].tolist() == pytest.approx([20.0, 20.0])
    assert tabla["bias_val"].tolist() == pytest.approx([20.0, -20.0])
    assert tabla["demanda_valorizada"].tolist() == pytest.approx([20.0, 20.0])


def test_error_por_temporada_agrega_desde_sumas():
    # feb es pico. El error del pico (4 valorizado sobre 40) NO es el promedio
    # de ratios: es la suma de errores sobre la suma de demandas de sus meses.
    real = _panel({"S|Z|LP": [10.0, 20.0, 10.0]}, inicio="2026-01")
    predicciones = {"m": _panel({"S|Z|LP": [10.0, 18.0, 15.0]}, inicio="2026-01")}
    costos = pd.Series({"S|Z|LP": 2.0})

    tabla = explicacion.error_por_temporada(
        real, predicciones, costos, meses=[2], modelos=["m"]
    )

    pico = tabla.set_index("temporada").loc["pico"]
    resto = tabla.set_index("temporada").loc["resto"]
    assert pico["meses"] == 1
    assert pico["wmape_val"] == pytest.approx(10.0)     # 4 / 40
    assert pico["bias_val"] == pytest.approx(-10.0)     # subestima el pico
    assert resto["meses"] == 2
    assert resto["wmape_val"] == pytest.approx(25.0)    # 10 / 40
    assert resto["bias_val"] == pytest.approx(25.0)


# --- pareto_valorizado ------------------------------------------------------

def test_pareto_agrupa_combinaciones_y_cierra_en_cien():
    panel = _panel({
        "SKU-1|Z|LP": [8.0] * 12,     # SKU-1 en dos combinaciones
        "SKU-1|Z|SC": [1.0] * 12,
        "SKU-2|Z|LP": [1.0] * 12,
    })
    costos = pd.Series({"SKU-1|Z|LP": 1.0, "SKU-1|Z|SC": 1.0, "SKU-2|Z|LP": 1.0})

    pareto = explicacion.pareto_valorizado(panel, costos)

    assert list(pareto["sku"]) == ["SKU-1", "SKU-2"]
    assert pareto["participacion_pct"].iloc[0] == pytest.approx(90.0)
    assert pareto["acumulada_pct"].iloc[-1] == pytest.approx(100.0)
    assert (pareto["acumulada_pct"].diff().dropna() >= 0).all()
    assert pareto["fraccion_skus_pct"].iloc[-1] == pytest.approx(100.0)


def test_pareto_excluye_series_sin_costo():
    panel = _panel({
        "SKU-1|Z|LP": [5.0] * 12,
        "SKU-3|Z|LP": [500.0] * 12,   # sin costo en el maestro
    })
    costos = pd.Series({"SKU-1|Z|LP": 2.0})

    pareto = explicacion.pareto_valorizado(panel, costos)

    assert list(pareto["sku"]) == ["SKU-1"]
    assert pareto["demanda_valorizada"].iloc[0] == pytest.approx(120.0)
