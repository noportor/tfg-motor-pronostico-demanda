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


# --- escala de las contribuciones TreeSHAP con el objetivo Tweedie ----------

def _modelo_tweedie_sintetico():
    """LightGBM Tweedie sobre un panel sintético, envuelto en ModeloLightGBM.

    Se arma la tabla larga a mano (serie, periodo, features, y) y se inyecta el
    booster: lo que se prueba es la escala de ``contribuciones()``, no la
    construcción de features.
    """
    lgb = pytest.importorskip("lightgbm")
    from src.modelos.lightgbm_modelo import ModeloLightGBM

    rng = np.random.default_rng(20260408)
    series = [f"SKU-{i}|C|R" for i in range(12)]
    periodos = pd.period_range("2020-01", periods=36, freq="M")
    filas = []
    for i, serie in enumerate(series):
        nivel = 5.0 + 3.0 * i
        for t, periodo in enumerate(periodos):
            estacion = 1.0 + 0.6 * np.sin(2 * np.pi * periodo.month / 12)
            rezago = nivel * (1.0 + 0.6 * np.sin(2 * np.pi * (periodo.month - 1) / 12))
            filas.append({
                "serie": serie, "periodo": periodo,
                "nivel": nivel, "mes": periodo.month, "rezago_1": rezago,
                "y": float(rng.poisson(nivel * estacion)),
            })
    tabla = pd.DataFrame(filas)
    columnas = ["nivel", "mes", "rezago_1"]

    # El objetivo y la potencia salen de la configuración publicada: la prueba
    # verifica la escala del modelo que produjo la Figura 6, no la de otro.
    from src.config import cargar_config
    parametros = cargar_config().modelos["lightgbm"]
    booster = lgb.train(
        {"objective": parametros["objective"],
         "tweedie_variance_power": parametros["tweedie_variance_power"],
         "learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 5,
         "seed": 1, "num_threads": 1, "deterministic": True,
         "force_row_wise": True, "verbosity": -1},
        lgb.Dataset(tabla[columnas], label=tabla["y"]),
        num_boost_round=60,
    )
    modelo = ModeloLightGBM.__new__(ModeloLightGBM)
    modelo.tabla = tabla
    modelo.columnas = columnas
    modelo.booster = booster
    panel = pd.DataFrame(
        0.0, index=periodos[-12:], columns=series
    )
    return modelo, panel, booster, columnas


def test_la_configuracion_publicada_usa_el_objetivo_tweedie():
    """Los rótulos en escala logarítmica valen mientras el objetivo sea Tweedie.

    Si alguien cambia el objetivo de LightGBM, esta prueba obliga a revisar la
    lectura de las contribuciones (docstrings, flujo.json y Figura 6).
    """
    from src.config import cargar_config

    assert cargar_config().modelos["lightgbm"]["objective"] == "tweedie"


def test_contribuciones_tweedie_suman_el_puntaje_crudo_logaritmico():
    """valor base + suma(contribuciones) = raw_score, y exp(raw_score) = predicción.

    Con el objetivo Tweedie, TreeSHAP de LightGBM devuelve las contribuciones en
    la escala cruda del modelo (enlace logarítmico): su suma más el valor base
    es el LOGARITMO de la predicción, no la predicción en unidades.
    """
    modelo, panel, booster, columnas = _modelo_tweedie_sintetico()

    contribuciones = modelo.contribuciones(panel)
    filas = modelo._filas_de(panel)
    crudo = booster.predict(filas[columnas], raw_score=True,
                            num_iteration=booster.best_iteration)
    prediccion = booster.predict(filas[columnas],
                                 num_iteration=booster.best_iteration)

    suma = contribuciones[columnas].sum(axis=1).to_numpy() + contribuciones[
        "valor_base"].to_numpy()
    np.testing.assert_allclose(suma, crudo, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.exp(suma), prediccion, rtol=1e-9)
    # Y por eso NO suman la predicción en unidades: leerlas así es un error.
    assert not np.allclose(suma, prediccion, rtol=1e-3)


def test_una_contribucion_tweedie_es_un_factor_multiplicativo():
    """Quitar la contribución c de una variable divide la predicción por exp(c)."""
    modelo, panel, booster, columnas = _modelo_tweedie_sintetico()

    contribuciones = modelo.contribuciones(panel)
    prediccion = np.exp(contribuciones[columnas].sum(axis=1)
                        + contribuciones["valor_base"])
    c = contribuciones["rezago_1"]
    sin_rezago = np.exp(contribuciones[["nivel", "mes"]].sum(axis=1)
                        + contribuciones["valor_base"])
    np.testing.assert_allclose(prediccion / sin_rezago, np.exp(c), rtol=1e-9)
    # +0,20 en la escala del modelo es un 22 % más sobre la predicción.
    assert np.exp(0.20) == pytest.approx(1.2214, abs=1e-4)
