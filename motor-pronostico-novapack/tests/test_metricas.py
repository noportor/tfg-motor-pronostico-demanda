"""RF-7 — Criterio de aceptación.

«Con predicción perfecta, MAE, RMSE y Bias son cero y MAPE es cero. Con una serie
que tiene meses en cero, MAPE no devuelve infinito ni NaN silencioso: devuelve el
valor sobre los meses válidos y el conteo de excluidos.»
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metricas import matriz_de_errores, metricas_por_serie, resumen, tabla8


def _paneles(reales, predichos, nombre="A"):
    indice = pd.period_range("2025-04", periods=len(reales), freq="M")
    real = pd.DataFrame({nombre: [float(v) for v in reales]}, index=indice)
    predicho = pd.DataFrame({nombre: [float(v) for v in predichos]}, index=indice)
    return real, predicho


def test_prediccion_perfecta_da_error_cero():
    real, predicho = _paneles([10, 20, 30, 40], [10, 20, 30, 40])
    escala = pd.Series({"A": 5.0})
    tabla = metricas_por_serie(real, predicho, escala)

    assert tabla["mae"].item() == pytest.approx(0.0)
    assert tabla["rmse"].item() == pytest.approx(0.0)
    assert tabla["mape"].item() == pytest.approx(0.0)
    assert tabla["bias"].item() == pytest.approx(0.0)
    assert tabla["mase"].item() == pytest.approx(0.0)


def test_mape_excluye_los_meses_en_cero_y_los_cuenta():
    real, predicho = _paneles([0, 0, 100, 200], [10, 10, 110, 180])
    escala = pd.Series({"A": 50.0})
    tabla = metricas_por_serie(real, predicho, escala)

    # |110-100|/100 = .10 ; |180-200|/200 = .10  ->  MAPE = 10 %
    assert tabla["mape"].item() == pytest.approx(10.0)
    assert np.isfinite(tabla["mape"].item()), "El MAPE no puede salir infinito"
    assert tabla["n_observaciones"].item() == 4
    assert tabla["n_meses_con_demanda"].item() == 2
    assert tabla["meses_excluidos_del_mape"].item() == 2
    assert tabla["pct_excluido_del_mape"].item() == pytest.approx(50.0)


def test_una_serie_toda_en_cero_no_produce_un_mape_silencioso():
    real, predicho = _paneles([0, 0, 0], [1, 2, 3])
    tabla = metricas_por_serie(real, predicho, pd.Series({"A": 1.0}))

    assert np.isnan(tabla["mape"].item()), "Sin demanda positiva el MAPE es indefinido"
    assert tabla["meses_excluidos_del_mape"].item() == 3
    assert np.isnan(tabla["bias"].item()), "Con media real cero el Bias es indefinido"
    assert tabla["mae"].item() == pytest.approx(2.0)


def test_bias_es_porcentual_sobre_las_medias():
    real, predicho = _paneles([100, 100], [110, 130])
    tabla = metricas_por_serie(real, predicho, pd.Series({"A": 10.0}))
    # media predicha 120 vs media real 100  ->  +20 %
    assert tabla["bias"].item() == pytest.approx(20.0)
    assert tabla["bias_unidades"].item() == pytest.approx(20.0)


def test_mase_escala_con_el_error_del_naive_en_entrenamiento():
    real, predicho = _paneles([100, 100, 100], [110, 90, 100])
    # MAE = (10 + 10 + 0) / 3 = 6.6667
    tabla = metricas_por_serie(real, predicho, pd.Series({"A": 20.0}))
    assert tabla["mase"].item() == pytest.approx((20 / 3) / 20.0)


def test_mase_indefinido_si_el_naive_no_se_equivoca_nunca():
    real, predicho = _paneles([100, 100], [110, 90])
    tabla = metricas_por_serie(real, predicho, pd.Series({"A": 0.0}))
    assert np.isnan(tabla["mase"].item()), "Dividir por cero no puede pasar en silencio"


def test_rmse_se_calcula_por_serie_y_luego_se_promedia():
    """Nunca sobre el conjunto agrupado: ahí las series grandes dominarían."""
    indice = pd.period_range("2025-04", periods=2, freq="M")
    real = pd.DataFrame({"CHICA": [10.0, 10.0], "GRANDE": [1000.0, 1000.0]}, index=indice)
    predicho = pd.DataFrame({"CHICA": [12.0, 8.0], "GRANDE": [1200.0, 800.0]}, index=indice)
    escala = pd.Series({"CHICA": 1.0, "GRANDE": 1.0})

    tabla = metricas_por_serie(real, predicho, escala)
    assert tabla.loc["CHICA", "rmse"] == pytest.approx(2.0)
    assert tabla.loc["GRANDE", "rmse"] == pytest.approx(200.0)

    agregado = resumen({"m": tabla})
    assert agregado["rmse_media"].item() == pytest.approx(101.0)
    assert agregado["rmse_mediana"].item() == pytest.approx(101.0)


def test_el_resumen_reporta_media_y_mediana():
    """El caso que motiva la regla: media casi igual, mediana muy distinta."""
    indice = pd.period_range("2025-04", periods=4, freq="M")
    series = [f"S{i}" for i in range(10)]
    real = pd.DataFrame(100.0, index=indice, columns=series)

    # Modelo A: se equivoca poco en casi todas y muchísimo en una.
    a = real.copy()
    for s in series[:-1]:
        a[s] = 101.0
    a[series[-1]] = 300.0
    # Modelo B: se equivoca de forma pareja.
    b = real + 20.0

    escala = pd.Series(1.0, index=series)
    errores = {
        "A": metricas_por_serie(real, a, escala),
        "B": metricas_por_serie(real, b, escala),
    }
    agregado = resumen(errores).set_index("modelo")

    assert agregado.loc["A", "mae_mediana"] < agregado.loc["B", "mae_mediana"]
    assert agregado.loc["A", "mae_media"] > agregado.loc["B", "mae_media"], (
        "Este es exactamente el caso donde mirar solo la media invierte la conclusión"
    )


def test_el_resumen_usa_las_mismas_series_para_todos_los_modelos():
    indice = pd.period_range("2025-04", periods=3, freq="M")
    real = pd.DataFrame({"A": [10.0, 10.0, 10.0], "B": [20.0, 20.0, 20.0]}, index=indice)
    escala = pd.Series({"A": 1.0, "B": 1.0})

    completo = metricas_por_serie(real, real, escala)
    parcial = completo.loc[["A"]]

    agregado = resumen({"completo": completo, "parcial": parcial})
    assert set(agregado["series"]) == {1}, (
        "El resumen debe restringirse a la intersección de series evaluables"
    )


def test_matriz_de_errores_solo_deja_bloques_completos():
    indice = pd.period_range("2025-04", periods=2, freq="M")
    real = pd.DataFrame({"A": [10.0, 10.0], "B": [0.0, 0.0]}, index=indice)
    escala = pd.Series({"A": 1.0, "B": 1.0})
    tabla = metricas_por_serie(real, real, escala)

    matriz = matriz_de_errores({"m1": tabla, "m2": tabla}, metrica="mape")
    # La serie B no tiene MAPE (toda en cero) y por tanto no forma bloque completo.
    assert list(matriz.index) == ["A"]


def test_tabla8_tiene_las_columnas_del_documento():
    indice = pd.period_range("2025-04", periods=3, freq="M")
    real = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=indice)
    tabla = metricas_por_serie(real, real, pd.Series({"A": 5.0}))
    t8 = tabla8(resumen({"naive_m1": tabla}))

    for columna in ("Modelo", "MAE medio (unidades)", "MAPE medio (%)",
                    "RMSE medio (unidades)", "Bias medio (%)"):
        assert columna in t8.columns
