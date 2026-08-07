"""RF-6 — Criterio de aceptación.

«Cada modelo tiene una prueba con una serie construida a mano cuyo resultado
esperado se conoce de antemano.»
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import panel_de, serie_estacional
from src.modelos.base import (
    aplicar_respaldo, enmascarar_fuera_de_vida, factor_de_crecimiento,
    mae_naive_entrenamiento, truncar_en_cero,
)
from src.modelos.croston import Croston
from src.modelos.motor import Motor
from src.modelos.naive import Naive, NaiveConCrecimiento
from src.modelos.promedio_movil import PromedioMovil
from src.modelos.suavizado import HoltWinters, SuavizadoExponencial
from src.series import a_panel_ancho


def _ancho(valores: list[float], nombre: str = "A") -> pd.DataFrame:
    return a_panel_ancho(panel_de({nombre: valores}))


# ---------------------------------------------------------------------------
# Naïve
# ---------------------------------------------------------------------------

def test_naive_devuelve_el_valor_del_mes_anterior():
    panel = _ancho([10.0, 20.0, 30.0, 40.0])
    modelo = Naive()
    modelo.ajustar(panel, panel)
    predicho = modelo.predecir(panel).iloc[:, 0].tolist()

    assert np.isnan(predicho[0])
    assert predicho[1:] == [10.0, 20.0, 30.0]


def test_factor_de_crecimiento_sobre_una_serie_lineal():
    """Con y(i) = i+1, la diferencia entre dos años móviles es 144/12 = 12."""
    panel = _ancho([float(i + 1) for i in range(30)])
    gf = factor_de_crecimiento(panel).iloc[:, 0]

    assert gf.iloc[:24].isna().all(), "El GF necesita 24 meses de historia"
    assert gf.iloc[24] == pytest.approx(12.0)
    assert gf.iloc[29] == pytest.approx(12.0)


def test_naive_con_crecimiento():
    panel = _ancho([float(i + 1) for i in range(30)])
    modelo = NaiveConCrecimiento()
    modelo.ajustar(panel, panel)
    predicho = modelo.predecir(panel).iloc[:, 0]

    # En el mes 24: y(23) = 24, GF = 12  ->  36
    assert predicho.iloc[24] == pytest.approx(36.0)


# ---------------------------------------------------------------------------
# Promedio móvil
# ---------------------------------------------------------------------------

def test_promedio_movil_de_dos_meses():
    panel = _ancho([10.0, 20.0, 30.0, 40.0, 50.0])
    modelo = PromedioMovil(2)
    modelo.ajustar(panel, panel)
    predicho = modelo.predecir(panel).iloc[:, 0].tolist()

    assert np.isnan(predicho[0]) and np.isnan(predicho[1])
    assert predicho[2] == pytest.approx(15.0)   # (10 + 20) / 2
    assert predicho[3] == pytest.approx(25.0)   # (20 + 30) / 2
    assert predicho[4] == pytest.approx(35.0)


def test_promedio_movil_de_doce_meses():
    valores = [float(i + 1) for i in range(15)]   # 1 .. 15
    panel = _ancho(valores)
    modelo = PromedioMovil(12)
    modelo.ajustar(panel, panel)
    predicho = modelo.predecir(panel).iloc[:, 0]

    assert predicho.iloc[:12].isna().all()
    assert predicho.iloc[12] == pytest.approx(np.mean(valores[0:12]))   # 6.5
    assert predicho.iloc[14] == pytest.approx(np.mean(valores[2:14]))


def test_el_promedio_movil_no_emite_con_historia_insuficiente():
    """§10: no se rellena un valor faltante con una estimación plausible."""
    panel = _ancho([10.0, 20.0])
    modelo = PromedioMovil(12)
    modelo.ajustar(panel, panel)
    assert modelo.predecir(panel).iloc[:, 0].isna().all()


# ---------------------------------------------------------------------------
# Suavizado exponencial simple
# ---------------------------------------------------------------------------

def test_suavizado_exponencial_sigue_la_recursion_del_libro():
    valores = [10.0, 20.0, 30.0, 40.0]
    panel = _ancho(valores)
    modelo = SuavizadoExponencial([0.5])       # rejilla de un solo alfa
    modelo.ajustar(panel, panel)
    predicho = modelo.predecir(panel).iloc[:, 0].tolist()

    # nivel(0) = 10          -> ŷ(1) = 10
    # nivel(1) = .5·20 + .5·10 = 15   -> ŷ(2) = 15
    # nivel(2) = .5·30 + .5·15 = 22.5 -> ŷ(3) = 22.5
    assert np.isnan(predicho[0])
    assert predicho[1] == pytest.approx(10.0)
    assert predicho[2] == pytest.approx(15.0)
    assert predicho[3] == pytest.approx(22.5)


def test_suavizado_exponencial_elige_alfa_por_serie():
    """Serie con nivel constante: gana el alfa que menos reacciona al ruido."""
    constante = [50.0] * 40
    escalon = [10.0] * 20 + [90.0] * 20
    panel = a_panel_ancho(panel_de({"CONST": constante, "SALTO": escalon}))

    modelo = SuavizadoExponencial([0.05, 0.95])
    modelo.ajustar(panel, panel)

    # En una serie perfectamente constante ambos alfas aciertan igual: gana el
    # primero de la rejilla por el desempate declarado.
    assert modelo.alfa_por_serie["CONST|CANAL-1|REGIONAL-A"] == pytest.approx(0.05)
    # Ante un escalón conviene reaccionar rápido.
    assert modelo.alfa_por_serie["SALTO|CANAL-1|REGIONAL-A"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Holt-Winters
# ---------------------------------------------------------------------------

def test_holt_winters_reproduce_una_serie_estacional_perfecta():
    """Con nivel + tendencia + estacionalidad exactos, el error debe ser mínimo."""
    m = 12
    estacional = [0, 5, 15, 30, 10, -5, -10, -5, 0, 5, 20, 40]
    valores = [100.0 + 2.0 * t + estacional[t % m] for t in range(60)]
    panel = _ancho(valores)

    modelo = HoltWinters(periodo_estacional=m,
                         alfas=[0.1, 0.3], betas=[0.05, 0.2], gammas=[0.05, 0.2])
    modelo.ajustar(panel, panel)
    predicho = modelo.predecir(panel).iloc[:, 0]

    # Los primeros 2m meses se consumen en la inicialización.
    assert predicho.iloc[: 2 * m].isna().all()
    real = pd.Series(valores, index=predicho.index)
    error = (predicho - real).abs().iloc[2 * m:]
    naive = (real - real.shift(1)).abs().iloc[2 * m:]

    assert error.mean() < naive.mean(), (
        "Holt-Winters debería batir al Naïve en una serie estacional perfecta"
    )


def test_holt_winters_no_usa_informacion_futura():
    m = 12
    valores = serie_estacional(48)
    panel = _ancho(valores)
    modelo = HoltWinters(periodo_estacional=m, alfas=[0.3], betas=[0.1], gammas=[0.1])
    modelo.ajustar(panel, panel)
    original = modelo.predecir(panel).iloc[:, 0]

    alterado = list(valores)
    alterado[40] = 999_999.0
    modificado = modelo.predecir(_ancho(alterado)).iloc[:, 0]

    pd.testing.assert_series_equal(
        original.iloc[:41], modificado.iloc[:41], check_names=False,
        obj="Alterar el mes 40 movió pronósticos anteriores",
    )


# ---------------------------------------------------------------------------
# Croston
# ---------------------------------------------------------------------------

def test_croston_sigue_la_recursion_del_articulo():
    # Demanda en los meses 0 y 3; ceros en medio.
    valores = [10.0, 0.0, 0.0, 20.0, 0.0, 0.0]
    panel = _ancho(valores)
    modelo = Croston([0.5])
    modelo.ajustar(panel, panel)
    predicho = modelo.predecir(panel).iloc[:, 0].tolist()

    # t=0: sin estado -> NaN. Se inicializa z=10, p=1 (q valía 1).
    assert np.isnan(predicho[0])
    # t=1,2,3: ŷ = 10 / 1 = 10, constante hasta el siguiente pedido.
    assert predicho[1] == pytest.approx(10.0)
    assert predicho[2] == pytest.approx(10.0)
    assert predicho[3] == pytest.approx(10.0)
    # En t=3 hay pedido con q = 3: z = .5·20 + .5·10 = 15 ; p = .5·3 + .5·1 = 2
    assert predicho[4] == pytest.approx(15.0 / 2.0)
    assert predicho[5] == pytest.approx(7.5)


def test_croston_es_estable_en_demanda_intermitente():
    """Frente al Naïve, no colapsa a cero después de un mes sin venta."""
    valores = ([0.0] * 3 + [40.0]) * 10
    panel = _ancho(valores)
    modelo = Croston([0.1, 0.2])
    modelo.ajustar(panel, panel)
    predicho = modelo.predecir(panel).iloc[:, 0]

    real = pd.Series(valores, index=predicho.index)
    error_croston = (predicho - real).abs().iloc[8:].mean()
    error_naive = (real.shift(1) - real).abs().iloc[8:].mean()
    assert error_croston < error_naive


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------

def test_las_predicciones_se_truncan_en_cero():
    panel = pd.DataFrame({"A": [-5.0, 3.0, np.nan]})
    truncado = truncar_en_cero(panel)
    assert truncado["A"].tolist()[:2] == [0.0, 3.0]
    assert np.isnan(truncado["A"].iloc[2]), "El NaN no es lo mismo que el cero"


def test_se_enmascaran_los_meses_fuera_de_la_vida_de_la_serie():
    real = pd.DataFrame({"A": [10.0, 20.0, np.nan]})
    prediccion = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    enmascarado = enmascarar_fuera_de_vida(prediccion, real)
    assert np.isnan(enmascarado["A"].iloc[2])


def test_el_respaldo_completa_los_huecos_y_los_cuenta():
    real = pd.DataFrame({"A": [10.0, 20.0, 30.0]})
    prediccion = pd.DataFrame({"A": [np.nan, np.nan, 25.0]})
    completado, cuantos = aplicar_respaldo(prediccion, real, real)

    assert cuantos == 2
    assert completado["A"].iloc[1] == pytest.approx(10.0)   # naive
    assert completado["A"].iloc[0] == pytest.approx(20.0)   # media de entrenamiento
    assert completado["A"].isna().sum() == 0


def test_escala_del_mase():
    entrenamiento = pd.DataFrame({"A": [10.0, 12.0, 9.0, 15.0]})
    escala = mae_naive_entrenamiento(entrenamiento)
    # |12-10| + |9-12| + |15-9| = 2 + 3 + 6 = 11 ; 11 / 3
    assert escala["A"] == pytest.approx(11.0 / 3.0)


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class _ModeloFijo:
    """Modelo de laboratorio que devuelve un panel prefijado."""

    def __init__(self, nombre, panel):
        self.nombre = nombre
        self._panel = panel

    def ajustar(self, entrenamiento, validacion):
        return None

    def predecir(self, datos):
        return self._panel.reindex(index=datos.index, columns=datos.columns)


def test_el_motor_elige_por_serie_el_ganador_en_validacion(cfg):
    corte_train = cfg.particion.fin_entrenamiento
    corte_val = cfg.particion.fin_validacion
    meses = pd.period_range(cfg.periodo.inicio, cfg.periodo.fin, freq="M")

    real = pd.DataFrame(100.0, index=meses, columns=["S1", "S2"])
    bueno_en_s1 = real.copy()
    bueno_en_s1["S1"] = 100.0     # perfecto en S1
    bueno_en_s1["S2"] = 180.0     # pésimo en S2
    bueno_en_s2 = real.copy()
    bueno_en_s2["S1"] = 180.0
    bueno_en_s2["S2"] = 100.0

    predicciones = {"ma_12": bueno_en_s1, "naive_m1": bueno_en_s2}
    motor = Motor(cfg, predicciones, real)
    motor.ajustar(
        real.loc[real.index <= corte_train],
        real.loc[(real.index > corte_train) & (real.index <= corte_val)],
    )

    assert motor._elegido["S1"] == "ma_12"
    assert motor._elegido["S2"] == "naive_m1"

    salida = motor.predecir(real)
    assert salida["S1"].iloc[-1] == pytest.approx(100.0)
    assert salida["S2"].iloc[-1] == pytest.approx(100.0)


def test_el_motor_rechaza_una_ventana_que_toque_la_prueba(cfg):
    meses = pd.period_range(cfg.periodo.inicio, cfg.periodo.fin, freq="M")
    real = pd.DataFrame(100.0, index=meses, columns=["S1"])
    predicciones = {"naive_m1": real.copy()}
    motor = Motor(cfg, predicciones, real)

    # Ventana de selección contaminada con meses de prueba.
    with pytest.raises(ValueError) as excepcion:
        motor.ajustar(real, real)
    assert "RN-2" in str(excepcion.value)
