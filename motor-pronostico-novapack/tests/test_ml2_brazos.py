"""Contrato de los brazos ML2 (neuronales y fundacional) — pruebas de humo.

Verifican el CONTRATO con el pipeline (formas, índices, dónde hay pronóstico y
dónde NaN), no la calidad predictiva: esa se mide en la corrida real. Datos
sintéticos, como exige la RN-1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import serie_estacional

from src.modelos.neuronales import _a_largo, _pivotear


# ---------------------------------------------------------------------------
# Utilidades puras (sin torch)
# ---------------------------------------------------------------------------

def _panel_ancho(n_series: int = 8, n_meses: int = 72) -> pd.DataFrame:
    columnas = {
        f"SKU-{i:03d}|X|SANTA CRUZ": serie_estacional(n_meses, base=50 + 10 * i,
                                                      semilla=i)
        for i in range(n_series)
    }
    indice = pd.period_range("2018-01", periods=n_meses, freq="M")
    return pd.DataFrame(columnas, index=indice)


def test_ida_y_vuelta_largo_ancho():
    panel = _panel_ancho(n_series=3, n_meses=24)
    largo = _a_largo(panel)
    assert list(largo.columns) == ["unique_id", "ds", "y"]
    assert len(largo) == 3 * 24

    reconstruido = _pivotear(
        largo.rename(columns={"y": "MODELO"}), "MODELO"
    ).reindex(columns=panel.columns)
    pd.testing.assert_frame_equal(
        reconstruido, panel, check_freq=False, check_names=False,
    )


def test_el_largo_descarta_la_vida_no_nacida():
    panel = _panel_ancho(n_series=2, n_meses=24)
    panel.iloc[:6, 0] = np.nan          # la serie 0 nace en el mes 7
    largo = _a_largo(panel)
    serie0 = panel.columns[0]
    assert (largo["unique_id"] == serie0).sum() == 24 - 6


# ---------------------------------------------------------------------------
# Contrato completo con un brazo liviano (requiere neuralforecast)
# ---------------------------------------------------------------------------

class _Particion:
    def __init__(self, fin_entrenamiento: pd.Period,
                 fin_validacion: pd.Period | None = None):
        self.fin_entrenamiento = fin_entrenamiento
        self.fin_validacion = fin_validacion


class _CfgStub:
    """Lo mínimo que ModeloNeural consulta de la configuración."""

    def __init__(self, fin_entrenamiento: pd.Period, neuronales: dict | None = None):
        self.particion = _Particion(fin_entrenamiento)
        self.modelos = {
            "neuronales": neuronales if neuronales is not None else {
                "semilla": 20260408,
                "horizonte": 6,
                "input_size": 12,
                "max_steps": 30,
                "dlinear": {
                    "perdida": "mae",
                    "moving_avg_window": 13,
                    "val_check_steps": 10,
                    "early_stop_patience_steps": 2,
                },
            },
        }


def test_mezclador_pesos_composicion_y_guardia():
    """V4: pesos de validación, composición consciente de NaN y RN-2 activa."""
    from src.modelos.mezclador import Mezclador

    indice = pd.period_range("2024-01", periods=6, freq="M")
    series = ["S1|X|SC", "S2|X|SC"]
    real = pd.DataFrame(
        {"S1|X|SC": [10, 10, 10, 10, 10, 10],
         "S2|X|SC": [20, 20, 20, 20, 20, 20]},
        index=indice, dtype=float,
    )
    entrenamiento = real.iloc[:3]
    validacion = real.iloc[3:]

    # A clava la validación; B yerra por 10 en todos lados.
    predicciones = {
        "A": real.copy(),
        "B": real + 10.0,
    }

    class _Cfg:
        particion = _Particion(indice[2], fin_validacion=indice[-1])
        modelos = {"mezclador": {"candidatos": ["A", "B"]}}

    ponderado = Mezclador(_Cfg(), "mezcla_pond", predicciones, real)
    ponderado.ajustar(entrenamiento, validacion)
    pesos = ponderado.pesos.div(ponderado.pesos.sum(axis=1), axis=0)
    assert (pesos["A"] > 0.99).all()          # el error ~0 domina el peso

    promedio = Mezclador(_Cfg(), "mezcla_prom", predicciones, real)
    promedio.ajustar(entrenamiento, validacion)
    salida = promedio.predecir(real)
    assert salida.loc[indice[4], "S1|X|SC"] == pytest.approx(15.0)  # (10+20)/2

    # NaN en un candidato: su peso se redistribuye (no contamina la mezcla).
    predicciones["B"].loc[indice[4], "S1|X|SC"] = np.nan
    salida = promedio.predecir(real)
    assert salida.loc[indice[4], "S1|X|SC"] == pytest.approx(10.0)

    # RN-2: una ventana de pesos que pise la prueba tiene que reventar.
    class _CfgCorta:
        particion = _Particion(indice[2], fin_validacion=indice[3])
        modelos = {"mezclador": {"candidatos": ["A", "B"]}}

    with pytest.raises(ValueError, match="RN-2"):
        Mezclador(_CfgCorta(), "mezcla_pond", predicciones, real).ajustar(
            entrenamiento, validacion
        )


def test_directo_alinea_objetivo_sin_fuga(cfg):
    """V4: el par del horizonte h apunta a y(p+h−1) — la prueba anti-fuga."""
    from src.modelos.lightgbm_directo import ModeloLightGBMDirecto

    meses = pd.period_range("2020-01", periods=6, freq="M")
    tabla = pd.DataFrame({
        "serie": ["A|X|SC"] * 6,
        "periodo": meses,
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    modelo = ModeloLightGBMDirecto(cfg, tabla)

    h1 = modelo._pares(1)
    assert (h1["y_objetivo"] == h1["y"]).all()          # h=1 ≡ brazo base

    h3 = modelo._pares(3).dropna(subset=["y_objetivo"])
    # La fila de enero (features con info <= diciembre) apunta a marzo.
    assert h3["y_objetivo"].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert (h3["mes_objetivo"] == h3["periodo"] + 2).all()


@pytest.mark.ml2
def test_perdida_tweedie_y_exogenas_estaticas():
    """V2: la pérdida Tweedie se construye y las estáticas se codifican."""
    pytest.importorskip("neuralforecast")
    from src.modelos.neuronales import ModeloNeural

    cfg = _CfgStub(pd.Period("2024-03", freq="M"), neuronales={
        "tweedie_rho": 1.465,
        "tft": {
            "perdida": "tweedie",
            "exogenas": {"mes": True, "estaticas": ["categoria", "canal"]},
        },
    })
    tabla = pd.DataFrame({
        "serie": ["A|X|SC", "A|X|SC", "B|Z|LP"],
        "categoria": ["CUADERNOS", "CUADERNOS", "PAPELES"],
        "canal": ["X", "X", "Z"],
    })
    modelo = ModeloNeural(cfg, "tft", tabla)

    assert type(modelo._perdida()).__name__ == "DistributionLoss"
    assert modelo.usar_mes

    estaticas = modelo._armar_static_df()
    assert list(estaticas.columns) == ["unique_id", "categoria", "canal"]
    assert len(estaticas) == 2                      # una fila por serie
    assert estaticas["categoria"].dtype == "int64"  # codificadas, no texto

    # Sin tabla no puede haber estáticas: el error debe ser inmediato y claro.
    with pytest.raises(ValueError, match="tabla de features"):
        ModeloNeural(cfg, "tft", None)


@pytest.mark.ml2
def test_contrato_dlinear_formas_y_mascara():
    pytest.importorskip("neuralforecast")
    from src.modelos.neuronales import ModeloNeural

    panel = _panel_ancho(n_series=8, n_meses=72)
    fin_entrenamiento = panel.index[59]           # 60 meses de entrenamiento
    entrenamiento = panel.loc[panel.index <= fin_entrenamiento]
    validacion = panel.loc[
        (panel.index > fin_entrenamiento) & (panel.index <= panel.index[65])
    ]                                             # 6 meses de validación

    modelo = ModeloNeural(_CfgStub(fin_entrenamiento), "dlinear")
    modelo.ajustar(entrenamiento, validacion)

    pred = modelo.predecir(panel)
    assert pred.shape == panel.shape
    assert list(pred.index) == list(panel.index)
    assert list(pred.columns) == list(panel.columns)

    # Meses de entrenamiento: sin pronóstico (los completa el respaldo del
    # pipeline y no participan de ninguna métrica).
    assert pred.loc[pred.index <= fin_entrenamiento].isna().all().all()
    # Meses evaluados: pronóstico finito para todas las series vivas.
    evaluado = pred.loc[pred.index > fin_entrenamiento]
    assert np.isfinite(evaluado.to_numpy(dtype=float)).all()

    # Proyección directa: abanico h=1..H desde el origen, sin recursión.
    historia = panel.loc[panel.index <= fin_entrenamiento]
    abanico = modelo.proyectar_directo(historia, horizonte=6)
    esperados = pd.period_range(fin_entrenamiento + 1, fin_entrenamiento + 6,
                                freq="M")
    assert list(abanico.index) == list(esperados)
    assert list(abanico.columns) == list(panel.columns)
    assert np.isfinite(abanico.to_numpy(dtype=float)).all()
