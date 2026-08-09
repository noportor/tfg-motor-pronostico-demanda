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
    def __init__(self, fin_entrenamiento: pd.Period):
        self.fin_entrenamiento = fin_entrenamiento


class _CfgStub:
    """Lo mínimo que ModeloNeural consulta de la configuración."""

    def __init__(self, fin_entrenamiento: pd.Period):
        self.particion = _Particion(fin_entrenamiento)
        self.modelos = {
            "neuronales": {
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
