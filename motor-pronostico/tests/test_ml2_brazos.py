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
        f"SKU-{i:03d}|X|SC": serie_estacional(n_meses, base=50 + 10 * i,
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


def test_mezcla_h_pesos_por_horizonte_composicion_y_guardia():
    """V5: pesos por candidato × horizonte del abanico de validación (RN-2)."""
    from src.modelos.mezclador import MezcladorHorizonte

    indice_val = pd.period_range("2024-04", periods=4, freq="M")
    series = ["S1|X|SC", "S2|X|SC"]
    real_fan = pd.DataFrame(
        {"S1|X|SC": [10.0] * 4, "S2|X|SC": [20.0] * 4}, index=indice_val
    )
    # A clava los horizontes cortos y yerra los largos; B al revés.
    fan_a = real_fan.copy()
    fan_a.iloc[2:] += 10.0
    fan_b = real_fan.copy()
    fan_b.iloc[:2] += 10.0

    class _Cfg:
        particion = _Particion(
            pd.Period("2024-03", freq="M"),
            fin_validacion=pd.Period("2024-07", freq="M"),
        )
        modelos = {"mezclador": {"candidatos": ["A", "B"]}}

    predicciones = {"A": real_fan.copy(), "B": real_fan + 10.0}
    costo = pd.Series(1.0, index=series)

    mezcla = MezcladorHorizonte(_Cfg(), predicciones, real_fan, costo)
    mezcla.ajustar({"A": fan_a, "B": fan_b}, real_fan)
    pesos = mezcla.pesos_h.div(mezcla.pesos_h.sum(axis=1), axis=0)
    assert pesos.loc[1, "A"] > 0.99          # D≈0 domina el peso en h=1
    assert pesos.loc[4, "B"] > 0.99          # ... y B domina en h=4

    # componer aplica el peso de CADA horizonte: fila 1 ≈ A, fila 4 ≈ B.
    proy_a = pd.DataFrame(
        {"S1|X|SC": [1.0] * 4, "S2|X|SC": [2.0] * 4}, index=indice_val
    )
    proy_b = proy_a + 100.0
    compuesto = mezcla.componer({"A": proy_a, "B": proy_b})
    assert compuesto.iloc[0, 0] == pytest.approx(1.0, abs=0.1)
    assert compuesto.iloc[3, 0] == pytest.approx(101.0, abs=0.1)

    # NaN en el dominante: su peso se redistribuye (no deja hueco).
    proy_a.iloc[0, 0] = np.nan
    compuesto = mezcla.componer({"A": proy_a, "B": proy_b})
    assert compuesto.iloc[0, 0] == pytest.approx(101.0)

    # A un paso aplican los pesos de h=1: la mezcla sigue al candidato A
    # (10), no al B (20).
    a_un_paso = mezcla.predecir(real_fan)
    assert a_un_paso.iloc[0, 0] == pytest.approx(10.0, abs=0.1)

    # RN-2: un abanico que pise la prueba tiene que reventar.
    indice_largo = pd.period_range("2024-05", periods=4, freq="M")
    with pytest.raises(ValueError, match="RN-2"):
        mezcla.ajustar(
            {"A": fan_a.set_axis(indice_largo), "B": fan_b.set_axis(indice_largo)},
            real_fan.set_axis(indice_largo),
        )

    # Candidato declarado sin abanico: error inmediato y claro.
    with pytest.raises(ValueError, match="sin abanico"):
        mezcla.ajustar({"A": fan_a}, real_fan)


def test_directo_reentrena_por_origen(cfg, monkeypatch):
    """V5: re-entrena los 12 boosters por origen NUEVO, con árboles congelados
    y sin ver un solo mes posterior al origen (anti-fuga)."""
    import lightgbm as lgb

    from src.modelos.lightgbm_directo import ModeloLightGBMDirecto

    fin_ent = cfg.particion.fin_entrenamiento
    fin_val = cfg.particion.fin_validacion
    meses = pd.period_range(fin_ent - 23, fin_val + 3, freq="M")
    tabla = pd.DataFrame({
        "serie": ["A|X|SC"] * len(meses),
        "periodo": meses,
        "y": np.arange(len(meses), dtype=float),
    })
    modelo = ModeloLightGBMDirecto(cfg, tabla)
    for columna in modelo.columnas:
        if columna not in tabla.columns:
            tabla[columna] = 0.0

    class _BoosterFalso:
        best_iteration = 7

        def predict(self, datos, num_iteration=None):
            return np.zeros(len(datos))

    horizontes = range(1, modelo.horizonte + 1)
    modelo.boosters = {h: _BoosterFalso() for h in horizontes}
    modelo.mejor_iteracion = {h: 7 for h in horizontes}
    modelo._parametros = {"objective": "regression"}
    modelo._fin_ajuste = fin_ent
    modelo.reentrenar_por_origen = True

    llamadas: list[tuple[int, float]] = []

    class _DatasetFalso:
        def __init__(self, datos, label=None, **kw):
            self.label = np.asarray(label, dtype=float)

    def _entrenar_falso(params, conjunto, num_boost_round=None, **kw):
        llamadas.append((num_boost_round, float(np.nanmax(conjunto.label))))
        return _BoosterFalso()

    monkeypatch.setattr(lgb, "Dataset", _DatasetFalso)
    monkeypatch.setattr(lgb, "train", _entrenar_falso)

    def _historia(hasta):
        idx = pd.period_range(meses[0], hasta, freq="M")
        return pd.DataFrame(
            {"A|X|SC": np.arange(len(idx), dtype=float)}, index=idx
        )

    # Origen sin meses nuevos (== fin del ajuste base): NO re-entrena.
    modelo.proyectar_directo(_historia(fin_ent), 3)
    assert llamadas == [] and modelo.reentrenos == 0

    # Origen nuevo: un booster por horizonte, árboles congelados (7), y
    # ningún objetivo posterior al origen (la y es el índice del mes).
    y_en_origen = float(meses.get_loc(fin_val))
    modelo.proyectar_directo(_historia(fin_val), 3)
    assert len(llamadas) == modelo.horizonte
    assert all(n == 7 for n, _ in llamadas)
    assert all(maximo == y_en_origen for _, maximo in llamadas)
    assert modelo.reentrenos == modelo.horizonte

    # Mismo origen otra vez: la caché evita repetir el trabajo.
    modelo.proyectar_directo(_historia(fin_val), 3)
    assert len(llamadas) == modelo.horizonte


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
