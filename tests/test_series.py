"""RF-3 — Criterio de aceptación.

«Una serie con ventas en enero y marzo produce tres filas (enero, febrero=0,
marzo). Una serie que empieza en marzo no genera filas de enero ni febrero.»
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.series import a_panel_ancho, construir_series


def _crudo(filas):
    return pd.DataFrame([
        {"fecha": pd.Timestamp(f), "sku": s, "canal": c, "regional": r,
         "cantidad": float(q)}
        for f, s, c, r, q in filas
    ])


def test_hueco_intermedio_se_rellena_con_cero(cfg):
    df = _crudo([
        ("2018-01-15", "A", "CANAL-1", "REGIONAL-A", 10),
        ("2018-03-20", "A", "CANAL-1", "REGIONAL-A", 30),
    ])
    panel, informe = construir_series(df, cfg)

    assert len(panel) == 3
    assert list(panel["periodo"].astype(str)) == ["2018-01", "2018-02", "2018-03"]
    assert panel["y"].tolist() == [10.0, 0.0, 30.0]
    assert informe.meses_rellenados_con_cero == 1


def test_no_se_extrapola_hacia_atras(cfg):
    """Un mes anterior al lanzamiento no es demanda cero: es ausencia de producto."""
    df = _crudo([
        ("2018-03-01", "A", "CANAL-1", "REGIONAL-A", 30),
        ("2018-04-01", "A", "CANAL-1", "REGIONAL-A", 40),
    ])
    panel, _ = construir_series(df, cfg)

    assert list(panel["periodo"].astype(str)) == ["2018-03", "2018-04"]
    assert "2018-01" not in set(panel["periodo"].astype(str))
    assert "2018-02" not in set(panel["periodo"].astype(str))


def test_no_se_extrapola_hacia_adelante(cfg):
    """Tampoco después de la última venta: un producto descontinuado no vende cero."""
    df = _crudo([
        ("2018-03-01", "A", "CANAL-1", "REGIONAL-A", 30),
        ("2018-05-01", "A", "CANAL-1", "REGIONAL-A", 10),
    ])
    panel, _ = construir_series(df, cfg)
    assert str(panel["periodo"].max()) == "2018-05"
    assert len(panel) == 3


def test_agrega_por_mes_y_combinacion(cfg):
    df = _crudo([
        ("2018-01-03", "A", "CANAL-1", "REGIONAL-A", 10),
        ("2018-01-20", "A", "CANAL-1", "REGIONAL-A", 5),
        ("2018-01-20", "A", "CANAL-2", "REGIONAL-A", 7),
    ])
    panel, _ = construir_series(df, cfg)

    por_serie = panel.set_index("serie")["y"]
    assert por_serie["A|CANAL-1|REGIONAL-A"] == 15.0
    assert por_serie["A|CANAL-2|REGIONAL-A"] == 7.0


def test_devoluciones_se_descartan_y_se_cuentan(cfg):
    df = _crudo([
        ("2018-01-01", "A", "CANAL-1", "REGIONAL-A", 10),
        ("2018-02-01", "A", "CANAL-1", "REGIONAL-A", -4),
        ("2018-03-01", "A", "CANAL-1", "REGIONAL-A", 20),
    ])
    panel, informe = construir_series(df, cfg)

    assert informe.filas_negativas == 1
    assert informe.unidades_negativas == -4.0
    if cfg.series.devoluciones == "descartar":
        assert panel.loc[panel["periodo"].astype(str) == "2018-02", "y"].item() == 0.0
    else:
        assert panel.loc[panel["periodo"].astype(str) == "2018-02", "y"].item() == -4.0


def test_el_nacimiento_es_la_primera_venta_positiva(cfg):
    """Un mes con solo devoluciones no marca el lanzamiento de un producto."""
    if cfg.series.devoluciones != "netear":
        pytest.skip("Solo aplica cuando las devoluciones se netean.")
    df = _crudo([
        ("2018-01-01", "A", "CANAL-1", "REGIONAL-A", -5),
        ("2018-03-01", "A", "CANAL-1", "REGIONAL-A", 20),
    ])
    panel, _ = construir_series(df, cfg)
    assert str(panel["periodo"].min()) == "2018-03"


def test_panel_ancho_deja_nan_fuera_de_la_vida(cfg):
    df = _crudo([
        ("2018-01-01", "A", "CANAL-1", "REGIONAL-A", 10),
        ("2018-02-01", "A", "CANAL-1", "REGIONAL-A", 20),
        ("2018-04-01", "B", "CANAL-1", "REGIONAL-A", 50),
    ])
    panel, _ = construir_series(df, cfg)
    ancho = a_panel_ancho(panel)

    assert ancho.shape[0] == 4          # enero .. abril
    assert pd.isna(ancho.loc[pd.Period("2018-01", "M"), "B|CANAL-1|REGIONAL-A"])
    assert pd.isna(ancho.loc[pd.Period("2018-04", "M"), "A|CANAL-1|REGIONAL-A"])
    assert ancho.loc[pd.Period("2018-04", "M"), "B|CANAL-1|REGIONAL-A"] == 50.0
    # Columnas en orden fijo: sin esto la reproducibilidad depende del orden de
    # iteración de un diccionario (RN-5).
    assert list(ancho.columns) == sorted(ancho.columns)


def test_recorta_al_periodo_declarado(cfg):
    df = _crudo([
        ("2010-01-01", "A", "CANAL-1", "REGIONAL-A", 99),
        ("2018-01-01", "A", "CANAL-1", "REGIONAL-A", 10),
    ])
    panel, informe = construir_series(df, cfg)
    assert informe.filas_fuera_de_periodo == 1
    assert str(panel["periodo"].min()) == "2018-01"
