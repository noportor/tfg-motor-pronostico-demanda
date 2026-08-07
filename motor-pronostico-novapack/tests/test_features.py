"""RF-5 — La prueba de no-fuga temporal.

Esta es la garantía de la RN-3 y, según el plan de trabajo, se escribe ANTES que
el módulo que verifica.

La idea es simple y no admite matices: se alteran valores FUTUROS de una serie y
se comprueba que **ninguna** feature de los meses anteriores cambia. Si una sola
cambiara, el modelo estaría viendo el futuro y todos los números del documento
serían inválidos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import panel_de, serie_estacional
from src.features import COLUMNAS_CONTEXTO, construir_features


def _features(panel, cfg):
    return construir_features(panel, cfg).sort_values(["serie", "periodo"]).reset_index(drop=True)


def test_alterar_el_futuro_no_cambia_ninguna_feature_del_presente(cfg):
    """El corazón de la RN-3."""
    valores = serie_estacional(60)
    panel = panel_de({"P1": valores})
    original = _features(panel, cfg)

    # Se altera el mes 40 (índice 40) de forma brutal: ×1000 y con signo.
    alterado = list(valores)
    alterado[40] = -999_999.0
    modificado = _features(panel_de({"P1": alterado}), cfg)

    columnas_feature = [
        c for c in original.columns if c not in COLUMNAS_CONTEXTO and c != "y"
    ]
    assert columnas_feature, "El módulo no produjo ninguna feature."

    presente = original.loc[original.index <= 40, columnas_feature]
    presente_mod = modificado.loc[modificado.index <= 40, columnas_feature]

    pd.testing.assert_frame_equal(
        presente, presente_mod,
        obj="Features de los meses 0..40 tras alterar el valor del mes 40",
    )


def test_alterar_el_ultimo_valor_no_cambia_nada_salvo_su_propio_objetivo(cfg):
    """Caso límite: solo cambia `y` de la última fila, ninguna feature."""
    valores = serie_estacional(48)
    original = _features(panel_de({"P1": valores}), cfg)

    alterado = list(valores)
    alterado[-1] = alterado[-1] + 12_345.0
    modificado = _features(panel_de({"P1": alterado}), cfg)

    columnas_feature = [
        c for c in original.columns if c not in COLUMNAS_CONTEXTO and c != "y"
    ]
    pd.testing.assert_frame_equal(
        original[columnas_feature], modificado[columnas_feature],
        obj="Alterar el ÚLTIMO valor no puede mover ninguna feature de la tabla",
    )
    assert original["y"].iloc[-1] != modificado["y"].iloc[-1]


def test_cada_alteracion_futura_deja_intacto_todo_lo_anterior(cfg):
    """Barrido: se altera cada mes t y se verifica que 0..t-1 no se movió."""
    valores = serie_estacional(36)
    original = _features(panel_de({"P1": valores}), cfg)
    columnas_feature = [
        c for c in original.columns if c not in COLUMNAS_CONTEXTO and c != "y"
    ]

    for t in range(12, 36):
        alterado = list(valores)
        alterado[t] = alterado[t] * 50 + 7777
        modificado = _features(panel_de({"P1": alterado}), cfg)
        pd.testing.assert_frame_equal(
            original.loc[: t - 1, columnas_feature],
            modificado.loc[: t - 1, columnas_feature],
            obj=f"Alterar el mes {t} movió features de meses anteriores",
        )


def test_ninguna_feature_usa_estadisticos_del_conjunto_completo(cfg):
    """Una serie NUEVA no puede alterar las features de otra serie.

    Es la comprobación de que no hay normalizaciones ni codificaciones ajustadas
    sobre el panel entero, que es la otra forma —menos evidente— de fuga que
    prohíbe la RN-3.
    """
    valores = serie_estacional(48)
    solo_una = _features(panel_de({"P1": valores}), cfg)

    con_companera = _features(
        panel_de({"P1": valores, "P2": [v * 1000 for v in serie_estacional(48, semilla=99)]}),
        cfg,
    )
    con_companera = con_companera.loc[con_companera["sku"] == "P1"].reset_index(drop=True)

    numericas = [
        c for c in solo_una.columns
        if c not in COLUMNAS_CONTEXTO and pd.api.types.is_numeric_dtype(solo_una[c])
    ]
    pd.testing.assert_frame_equal(
        solo_una[numericas], con_companera[numericas],
        obj="Agregar otra serie al panel cambió las features de la primera",
    )


def test_rezagos_son_exactamente_el_valor_de_hace_k_meses(cfg):
    """Verificación directa contra una serie construida a mano."""
    valores = [float(i) for i in range(1, 25)]  # 1, 2, 3, ... 24
    tabla = _features(panel_de({"P1": valores}), cfg)

    for k in cfg.features.rezagos:
        columna = f"rezago_{k}"
        assert columna in tabla.columns
        # En la fila i (0-based), el rezago k debe valer valores[i - k].
        for i in range(len(valores)):
            esperado = valores[i - k] if i - k >= 0 else np.nan
            obtenido = tabla[columna].iloc[i]
            if np.isnan(esperado):
                assert np.isnan(obtenido), f"{columna} fila {i}: debía ser NaN"
            else:
                assert obtenido == pytest.approx(esperado), f"{columna} fila {i}"


def test_medias_moviles_se_calculan_sobre_valores_desplazados(cfg):
    """La media móvil de 3 en el mes t es la media de t-1, t-2 y t-3."""
    valores = [float(i) for i in range(1, 25)]
    tabla = _features(panel_de({"P1": valores}), cfg)

    if 3 not in cfg.features.ventanas_moviles:
        pytest.skip("La ventana 3 no está configurada.")

    # fila 5 (0-based) -> valores[2], valores[3], valores[4] = 3, 4, 5 -> 4.0
    assert tabla["media_movil_3"].iloc[5] == pytest.approx(4.0)
    # las tres primeras filas no tienen tres valores previos
    assert tabla["media_movil_3"].iloc[:3].isna().all()


def test_la_tendencia_no_depende_de_valores_futuros(cfg):
    """La tendencia es la antigüedad de la serie, no un ajuste sobre los datos."""
    if not cfg.features.incluir_tendencia:
        pytest.skip("La tendencia no está configurada.")

    valores = serie_estacional(30)
    original = _features(panel_de({"P1": valores}), cfg)

    alterado = list(valores)
    alterado[25] = 10_000_000.0
    modificado = _features(panel_de({"P1": alterado}), cfg)

    pd.testing.assert_series_equal(original["tendencia"], modificado["tendencia"])
    assert original["tendencia"].iloc[0] == 0
    assert original["tendencia"].is_monotonic_increasing
