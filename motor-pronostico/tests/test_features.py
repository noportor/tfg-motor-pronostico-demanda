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


def _exogenas_de(panel, precios=None, perdidas=None, quiebres=None, tc=None,
                 categorias=None):
    """Paquete de exógenas sintético con la misma forma que src.exogenas."""
    from types import SimpleNamespace

    def _largo(valores, columna):
        if valores is None:
            return None
        filas = []
        for serie, lista in valores.items():
            meses = pd.period_range("2017-04", periods=len(lista), freq="M")
            for periodo, valor in zip(meses, lista):
                filas.append({"serie": serie, "periodo": periodo, columna: valor})
        return pd.DataFrame(filas)

    return SimpleNamespace(
        precio=_largo(precios, "precio"),
        perdidas=_largo(perdidas, "perdidas"),
        quiebres=quiebres,
        tipo_cambio=tc,
        categoria_por_serie=categorias,
    )


def test_precio_rezagado_a_mano(cfg):
    if not cfg.features.v2.get("precio"):
        pytest.skip("features.v2.precio apagada.")
    panel = panel_de({"P1": [10.0] * 6})
    serie = panel["serie"].iloc[0]
    exogenas = _exogenas_de(panel, precios={serie: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    tabla = _features_con(panel, cfg, exogenas)
    # Fila 3: precio_rezago_1 = precio del mes 2 (3.0); delta = (3-2)/2.
    assert tabla["precio_rezago_1"].iloc[3] == pytest.approx(3.0)
    assert tabla["precio_delta_pct"].iloc[3] == pytest.approx(0.5)
    assert np.isnan(tabla["precio_rezago_1"].iloc[0])


def test_exogenas_ausentes_son_nan_no_cero(cfg):
    """Sin fuentes, las columnas exógenas existen y son NaN: «sin dato»
    declarado — un cero inventado diría «no hubo quiebre/pérdida»."""
    panel = panel_de({"P1": [10.0] * 6})
    tabla = _features_con(panel, cfg, None)
    for columna in ("precio_rezago_1", "perdidas_rezago_1",
                    "quiebre_pct_rezago_1", "tc_rezago_1"):
        if columna in tabla.columns:
            assert tabla[columna].isna().all(), f"{columna} debía ser NaN"


def test_alterar_exogenas_futuras_no_cambia_el_presente(cfg):
    """La no-fuga también rige para las exógenas: cambiar el precio de un mes
    futuro no puede mover ninguna feature de meses anteriores."""
    if not cfg.features.v2.get("precio"):
        pytest.skip("features.v2.precio apagada.")
    valores = serie_estacional(24)
    panel = panel_de({"P1": valores})
    serie = panel["serie"].iloc[0]

    precios_a = [2.0] * 24
    precios_b = list(precios_a)
    precios_b[20] = 999.0                      # solo el mes 20 cambia

    tabla_a = _features_con(panel, cfg, _exogenas_de(panel, precios={serie: precios_a}))
    tabla_b = _features_con(panel, cfg, _exogenas_de(panel, precios={serie: precios_b}))

    columnas = [c for c in tabla_a.columns if c not in COLUMNAS_CONTEXTO and c != "y"]
    pd.testing.assert_frame_equal(
        tabla_a.loc[:19, columnas], tabla_b.loc[:19, columnas],
        obj="Features de los meses 0..19 tras alterar el precio del mes 20",
    )


def test_perdidas_y_quiebres_rezagados_a_mano(cfg):
    """Los dos grupos exógenos que el guardián de no-fuga no cubría por
    defecto: valor rezagado verificado a mano (hallazgo de auditoría)."""
    v2 = cfg.features.v2
    if not (v2.get("perdidas") and v2.get("quiebres")):
        pytest.skip("features.v2.perdidas/quiebres apagadas.")
    panel = panel_de({"P1": [10.0] * 6})
    serie = panel["serie"].iloc[0]
    sku, _, regional = serie.split("|")

    meses = pd.period_range("2017-04", periods=6, freq="M")
    quiebres = pd.DataFrame({
        "sku": pd.array([sku] * 6, dtype="string"),
        "regional": pd.array([regional] * 6, dtype="string"),
        "periodo": meses,
        "semanas_medidas": [4] * 6,
        "semanas_en_cero": [0, 2, 4, 0, 1, 3],
    })
    exogenas = _exogenas_de(
        panel, perdidas={serie: [0.0, 5.0, 0.0, 7.0, 0.0, 0.0]},
        quiebres=quiebres,
    )
    tabla = _features_con(panel, cfg, exogenas)

    # Fila 2: perdidas_rezago_1 = pérdidas del mes 1 (5.0).
    assert tabla["perdidas_rezago_1"].iloc[2] == pytest.approx(5.0)
    # Fila 3: quiebre del mes 2 = 4/4 = 1.0.
    assert tabla["quiebre_pct_rezago_1"].iloc[3] == pytest.approx(1.0)
    # Fila 4: móvil 3 de quiebres de los meses 1..3 = (0.5 + 1.0 + 0.0)/3.
    assert tabla["quiebre_pct_movil_3"].iloc[4] == pytest.approx(0.5)


def test_alterar_perdidas_y_quiebres_futuros_no_cambia_el_presente(cfg):
    """El barrido de no-fuga para los grupos exógenos restantes."""
    v2 = cfg.features.v2
    if not (v2.get("perdidas") and v2.get("quiebres")):
        pytest.skip("features.v2.perdidas/quiebres apagadas.")
    valores = serie_estacional(24)
    panel = panel_de({"P1": valores})
    serie = panel["serie"].iloc[0]
    sku, _, regional = serie.split("|")
    meses = pd.period_range("2017-04", periods=24, freq="M")

    def armar(perdidas_20, quiebre_20):
        perdidas = [1.0] * 24
        perdidas[20] = perdidas_20
        en_cero = [1] * 24
        en_cero[20] = quiebre_20
        quiebres = pd.DataFrame({
            "sku": pd.array([sku] * 24, dtype="string"),
            "regional": pd.array([regional] * 24, dtype="string"),
            "periodo": meses,
            "semanas_medidas": [4] * 24,
            "semanas_en_cero": en_cero,
        })
        return _exogenas_de(panel, perdidas={serie: perdidas}, quiebres=quiebres)

    tabla_a = _features_con(panel, cfg, armar(1.0, 1))
    tabla_b = _features_con(panel, cfg, armar(999.0, 4))

    columnas = [c for c in tabla_a.columns if c not in COLUMNAS_CONTEXTO and c != "y"]
    pd.testing.assert_frame_equal(
        tabla_a.loc[:19, columnas], tabla_b.loc[:19, columnas],
        obj="Features de los meses 0..19 tras alterar pérdidas/quiebres del mes 20",
    )


def test_tipo_cambio_es_el_del_mes_anterior(cfg):
    if not cfg.features.v2.get("tipo_cambio"):
        pytest.skip("features.v2.tipo_cambio apagada.")
    panel = panel_de({"P1": [10.0] * 6})
    meses = pd.period_range("2017-04", periods=6, freq="M")
    tc = pd.Series([6.0, 7.0, 8.0, 9.0, 10.0, 11.0], index=meses)

    tabla = _features_con(panel, cfg, _exogenas_de(panel, tc=tc))
    # Fila 2 (mes 2017-06): tc_rezago_1 = tc de 2017-05 = 7.0;
    # delta = (7-6)/6.
    assert tabla["tc_rezago_1"].iloc[2] == pytest.approx(7.0)
    assert tabla["tc_delta_pct"].iloc[2] == pytest.approx(1 / 6)


def test_recencia_y_media_del_mismo_mes_a_mano(cfg):
    if not cfg.features.v2.get("derivadas"):
        pytest.skip("features.v2.derivadas apagadas.")
    # 26 meses: venta en los meses 0, 3 y luego nada.
    valores = [5.0, 0.0, 0.0, 8.0] + [0.0] * 22
    tabla = _features_con(panel_de({"P1": valores}), cfg, None)

    # Mes 2: última venta ANTERIOR fue el mes 0 -> recencia 2.
    assert tabla["meses_desde_venta"].iloc[2] == pytest.approx(2.0)
    # Mes 4: última venta anterior fue el mes 3 -> recencia 1.
    assert tabla["meses_desde_venta"].iloc[4] == pytest.approx(1.0)
    # Mes 0: sin venta previa -> NaN.
    assert np.isnan(tabla["meses_desde_venta"].iloc[0])

    # media_mismo_mes: la fila 24 es el MISMO mes calendario que la fila 12 y
    # la 0 -> media de y[0] y y[12] = (5 + 0) / 2.
    assert tabla["media_mismo_mes"].iloc[24] == pytest.approx(2.5)
    # La fila 12 solo tiene un año previo: y[0] = 5.
    assert tabla["media_mismo_mes"].iloc[12] == pytest.approx(5.0)


def _features_con(panel, cfg, exogenas):
    return (
        construir_features(panel, cfg, exogenas)
        .sort_values(["serie", "periodo"]).reset_index(drop=True)
    )


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


def test_calendario_comercial_declarado(cfg):
    """V6: temporada por categoria x mes y cuenta regresiva a clases.

    Ambas features son deterministas del calendario: no usan NINGUNA
    observacion (la prueba altera `y` y exige igualdad exacta) y son
    computables para meses futuros sin persistencia.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    features_cal = replace(cfg.features, calendario={
        "inicio_clases_mes": 2,
        "temporadas": {
            "CUADERNOS": [12, 1, 2],
            "PLASTICOS": [10, 11, 12],
        },
    })
    cfg_cal = replace(cfg, features=features_cal)

    valores = serie_estacional(24)  # 2017-04 .. 2019-03
    panel = panel_de({"CUAD-1": valores, "FOTO-1": valores})
    exogenas = SimpleNamespace(categoria_por_serie={
        "CUAD-1|CANAL-1|REGIONAL-A": "CUADERNOS",
        "FOTO-1|CANAL-1|REGIONAL-A": "PAPEL FOTOCOPIA",
    })
    tabla = (
        construir_features(panel, cfg_cal, exogenas)
        .sort_values(["serie", "periodo"]).reset_index(drop=True)
    )
    assert {"temporada_alta", "meses_a_clases"} <= set(tabla.columns)

    cuadernos = tabla[tabla["serie"] == "CUAD-1|CANAL-1|REGIONAL-A"]
    fila = cuadernos.set_index(
        cuadernos["periodo"].astype(str)
    )

    # CUADERNOS: temporada en dic-ene-feb, fuera en marzo y octubre.
    assert fila.loc["2017-12", "temporada_alta"] == 1.0
    assert fila.loc["2018-01", "temporada_alta"] == 1.0
    assert fila.loc["2018-02", "temporada_alta"] == 1.0
    assert fila.loc["2018-03", "temporada_alta"] == 0.0
    assert fila.loc["2017-10", "temporada_alta"] == 0.0

    # FOTOCOPIA no esta en el calendario: nunca temporada (venta regular).
    fotocopia = tabla[tabla["serie"] == "FOTO-1|CANAL-1|REGIONAL-A"]
    assert (fotocopia["temporada_alta"] == 0.0).all()

    # Cuenta regresiva circular al inicio de clases (febrero).
    assert fila.loc["2018-02", "meses_a_clases"] == 0.0
    assert fila.loc["2018-01", "meses_a_clases"] == 1.0
    assert fila.loc["2017-12", "meses_a_clases"] == 2.0
    assert fila.loc["2018-03", "meses_a_clases"] == 11.0

    # RN-3 trivial y verificada: alterar la demanda no mueve el calendario.
    alterado = [v * 1000.0 for v in valores]
    panel2 = panel_de({"CUAD-1": alterado, "FOTO-1": alterado})
    tabla2 = (
        construir_features(panel2, cfg_cal, exogenas)
        .sort_values(["serie", "periodo"]).reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        tabla["temporada_alta"], tabla2["temporada_alta"]
    )
    pd.testing.assert_series_equal(
        tabla["meses_a_clases"], tabla2["meses_a_clases"]
    )

    # Serie sin categoria conocida: NaN declarado, nunca cero inventado.
    exogenas_sin = SimpleNamespace(categoria_por_serie={})
    tabla3 = construir_features(panel, cfg_cal, exogenas_sin)
    assert tabla3["temporada_alta"].isna().all()
