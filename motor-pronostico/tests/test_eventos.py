"""Tratamiento del evento exógeno: la práctica de la empresa, verificada a mano."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import panel_de
from src.config import ErrorDeConfiguracion, cargar_config
from src.eventos import aplicar_tratamiento


def _con_evento(cfg, desde="2020-03", hasta="2021-02", tratamiento="copiar_gestion_previa"):
    return cargar_config(
        cfg.ruta_config,
        anulaciones=[
            f"evento_exogeno.desde={desde}",
            f"evento_exogeno.hasta={hasta}",
            f"evento_exogeno.tratamiento={tratamiento}",
        ],
    )


def test_copia_el_mismo_mes_del_anio_previo(cfg):
    cfg_evento = _con_evento(cfg)
    # Serie desde 2019-01: el valor de cada mes es su índice (1, 2, 3, …).
    valores = [float(i + 1) for i in range(40)]           # 2019-01 .. 2022-04
    panel = panel_de({"P1": valores}, inicio="2019-01")

    ajustado, informe = aplicar_tratamiento(panel, cfg_evento)

    original = panel.set_index("periodo")["y"]
    corregido = ajustado.set_index("periodo")["y"]

    for mes in pd.period_range("2020-03", "2021-02", freq="M"):
        assert corregido[mes] == original[mes - 12], f"{mes} no copió su año previo"
    # Fuera de la ventana, intacto.
    assert corregido[pd.Period("2020-02", "M")] == original[pd.Period("2020-02", "M")]
    assert corregido[pd.Period("2021-03", "M")] == original[pd.Period("2021-03", "M")]

    assert informe.observaciones_reemplazadas == 12
    assert informe.sin_historia_previa == 0
    assert informe.series_afectadas == 1


def test_sin_historia_previa_no_se_inventa_nada(cfg):
    """Una serie nacida en 2019-10 no tiene marzo-2019: esos meses quedan
    crudos y CONTADOS — copiar desde la nada sería fabricar datos (RN-1)."""
    cfg_evento = _con_evento(cfg)
    panel = panel_de({"N1": [10.0] * 24}, inicio="2019-10")   # 2019-10..2021-09

    ajustado, informe = aplicar_tratamiento(panel, cfg_evento)

    original = panel.set_index("periodo")["y"]
    corregido = ajustado.set_index("periodo")["y"]
    # 2020-03..2020-09 no tienen año previo (la serie nace 2019-10): crudos.
    for mes in pd.period_range("2020-03", "2020-09", freq="M"):
        assert corregido[mes] == original[mes]
    # 2020-10..2021-02 sí lo tienen: copiados.
    for mes in pd.period_range("2020-10", "2021-02", freq="M"):
        assert corregido[mes] == original[mes - 12]

    assert informe.sin_historia_previa == 7
    assert informe.observaciones_reemplazadas == 5


def test_tratamiento_nada_es_passthrough(cfg):
    cfg_evento = _con_evento(cfg, tratamiento="nada")
    panel = panel_de({"P1": [5.0] * 48}, inicio="2019-01")
    ajustado, informe = aplicar_tratamiento(panel, cfg_evento)
    pd.testing.assert_frame_equal(ajustado, panel)
    assert informe.tratamiento == "nada"
    assert informe.observaciones_reemplazadas == 0


def test_el_apagon_covid_recupera_su_historia(cfg):
    """El caso de las 181 series: activa antes, CERO durante el confinamiento.
    Con la práctica de la empresa, el ajuste ve su año previo, no el apagón."""
    cfg_evento = _con_evento(cfg)
    valores = [20.0] * 14 + [0.0] * 6 + [15.0] * 20      # apagón 2020-03..2020-08
    panel = panel_de({"A1": valores}, inicio="2019-01")

    ajustado, _ = aplicar_tratamiento(panel, cfg_evento)
    corregido = ajustado.set_index("periodo")["y"]
    for mes in pd.period_range("2020-03", "2020-08", freq="M"):
        assert corregido[mes] == 20.0, "El apagón exógeno debía reemplazarse"


def test_la_ventana_no_puede_tocar_validacion(cfg):
    with pytest.raises(ErrorDeConfiguracion) as excepcion:
        _con_evento(cfg, desde="2024-01", hasta="2024-06")
    assert "validación" in str(excepcion.value) or "corregidos" in str(excepcion.value)


def test_tratamiento_desconocido_falla(cfg):
    with pytest.raises(ErrorDeConfiguracion):
        _con_evento(cfg, tratamiento="interpolar")
