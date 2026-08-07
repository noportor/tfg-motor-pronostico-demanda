"""Validación de la configuración y anulaciones para ablaciones."""

from __future__ import annotations

import pytest

from src.config import (
    ErrorDeConfiguracion, aplicar_anulaciones, cargar_config, gestion_de, mes_a_periodo,
)


# ---------------------------------------------------------------------------
# Gestión fiscal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mes, gestion_esperada",
    [
        ("2017-03", 2017),   # cierra la gestión 2017
        ("2017-04", 2018),   # abre la gestión 2018
        ("2017-12", 2018),
        ("2018-01", 2018),
        ("2018-03", 2018),   # cierra la gestión 2018
        ("2018-04", 2019),
        ("2026-03", 2026),   # última gestión cerrada
    ],
)
def test_la_gestion_fiscal_va_de_abril_a_marzo(mes, gestion_esperada):
    assert gestion_de(mes_a_periodo(mes), 4) == gestion_esperada


def test_la_configuracion_real_cubre_gestiones_completas(cfg):
    inicio = gestion_de(cfg.periodo.inicio, cfg.periodo.mes_inicio_gestion)
    fin = gestion_de(cfg.periodo.fin, cfg.periodo.mes_inicio_gestion)
    assert cfg.periodo.inicio.month == cfg.periodo.mes_inicio_gestion
    assert cfg.periodo.fin.month == (cfg.periodo.mes_inicio_gestion - 1) or 12
    assert fin > inicio


# ---------------------------------------------------------------------------
# Anulaciones
# ---------------------------------------------------------------------------

def test_anular_cambia_solo_la_clave_indicada(cfg):
    anulado = aplicar_anulaciones(cfg.crudo, ["modelos.motor_regla=mae_mas_bias"])

    assert anulado["modelos"]["motor_regla"] == "mae_mas_bias"
    assert anulado["periodo"] == cfg.crudo["periodo"], "Se tocó otra sección"
    assert cfg.crudo["modelos"]["motor_regla"] != "mae_mas_bias", (
        "La anulación mutó el diccionario original"
    )


def test_anular_convierte_los_tipos(cfg):
    anulado = aplicar_anulaciones(
        cfg.crudo,
        ["inclusion.historial_minimo_meses=48", "inclusion.proporcion_maxima_ceros=0.5"],
    )
    assert anulado["inclusion"]["historial_minimo_meses"] == 48
    assert anulado["inclusion"]["proporcion_maxima_ceros"] == pytest.approx(0.5)


def test_anular_una_clave_inexistente_falla(cfg):
    """Una errata no puede pasar inadvertida y correr con la configuración vieja."""
    with pytest.raises(ErrorDeConfiguracion) as excepcion:
        aplicar_anulaciones(cfg.crudo, ["modelos.motor_regIa=mae"])
    assert "motor_regIa" in str(excepcion.value)


def test_anular_una_seccion_inexistente_falla(cfg):
    with pytest.raises(ErrorDeConfiguracion):
        aplicar_anulaciones(cfg.crudo, ["modelosss.motor_regla=mae"])


def test_anulacion_mal_formada_falla(cfg):
    with pytest.raises(ErrorDeConfiguracion) as excepcion:
        aplicar_anulaciones(cfg.crudo, ["modelos.motor_regla"])
    assert "clave.subclave=valor" in str(excepcion.value)


def test_las_anulaciones_quedan_registradas_para_el_manifiesto(cfg, tmp_path):
    """RN-6: lo que se guarda es la configuración REALMENTE usada."""
    modificada = cargar_config(
        cfg.ruta_config, anulaciones=["modelos.motor_regla=mae_mas_bias"]
    )
    assert modificada.crudo["_anulaciones"] == ["modelos.motor_regla=mae_mas_bias"]
    assert modificada.modelos["motor_regla"] == "mae_mas_bias"
    # Y el hash de la configuración cambia, de modo que dos corridas con distinta
    # ablación no pueden confundirse entre sí.
    assert modificada.hash_configuracion() != cfg.hash_configuracion()


# ---------------------------------------------------------------------------
# Validaciones que deben rechazar
# ---------------------------------------------------------------------------

def test_rechaza_un_rezago_cero(cfg):
    with pytest.raises(ErrorDeConfiguracion) as excepcion:
        cargar_config(cfg.ruta_config, anulaciones=["features.rezagos=[0, 1, 2]"])
    assert "fuga" in str(excepcion.value).lower()


def test_rechaza_una_regla_de_motor_desconocida(cfg):
    with pytest.raises(ErrorDeConfiguracion):
        cargar_config(cfg.ruta_config, anulaciones=["modelos.motor_regla=lo_que_sea"])


def test_rechaza_un_benchmark_que_no_esta_entre_los_modelos(cfg):
    with pytest.raises(ErrorDeConfiguracion) as excepcion:
        cargar_config(
            cfg.ruta_config, anulaciones=["modelos.benchmark_promedio_movil=ma_99"]
        )
    assert "ma_99" in str(excepcion.value)


def test_rechaza_un_tratamiento_de_devoluciones_desconocido(cfg):
    with pytest.raises(ErrorDeConfiguracion):
        cargar_config(cfg.ruta_config, anulaciones=["series.devoluciones=promediar"])
