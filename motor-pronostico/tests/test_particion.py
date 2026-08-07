"""RF-4 — Criterio de aceptación.

«Existe una prueba que verifica que ninguna fecha de prueba es anterior a una de
validación, y ninguna de validación es anterior a una de entrenamiento.»

Es la RN-2, la restricción más fácil de romper por accidente y la más grave.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import ErrorDeConfiguracion, cargar_config
from src.particion import aplicar_criterios_inclusion, etiquetar_bloques, particionar
from src.series import construir_series

from conftest import panel_de, serie_estacional


def test_orden_temporal_estricto_entre_los_tres_bloques(cfg):
    corte = particionar(cfg)

    assert corte.entrenamiento.max() < corte.validacion.min(), (
        "Hay meses de entrenamiento posteriores a validación"
    )
    assert corte.validacion.max() < corte.prueba.min(), (
        "Hay meses de validación posteriores a prueba"
    )
    # Y la comprobación por elemento, que es la que pide el requerimiento.
    for mes_prueba in corte.prueba:
        assert all(mes_prueba > m for m in corte.validacion)
        assert all(mes_prueba > m for m in corte.entrenamiento)
    for mes_validacion in corte.validacion:
        assert all(mes_validacion > m for m in corte.entrenamiento)


def test_los_bloques_no_se_solapan_ni_dejan_huecos(cfg):
    corte = particionar(cfg)
    todos = list(corte.entrenamiento) + list(corte.validacion) + list(corte.prueba)

    assert len(todos) == len(set(todos)), "Hay meses repetidos entre bloques"
    esperados = pd.period_range(cfg.periodo.inicio, cfg.periodo.fin, freq="M")
    assert sorted(todos) == list(esperados), "Los bloques no cubren el período completo"


def test_los_cortes_caen_en_frontera_de_gestion_fiscal(cfg):
    """Cada bloque debe contener un número entero de ciclos comerciales."""
    corte = particionar(cfg)
    mes_cierre = (cfg.periodo.mes_inicio_gestion - 1) or 12

    assert corte.entrenamiento[-1].month == mes_cierre
    assert corte.validacion[-1].month == mes_cierre
    assert corte.prueba[-1].month == mes_cierre
    for bloque in (corte.entrenamiento, corte.validacion, corte.prueba):
        assert len(bloque) % 12 == 0, "Un bloque no contiene gestiones completas"


def test_la_configuracion_rechaza_cortes_desordenados(tmp_path, cfg):
    """Si alguien invierte los cortes, el programa falla al arrancar."""
    contenido = cfg.ruta_config.read_text(encoding="utf-8")
    roto = contenido.replace(
        f'fin_entrenamiento: "{cfg.particion.fin_entrenamiento}"',
        f'fin_entrenamiento: "{cfg.particion.fin_validacion}"',
    ).replace(
        f'fin_validacion: "{cfg.particion.fin_validacion}"',
        f'fin_validacion: "{cfg.particion.fin_entrenamiento}"',
    )
    ruta = tmp_path / "config_roto.yaml"
    ruta.write_text(roto, encoding="utf-8")

    with pytest.raises(ErrorDeConfiguracion) as excepcion:
        cargar_config(ruta)
    assert "fin_entrenamiento" in str(excepcion.value)


def test_etiquetado_de_bloques_es_coherente(cfg):
    corte = particionar(cfg)
    n = len(pd.period_range(cfg.periodo.inicio, cfg.periodo.fin, freq="M"))
    panel = panel_de({"A": serie_estacional(n)}, inicio=str(cfg.periodo.inicio))
    etiquetado = etiquetar_bloques(panel, corte)

    for bloque, meses in (("entrenamiento", corte.entrenamiento),
                          ("validacion", corte.validacion),
                          ("prueba", corte.prueba)):
        seleccion = etiquetado.loc[etiquetado["bloque"] == bloque, "periodo"]
        assert set(seleccion) == set(meses)


def test_los_criterios_de_inclusion_se_miden_solo_en_entrenamiento(cfg):
    """Una serie que se apaga en prueba no puede quedar excluida por eso.

    Si los criterios miraran la historia completa, la muestra se estaría
    eligiendo con información de los bloques que aún no se pueden mirar.
    """
    corte = particionar(cfg)
    n_train = len(corte.entrenamiento)
    n_total = n_train + len(corte.validacion) + len(corte.prueba)

    # Serie sana en entrenamiento; en validación y prueba vende una sola vez.
    valores = serie_estacional(n_train) + [0.0] * (n_total - n_train - 1) + [5.0]
    panel = panel_de({"A": valores}, inicio=str(cfg.periodo.inicio))

    cohorte, informe = aplicar_criterios_inclusion(panel, cfg, corte)
    assert informe.series_finales == 1, (
        "La serie fue excluida por comportamiento POSTERIOR al entrenamiento"
    )
    assert not cohorte.empty


def test_reporta_las_exclusiones_por_criterio(cfg):
    corte = particionar(cfg)
    n_total = len(pd.period_range(cfg.periodo.inicio, cfg.periodo.fin, freq="M"))
    n_train = len(corte.entrenamiento)

    series_sinteticas = {
        "SANA": serie_estacional(n_total),
        # historial corto: nace justo antes de validación
        "CORTA": [0.0] * (n_train - 5) + serie_estacional(n_total - n_train + 5),
        # volumen ridículo
        "CHICA": [0.01] * n_total,
    }
    panel = panel_de(series_sinteticas, inicio=str(cfg.periodo.inicio))
    # el relleno con ceros de arriba no representa nacimiento tardío, así que se
    # recorta explícitamente la serie CORTA
    panel = panel.loc[~(
        (panel["sku"] == "CORTA")
        & (panel["periodo"] < corte.entrenamiento[n_train - 5])
    )]

    _, informe = aplicar_criterios_inclusion(panel, cfg, corte)
    flujo = informe.tabla_flujo()

    assert "paso" in flujo.columns and "series" in flujo.columns
    assert flujo["series"].is_monotonic_decreasing, (
        "La cascada de criterios no puede aumentar el número de series"
    )
    assert informe.series_finales >= 1
    assert set(informe.excluidas_aislado), "Falta el efecto aislado de cada criterio"
