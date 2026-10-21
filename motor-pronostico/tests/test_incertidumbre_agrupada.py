"""Pruebas del remuestreo agrupado de la Tabla H3, Tabla 20 del documento final (scripts/incertidumbre_agrupada.py).

Sólo paneles sintéticos, construidos aquí: ninguna cifra ni clave de la
corrida real.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUTA = Path(__file__).resolve().parent.parent / "scripts" / "incertidumbre_agrupada.py"
_spec = importlib.util.spec_from_file_location("incertidumbre_agrupada", RUTA)
ia = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ia          # dataclasses resuelve el módulo por nombre
_spec.loader.exec_module(ia)


def _panel_sintetico(n_sku: int = 60, por_sku: int = 3, semilla: int = 7):
    """Claves SKU|CANAL|REGIONAL ficticias y un código de conglomerado por serie."""
    claves = pd.Index([f"P{k:03d}|C{j}|R{j}" for k in range(n_sku) for j in range(por_sku)])
    codigos = pd.factorize(ia.sku_de_serie(claves))[0]
    return claves, codigos, np.random.default_rng(semilla)


def test_sku_de_serie_toma_el_primer_campo_y_exige_tres():
    assert list(ia.sku_de_serie(pd.Index(["A|x|y", "B|x|z"]))) == ["A", "B"]
    with pytest.raises(ValueError):
        ia.sku_de_serie(pd.Index(["A|x"]))


def test_mediana_ponderada_igual_a_repetir():
    rng = np.random.default_rng(1)
    for _ in range(50):
        v = rng.normal(size=25)
        w = rng.integers(0, 4, size=25)
        if w.sum() == 0:
            continue
        assert ia.mediana_ponderada(v, w) == pytest.approx(np.median(np.repeat(v, w)))


def test_pesos_por_conglomerado_mueven_el_sku_entero():
    _, codigos, rng = _panel_sintetico()
    pesos = ia.pesos_conglomerados(codigos, 200, rng)
    n_sku = codigos.max() + 1
    for fila in pesos:
        # Las tres series de un SKU llevan siempre el mismo peso...
        por_sku = pd.Series(fila).groupby(codigos).nunique()
        assert (por_sku == 1).all()
        # ...y se sortean tantos SKU como hay.
        assert pd.Series(fila).groupby(codigos).first().sum() == n_sku


def test_pesos_bloques_moviles_suman_los_meses_y_son_circulares():
    rng = np.random.default_rng(3)
    pesos = ia.pesos_bloques_moviles(12, 3, 500, rng)
    assert (pesos.sum(axis=1) == 12).all()
    # Con bloques circulares todo mes tiene la misma probabilidad de salir.
    frecuencia = pesos.mean(axis=0)
    assert np.allclose(frecuencia, 1.0, atol=0.15)


def test_compuesto_con_pesos_unitarios_es_mae_mas_sesgo_absoluto():
    rng = np.random.default_rng(5)
    y = rng.poisson(20, size=(8, 12)).astype(float)
    p = y + rng.normal(2, 3, size=y.shape)
    y[0, :4] = np.nan                       # serie que nace tarde
    ok = ~np.isnan(y)
    firmado = np.where(ok, p - y, 0.0)
    obtenido = ia.compuesto_por_serie(np.abs(firmado), firmado, ok.astype(float),
                                      np.ones((1, 12)))[:, 0]
    e = np.where(ok, p - y, np.nan)
    esperado = np.nanmean(np.abs(e), axis=1) + np.abs(np.nanmean(e, axis=1))
    assert np.allclose(obtenido, esperado)


def test_d_portafolio():
    assert ia.d_portafolio(np.array([30.0]), np.array([100.0]), np.array([90.0]))[0] == pytest.approx(40.0)


def test_mejora_constante_excluye_el_cero_y_la_nula_no():
    claves, codigos, rng = _panel_sintetico()
    comun = rng.normal(0, 3, size=codigos.max() + 1)[codigos]   # choque común del SKU
    referencia = rng.gamma(4, 5, size=len(claves))
    mejora = {("prop", "ref"): (referencia - 2 + comun) - referencia}
    nula = {("prop", "ref"): comun.copy()}
    filas = ia.bootstrap_por_sku(mejora, None, codigos, 400, 11)
    media = next(f for f in filas if f.magnitud == "media")
    assert ia.excluye_cero(media.ic_inferior, media.ic_superior)
    assert media.ic_superior < 0
    filas = ia.bootstrap_por_sku({k: v - v.mean() for k, v in nula.items()}, None, codigos, 400, 11)
    media = next(f for f in filas if f.magnitud == "media")
    assert not ia.excluye_cero(media.ic_inferior, media.ic_superior)


def test_agrupar_por_sku_ensancha_el_intervalo_si_las_series_del_sku_se_mueven_juntas():
    claves, codigos, rng = _panel_sintetico(n_sku=80, por_sku=4)
    diferencia = rng.normal(0, 1, size=codigos.max() + 1)[codigos]   # idéntica dentro del SKU
    agrupado = ia.bootstrap_por_sku({("a", "b"): diferencia}, None, codigos, 1000, 2)[0]
    independiente = ia.bootstrap_por_sku({("a", "b"): diferencia}, None,
                                         np.arange(len(claves)), 1000, 2)[0]
    ancho_agrupado = agrupado.ic_superior - agrupado.ic_inferior
    ancho_independiente = independiente.ic_superior - independiente.ic_inferior
    # Con correlación total dentro del SKU, el ancho crece como raíz de 4 = 2.
    assert ancho_agrupado > 1.6 * ancho_independiente


def test_por_sku_igual_a_remuestrear_los_sku_a_mano():
    """Los pesos por conglomerado equivalen a concatenar las series de los SKU sorteados."""
    claves, codigos, rng = _panel_sintetico(n_sku=40, por_sku=3)
    diferencia = rng.normal(0, 2, size=len(claves)) + rng.normal(0, 1, size=40)[codigos]
    replicas, semilla = 300, 17
    filas = ia.bootstrap_por_sku({("a", "b"): diferencia}, None, codigos, replicas, semilla)
    sorteo = np.random.default_rng(semilla).integers(0, 40, size=(replicas, 40))
    por_sku = [np.flatnonzero(codigos == k) for k in range(40)]
    medias, medianas = [], []
    for fila in sorteo:
        indices = np.concatenate([por_sku[k] for k in fila])
        medias.append(diferencia[indices].mean())
        medianas.append(np.median(diferencia[indices]))
    media = next(f for f in filas if f.magnitud == "media")
    mediana = next(f for f in filas if f.magnitud == "mediana")
    assert (media.ic_inferior, media.ic_superior) == pytest.approx(tuple(np.percentile(medias, [2.5, 97.5])))
    assert (mediana.ic_inferior, mediana.ic_superior) == pytest.approx(
        tuple(np.percentile(medianas, [2.5, 97.5])))


def test_d_por_sku_con_pesos_unitarios_reproduce_la_d():
    claves, codigos, rng = _panel_sintetico()
    n = len(claves)
    real = rng.gamma(3, 10, size=n)
    comp = {
        "a": (0.5 * real, real, 0.9 * real),        # D = 50 + 10
        "b": (0.7 * real, real, 1.2 * real),        # D = 70 + 20
    }
    filas = ia.bootstrap_por_sku({("a", "b"): np.zeros(n)}, comp, codigos, 300, 4)
    d = next(f for f in filas if f.magnitud == "d_portafolio")
    assert d.estimacion == pytest.approx(-30.0)
    # Proporciones fijas por serie: toda réplica da la misma D.
    assert d.ic_inferior == pytest.approx(-30.0) and d.ic_superior == pytest.approx(-30.0)


def test_bloques_temporales_sobre_panel_sintetico():
    rng = np.random.default_rng(9)
    y = rng.poisson(30, size=(20, 12)).astype(float)
    ok = np.ones_like(y)

    def modelo(desplazamiento, ruido):
        p = y + desplazamiento + rng.normal(0, ruido, size=y.shape)
        f = p - y
        return {"absoluto": np.abs(f), "firmado": f, "observado": ok,
                "error_val": np.abs(f).sum(axis=0), "real_val": y.sum(axis=0),
                "pred_val": p.sum(axis=0)}

    mensual = {"prop": modelo(0.0, 1.0), "ref": modelo(8.0, 1.0)}
    estimaciones = {("prop", "ref", k): -1.0 for k in ("media", "mediana", "d_portafolio")}
    filas = ia.bootstrap_por_bloques(mensual, [("prop", "ref")], estimaciones, 300, 8)
    assert {f.magnitud for f in filas} == {"media", "mediana", "d_portafolio"}
    for f in filas:
        assert f.remuestreo == "bloques"
        assert f.ic_superior < 0          # la referencia erra siempre más
