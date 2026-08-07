"""RF-8 — Criterio de aceptación.

«Con dos vectores idénticos, ninguna prueba reporta significancia.»
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pruebas import (
    bateria_completa, friedman_con_nemenyi, porcentaje_de_victorias,
    shapiro_wilk, wilcoxon_pareado,
)


def _serie(valores, nombre):
    return pd.Series(valores, index=[f"S{i}" for i in range(len(valores))], name=nombre)


# ---------------------------------------------------------------------------
# El criterio de aceptación
# ---------------------------------------------------------------------------

def test_vectores_identicos_no_producen_significancia():
    valores = list(np.linspace(1, 100, 60))
    a = _serie(valores, "modelo_a")
    b = _serie(valores, "modelo_b")

    resultado = wilcoxon_pareado(a, b)
    assert resultado.p == pytest.approx(1.0)
    assert not resultado.significativo
    assert resultado.r == pytest.approx(0.0)
    assert resultado.rango_biserial == pytest.approx(0.0)
    assert resultado.empates == len(valores)


def test_friedman_con_columnas_identicas_no_rechaza():
    valores = np.linspace(1, 100, 40)
    matriz = pd.DataFrame({"a": valores, "b": valores, "c": valores})
    resultado = friedman_con_nemenyi(matriz)

    assert resultado.chi2 == pytest.approx(0.0, abs=1e-9)
    assert resultado.p == pytest.approx(1.0)
    assert not resultado.significativo
    assert resultado.nemenyi is None, "El post hoc es protegido: no corre si no rechaza"


def test_shapiro_con_diferencias_constantes_no_revienta():
    resultado = shapiro_wilk(np.zeros(50))
    assert np.isnan(resultado["estadistico"])
    assert "no aplica" in resultado["nota"]


# ---------------------------------------------------------------------------
# Comportamiento con diferencias reales
# ---------------------------------------------------------------------------

def test_wilcoxon_detecta_una_mejora_sistematica():
    generador = np.random.default_rng(11)
    referencia = generador.gamma(2.0, 10.0, 200)
    propuesto = referencia * 0.7                       # 30 % menos error siempre

    resultado = wilcoxon_pareado(
        _serie(propuesto, "propuesto"), _serie(referencia, "referencia"),
        alternativa="less",
    )
    assert resultado.significativo
    assert resultado.p < 1e-10
    assert resultado.gana_propuesto == 200
    assert resultado.rango_biserial == pytest.approx(-1.0)
    assert resultado.r < 0, "El signo debe indicar que el propuesto tiene MENOS error"


def test_wilcoxon_unilateral_no_confunde_la_direccion():
    generador = np.random.default_rng(3)
    referencia = generador.gamma(2.0, 10.0, 150)
    peor = referencia * 1.4

    resultado = wilcoxon_pareado(
        _serie(peor, "peor"), _serie(referencia, "referencia"), alternativa="less",
    )
    assert not resultado.significativo, (
        "Un modelo PEOR no puede salir significativo en el contraste unilateral"
    )


def test_el_tamano_del_efecto_no_depende_solo_del_n():
    """Con la misma diferencia relativa, r no debe dispararse al crecer n."""
    generador = np.random.default_rng(5)
    base = generador.gamma(2.0, 10.0, 2000)

    chico = wilcoxon_pareado(
        _serie(base[:100] * 0.95, "p"), _serie(base[:100], "r"), alternativa="less"
    )
    grande = wilcoxon_pareado(
        _serie(base * 0.95, "p"), _serie(base, "r"), alternativa="less"
    )
    # El p-valor sí cae con n; r se mantiene en el mismo orden de magnitud.
    assert grande.p < chico.p
    assert abs(abs(grande.r) - abs(chico.r)) < 0.35


def test_friedman_ordena_los_modelos_por_rango_medio():
    generador = np.random.default_rng(7)
    base = generador.gamma(2.0, 10.0, 300)
    matriz = pd.DataFrame({
        "bueno": base * 0.5,
        "medio": base * 1.0,
        "malo": base * 2.0,
    })
    resultado = friedman_con_nemenyi(matriz)

    assert resultado.significativo
    assert list(resultado.rangos_medios.index) == ["bueno", "medio", "malo"]
    assert resultado.nemenyi is not None
    assert resultado.diferencia_critica > 0
    assert 0.0 <= resultado.kendall_w <= 1.0


def test_las_victorias_reparten_los_empates():
    matriz = pd.DataFrame({
        "a": [1.0, 1.0, 5.0],
        "b": [1.0, 2.0, 1.0],
    })
    victorias = porcentaje_de_victorias(matriz).set_index("modelo")
    # fila 0: empate (0,5 cada uno) · fila 1: gana a · fila 2: gana b
    assert victorias.loc["a", "victorias"] == pytest.approx(1.5)
    assert victorias.loc["b", "victorias"] == pytest.approx(1.5)
    assert victorias["porcentaje"].sum() == pytest.approx(100.0)


def test_bateria_completa_devuelve_las_tres_familias():
    generador = np.random.default_rng(13)
    base = generador.gamma(2.0, 10.0, 250)
    matriz = pd.DataFrame({
        "motor": base * 0.6,
        "lightgbm": base * 0.7,
        "ma_12": base * 1.0,
        "naive_m1": base * 1.2,
    })
    resultados = bateria_completa(
        matriz, propuestos=["motor", "lightgbm"], referencias=["ma_12", "naive_m1"],
    )

    assert resultados["n_bloques"] == 250
    assert len(resultados["wilcoxon"]) == 4
    assert all(r.significativo for r in resultados["wilcoxon"])
    assert resultados["friedman"].significativo
    assert len(resultados["victorias"]) == 4


def test_shapiro_submuestrea_y_lo_declara():
    generador = np.random.default_rng(17)
    diferencias = generador.normal(0, 1, 20_000)
    resultado = shapiro_wilk(diferencias, n_maximo=5000, semilla=1)

    assert resultado["n"] == 20_000
    assert resultado["n_usado"] == 5000
    assert resultado["submuestreado"] is True
    assert "semilla" in resultado["nota"]

    # Reproducible: misma semilla, mismo resultado (RN-5).
    repetido = shapiro_wilk(diferencias, n_maximo=5000, semilla=1)
    assert resultado["estadistico"] == pytest.approx(repetido["estadistico"])
