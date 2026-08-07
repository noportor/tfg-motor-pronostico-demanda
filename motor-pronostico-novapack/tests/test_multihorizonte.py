"""Proyección multihorizonte: cada mecánica verificada contra la fórmula a mano.

La propiedad central que se defiende aquí: la proyección recursiva (realimentar
las propias predicciones) reproduce la forma cerrada de cada modelo — plana para
el suavizado exponencial simple, ``ℓ + h·b + s`` para Holt-Winters — y ningún
pronóstico usa información posterior a su origen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import panel_de, serie_estacional
from src.modelos.croston import Croston
from src.modelos.naive import Naive
from src.modelos.promedio_movil import PromedioMovil
from src.modelos.suavizado import HoltWinters, SuavizadoExponencial
from src.multihorizonte import (
    agregado_valorizado, proyectar_constante, proyectar_recursivo,
)


def _ancho(valores_por_serie: dict[str, list[float]], inicio="2017-04") -> pd.DataFrame:
    largo = panel_de(valores_por_serie, inicio=inicio)
    return largo.pivot(index="periodo", columns="serie", values="y")


# ---------------------------------------------------------------------------
# Formas cerradas
# ---------------------------------------------------------------------------

def test_naive_proyecta_plano_en_el_ultimo_valor():
    panel = _ancho({"A": [5.0, 8.0, 11.0, 40.0]})
    proyeccion = proyectar_recursivo(Naive(), panel, horizonte=6)
    assert proyeccion.shape == (6, 1)
    assert (proyeccion.iloc[:, 0] == 40.0).all(), (
        "El Naïve multihorizonte es el último valor observado, congelado"
    )


def test_ma2_recursivo_a_mano():
    """(a, b) = (10, 20): f1 = 15, f2 = (20+15)/2 = 17.5, f3 = (15+17.5)/2."""
    panel = _ancho({"A": [10.0, 20.0]})
    proyeccion = proyectar_recursivo(PromedioMovil(2), panel, horizonte=3)
    esperado = [15.0, 17.5, 16.25]
    assert np.allclose(proyeccion.iloc[:, 0].to_numpy(), esperado), (
        "La recursión del promedio móvil debe realimentar sus pronósticos, "
        "como el sistema en producción"
    )


def test_ses_proyecta_plano():
    """Realimentar ŷ = ℓ deja el nivel intacto: la forma cerrada del SES."""
    modelo = SuavizadoExponencial(alfas=[0.3])
    panel = _ancho({"A": [float(10 + (i % 5)) for i in range(24)]})
    modelo.ajustar(panel, panel)

    proyeccion = proyectar_recursivo(modelo, panel, horizonte=8)
    primera = proyeccion.iloc[0, 0]
    assert np.allclose(proyeccion.iloc[:, 0].to_numpy(), primera), (
        "El SES multihorizonte es una línea plana en el nivel del origen"
    )


def test_croston_es_constante_y_no_se_contamina():
    modelo = Croston(alfas=[0.2])
    valores = [0.0, 6.0, 0.0, 0.0, 9.0, 0.0, 3.0, 0.0, 0.0, 12.0, 0.0, 0.0]
    panel = _ancho({"A": valores * 2})
    modelo.ajustar(panel, panel)

    proyeccion = proyectar_constante(modelo, panel, horizonte=6)
    primera = proyeccion.iloc[0, 0]
    assert primera > 0
    assert (proyeccion.iloc[:, 0] == primera).all(), (
        "Croston solo actualiza estados con demanda observada: su proyección "
        "multihorizonte es su constante congelada en el origen"
    )


def test_holt_winters_reproduce_su_forma_cerrada():
    """La recursión realimentada debe dar ℓ + h·b + s(t+h), con los estados
    del origen calculados a mano con las ecuaciones del libro."""
    m = 4  # estacionalidad corta para que la mano no se canse
    alfa, beta, gamma = 0.3, 0.1, 0.2
    valores = [
        10.0, 20.0, 30.0, 16.0, 12.0, 24.0, 33.0, 18.0,
        14.0, 26.0, 37.0, 21.0, 15.0, 28.0, 40.0, 22.0,
    ]

    # Estados a mano, replicando la inicialización y las actualizaciones del
    # módulo (implementación independiente: bucle escalar, no vectorizado).
    nivel = float(np.mean(valores[:m]))
    pendiente = (float(np.mean(valores[m:2 * m])) - nivel) / m
    estacional = {j % m: valores[j] - float(np.mean(valores[:m])) for j in range(m)}
    for t in range(2 * m, len(valores)):
        ranura = t % m
        y = valores[t]
        nivel_previo = nivel
        nivel = alfa * (y - estacional[ranura]) + (1 - alfa) * (nivel + pendiente)
        pendiente = beta * (nivel - nivel_previo) + (1 - beta) * pendiente
        estacional[ranura] = gamma * (y - nivel) + (1 - gamma) * estacional[ranura]

    horizonte = 6
    esperado = [
        nivel + h * pendiente + estacional[(len(valores) + h - 1) % m]
        for h in range(1, horizonte + 1)
    ]

    modelo = HoltWinters(periodo_estacional=m, alfas=[alfa], betas=[beta], gammas=[gamma])
    panel = _ancho({"A": valores})
    modelo.ajustar(panel, panel)
    proyeccion = proyectar_recursivo(modelo, panel, horizonte=horizonte)

    assert np.allclose(
        proyeccion.iloc[:, 0].to_numpy(), np.clip(esperado, 0, None), atol=1e-9
    ), "La recursión realimentada debe reproducir ℓ + h·b + s exactamente"


# ---------------------------------------------------------------------------
# Propiedades del protocolo
# ---------------------------------------------------------------------------

def test_h1_coincide_con_el_protocolo_a_un_paso():
    """El primer mes proyectado es el pronóstico a un paso del mes siguiente
    al origen: el protocolo multihorizonte GENERALIZA al principal."""
    valores = serie_estacional(48)
    panel = _ancho({"A": valores})
    origen_mas_uno = panel.index[-1] + 1

    for modelo in (Naive(), PromedioMovil(2), PromedioMovil(12)):
        proyeccion = proyectar_recursivo(modelo, panel, horizonte=4)
        extendido = panel.reindex(pd.period_range(panel.index[0], origen_mas_uno, freq="M"))
        un_paso = modelo.predecir(extendido).loc[origen_mas_uno]
        assert np.allclose(
            proyeccion.iloc[0].to_numpy(dtype=float),
            un_paso.to_numpy(dtype=float),
        ), f"h=1 de {modelo.nombre} debe coincidir con el pronóstico a un paso"


def test_la_proyeccion_no_mira_despues_del_origen():
    """Cambiar el futuro no puede cambiar la proyección hecha en el origen."""
    valores = serie_estacional(60)
    historia = _ancho({"A": valores})

    futuro_a = _ancho({"A": valores + [999.0, 999.0, 999.0]})
    futuro_b = _ancho({"A": valores + [1.0, 2.0, 3.0]})
    origen = historia.index[-1]

    ses = SuavizadoExponencial(alfas=[0.5])
    ses.ajustar(historia, historia)

    for modelo in (Naive(), PromedioMovil(12), ses):
        desde_a = proyectar_recursivo(
            modelo, futuro_a.loc[futuro_a.index <= origen], horizonte=3
        )
        desde_b = proyectar_recursivo(
            modelo, futuro_b.loc[futuro_b.index <= origen], horizonte=3
        )
        # Si esto falla, la proyección desde el origen dependió del futuro:
        # fuga temporal.
        pd.testing.assert_frame_equal(desde_a, desde_b)


def test_agregado_valorizado_a_mano():
    """Dos series de costo distinto, un origen: D calculada a mano."""
    obs = pd.DataFrame({
        "modelo": ["x"] * 4,
        "origen": ["2025-03"] * 4,
        "horizonte": [1, 1, 2, 2],
        "serie": ["CARA", "BARATA", "CARA", "BARATA"],
        "y_real": [10.0, 100.0, 10.0, 100.0],
        "y_pred": [12.0, 90.0, 8.0, 110.0],
        "costo": [10.0, 1.0, 10.0, 1.0],
    })

    resumen = agregado_valorizado(obs, ["modelo", "horizonte"]).set_index("horizonte")

    # h=1: demanda = 10·10 + 1·100 = 200; error = 10·2 + 1·10 = 30;
    #      pred = 10·12 + 1·90 = 210 → WMAPE 15 %, Bias +5 %, D = 20.
    assert resumen.loc[1, "wmape_val"] == pytest.approx(15.0)
    assert resumen.loc[1, "bias_val"] == pytest.approx(5.0)
    assert resumen.loc[1, "D"] == pytest.approx(20.0)
    # h=2: error = 10·2 + 1·10 = 30; pred = 10·8 + 1·110 = 190
    #      → WMAPE 15 %, Bias −5 %, D = 20 (el |·| del sesgo, verificado).
    assert resumen.loc[2, "bias_val"] == pytest.approx(-5.0)
    assert resumen.loc[2, "D"] == pytest.approx(20.0)
    assert int(resumen.loc[1, "n_origenes"]) == 1


def test_grupo_sin_demanda_queda_visible_no_descartado():
    obs = pd.DataFrame({
        "modelo": ["x", "x"],
        "origen": ["2025-03", "2025-03"],
        "horizonte": [1, 2],
        "serie": ["A", "A"],
        "y_real": [50.0, 0.0],
        "y_pred": [40.0, 5.0],
        "costo": [2.0, 2.0],
    })
    resumen = agregado_valorizado(obs, ["modelo", "horizonte"]).set_index("horizonte")
    assert np.isnan(resumen.loc[2, "D"]), (
        "Sin demanda valorizada el porcentaje no existe: NaN visible, "
        "nunca un número inventado ni una fila descartada"
    )
    assert 2 in resumen.index
