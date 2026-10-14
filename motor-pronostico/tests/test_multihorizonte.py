"""Proyección multihorizonte: cada mecánica verificada contra la fórmula a mano.

La propiedad central que se defiende aquí: la proyección recursiva (realimentar
las propias predicciones) reproduce la forma cerrada de cada modelo — plana para
el suavizado exponencial simple, ``ℓ + h·b + s`` para Holt-Winters — y ningún
pronóstico usa información posterior a su origen.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import panel_de, serie_estacional
from src.config import ErrorDeConfiguracion, cargar_config
from src.modelos.base import truncar_en_cero
from src.modelos.croston import Croston
from src.modelos.naive import Naive
from src.modelos.promedio_movil import PromedioMovil
from src.modelos.suavizado import HoltWinters, SuavizadoExponencial
from src.multihorizonte import (
    _reestimar_en_origen, _tiene_parametros, agregado_valorizado, criterio_por_serie, evaluar,
    proyectar_constante, proyectar_recursivo,
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
        "como la práctica vigente"
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


def test_criterio_por_serie_a_mano():
    """mae + |bias| de una proyección contra la realidad, verificado a mano."""
    meses = pd.period_range("2024-04", periods=3, freq="M")
    real = pd.DataFrame({"S": [10.0, 20.0, 30.0]}, index=meses)
    proyeccion = pd.DataFrame({"S": [12.0, 18.0, 33.0]}, index=meses)

    # mae = (2+2+3)/3 = 7/3; medias: pred 21, real 20 → bias 1.
    puro = criterio_por_serie(proyeccion, real, regla="mae")
    compuesto = criterio_por_serie(proyeccion, real, regla="mae_mas_bias")
    assert puro["S"] == pytest.approx(7 / 3)
    assert compuesto["S"] == pytest.approx(7 / 3 + 1.0)


def test_criterio_sin_meses_comparables_es_nan():
    meses = pd.period_range("2024-04", periods=2, freq="M")
    real = pd.DataFrame({"S": [np.nan, np.nan]}, index=meses)
    proyeccion = pd.DataFrame({"S": [5.0, 5.0]}, index=meses)
    criterio = criterio_por_serie(proyeccion, real, regla="mae_mas_bias")
    assert np.isnan(criterio["S"]), (
        "Sin realidad con la que comparar no hay criterio: NaN, y el motor "
        "asigna su respaldo declarado"
    )


def test_seleccion_multihorizonte_castiga_al_plano(cfg):
    """El caso que motiva la ablación: una serie estacional donde el modelo
    plano gana a un paso pero pierde proyectando el año completo."""
    from src.modelos.motor import Motor

    valores = serie_estacional(48)
    panel = _ancho({"A": valores})
    origen = panel.index[35]                       # 36 meses de "entrenamiento"
    historia = panel.loc[panel.index <= origen]
    meses_validacion = panel.index[36:48]
    real_validacion = panel.reindex(meses_validacion)

    hw = HoltWinters(periodo_estacional=12, alfas=[0.3], betas=[0.1], gammas=[0.2])
    hw.ajustar(historia, historia)
    ses = SuavizadoExponencial(alfas=[0.3])
    ses.ajustar(historia, historia)

    criterios = {}
    for modelo in (hw, ses):
        proyeccion = proyectar_recursivo(modelo, historia, horizonte=12)
        proyeccion = proyeccion.reindex(index=meses_validacion)
        criterios[modelo.nombre] = criterio_por_serie(
            proyeccion, real_validacion, regla="mae_mas_bias"
        )
    matriz = pd.DataFrame(criterios)

    # Con estacionalidad fuerte, la proyección plana del SES tiene que perder
    # contra la forma estacional de HW en el año completo.
    assert matriz.loc[matriz.index[0], "holt_winters"] < matriz.loc[
        matriz.index[0], "exp_smooth_opt"
    ]

    # Y el motor, alimentado con esa matriz, elige HW para la serie.
    predicciones = {
        "holt_winters": hw.predecir(panel),
        "exp_smooth_opt": ses.predecir(panel),
        # Los demás candidatos declarados no participan de esta prueba: la
        # matriz externa solo trae dos columnas y el motor debe rechazarla si
        # le falta un candidato — por eso acá se recorta la lista.
    }
    cfg_dos = cargar_config(
        cfg.ruta_config,
        anulaciones=["modelos.activos=[holt_winters, exp_smooth_opt, motor]",
                     "modelos.benchmark_promedio_movil=holt_winters",
                     "modelos.benchmark_naive=exp_smooth_opt"],
    )
    motor = Motor(cfg_dos, predicciones, panel)
    validacion = panel.reindex(meses_validacion)
    motor.ajustar(historia, validacion, criterio_externo=matriz)
    assert motor.informe.seleccion["modelo_elegido"].iloc[0] == "holt_winters"
    assert "multihorizonte" in motor.informe.regla


def test_motor_rechaza_criterio_incompleto(cfg):
    from src.modelos.motor import Motor

    panel = _ancho({"A": [10.0] * 48})
    # El criterio externo trae naive_m1 pero NO ma_2, que sí es candidato con
    # predicciones: el motor debe rechazar la matriz incompleta, no elegir
    # entre los que casualmente estén.
    matriz = pd.DataFrame({"naive_m1": [1.0]}, index=pd.Index(
        [panel.columns[0]], name="serie"
    ))
    predicciones = {"naive_m1": panel.shift(1), "ma_2": panel.shift(1)}
    motor = Motor(cfg, predicciones, panel)
    with pytest.raises(ValueError):
        motor.ajustar(
            panel.iloc[:36], panel.iloc[36:48], criterio_externo=matriz
        )


def test_config_valida_motor_seleccion(cfg):
    with pytest.raises(ErrorDeConfiguracion):
        cargar_config(
            cfg.ruta_config, anulaciones=["modelos.motor_seleccion=magia"]
        )
    valida = cargar_config(
        cfg.ruta_config, anulaciones=["modelos.motor_seleccion=multihorizonte"]
    )
    assert valida.modelos["motor_seleccion"] == "multihorizonte"


def test_exogenas_futuras_persisten_el_valor_del_origen(cfg):
    """La política declarada del multihorizonte: las exógenas futuras valen su
    último valor CONOCIDO al origen — nunca el valor real futuro (fuga) ni un
    cero inventado."""
    from types import SimpleNamespace

    from src.multihorizonte import ProyectorLightGBM

    meses = pd.period_range("2024-01", periods=6, freq="M")
    origen = pd.Period("2024-03", "M")

    precio = pd.DataFrame({
        "serie": ["S"] * 6, "periodo": meses,
        # Tras el origen (2024-03, valor 3.0) el precio REAL salta a 100: si la
        # persistencia mirara el futuro, se delataría acá.
        "precio": [1.0, 2.0, 3.0, 100.0, 100.0, 100.0],
    })
    tc = pd.Series([6.0, 7.0, 8.0, 50.0, 50.0, 50.0], index=meses)
    exogenas = SimpleNamespace(
        precio=precio, perdidas=None, quiebres=None, tipo_cambio=tc,
        categoria_por_serie=None,
    )

    cohorte = panel_de({"S": [10.0] * 6}, inicio="2024-01")
    proyector = ProyectorLightGBM(
        cfg, cohorte, SimpleNamespace(tabla=None, booster=None,
                                      mejor_iteracion=None),
        exogenas,
    )
    persistidas = proyector._exogenas_persistidas(origen, horizonte=3)

    futuros = persistidas.precio.loc[persistidas.precio["periodo"] > origen]
    assert len(futuros) == 3
    assert (futuros["precio"] == 3.0).all(), (
        "El precio futuro debe persistir el valor del ORIGEN (3.0), no el "
        "real futuro (100.0) ni cero"
    )
    tc_futuro = persistidas.tipo_cambio.loc[
        persistidas.tipo_cambio.index > origen
    ]
    assert (tc_futuro == 8.0).all()


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


# ---------------------------------------------------------------------------
# Ablación: re-estimación de los estadísticos por origen
# ---------------------------------------------------------------------------
# Panel sintético de 48 meses (2017-04 .. 2021-03): 36 de entrenamiento, 6 de
# validación y 6 de prueba. La serie A está DISEÑADA para que el alfa óptimo
# del suavizado exponencial cambie cuando se agrega historia posterior al
# entrenamiento: 36 meses alternando 100/101 (gana el alfa bajo, que promedia
# la alternancia) seguidos de una rampa de +30 por mes (gana el alfa alto, que
# la sigue). Con la rejilla {0,1; 0,9} la decisión es nítida a mano.

class _SESVigilado(SuavizadoExponencial):
    """SES que registra el ÚLTIMO mes de cada historia con la que se le pidió
    ajustar(). El registro vive en la clase: las copias que hace la ablación
    (``copy.deepcopy``) escriben en la misma lista."""

    ajustes: list[pd.Period] = []

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        _SESVigilado.ajustes.append(entrenamiento.index[-1])
        super().ajustar(entrenamiento, validacion)


def _panel_alfa_cambiante() -> pd.DataFrame:
    a = [100.0 + (t % 2) for t in range(36)] + [100.0 + 30.0 * k for k in range(1, 13)]
    return _ancho({"A": a, "B": serie_estacional(48)})


def _cfg_sintetico(cfg, reestimar: bool, origenes: str = "rodantes",
                   horizonte: int = 3):
    """La configuración real con cortes cortos para el panel sintético."""
    return replace(
        cfg,
        periodo=replace(cfg.periodo, inicio=pd.Period("2017-04", "M"),
                        fin=pd.Period("2021-03", "M")),
        particion=replace(cfg.particion,
                          fin_entrenamiento=pd.Period("2020-03", "M"),
                          fin_validacion=pd.Period("2020-09", "M")),
        multihorizonte=replace(cfg.multihorizonte, activo=True,
                               horizonte_maximo=horizonte, origenes=origenes,
                               reestimar_estadisticos=reestimar),
    )


def _brazos_ajustados(entrenamiento: pd.DataFrame) -> dict[str, object]:
    modelos = {
        "naive_m1": Naive(),
        "ma_2": PromedioMovil(2),
        "exp_smooth_opt": _SESVigilado(alfas=[0.1, 0.9]),
        "holt_winters": HoltWinters(periodo_estacional=12, alfas=[0.3],
                                    betas=[0.1], gammas=[0.2]),
        "croston": Croston(alfas=[0.1, 0.3]),
    }
    for modelo in modelos.values():
        modelo.ajustar(entrenamiento, entrenamiento)
    return modelos


def _evaluar(cfg_mh, modelos, panel_ajuste, panel_real):
    entrenamiento = panel_real.loc[
        panel_real.index <= cfg_mh.particion.fin_entrenamiento
    ]
    elegido = pd.Series({panel_real.columns[0]: "exp_smooth_opt",
                         panel_real.columns[1]: "naive_m1"})
    costo = pd.Series(1.0, index=panel_real.columns)
    return evaluar(
        cfg_mh, modelos, elegido, panel_ajuste, panel_real, entrenamiento,
        pd.DataFrame(), costo,
    )


def _resumen_origen_congelado(cfg_mh, modelos, panel, entrenamiento):
    """El protocolo con parámetros congelados, armado a mano con las
    primitivas públicas: es la referencia de «lo mismo que antes»."""
    media = entrenamiento.mean(axis=0, skipna=True)
    origenes = pd.period_range(
        cfg_mh.particion.fin_validacion, cfg_mh.periodo.fin - 1, freq="M"
    )
    filas = []
    for origen in origenes:
        horizonte = min(cfg_mh.multihorizonte.horizonte_maximo,
                        (cfg_mh.periodo.fin - origen).n)
        historia = panel.loc[panel.index <= origen]
        meses = pd.period_range(origen + 1, origen + horizonte, freq="M")
        real = panel.reindex(index=meses)
        ultimo = historia.ffill().iloc[-1]
        for nombre, modelo in modelos.items():
            proyectar = proyectar_constante if nombre == "croston" else proyectar_recursivo
            cruda = proyectar(modelo, historia, horizonte).reindex(
                index=meses, columns=panel.columns
            )
            pred = truncar_en_cero(cruda.fillna(ultimo).fillna(media))
            for mes in meses:
                for serie in panel.columns:
                    if np.isnan(real.loc[mes, serie]) or np.isnan(pred.loc[mes, serie]):
                        continue
                    filas.append({
                        "modelo": nombre, "origen": str(origen),
                        "horizonte": (mes - origen).n, "serie": serie,
                        "y_real": real.loc[mes, serie],
                        "y_pred": pred.loc[mes, serie], "costo": 1.0,
                    })
    return agregado_valorizado(pd.DataFrame(filas), ["modelo", "origen"])


def _ordenado(resumen: pd.DataFrame, modelos) -> pd.DataFrame:
    return (
        resumen.loc[resumen["modelo"].isin(modelos)]
        .sort_values(["modelo", "origen"])
        .reset_index(drop=True)
        [["modelo", "origen", "n_observaciones", "wmape_val", "bias_val", "D"]]
    )


def test_reestimar_en_origen_reestima_con_la_historia_y_no_toca_el_original():
    """La copia re-estimada elige el alfa de la historia completa hasta el
    origen; el modelo ajustado en entrenamiento queda intacto."""
    panel = _panel_alfa_cambiante()
    entrenamiento = panel.iloc[:36]
    serie_a = panel.columns[0]

    ses = SuavizadoExponencial(alfas=[0.1, 0.9])
    ses.ajustar(entrenamiento, entrenamiento)
    assert ses.alfa_por_serie[serie_a] == pytest.approx(0.1), (
        "Con la alternancia 100/101 del entrenamiento gana el alfa bajo"
    )

    historia = panel.iloc[:47]                    # un origen tardío, con rampa
    copia = _reestimar_en_origen(ses, historia)
    assert copia is not ses
    assert copia.alfa_por_serie[serie_a] == pytest.approx(0.9), (
        "Con la rampa incorporada a la historia gana el alfa alto"
    )
    assert ses.alfa_por_serie[serie_a] == pytest.approx(0.1), (
        "El modelo original (parámetros congelados) no se toca"
    )


def test_flag_apagado_es_el_protocolo_congelado(cfg):
    """Con la ablación apagada nadie re-ajusta nada y la curva por origen es
    exactamente la del protocolo de parámetros congelados."""
    panel = _panel_alfa_cambiante()
    cfg_mh = _cfg_sintetico(cfg, reestimar=False)
    entrenamiento = panel.loc[panel.index <= cfg_mh.particion.fin_entrenamiento]

    _SESVigilado.ajustes.clear()
    modelos = _brazos_ajustados(entrenamiento)
    alfa_base = modelos["exp_smooth_opt"].alfa_por_serie.copy()

    resultado = _evaluar(cfg_mh, modelos, panel, panel)

    assert resultado.informe.reestimaciones_estadisticas == 0
    assert _SESVigilado.ajustes == [entrenamiento.index[-1]], (
        "Con el flag apagado el único ajuste es el del paso 7 (entrenamiento)"
    )
    assert not any("Re-estimaciones" in linea for linea in resultado.informe.lineas())
    pd.testing.assert_series_equal(modelos["exp_smooth_opt"].alfa_por_serie, alfa_base)

    esperado = _resumen_origen_congelado(cfg_mh, modelos, panel, entrenamiento)
    pd.testing.assert_frame_equal(
        _ordenado(resultado.resumen_origen, modelos),
        _ordenado(esperado, modelos),
        check_dtype=False,
    )


def test_reestimar_rodante_reajusta_una_vez_por_origen_con_la_historia_hasta_el_origen(cfg):
    panel = _panel_alfa_cambiante()
    cfg_mh = _cfg_sintetico(cfg, reestimar=True)
    entrenamiento = panel.loc[panel.index <= cfg_mh.particion.fin_entrenamiento]
    origenes = list(pd.period_range(
        cfg_mh.particion.fin_validacion, cfg_mh.periodo.fin - 1, freq="M"
    ))

    _SESVigilado.ajustes.clear()
    modelos = _brazos_ajustados(entrenamiento)
    resultado = _evaluar(cfg_mh, modelos, panel, panel)

    # Un re-ajuste por origen y brazo CON parámetros (los sin parámetros se
    # re-ajustan igual, pero no se cuentan: su ajustar() no hace nada).
    con_parametros = [m for m in modelos.values() if _tiene_parametros(m)]
    assert resultado.informe.reestimaciones_estadisticas == len(origenes) * len(con_parametros)
    assert _SESVigilado.ajustes == [entrenamiento.index[-1], *origenes], (
        "Cada re-ajuste del SES usa la historia que termina EXACTAMENTE en su "
        "origen: ni antes (sería tirar información) ni después (sería fuga)"
    )
    assert any("Re-estimaciones" in linea for linea in resultado.informe.lineas())


def test_reestimar_cambia_al_ses_en_origen_tardio_y_no_mira_el_futuro(cfg):
    """(1) Con el flag encendido la proyección del SES en un origen posterior
    al entrenamiento DIFIERE de la congelada (el alfa cambió); los brazos sin
    parámetros no cambian. (2) Ningún re-ajuste usa datos posteriores al
    origen: perturbar el panel de AJUSTE después del origen no altera nada."""
    panel = _panel_alfa_cambiante()
    serie_a = panel.columns[0]
    congelado = _cfg_sintetico(cfg, reestimar=False, origenes="fijo")
    ablacion = _cfg_sintetico(cfg, reestimar=True, origenes="fijo")
    entrenamiento = panel.loc[panel.index <= congelado.particion.fin_entrenamiento]
    origen = congelado.particion.fin_validacion

    _SESVigilado.ajustes.clear()
    modelos = _brazos_ajustados(entrenamiento)
    alfa_base = modelos["exp_smooth_opt"].alfa_por_serie.copy()
    assert alfa_base[serie_a] == pytest.approx(0.1)

    r_congelado = _evaluar(congelado, modelos, panel, panel)
    r_ablacion = _evaluar(ablacion, modelos, panel, panel)

    assert r_ablacion.informe.reestimaciones_estadisticas == sum(
        1 for m in modelos.values() if _tiene_parametros(m)
    )
    assert _SESVigilado.ajustes == [entrenamiento.index[-1], origen]
    pd.testing.assert_series_equal(
        modelos["exp_smooth_opt"].alfa_por_serie, alfa_base
    ), "El modelo del paso 7 sigue congelado aun con la ablación encendida"

    def d(resultado, modelo):
        return float(resultado.resumen_origen.set_index("modelo").loc[modelo, "D"])

    assert d(r_ablacion, "exp_smooth_opt") != pytest.approx(d(r_congelado, "exp_smooth_opt")), (
        "El SES re-estimado en el origen (alfa 0,9) proyecta distinto que el "
        "congelado (alfa 0,1): la ablación mide algo"
    )
    assert d(r_ablacion, "naive_m1") == pytest.approx(d(r_congelado, "naive_m1")), (
        "Sin parámetros que re-estimar, el Naïve no cambia"
    )

    # Sin fuga. El panel de ajuste perturbado DESPUÉS del origen alterna
    # 0/1000: si la re-estimación mirara esos meses, el alfa volvería a 0,1.
    perturbado = panel.copy()
    despues = perturbado.index > origen
    perturbado.loc[despues, serie_a] = [0.0, 1000.0] * (int(despues.sum()) // 2)

    con_fuga = SuavizadoExponencial(alfas=[0.1, 0.9])
    con_fuga.ajustar(perturbado, perturbado)
    honesto = SuavizadoExponencial(alfas=[0.1, 0.9])
    honesto.ajustar(panel.loc[panel.index <= origen], panel.iloc[0:0])
    assert con_fuga.alfa_por_serie[serie_a] != honesto.alfa_por_serie[serie_a], (
        "Precondición: la perturbación cambiaría el alfa si se mirara el futuro"
    )

    # panel_real intacto: la realidad contra la que se compara es la misma.
    r_perturbado = _evaluar(ablacion, modelos, perturbado, panel)
    pd.testing.assert_frame_equal(r_perturbado.resumen_origen, r_ablacion.resumen_origen)
    pd.testing.assert_frame_equal(r_perturbado.resumen_horizonte, r_ablacion.resumen_horizonte)
