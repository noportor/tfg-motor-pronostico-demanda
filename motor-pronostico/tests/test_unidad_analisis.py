"""Unidad de análisis: cada estadístico verificado contra el cálculo a mano.

La etapa justifica un supuesto de DISEÑO, así que su disciplina temporal se
prueba igual que la de los modelos: la estructura solo puede mirar
entrenamiento y el contrafactual solo puede evaluarse en validación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import panel_de
from src.unidad_analisis import (
    clasificar_regimen, contrafactual_topdown, estructura_por_sku,
    ganadores_por_sku,
)


def _largo(series: dict[str, list[float]], inicio="2017-04", **kwargs) -> pd.DataFrame:
    return panel_de(series, inicio=inicio, **kwargs)


# ---------------------------------------------------------------------------
# Regímenes ADI–CV²
# ---------------------------------------------------------------------------

def test_adi_y_cv2_a_mano():
    """Serie [5,0,5,0,…]: vida 8, 4 positivos → ADI 2; CV² de [5,5,5,5] = 0
    → cuadrante intermitente. Serie pareja → suave."""
    largo = _largo({
        "ESPORADICA": [5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0],
        "PAREJA": [10.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 9.0],
    })
    tabla = clasificar_regimen(largo)
    tabla.index = [s.split("|")[0] for s in tabla.index]

    assert tabla.loc["ESPORADICA", "adi"] == pytest.approx(2.0)
    assert tabla.loc["ESPORADICA", "cv2"] == pytest.approx(0.0)
    assert tabla.loc["ESPORADICA", "cuadrante"] == "intermitente"
    assert tabla.loc["PAREJA", "adi"] == pytest.approx(1.0)
    assert tabla.loc["PAREJA", "cuadrante"] == "suave"


def test_serie_con_pocos_positivos_queda_sin_clasificar():
    largo = _largo({"RALA": [0.0, 7.0, 0.0, 0.0, 3.0, 0.0]})
    tabla = clasificar_regimen(largo)
    assert (tabla["cuadrante"] == "sin_clasificar").all(), (
        "Con menos de 4 meses positivos el CV² es una anécdota, no un régimen"
    )


# ---------------------------------------------------------------------------
# Estructura por SKU
# ---------------------------------------------------------------------------

def test_deriva_de_participacion_a_mano():
    """Dos combinaciones del mismo SKU: participaciones 60/40 → 50/50 → 70/30.
    |Δ| por combinación = (10 + 20) / 2 = 15 pp."""
    a = _largo({"S1": [60.0, 50.0, 70.0]}, canal="C1")
    b = _largo({"S1": [40.0, 50.0, 30.0]}, canal="C2")
    largo = pd.concat([a, b], ignore_index=True)

    tabla, resumen = estructura_por_sku(largo)
    assert resumen["skus_multicombo"] == 1
    assert resumen["deriva_participacion_pp_media"] == pytest.approx(15.0)


def test_correlacion_intra_sku_a_mano():
    """Una combinación sube linealmente y la otra baja: r = −1 exacto."""
    n = 30
    a = _largo({"S1": [float(10 + t) for t in range(n)]}, canal="C1")
    b = _largo({"S1": [float(100 - t) for t in range(n)]}, canal="C2")
    largo = pd.concat([a, b], ignore_index=True)

    _, resumen = estructura_por_sku(largo)
    assert resumen["correlacion_mediana"] == pytest.approx(-1.0)
    assert resumen["pares_correlacion"] == 1


def test_par_sin_solape_no_aporta_correlacion():
    """Dos combinaciones que casi no conviven: el par se cuenta como sin
    solape, jamás como una correlación calculada sobre nada."""
    a = _largo({"S1": [10.0] * 30}, inicio="2017-04", canal="C1")
    b = _largo({"S1": [10.0] * 10}, inicio="2020-04", canal="C2")
    largo = pd.concat([a, b], ignore_index=True)

    tabla, resumen = estructura_por_sku(largo)
    assert resumen["pares_correlacion"] == 0
    assert int(tabla["pares_sin_solape"].fillna(0).sum()) == 1


def test_regimenes_mixtos_bajo_el_mismo_sku():
    esporadica = [5.0, 0.0, 0.0, 5.0, 0.0, 0.0] * 5
    pareja = [10.0, 11.0, 9.0, 10.0, 11.0, 10.0] * 5
    a = _largo({"S1": esporadica}, canal="C1")
    b = _largo({"S1": pareja}, canal="C2")
    largo = pd.concat([a, b], ignore_index=True)

    _, resumen = estructura_por_sku(largo)
    assert resumen["skus_con_regimenes_mixtos"] == 1


def test_ganadores_distintos_por_sku():
    eleccion = pd.Series(
        {"A|C1|R1": "naive_m1", "A|C2|R1": "holt_winters", "B|C1|R1": "ma_12",
         "B|C2|R1": "ma_12"},
        name="modelo_elegido",
    )
    mapa = pd.Series(
        {"A|C1|R1": "A", "A|C2|R1": "A", "B|C1|R1": "B", "B|C2|R1": "B"}
    )
    _, resumen = ganadores_por_sku(eleccion, mapa)
    assert resumen["skus_cohorte_multi"] == 2
    assert resumen["skus_multiganador"] == 1
    assert resumen["proporcion_multiganador"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Contrafactual arriba/abajo
# ---------------------------------------------------------------------------

def _cfg_cortes(cfg, fin_entrenamiento="2019-03", fin_validacion="2020-03"):
    """Config con cortes cortos para paneles sintéticos chicos."""
    from src.config import cargar_config
    return cargar_config(cfg.ruta_config, anulaciones=[
        f"particion.fin_entrenamiento={fin_entrenamiento}",
        f"particion.fin_validacion={fin_validacion}",
        "periodo.fin=2021-03",
        "evento_exogeno.desde=2018-03",
        "evento_exogeno.hasta=2019-02",
    ])


def test_contrafactual_combos_identicos_empata_exacto(cfg):
    """Dos combinaciones idénticas: participaciones 50/50 constantes, y el
    promedio móvil del agregado repartido == promedio móvil por combinación.
    El contrafactual DEBE dar empate exacto — si no, hay un error de cálculo."""
    n = 48  # 2017-04 .. 2021-03
    valores = [float(100 + 10 * np.sin(2 * np.pi * t / 12)) for t in range(n)]
    a = _largo({"S1": valores}, canal="C1")
    b = _largo({"S1": valores}, canal="C2")
    largo = pd.concat([a, b], ignore_index=True)
    panel_real = largo.pivot(index="periodo", columns="serie", values="y")
    costo = pd.Series(2.0, index=panel_real.columns)

    tabla = contrafactual_topdown(_cfg_cortes(cfg), largo, panel_real, costo)

    for modelo in tabla["modelo"].unique():
        fila = tabla.loc[tabla["modelo"] == modelo]
        arriba = float(fila.loc[fila["enfoque"] == "top_down", "D"].iloc[0])
        abajo = float(fila.loc[fila["enfoque"] == "bottom_up", "D"].iloc[0])
        assert arriba == pytest.approx(abajo, abs=1e-9), (
            f"{modelo}: con combinaciones idénticas arriba y abajo son el "
            "mismo pronóstico; una diferencia delata fuga o mal prorrateo"
        )


def test_contrafactual_no_evalua_fuera_de_validacion(cfg):
    """Cambiar los meses de PRUEBA no puede mover ni un decimal del
    contrafactual: se evalúa solo en validación."""
    n = 48
    base = [float(50 + t) for t in range(n)]
    otros = [float(80 + 2 * t) for t in range(n)]

    def armar(valores_b):
        a = _largo({"S1": base}, canal="C1")
        b = _largo({"S1": valores_b}, canal="C2")
        return pd.concat([a, b], ignore_index=True)

    cfg_corto = _cfg_cortes(cfg)
    fin_val = cfg_corto.particion.fin_validacion

    con_prueba_x = armar(otros)
    otros_alterados = list(otros)
    otros_alterados[-6:] = [9999.0] * 6          # solo meses de prueba
    con_prueba_y = armar(otros_alterados)

    panel_x = con_prueba_x.pivot(index="periodo", columns="serie", values="y")
    panel_y = con_prueba_y.pivot(index="periodo", columns="serie", values="y")
    assert not panel_x.equals(panel_y)
    assert panel_x.loc[panel_x.index <= fin_val].equals(
        panel_y.loc[panel_y.index <= fin_val]
    ), "La alteración debía caer íntegra en prueba"

    costo = pd.Series(1.0, index=panel_x.columns)
    tabla_x = contrafactual_topdown(cfg_corto, con_prueba_x, panel_x, costo)
    tabla_y = contrafactual_topdown(cfg_corto, con_prueba_y, panel_y, costo)
    pd.testing.assert_frame_equal(tabla_x, tabla_y)


def test_contrafactual_pareado_en_resurreccion(cfg):
    """El escenario que la auditoría reprodujo: un SKU con doce meses cerrados
    en cero y resurrección dentro de validación. La participación da 0/0 = NaN
    y el enfoque top-down no puede pronosticar ese mes. Sin máscara COMÚN, el
    top-down se salteaba justo el mes donde el bottom-up come su peor error
    (delta fabricado de hasta 66 puntos). Con proporciones constantes 60/40 el
    resultado correcto es empate EXACTO sobre el soporte común, con las
    exclusiones contadas."""
    meses = pd.period_range("2017-04", periods=48, freq="M")
    corte_muerte = pd.Period("2018-10", "M")
    resurreccion = pd.Period("2019-10", "M")

    def combo(nivel):
        valores = []
        for mes in meses:
            if corte_muerte <= mes < resurreccion:
                valores.append(0.0)
            else:
                valores.append(nivel)
        return valores

    a = _largo({"S1": combo(60.0)}, canal="C1")
    b = _largo({"S1": combo(40.0)}, canal="C2")
    largo = pd.concat([a, b], ignore_index=True)
    panel_real = largo.pivot(index="periodo", columns="serie", values="y")
    costo = pd.Series(1.0, index=panel_real.columns)

    tabla = contrafactual_topdown(_cfg_cortes(cfg), largo, panel_real, costo)

    assert (tabla["observaciones_excluidas_del_pareo"] > 0).all(), (
        "El mes de resurrección (participación 0/0) debía quedar excluido "
        "del pareo Y contado — nunca omitido en silencio"
    )
    obs = tabla.pivot(index="modelo", columns="enfoque", values="observaciones")
    assert (obs["bottom_up"] == obs["top_down"]).all(), (
        "Los dos enfoques deben evaluarse sobre EXACTAMENTE las mismas "
        "observaciones (comparación pareada)"
    )
    for _, fila in tabla.iterrows():
        assert fila["delta_D_topdown"] == pytest.approx(0.0, abs=1e-9), (
            f"{fila['modelo']}: con participaciones constantes, arriba y "
            "abajo son el mismo pronóstico sobre el soporte común"
        )


def test_evento_excluido_de_la_estructura():
    """La ventana del evento no puede aportar NADA a deriva ni correlaciones:
    dos variantes del panel que solo difieren DENTRO de la ventana deben dar
    la misma estructura cuando la ventana se excluye. (Sin exclusión, la copia
    de gestión previa fabricaba interanuales (0,0) que inflaban r_yoy.)"""
    n = 60
    ventana = pd.period_range("2019-04", "2020-03", freq="M")

    def panel_con_ventana(valor_ventana):
        filas = []
        for canal, base in (("C1", 100.0), ("C2", 50.0)):
            serie = _largo(
                {"S1": [base + 3 * ((t * 7) % 5) + 0.5 * t for t in range(n)]},
                canal=canal,
            )
            en_ventana = serie["periodo"].isin(ventana)
            serie.loc[en_ventana, "y"] = valor_ventana
            filas.append(serie)
        return pd.concat(filas, ignore_index=True)

    con_a = panel_con_ventana(999.0)
    con_b = panel_con_ventana(1.0)

    _, resumen_a = estructura_por_sku(con_a, excluir_meses=ventana)
    _, resumen_b = estructura_por_sku(con_b, excluir_meses=ventana)

    for clave in ("deriva_participacion_pp_media", "correlacion_mediana",
                  "correlacion_yoy_mediana", "pares_correlacion"):
        va, vb = resumen_a[clave], resumen_b[clave]
        iguales = va == vb or (
            isinstance(va, float) and np.isnan(va) and np.isnan(vb)
        )
        assert iguales, (
            f"'{clave}' cambió con el contenido de la ventana excluida: "
            "la exclusión no está funcionando"
        )

    # Y la exclusión importa: sin ella, el contenido de la ventana SÍ mueve
    # el número (eso era el defecto).
    _, sin_exclusion_a = estructura_por_sku(con_a)
    _, sin_exclusion_b = estructura_por_sku(con_b)
    assert (
        sin_exclusion_a["correlacion_mediana"]
        != sin_exclusion_b["correlacion_mediana"]
    )


def test_evaluar_con_costos_pero_sin_skus_multicombo(cfg):
    """La rama del aviso del visor: costos presentes, ningún SKU multi.
    El contrafactual queda vacío y lineas() degrada sin reventar."""
    from src.unidad_analisis import evaluar

    largo = _largo({"UNICO": [10.0 + t for t in range(48)]})
    panel = largo.pivot(index="periodo", columns="serie", values="y")
    costo = pd.Series(1.0, index=panel.columns)

    informe = evaluar(
        _cfg_cortes(cfg), largo, largo, panel,
        pd.Series(dtype=object), costo,
    )
    assert informe.contrafactual.empty
    assert informe.resumen["skus_multicombo"] == 0
    salida = informe.lineas()
    assert isinstance(salida, list) and salida, (
        "lineas() debe producir el informe aun sin SKUs multi-combinación"
    )


def test_estructura_no_mira_despues_de_entrenamiento(cfg):
    """La estructura se mide sobre entrenamiento: meses posteriores al corte
    no pueden cambiarla. Lo garantiza el recorte que hace la orquestación."""
    from src.unidad_analisis import evaluar

    n = 48
    a = _largo({"S1": [float(10 + t) for t in range(n)]}, canal="C1")
    b = _largo({"S1": [float(30 - 0.2 * t) for t in range(n)]}, canal="C2")
    largo_x = pd.concat([a, b], ignore_index=True)

    largo_y = largo_x.copy()
    despues = largo_y["periodo"] > pd.Period("2019-03", "M")
    largo_y.loc[despues, "y"] = largo_y.loc[despues, "y"] * 100

    cfg_corto = _cfg_cortes(cfg)
    eleccion = pd.Series(dtype=object)
    panel_x = largo_x.pivot(index="periodo", columns="serie", values="y")
    panel_y = largo_y.pivot(index="periodo", columns="serie", values="y")

    ix = evaluar(cfg_corto, largo_x, largo_x, panel_x, eleccion, None)
    iy = evaluar(cfg_corto, largo_y, largo_y, panel_y, eleccion, None)
    claves_estructura = [
        "skus_totales", "skus_multicombo", "deriva_participacion_pp_media",
        "correlacion_mediana", "pares_correlacion", "skus_con_regimenes_mixtos",
    ]
    for clave in claves_estructura:
        vx, vy = ix.resumen[clave], iy.resumen[clave]
        iguales = (vx == vy) or (
            isinstance(vx, float) and np.isnan(vx) and np.isnan(vy)
        )
        assert iguales, f"'{clave}' cambió con datos posteriores al corte: fuga"
