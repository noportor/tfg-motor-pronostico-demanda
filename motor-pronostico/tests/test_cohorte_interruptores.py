"""Interruptores de cohorte: sensibilidad «cohorte al origen».

Criterios de aceptación:

«Con los valores por defecto la cascada es la publicada (pasos 0 a 6, mismos
rótulos). Con ``inclusion.exigir_presencia_prueba: false`` el paso 6 no excluye
y queda declarado. Con ``series.extender_vida_hasta_fin_periodo: true`` cada
serie vive hasta el fin del período con ceros después de su última venta, sin
mover su nacimiento, y la presencia en validación se mide sobre la vida
observada. Con ``inclusion.estado_al_origen: activa`` sólo quedan las series con
venta en los seis meses previos al origen de la prueba. El informe de la cohorte
declara siempre el valor de cada interruptor.»

Datos 100 % sintéticos (RN-1).
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.config import ErrorDeConfiguracion, cargar_config
from src.particion import (
    aplicar_criterios_inclusion,
    cohorte_por_defecto,
    particionar,
    sensibilidad_umbrales,
    tabla_fuera_de_muestra,
)
from src.series import construir_series

PASOS_PUBLICADOS = [
    "0. series construidas",
    "1. con historia en entrenamiento",
    "2. historial >= 36 meses",
    "3. proporción de meses en cero <= 0.70",
    "4. volumen acumulado >= 50 unidades",
    "5. presente en validación",
    "6. presente en prueba",
]


# --- utilidades ---------------------------------------------------------------

def _variante(cfg, *, exigir_prueba=True, extender=False, estado="todas"):
    return replace(
        cfg,
        series=replace(cfg.series, extender_vida_hasta_fin_periodo=extender),
        inclusion=replace(cfg.inclusion, exigir_presencia_prueba=exigir_prueba,
                          estado_al_origen=estado),
    )


def _crudo_sintetico(cfg) -> pd.DataFrame:
    """Cuatro series que venden 20 unidades por mes hasta su última venta.

    - SANA: vende hasta el fin del período (marzo de 2026).
    - SIN_PRUEBA: última venta en enero de 2025 (validación); activa al origen.
    - DORMIDA: última venta en julio de 2024 (validación); al origen lleva ocho
      meses sin vender.
    - MUERTA: última venta en junio de 2023 (entrenamiento).
    """
    ultimas = {
        "SANA": str(cfg.periodo.fin),
        "SIN_PRUEBA": "2025-01",
        "DORMIDA": "2024-07",
        "MUERTA": "2023-06",
    }
    filas = []
    for sku, ultima in ultimas.items():
        for periodo in pd.period_range(cfg.periodo.inicio, ultima, freq="M"):
            filas.append({
                "fecha": periodo.to_timestamp() + pd.Timedelta(days=9),
                "sku": sku, "canal": "CANAL-1", "regional": "REGIONAL-A",
                "cantidad": 20.0,
            })
    return pd.DataFrame(filas)


def _cohorte(cfg_v):
    panel, informe_series = construir_series(_crudo_sintetico(cfg_v), cfg_v)
    corte = particionar(cfg_v)
    cohorte, informe = aplicar_criterios_inclusion(
        panel, cfg_v, corte, panel_estado=panel
    )
    skus = set(cohorte["sku"].unique())
    return panel, informe_series, corte, skus, informe


# --- configuración ------------------------------------------------------------

def test_los_valores_por_defecto_son_los_de_la_corrida_publicada(cfg):
    assert cfg.inclusion.exigir_presencia_prueba is True
    assert cfg.series.extender_vida_hasta_fin_periodo is False
    assert cfg.inclusion.estado_al_origen == "todas"
    assert cohorte_por_defecto(cfg)


@pytest.mark.parametrize("archivo, estado", [
    ("config/config_sensibilidad_a.yaml", "todas"),
    ("config/config_sensibilidad_b.yaml", "activa"),
])
def test_las_configuraciones_de_sensibilidad_solo_cambian_sus_claves(cfg, archivo, estado):
    variante = cargar_config(archivo)

    assert variante.inclusion.exigir_presencia_prueba is False
    assert variante.series.extender_vida_hasta_fin_periodo is True
    assert variante.inclusion.estado_al_origen == estado
    assert variante.salidas["directorio"] == (
        "salidas_sensibilidad_a" if estado == "todas" else "salidas_sensibilidad_b"
    )
    assert variante.crudo["_hereda_de"] == "config.yaml"

    # Todo lo demás es la configuración base, clave por clave.
    base = cfg.crudo
    for seccion in base:
        if seccion in ("inclusion", "series", "salidas"):
            continue
        assert variante.crudo[seccion] == base[seccion], seccion
    for seccion, claves in (
        ("inclusion", {"exigir_presencia_prueba", "estado_al_origen"}),
        ("series", {"extender_vida_hasta_fin_periodo"}),
        ("salidas", {"directorio"}),
    ):
        for clave, valor in base[seccion].items():
            if clave not in claves:
                assert variante.crudo[seccion][clave] == valor, (seccion, clave)
    assert variante.hash_configuracion() != cfg.hash_configuracion()


def test_un_archivo_derivado_con_una_errata_falla(tmp_path):
    derivado = tmp_path / "derivado.yaml"
    base = cargar_config().ruta_config
    derivado.write_text(
        f'hereda_de: "{base.as_posix()}"\n'
        "inclusion:\n  exigir_presencia_preuba: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ErrorDeConfiguracion, match="exigir_presencia_preuba"):
        cargar_config(derivado)


@pytest.mark.parametrize("anulacion, mensaje", [
    ('inclusion.exigir_presencia_prueba="false"', "booleano"),
    ('series.extender_vida_hasta_fin_periodo="si"', "booleano"),
    ("inclusion.estado_al_origen=dormida", "estado_al_origen"),
])
def test_rechaza_interruptores_mal_escritos(anulacion, mensaje):
    with pytest.raises(ErrorDeConfiguracion, match=mensaje):
        cargar_config(anulaciones=[anulacion])


def test_extender_la_vida_exige_el_relleno_de_ceros():
    with pytest.raises(ErrorDeConfiguracion, match="rellenar_ceros_intermedios"):
        cargar_config(anulaciones=[
            "series.extender_vida_hasta_fin_periodo=true",
            "series.rellenar_ceros_intermedios=false",
        ])


def test_los_interruptores_se_pueden_anular(cfg):
    variante = cargar_config(anulaciones=["inclusion.exigir_presencia_prueba=false"])
    assert variante.inclusion.exigir_presencia_prueba is False
    assert not cohorte_por_defecto(variante)


# --- series: vida extendida ---------------------------------------------------

def test_la_vida_extendida_agrega_ceros_hasta_el_fin_del_periodo(cfg):
    df = pd.DataFrame([
        {"fecha": pd.Timestamp("2018-03-05"), "sku": "A", "canal": "C",
         "regional": "R", "cantidad": 30.0},
        {"fecha": pd.Timestamp("2018-05-05"), "sku": "A", "canal": "C",
         "regional": "R", "cantidad": 10.0},
    ])
    panel_def, informe_def = construir_series(df, cfg)
    panel_ext, informe_ext = construir_series(df, _variante(cfg, extender=True))

    # Por defecto: vida de la primera a la última venta, como siempre.
    assert str(panel_def["periodo"].max()) == "2018-05"
    assert informe_def.meses_extendidos_con_cero == 0
    assert not informe_def.vida_extendida_hasta_fin_periodo

    # Extendida: el nacimiento no se mueve y después de la última venta hay ceros.
    assert str(panel_ext["periodo"].min()) == "2018-03"
    assert panel_ext["periodo"].max() == cfg.periodo.fin
    esperados = len(pd.period_range("2018-03", cfg.periodo.fin, freq="M"))
    assert len(panel_ext) == esperados
    despues = panel_ext.loc[panel_ext["periodo"] > pd.Period("2018-05", "M"), "y"]
    assert (despues == 0.0).all()
    assert informe_ext.meses_extendidos_con_cero == len(despues)
    assert informe_ext.vida_extendida_hasta_fin_periodo
    # Los meses observados no cambian.
    comunes = panel_ext.loc[panel_ext["periodo"] <= pd.Period("2018-05", "M")]
    assert comunes["y"].tolist() == panel_def["y"].tolist()


# --- cascada ------------------------------------------------------------------

def test_por_defecto_la_cascada_es_la_publicada(cfg):
    _, _, _, skus, informe = _cohorte(cfg)

    assert list(informe.tabla_flujo()["paso"]) == PASOS_PUBLICADOS
    assert skus == {"SANA"}
    assert informe.excluidas_sin_validacion == 1      # MUERTA
    assert informe.excluidas_sin_prueba == 2          # SIN_PRUEBA y DORMIDA
    assert informe.interruptores == {
        "inclusion.exigir_presencia_prueba": True,
        "series.extender_vida_hasta_fin_periodo": False,
        "inclusion.estado_al_origen": "todas",
    }
    assert any("exigir_presencia_prueba = True" in linea
               for linea in informe.lineas())


def test_sin_el_paso_6_la_cascada_lo_declara_y_no_excluye(cfg):
    """Sin extender la vida: las series que no llegan a prueba vuelven a entrar."""
    _, _, _, skus, informe = _cohorte(_variante(cfg, exigir_prueba=False))
    flujo = informe.tabla_flujo()

    assert skus == {"SANA", "SIN_PRUEBA", "DORMIDA"}
    paso6 = flujo.iloc[6]
    assert paso6["paso"].startswith("6. presente en prueba: no se aplica")
    assert paso6["excluidas"] == 0
    assert informe.excluidas_sin_prueba == 0
    assert len(flujo) == 7


def test_variante_a_cohorte_al_origen(cfg):
    cfg_a = _variante(cfg, exigir_prueba=False, extender=True)
    panel, _, corte, skus, informe = _cohorte(cfg_a)

    # La MUERTA no tiene vida observada en validación: sigue fuera aunque la
    # extensión le dé ceros en ese bloque.
    assert skus == {"SANA", "SIN_PRUEBA", "DORMIDA"}
    assert informe.excluidas_sin_validacion == 1
    assert informe.tabla_flujo().iloc[5]["paso"] == (
        "5. presente en validación (vida observada)"
    )
    # Las series que dejaron de vender se evalúan contra ceros en prueba.
    prueba = panel.loc[panel["periodo"].isin(corte.prueba)]
    assert set(prueba["sku"]) == {"SANA", "SIN_PRUEBA", "DORMIDA", "MUERTA"}
    assert (prueba.loc[prueba["sku"] == "DORMIDA", "y"] == 0.0).all()
    assert len(prueba.loc[prueba["sku"] == "SIN_PRUEBA"]) == len(corte.prueba)
    # La que muere en validación (enero de 2025) queda en la cohorte con ceros
    # en cada mes desde febrero de 2025 hasta el fin del período, sin huecos.
    sin_prueba = panel.loc[panel["sku"] == "SIN_PRUEBA"].set_index("periodo")["y"]
    despues = sin_prueba.loc[sin_prueba.index > pd.Period("2025-01", "M")]
    assert list(despues.index) == list(
        pd.period_range("2025-02", cfg_a.periodo.fin, freq="M")
    )
    assert (despues == 0.0).all()
    assert (sin_prueba.loc[sin_prueba.index <= pd.Period("2025-01", "M")] == 20.0).all()
    assert informe.interruptores["series.extender_vida_hasta_fin_periodo"] is True
    assert informe.interruptores["inclusion.exigir_presencia_prueba"] is False


def test_variante_b_solo_activas_al_origen(cfg):
    cfg_b = _variante(cfg, exigir_prueba=False, extender=True, estado="activa")
    _, _, _, skus, informe = _cohorte(cfg_b)
    flujo = informe.tabla_flujo()

    # DORMIDA lleva ocho meses sin vender al origen (abril de 2025): fuera.
    assert skus == {"SANA", "SIN_PRUEBA"}
    assert informe.excluidas_no_activas_al_origen == 1
    assert flujo.iloc[-1]["paso"] == "7. activa al origen (venta entre 2024-10 y 2025-03)"
    assert flujo.iloc[-1]["excluidas"] == 1
    assert informe.interruptores["inclusion.estado_al_origen"] == "activa"


@pytest.mark.parametrize("variante", [
    {},
    {"exigir_prueba": False, "extender": True},
    {"exigir_prueba": False, "extender": True, "estado": "activa"},
])
def test_fuera_de_muestra_y_sensibilidad_siguen_a_la_cascada(cfg, variante):
    cfg_v = _variante(cfg, **variante)
    panel, _, corte, skus, informe = _cohorte(cfg_v)

    fuera = tabla_fuera_de_muestra(panel, panel, {
        s for s in panel["serie"].unique() if s.split("|")[0] in skus
    }, cfg_v, corte)
    incluidas = set(fuera.loc[fuera["motivo_exclusion"] == "incluida", "serie"])
    assert {s.split("|")[0] for s in incluidas} == skus
    assert ("estado_al_origen" in fuera.columns) == bool(variante)

    sens = sensibilidad_umbrales(panel, corte, cfg=cfg_v, panel_estado=panel)
    fila = sens.loc[
        (sens["historial_minimo_meses"] == cfg.inclusion.historial_minimo_meses)
        & (sens["proporcion_maxima_ceros"] == cfg.inclusion.proporcion_maxima_ceros)
        & (sens["volumen_minimo_unidades"] == cfg.inclusion.volumen_minimo_unidades)
    ]
    assert int(fila["n_series"].iloc[0]) == informe.series_finales


def test_fuera_de_muestra_nombra_el_motivo_de_cada_variante(cfg):
    por_defecto = _cohorte(cfg)
    panel, _, corte, skus, _ = por_defecto
    fuera = tabla_fuera_de_muestra(panel, panel, set(), cfg, corte).set_index("serie")
    motivo = {s.split("|")[0]: m for s, m in fuera["motivo_exclusion"].items()}
    assert motivo["SIN_PRUEBA"] == "ausente_en_prueba"
    assert motivo["MUERTA"] == "ausente_en_validacion"

    cfg_b = _variante(cfg, exigir_prueba=False, extender=True, estado="activa")
    panel_b, _, corte_b, _, _ = _cohorte(cfg_b)
    fuera_b = tabla_fuera_de_muestra(panel_b, panel_b, set(), cfg_b, corte_b)
    fuera_b = fuera_b.set_index("serie")
    motivo_b = {s.split("|")[0]: m for s, m in fuera_b["motivo_exclusion"].items()}
    estado_b = {s.split("|")[0]: e for s, e in fuera_b["estado_al_origen"].items()}
    assert motivo_b["DORMIDA"] == "no_activa_al_origen"
    assert motivo_b["SIN_PRUEBA"] == "incluida"
    assert estado_b["DORMIDA"] == "dormida"
    assert estado_b["SIN_PRUEBA"] == "activa"


# --- de punta a punta: la variante B corre el pipeline completo ---------------

def test_la_variante_b_corre_el_pipeline_y_declara_sus_interruptores(tmp_path):
    """El pipeline completo acepta la cohorte al origen y lo deja en el registro.

    Corrida sintética reducida (mismo armado que tests/test_pipeline.py): lo que
    se prueba es que los interruptores atraviesan todas las etapas y quedan
    declarados en cohorte_flujo.csv, flujo.json y el manifiesto.
    """
    import json

    import yaml

    import main
    from test_pipeline import _ventas_sinteticas, json_seguro

    datos = tmp_path / "ventas.csv"
    _ventas_sinteticas(datos)
    # Dos series que dejan de vender en validación: activas al origen, con
    # TODO el bloque de prueba en cero. Son el caso que la variante agrega y
    # el pipeline tiene que evaluarlas sin romperse (métricas con real cero).
    ventas = pd.read_csv(datos)
    costos = pd.read_csv(tmp_path / "costos.csv")
    nuevas = []
    for sku, ultima in (("SKU-8000", "2025-01"), ("SKU-8001", "2024-12")):
        for t, periodo in enumerate(pd.period_range("2017-04", ultima, freq="M")):
            nuevas.append({
                "fecha": periodo.to_timestamp().date().isoformat(), "sku": sku,
                "canal": "CANAL-1", "regional": "REGIONAL-A",
                "cantidad": 80.0 + 30.0 * (periodo.month in (1, 2, 12)),
                "categoria": "CUADERNOS", "precio": 3.0, "perdidas": 0.0,
            })
        costos.loc[len(costos)] = [sku, "CANAL-1", "REGIONAL-A", 2.0]
    pd.concat([ventas, pd.DataFrame(nuevas)]).to_csv(datos, index=False)
    costos.to_csv(tmp_path / "costos.csv", index=False)

    crudo = json_seguro(cargar_config("config/config_sensibilidad_b.yaml").crudo)
    crudo.pop("_hereda_de", None)
    crudo["datos"]["archivo"] = str(datos)
    crudo["salidas"]["directorio"] = str(tmp_path / "salidas")
    crudo["features"]["archivo_quiebres"] = str(tmp_path / "quiebres.csv")
    crudo["features"]["archivo_tipo_cambio"] = str(tmp_path / "tipo_cambio.csv")
    crudo["modelos"]["lightgbm"]["num_boost_round"] = 40
    crudo["modelos"]["lightgbm"]["early_stopping_rounds"] = 10
    crudo["modelos"]["lightgbm"]["min_data_in_leaf"] = 5
    crudo["modelos"]["holt_winters"]["alfa_rejilla"] = [0.2, 0.5]
    crudo["modelos"]["holt_winters"]["beta_rejilla"] = [0.1]
    crudo["modelos"]["holt_winters"]["gamma_rejilla"] = [0.1]
    # Multihorizonte corto y sin re-entrenar por origen: acá no se mide el
    # horizonte, sólo que los interruptores atraviesen el pipeline.
    crudo["multihorizonte"]["horizonte_maximo"] = 3
    crudo["multihorizonte"]["reentrenar_lightgbm"] = False
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(crudo, allow_unicode=True), encoding="utf-8")

    cfg_b = cargar_config(ruta)
    assert main.comando_ejecutar(cfg_b) == 0

    salidas = cfg_b.ruta_salidas
    flujo_csv = pd.read_csv(salidas / "cohorte_flujo.csv")
    assert flujo_csv["paso"].iloc[6].startswith("6. presente en prueba: no se aplica")
    assert flujo_csv["paso"].iloc[7].startswith("7. activa al origen")

    flujo = json.loads((salidas / "flujo.json").read_text(encoding="utf-8"))
    etapas = {e["id"]: e for e in flujo["etapas"]}
    assert etapas["particion_cohorte"]["decisiones"]["interruptores_cohorte"] == {
        "inclusion.exigir_presencia_prueba": False,
        "series.extender_vida_hasta_fin_periodo": True,
        "inclusion.estado_al_origen": "activa",
    }
    assert etapas["series"]["decisiones"]["extender_vida_hasta_fin_periodo"] is True

    manifiesto = json.loads((salidas / "manifiesto.json").read_text(encoding="utf-8"))
    cohorte = manifiesto["resultados"]["cohorte"]
    assert cohorte["interruptores"]["inclusion.estado_al_origen"] == "activa"
    fuera = pd.read_csv(salidas / "fuera_de_muestra.csv")
    assert "estado_al_origen" in fuera.columns
    assert set(fuera.loc[fuera["incluida"], "estado_al_origen"]) == {"activa"}
    # Las dos series sin ventas en prueba quedaron en la cohorte y evaluadas.
    errores = pd.read_csv(salidas / "errores_por_serie.csv")
    for sku in ("SKU-8000", "SKU-8001"):
        assert sku in set(fuera.loc[fuera["incluida"], "serie"].str.split("|").str[0])
        assert errores["serie"].str.startswith(sku + "|").any()
