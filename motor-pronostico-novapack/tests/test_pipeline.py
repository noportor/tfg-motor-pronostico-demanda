"""Prueba de extremo a extremo del pipeline completo.

Las pruebas de los módulos verifican cada pieza por separado; esta verifica que
encajan. Se añadió después de que una corrida real muriera en el paso 10 —al
escribir una tabla— con todos los cálculos ya hechos y ninguna prueba unitaria
en rojo: los módulos estaban bien y el pegamento no.

Corre sobre datos sintéticos generados aquí, en un directorio temporal y con una
configuración propia. Ningún número de esta prueba llega al documento (RN-1).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import main
from src.config import cargar_config

SALIDAS_ESPERADAS = (
    "inspeccion_datos.txt",
    "inspeccion.json",
    "flujo.json",
    "cohorte_flujo.csv",
    "particion.csv",
    "fuera_de_muestra.csv",
    "sensibilidad_umbrales.csv",
    "errores_por_serie.csv",
    "resumen_metricas.csv",
    "tabla8_resultados.csv",
    "tabla_valorizada.csv",
    "pruebas_estadisticas.txt",
    "seleccion_motor.csv",
    "unidad_estructura_sku.csv",
    "unidad_contrafactual.csv",
    "multihorizonte_horizonte.csv",
    "multihorizonte_origen.csv",
    "victorias_por_modelo.csv",
    "rangos_friedman.csv",
    "parametros_por_serie.csv",
    "lightgbm_importancias.csv",
    "figura2_error.png",
    "figura3_dispersion.png",
    "figura4_diferencia_critica.png",
    "figura5_horizonte.png",
    "manifiesto.json",
)


def _ventas_sinteticas(ruta, n_series: int = 40, meses: int = 108,
                       inicio: str = "2017-04") -> None:
    """Panel sintético con nivel, tendencia, estacionalidad e intermitencia.

    Se mezclan series regulares y esporádicas a propósito: si todas fueran
    suaves, Croston nunca se ejercitaría y la prueba no cubriría esa rama.
    """
    generador = np.random.default_rng(42)
    fechas = pd.period_range(inicio, periods=meses, freq="M").to_timestamp()
    filas = []

    # Las categorías REALES de la población objetivo (config.yaml): la corrida
    # sintética pasa por el mismo filtro de población que la real. Se agrega
    # además una categoría FUERA de la población para ejercitar la exclusión.
    poblacion = ("PLASTICOS", "PAPELES", "PAPEL FOTOCOPIA",
                 "ESCRITURA Y TRAZADO", "CUADERNOS")

    for i in range(n_series):
        intermitente = i % 3 == 0
        # Un tercio de los SKUs vive en DOS combinaciones canal-regional: sin
        # SKUs multi-combinación la etapa de unidad de análisis no tendría
        # nada que medir y el contrafactual arriba/abajo quedaría vacío.
        numero_sku = i - i % 3 if i % 3 < 2 else i
        sku = f"SKU-{numero_sku:04d}"
        categoria = poblacion[numero_sku % len(poblacion)]
        nivel = float(generador.integers(40, 400))
        estacional = 0.35 * nivel * np.sin(2 * np.pi * np.arange(meses) / 12)
        tendencia = generador.normal(0, 0.4) * np.arange(meses)
        ruido = generador.normal(0, 0.10 * nivel, meses)
        valores = np.clip(nivel + estacional + tendencia + ruido, 0, None)
        if intermitente:
            valores = valores * (generador.random(meses) < 0.45)

        canal = ("CANAL-1", "CANAL-2")[i % 2]
        regional = ("REGIONAL-A", "REGIONAL-B", "REGIONAL-C")[i % 3]
        precio_base = round(1.0 + (numero_sku % 7) * 0.8, 2)
        for t, (fecha, valor) in enumerate(zip(fechas, valores)):
            if valor <= 0:
                continue  # solo se registran los meses CON venta
            filas.append({
                "fecha": fecha.date().isoformat(),
                "sku": sku,
                "canal": canal,
                "regional": regional,
                "cantidad": round(float(valor), 2),
                "categoria": categoria,
                "precio": round(precio_base * (1 + 0.002 * t), 4),
                "perdidas": round(float(valor) * 0.1, 2) if t % 9 == 0 else 0.0,
            })

    # Dos series sanas FUERA de la población objetivo: si el filtro no las
    # excluye (y las cuenta), la e2e tiene que enterarse.
    for j, (canal, regional) in enumerate(
        (("CANAL-1", "REGIONAL-A"), ("CANAL-2", "REGIONAL-B"))
    ):
        for fecha in fechas:
            filas.append({
                "fecha": fecha.date().isoformat(),
                "sku": f"SKU-99{j}0",
                "canal": canal,
                "regional": regional,
                "cantidad": 120.0,
                "categoria": "LIMPIEZA",
                "precio": 2.5,
                "perdidas": 0.0,
            })

    pd.DataFrame(filas).to_csv(ruta, index=False)

    # Maestro de costos sintético junto al archivo de ventas: habilita la suite
    # valorizada en la corrida de prueba (RN-1: sintético, solo en tests).
    combinaciones = (
        pd.DataFrame(filas)[["sku", "canal", "regional"]].drop_duplicates()
    )
    combinaciones["costo_unitario"] = [
        round(0.5 + 3.7 * ((i * 7919) % 100) / 100, 2)
        for i in range(len(combinaciones))
    ]
    combinaciones.to_csv(ruta.parent / "costos.csv", index=False)

    # Fuentes exógenas sintéticas (features v2): quiebres SOLO en el tramo
    # final (como la fuente real, que nace a mitad del período) y un tipo de
    # cambio que arranca fijo y se vuelve variable.
    quiebres = []
    combos_sr = (
        pd.DataFrame(filas)[["sku", "regional"]].drop_duplicates().reset_index(drop=True)
    )
    for k, fila_sr in combos_sr.iterrows():
        for fecha in fechas[60:]:
            quiebres.append({
                "fecha": fecha.date().isoformat(),
                "sku": fila_sr["sku"],
                "regional": fila_sr["regional"],
                "semanas_medidas": 4,
                "semanas_en_cero": (k + fecha.month) % 5 == 0 and 2 or 0,
            })
    pd.DataFrame(quiebres).to_csv(ruta.parent / "quiebres.csv", index=False)

    tc = [{"fecha": fecha.date().isoformat(),
           "tc": round(6.96 if t < 72 else 6.96 + 0.35 * (t - 72), 2)}
          for t, fecha in enumerate(fechas[24:], start=24)]
    pd.DataFrame(tc).to_csv(ruta.parent / "tipo_cambio.csv", index=False)


@pytest.fixture(scope="module")
def corrida(tmp_path_factory, request):
    """Ejecuta el pipeline completo una vez y devuelve la configuración usada."""
    base = tmp_path_factory.mktemp("pipeline")
    datos = base / "ventas.csv"
    _ventas_sinteticas(datos)

    real = cargar_config()
    crudo = json_seguro(real.crudo)
    crudo["datos"]["archivo"] = str(datos)
    crudo["salidas"]["directorio"] = str(base / "salidas")
    crudo["features"]["archivo_quiebres"] = str(base / "quiebres.csv")
    crudo["features"]["archivo_tipo_cambio"] = str(base / "tipo_cambio.csv")
    # LightGBM con muy pocos árboles: aquí interesa que el pegamento funcione,
    # no la calidad del ajuste.
    crudo["modelos"]["lightgbm"]["num_boost_round"] = 40
    crudo["modelos"]["lightgbm"]["early_stopping_rounds"] = 10
    crudo["modelos"]["lightgbm"]["min_data_in_leaf"] = 5
    crudo["modelos"]["holt_winters"]["alfa_rejilla"] = [0.2, 0.5]
    crudo["modelos"]["holt_winters"]["beta_rejilla"] = [0.1]
    crudo["modelos"]["holt_winters"]["gamma_rejilla"] = [0.1]

    ruta_config = base / "config.yaml"
    ruta_config.write_text(yaml.safe_dump(crudo, allow_unicode=True), encoding="utf-8")

    cfg = cargar_config(ruta_config)
    codigo = main.comando_ejecutar(cfg)
    assert codigo == 0
    return cfg


def json_seguro(objeto):
    """Copia profunda del YAML crudo, con los tipos ya normalizados."""
    return json.loads(json.dumps(objeto, default=str))


def test_el_pipeline_produce_todas_las_salidas(corrida):
    faltan = [
        nombre for nombre in SALIDAS_ESPERADAS
        if not (corrida.ruta_salidas / nombre).exists()
    ]
    assert not faltan, f"El pipeline no escribió: {faltan}"


def test_todas_las_salidas_tienen_contenido(corrida):
    vacias = [
        nombre for nombre in SALIDAS_ESPERADAS
        if (corrida.ruta_salidas / nombre).stat().st_size == 0
    ]
    assert not vacias, f"Salidas vacías: {vacias}"


def test_la_tabla8_tiene_una_fila_por_modelo(corrida):
    tabla = pd.read_csv(corrida.ruta_salidas / "tabla8_resultados.csv")
    assert set(tabla["Modelo"]) == set(corrida.modelos_activos)
    assert tabla["Series"].nunique() == 1, (
        "Todos los modelos deben resumirse sobre el MISMO conjunto de series"
    )


def test_el_manifiesto_registra_la_trazabilidad(corrida):
    """RN-6: cada número debe poder rastrearse hasta su corrida."""
    manifiesto = json.loads(
        (corrida.ruta_salidas / "manifiesto.json").read_text(encoding="utf-8")
    )

    assert manifiesto["datos"]["sha256"], "Falta el hash del archivo de datos"
    assert manifiesto["configuracion"]["sha256"], "Falta el hash de la configuración"
    assert manifiesto["dependencias"]["pandas"], "Faltan las versiones cargadas"
    assert "codigo" in manifiesto
    # El commit tiene que quedar registrado por alguna vía; si no se pudo, el
    # manifiesto debe decir POR QUÉ en lugar de callarlo.
    codigo = manifiesto["codigo"]
    assert codigo.get("commit") or codigo.get("advertencia"), (
        "Sin commit y sin explicación no hay trazabilidad (RN-6)"
    )
    assert manifiesto["salidas"], "El manifiesto no lista ninguna salida"
    for salida in manifiesto["salidas"]:
        assert len(salida["sha256"]) == 64


def test_el_motor_eligio_entre_los_candidatos_declarados(corrida):
    seleccion = pd.read_csv(corrida.ruta_salidas / "seleccion_motor.csv")
    elegidos = set(seleccion["modelo_elegido"])
    assert elegidos <= set(corrida.modelos_candidatos)
    assert "motor" not in elegidos, "El motor no puede elegirse a sí mismo"


def test_los_errores_por_serie_cubren_todos_los_modelos(corrida):
    errores = pd.read_csv(corrida.ruta_salidas / "errores_por_serie.csv")
    assert set(errores["modelo"]) == set(corrida.modelos_activos)
    assert (errores["mae"] >= 0).all()
    assert (errores["rmse"] >= errores["mae"] - 1e-9).all(), (
        "El RMSE no puede ser menor que el MAE"
    )


def test_ningun_modelo_predice_demanda_negativa(corrida):
    errores = pd.read_csv(corrida.ruta_salidas / "errores_por_serie.csv")
    assert (errores["media_predicha"] >= -1e-9).all()


# ---------------------------------------------------------------------------
# El contrato con el tablero: flujo.json e inspeccion.json
# ---------------------------------------------------------------------------

ETAPAS_MINIMAS = (
    "carga", "poblacion", "series", "inspeccion", "evento_exogeno",
    "particion_cohorte", "features", "modelos", "motor", "unidad_analisis",
    "metricas", "contraste", "evaluacion_multihorizonte", "figuras",
)


def test_flujo_json_cumple_el_contrato(corrida):
    """El tablero renderiza las etapas genéricamente: si el schema se rompe,
    la interfaz se rompe con él. Este es el contrato."""
    flujo = json.loads(
        (corrida.ruta_salidas / "flujo.json").read_text(encoding="utf-8")
    )

    assert flujo["version"] == 1
    assert flujo["corrida"]["config_sha256"]
    assert flujo["corrida"]["datos_sha256"]

    ids = [etapa["id"] for etapa in flujo["etapas"]]
    assert len(ids) == len(set(ids)), "Los ids de etapa deben ser únicos"
    for esperada in ETAPAS_MINIMAS:
        assert esperada in ids, f"Falta la etapa '{esperada}' en flujo.json"

    for etapa in flujo["etapas"]:
        assert etapa["titulo"], f"La etapa {etapa['id']} no tiene título"
        for clave in ("entrada", "salida", "decisiones", "conteos"):
            assert isinstance(etapa[clave], dict), (
                f"{etapa['id']}.{clave} debe ser un dict"
            )
        assert isinstance(etapa["artefactos"], list)
        # Todo artefacto declarado tiene que existir: una etapa que promete un
        # archivo que no está rompería los enlaces del tablero.
        for artefacto in etapa["artefactos"]:
            assert (corrida.ruta_salidas / artefacto).exists(), (
                f"La etapa {etapa['id']} declara '{artefacto}' y no existe"
            )


def test_la_poblacion_objetivo_excluye_y_cuenta(corrida):
    """Las series LIMPIEZA de la fixture están fuera de la población: la etapa
    debe excluirlas Y contarlas — nunca un filtro silencioso."""
    flujo = json.loads(
        (corrida.ruta_salidas / "flujo.json").read_text(encoding="utf-8")
    )
    etapa = next(e for e in flujo["etapas"] if e["id"] == "poblacion")
    assert etapa["conteos"]["series_fuera"] == 2
    assert "LIMPIEZA" in etapa["conteos"]["filas_por_categoria_excluida"]
    assert etapa["salida"]["filas_en_poblacion"] < etapa["entrada"]["filas_validas"]


def test_fuera_de_muestra_clasifica_todas_las_series(corrida):
    """Contracara declarativa de la cascada: toda serie con estado y motivo."""
    tabla = pd.read_csv(corrida.ruta_salidas / "fuera_de_muestra.csv")
    seleccion = pd.read_csv(corrida.ruta_salidas / "seleccion_motor.csv")

    assert tabla["serie"].is_unique
    assert set(seleccion["serie"]) <= set(tabla["serie"]), (
        "Toda serie de la cohorte debe estar clasificada"
    )
    incluidas = tabla.loc[tabla["incluida"]]
    assert (incluidas["motivo_exclusion"] == "incluida").all()
    excluidas = tabla.loc[~tabla["incluida"]]
    assert (excluidas["motivo_exclusion"] != "incluida").all(), (
        "Ninguna serie puede salir del estudio sin motivo atribuido"
    )
    assert set(tabla["estado"]) <= {
        "activa", "dormida", "descontinuada", "nueva", "reciente", "insuficiente"
    }


def test_sensibilidad_reproduce_el_n_de_la_cascada(corrida):
    """La fila de los umbrales BASE de la tabla de sensibilidad tiene que dar
    exactamente el N de la cascada: si divergen, la sensibilidad está midiendo
    otra cosa que los criterios reales."""
    sensibilidad = pd.read_csv(corrida.ruta_salidas / "sensibilidad_umbrales.csv")
    flujo = pd.read_csv(corrida.ruta_salidas / "cohorte_flujo.csv")

    base = sensibilidad.loc[
        (sensibilidad["historial_minimo_meses"]
         == corrida.inclusion.historial_minimo_meses)
        & (sensibilidad["proporcion_maxima_ceros"]
           == corrida.inclusion.proporcion_maxima_ceros)
        & (sensibilidad["volumen_minimo_unidades"]
           == corrida.inclusion.volumen_minimo_unidades)
    ]
    assert len(base) == 1, "La grilla debe contener la combinación base"
    n_cascada = int(flujo["series"].iloc[-1])
    assert int(base["n_series"].iloc[0]) == n_cascada


def test_los_json_emitidos_son_json_estricto(corrida):
    """El visor los parsea con JSON.parse, que rechaza los tokens NaN e
    Infinity. json.loads de Python los ACEPTA, así que sin este parser
    estricto un NaN en una nota rompería todas las vistas sin que la suite
    se entere (hallazgo de la auditoría adversarial)."""
    def parser_estricto(token):
        raise AssertionError(
            f"Token '{token}' en un JSON emitido: JSON.parse del navegador "
            "lo rechaza. Un valor incalculable debe publicarse como null."
        )

    for nombre in ("manifiesto.json", "flujo.json", "inspeccion.json"):
        texto = (corrida.ruta_salidas / nombre).read_text(encoding="utf-8")
        json.loads(texto, parse_constant=parser_estricto)


def test_la_cohorte_sintetica_retiene_skus_multicombo(corrida):
    """SALIDAS_ESPERADAS exige unidad_contrafactual.csv, que main solo escribe
    si la cohorte retiene SKUs multi-combinación. Este assert hace EXPLÍCITO
    ese acoplamiento con la fixture: si un cambio de umbrales lo rompe, el
    mensaje señala la causa real y no un archivo faltante."""
    manifiesto = json.loads(
        (corrida.ruta_salidas / "manifiesto.json").read_text(encoding="utf-8")
    )
    nota = manifiesto["resultados"]["unidad_analisis"]
    assert nota["skus_cohorte_multi"] >= 1, (
        "La fixture sintética dejó de producir SKUs multi-combinación que "
        "sobrevivan los criterios de inclusión: el contrafactual queda vacío "
        "y unidad_contrafactual.csv no se escribe. Ajustá la fixture (o los "
        "umbrales del config de prueba), no el contrato."
    )
    assert nota["contrafactual"], "El contrafactual no produjo filas"


def test_flujo_json_esta_listado_en_el_manifiesto(corrida):
    """RN-6: flujo.json es una salida más, con su hash en el manifiesto."""
    manifiesto = json.loads(
        (corrida.ruta_salidas / "manifiesto.json").read_text(encoding="utf-8")
    )
    listados = {s["archivo"] for s in manifiesto["salidas"]}
    assert "flujo.json" in listados
    assert "inspeccion.json" in listados


def test_inspeccion_json_refleja_el_informe(corrida):
    datos = json.loads(
        (corrida.ruta_salidas / "inspeccion.json").read_text(encoding="utf-8")
    )
    for clave in ("rango_fechas", "n_sku", "n_combinaciones",
                  "estacionalidad_mensual", "volumen_por_gestion"):
        assert clave in datos, f"inspeccion.json no tiene '{clave}'"


def test_humo_del_ajuste_de_hiperparametros(corrida, tmp_path):
    """El script de tuning corre de punta a punta sobre la corrida sintética
    (2 ensayos) y emite sus artefactos parseables. Es la única cobertura del
    contrato Optuna/espacio/artefactos que no depende de la corrida real."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import ajustar_lightgbm

    codigo = ajustar_lightgbm.main([
        "--config", str(corrida.ruta_config),
        "--n-trials", "2",
        "--destino", str(tmp_path / "tuning"),
    ])
    assert codigo == 0

    ensayos = pd.read_csv(tmp_path / "tuning" / "lightgbm_tuning_ensayos.csv")
    assert len(ensayos) == 2
    assert {"objective", "D_interna", "mejor_iteracion"} <= set(ensayos.columns)

    mejor = json.loads(
        (tmp_path / "tuning" / "lightgbm_tuning_mejor.json").read_text(
            encoding="utf-8")
    )
    assert mejor["n_trials"] == 2
    assert mejor["mejores_parametros"]
    assert mejor["sha256_datos"], "El hash del snapshot debe quedar certificado"


def test_registrar_dos_veces_la_misma_etapa_falla(cfg):
    """Un id duplicado produciría dos nodos iguales en el diagrama del flujo."""
    from src.reporte import Reporte

    reporte = Reporte(cfg)
    reporte.etapa("x", "Una etapa")
    with pytest.raises(ValueError):
        reporte.etapa("x", "La misma otra vez")
