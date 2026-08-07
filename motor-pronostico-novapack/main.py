"""Ejecución del experimento.

    python main.py inspeccionar     # fases 1–2: carga e informe de inspección
    python main.py ejecutar         # pipeline completo

El plan de trabajo indica detenerse a leer el informe de inspección antes de
seguir: ahí se calibran los criterios de inclusión, y el N resultante es el
tamaño de la muestra que se declara en la tesis. Por eso ``inspeccionar`` es un
comando aparte y no un paso silencioso del pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import (
    carga, eventos, features, figuras, inspeccion, metricas, multihorizonte,
    particion, pruebas, series, unidad_analisis,
)
from src.config import cargar_config
from src.costos import cargar_costos
from src.modelos import construir_modelos
from src.modelos.base import (
    aplicar_respaldo, enmascarar_fuera_de_vida, mae_naive_entrenamiento, truncar_en_cero,
)
from src.modelos.motor import Motor
from src.reporte import Reporte, informe_pruebas


def _paso(numero: str, texto: str) -> float:
    print(f"\n[{numero}] {texto}", flush=True)
    return time.perf_counter()


def _fin(inicio: float) -> None:
    print(f"      ({time.perf_counter() - inicio:.1f} s)", flush=True)


# ---------------------------------------------------------------------------

def comando_inspeccionar(cfg) -> int:
    inicio = _paso("1/3", "Cargando el archivo crudo")
    df, informe_carga = carga.cargar(cfg)
    for linea in informe_carga.lineas():
        print("      " + linea)
    df, informe_poblacion = carga.aplicar_poblacion(df, cfg)
    for linea in informe_poblacion.lineas():
        print("      " + linea)
    _fin(inicio)

    inicio = _paso("2/3", "Construyendo las series mensuales")
    panel_largo, informe_series = series.construir_series(df, cfg)
    for linea in informe_series.lineas():
        print("      " + linea)
    _fin(inicio)

    inicio = _paso("3/3", "Inspeccionando")
    informe = inspeccion.inspeccionar(df, cfg, panel_largo)
    _fin(inicio)

    reporte = Reporte(cfg)
    encabezado = [
        "INFORME DE CARGA",
        "-" * 78,
        *informe_carga.lineas(),
        "",
        "CONSTRUCCIÓN DE LAS SERIES",
        "-" * 78,
        *informe_series.lineas(),
    ]
    ruta = reporte.texto("inspeccion_datos.txt", encabezado + informe.lineas)
    reporte.escribir_json("inspeccion.json", informe.datos)
    print(f"\nInforme escrito en {ruta}")
    print("\nLeé el informe y ajustá `inclusion` en config/config.yaml antes de "
          "correr `main.py ejecutar`.")
    return 0


# ---------------------------------------------------------------------------

def comando_ejecutar(cfg) -> int:
    reporte = Reporte(cfg)
    total = "12"

    # --- 1. Carga -----------------------------------------------------------
    inicio = _paso(f"1/{total}", "Cargando el archivo crudo")
    df, informe_carga = carga.cargar(cfg)
    print(f"      {informe_carga.filas_validas:,} filas válidas de "
          f"{informe_carga.filas_leidas:,}")
    reporte.etapa(
        "carga", "Carga del archivo crudo", rf="RF-1",
        entrada={"filas_leidas": informe_carga.filas_leidas},
        salida={"filas_validas": informe_carga.filas_validas},
        decisiones={
            "mapeo_columnas": informe_carga.mapeo_aplicado,
            "separador": cfg.datos.separador,
            "codificacion": cfg.datos.codificacion,
        },
        conteos={
            "fechas_no_convertidas": informe_carga.fechas_no_convertidas,
            "cantidades_no_convertidas": informe_carga.cantidades_no_convertidas,
            "filas_con_clave_nula": informe_carga.filas_con_clave_nula,
            "filas_descartadas": informe_carga.filas_descartadas,
        },
        notas=[f"SHA-256 del archivo crudo: {informe_carga.sha256}"],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 1b. Población objetivo ---------------------------------------------
    inicio = _paso(f"1b/{total}", "Población objetivo")
    df, informe_poblacion = carga.aplicar_poblacion(df, cfg)
    for linea in informe_poblacion.lineas():
        print("      " + linea)
    reporte.etapa(
        "poblacion", "Población objetivo del estudio",
        entrada={"filas_validas": informe_poblacion.filas_antes},
        salida={"filas_en_poblacion": informe_poblacion.filas_en_poblacion},
        decisiones={
            "categorias": (informe_poblacion.categorias_incluidas
                           or "portafolio completo (ablación)"),
        },
        conteos={
            "filas_fuera": informe_poblacion.filas_fuera,
            "series_fuera": informe_poblacion.series_fuera,
            "skus_fuera": informe_poblacion.skus_fuera,
            "filas_por_categoria_excluida":
                informe_poblacion.filas_por_categoria_excluida,
        },
        notas=[
            "Decisión de DISEÑO, no criterio de inclusión: el estudio se acota "
            "a un grupo representativo de categorías con comportamiento "
            "comercial conocido (dos regímenes verificados: recurrente y "
            "estacional de campaña). Analizar el portafolio completo agrega "
            "ruido de series sin patrón interpretable. La ablación "
            "poblacion.categorias=[] vuelve al portafolio completo.",
        ],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 2. Series ----------------------------------------------------------
    inicio = _paso(f"2/{total}", "Construyendo las series mensuales")
    panel_largo, informe_series = series.construir_series(df, cfg)
    print(f"      {informe_series.series_construidas:,} series · "
          f"{informe_series.filas_panel:,} filas · {informe_series.rango[0]}"
          f" .. {informe_series.rango[1]}")
    reporte.etapa(
        "series", "Construcción de las series mensuales", rf="RF-3",
        entrada={"filas_validas": informe_carga.filas_validas},
        salida={
            "series": informe_series.series_construidas,
            "filas_panel": informe_series.filas_panel,
        },
        decisiones={
            "devoluciones": informe_series.tratamiento_devoluciones,
            "rellenar_ceros_intermedios": cfg.series.rellenar_ceros_intermedios,
        },
        conteos={
            "filas_negativas": informe_series.filas_negativas,
            "unidades_negativas": informe_series.unidades_negativas,
            "filas_fuera_de_periodo": informe_series.filas_fuera_de_periodo,
            "combinaciones_totales": informe_series.combinaciones_totales,
            "sin_venta_positiva": informe_series.combinaciones_sin_venta_positiva,
            "meses_rellenados_con_cero": informe_series.meses_rellenados_con_cero,
        },
        notas=[f"Rango del panel: {informe_series.rango[0]} .. {informe_series.rango[1]}"],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 3. Inspección ------------------------------------------------------
    inicio = _paso(f"3/{total}", "Informe de inspección")
    informe_inspeccion = inspeccion.inspeccionar(df, cfg, panel_largo)
    reporte.texto(
        "inspeccion_datos.txt",
        ["INFORME DE CARGA", "-" * 78, *informe_carga.lineas(), "",
         "CONSTRUCCIÓN DE LAS SERIES", "-" * 78, *informe_series.lineas(),
         *informe_inspeccion.lineas],
    )
    reporte.escribir_json("inspeccion.json", informe_inspeccion.datos)
    reporte.etapa(
        "inspeccion", "Inspección de los datos (CRISP-DM)", rf="RF-2",
        conteos={
            "skus": informe_inspeccion.datos.get("n_sku"),
            "canales": informe_inspeccion.datos.get("n_canales"),
            "regionales": informe_inspeccion.datos.get("n_regionales"),
            "combinaciones": informe_inspeccion.datos.get("n_combinaciones"),
            "duplicados": informe_inspeccion.datos.get("registros_duplicados"),
            "series_intermitentes_adi_1_32":
                informe_inspeccion.datos.get("series_intermitentes_adi_1_32"),
        },
        artefactos=["inspeccion_datos.txt", "inspeccion.json"],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 3b. Evento exógeno (COVID-19) --------------------------------------
    inicio = _paso(f"3b/{total}", "Tratamiento del evento exógeno")
    panel_ajuste_largo, informe_evento = eventos.aplicar_tratamiento(panel_largo, cfg)
    for linea in informe_evento.lineas():
        print("      " + linea)
    reporte.etapa(
        "evento_exogeno", "Tratamiento del evento exógeno (ajuste)",
        entrada={"observaciones_en_ventana": informe_evento.observaciones_en_ventana},
        salida={"observaciones_reemplazadas": informe_evento.observaciones_reemplazadas},
        decisiones={
            "evento": informe_evento.nombre or "ninguno",
            "ventana": f"{informe_evento.desde}..{informe_evento.hasta}",
            "tratamiento": informe_evento.tratamiento,
        },
        conteos={
            "series_afectadas": informe_evento.series_afectadas,
            "sin_historia_previa": informe_evento.sin_historia_previa,
        },
        notas=[
            "El tratamiento reproduce la práctica documentada de la empresa "
            "durante el evento (copiar la historia de la gestión pasada). La "
            "serie corregida existe SOLO para el ajuste: la evaluación compara "
            "siempre contra la serie observada, y la ventana cae íntegra en "
            "entrenamiento (lo valida la configuración).",
        ],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 4. Partición y cohorte --------------------------------------------
    inicio = _paso(f"4/{total}", "Partición temporal y criterios de inclusión")
    corte = particion.particionar(cfg)
    resumen_particion = corte.resumen(cfg.periodo.mes_inicio_gestion)
    for linea in resumen_particion.to_string(index=False).split("\n"):
        print("      " + linea)

    # Los criterios se miden sobre el panel de AJUSTE: con el tratamiento de la
    # empresa, una serie apagada por el confinamiento no pierde su lugar en la
    # muestra por ceros exógenos.
    cohorte_ajustada, informe_cohorte = particion.aplicar_criterios_inclusion(
        panel_ajuste_largo, cfg, corte
    )
    for linea in informe_cohorte.lineas():
        print("      " + linea)
    reporte.tabla("cohorte_flujo.csv", informe_cohorte.tabla_flujo())
    reporte.tabla("particion.csv", resumen_particion)

    # --- Clasificar y declarar, no descartar en silencio ---------------------
    # El maestro de costos se carga acá (lo reusan 8b y 9): pesa lo excluido.
    maestro = cargar_costos(cfg)
    valor_por_serie = None
    if maestro.disponible:
        train_real = panel_largo.loc[
            panel_largo["periodo"] <= cfg.particion.fin_entrenamiento
        ]
        costo_train = train_real["serie"].map(maestro.costo_por_serie)
        valor_por_serie = (
            (train_real["y"] * costo_train).groupby(train_real["serie"]).sum()
            .dropna()
        )

    fuera = particion.tabla_fuera_de_muestra(
        panel_largo, panel_ajuste_largo,
        set(cohorte_ajustada["serie"].unique()), cfg, corte, valor_por_serie,
    )
    reporte.tabla("fuera_de_muestra.csv", fuera)
    reparto_estados = (
        fuera.groupby(["estado", "incluida"], observed=True).size()
        .rename("series").reset_index()
        .to_dict("records")
    )

    sensibilidad = particion.sensibilidad_umbrales(
        panel_ajuste_largo, corte, valor_por_serie
    )
    reporte.tabla("sensibilidad_umbrales.csv", sensibilidad)

    reporte.etapa(
        "particion_cohorte", "Partición temporal y criterios de inclusión", rf="RF-4",
        entrada={"series": informe_cohorte.series_iniciales},
        salida={"series_muestra_N": informe_cohorte.series_finales},
        decisiones={
            "fin_entrenamiento": str(cfg.particion.fin_entrenamiento),
            "fin_validacion": str(cfg.particion.fin_validacion),
            "fin_prueba": str(cfg.periodo.fin),
            "umbrales_inclusion": informe_cohorte.umbrales,
        },
        conteos={
            "cascada": informe_cohorte.flujo,
            "excluidas_aislado": informe_cohorte.excluidas_aislado,
            "estados_por_inclusion": reparto_estados,
        },
        artefactos=["cohorte_flujo.csv", "particion.csv", "fuera_de_muestra.csv",
                    "sensibilidad_umbrales.csv"],
        notas=[
            "Los criterios se miden SOLO sobre el bloque de entrenamiento.",
            "fuera_de_muestra.csv clasifica TODAS las series (estado de "
            "actividad al cierre de entrenamiento, molde del sistema en "
            "producción) con el criterio que las excluyó y su peso valorizado: "
            "ninguna serie sale del estudio en silencio.",
            "sensibilidad_umbrales.csv: el N y la cobertura valorizada bajo "
            "umbrales alternativos — la muestra declarada no es un punto "
            "arbitrario.",
        ],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 5. Paneles ---------------------------------------------------------
    inicio = _paso(f"5/{total}", "Panel ancho y bloques")
    # Dos mundos con las MISMAS series: el de AJUSTE (serie corregida por el
    # evento) alimenta los modelos; el REAL es contra el que se evalúa.
    series_cohorte = set(cohorte_ajustada["serie"].unique())
    cohorte_real = panel_largo.loc[
        panel_largo["serie"].isin(series_cohorte)
    ].reset_index(drop=True)

    panel = series.a_panel_ancho(cohorte_ajustada)          # ajuste
    panel_real = series.a_panel_ancho(cohorte_real)         # evaluación
    if not panel.index.equals(panel_real.index) or list(panel.columns) != list(panel_real.columns):
        raise RuntimeError(
            "El panel de ajuste y el real divergen en forma: el tratamiento del "
            "evento no puede alterar la vida de las series."
        )

    panel_entrenamiento = panel.loc[panel.index <= corte.entrenamiento[-1]]
    panel_validacion = panel.loc[
        (panel.index >= corte.validacion[0]) & (panel.index <= corte.validacion[-1])
    ]
    panel_prueba = panel_real.loc[panel_real.index >= corte.prueba[0]]
    # La ventana del evento cae en entrenamiento: en prueba ambos mundos deben
    # ser idénticos. Verificación activa, no supuesta.
    if not panel.loc[panel.index >= corte.prueba[0]].equals(panel_prueba):
        raise RuntimeError(
            "El bloque de prueba difiere entre el panel de ajuste y el real: "
            "la evaluación estaría tocando datos corregidos (RN-1)."
        )
    print(f"      panel {panel.shape[0]} meses × {panel.shape[1]:,} series  "
          f"(train {len(panel_entrenamiento)} · val {len(panel_validacion)} · "
          f"test {len(panel_prueba)})")
    _fin(inicio)

    # --- 6. Features --------------------------------------------------------
    inicio = _paso(f"6/{total}", "Construyendo features (sin fuga temporal)")
    tabla = features.construir_features(cohorte_ajustada, cfg)
    print(f"      {len(tabla):,} filas × {len(features.nombres_de_features(cfg))} features")
    reporte.etapa(
        "features", "Variables derivadas, sin fuga temporal", rf="RF-5",
        entrada={"filas_panel": len(cohorte_ajustada)},
        salida={
            "filas": len(tabla),
            "features": len(features.nombres_de_features(cfg)),
        },
        decisiones={
            "rezagos": cfg.features.rezagos,
            "ventanas_moviles": cfg.features.ventanas_moviles,
            "calendario": {"mes": cfg.features.incluir_mes,
                           "anio": cfg.features.incluir_anio,
                           "tendencia": cfg.features.incluir_tendencia},
            "categoricas": cfg.features.categoricas,
        },
        notas=["Toda feature usa solo información ANTERIOR al mes que predice "
               "(RN-3); la garantía es tests/test_features.py."],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 7. Modelos ---------------------------------------------------------
    inicio = _paso(f"7/{total}", "Ajustando modelos y pronosticando a un paso")
    modelos = construir_modelos(cfg, tabla)
    predicciones: dict[str, pd.DataFrame] = {}
    respaldos: dict[str, int] = {}
    tiempos: dict[str, float] = {}
    parametros: list[pd.DataFrame] = []

    meses_evaluados = corte.validacion.append(corte.prueba)

    for nombre, modelo in modelos.items():
        marca = time.perf_counter()
        modelo.ajustar(panel_entrenamiento, panel_validacion)
        crudo = modelo.predecir(panel)
        crudo = truncar_en_cero(crudo)
        crudo = enmascarar_fuera_de_vida(crudo, panel)
        completo, n_respaldos = aplicar_respaldo(
            crudo, panel, panel_entrenamiento, meses_evaluados=meses_evaluados
        )
        predicciones[nombre] = truncar_en_cero(completo)
        respaldos[nombre] = n_respaldos
        tiempos[nombre] = round(time.perf_counter() - marca, 2)
        print(f"      {nombre:<18} {time.perf_counter() - marca:7.1f} s"
              f"   respaldos en validación+prueba: {n_respaldos:,}")

        if getattr(modelo, "alfa_por_serie", None) is not None:
            parametros.append(
                modelo.alfa_por_serie.rename("valor").to_frame()
                .assign(modelo=nombre, parametro="alfa").reset_index()
            )
        if getattr(modelo, "parametros_por_serie", None) is not None:
            parametros.append(
                modelo.parametros_por_serie.reset_index()
                .melt(id_vars="serie", var_name="parametro", value_name="valor")
                .assign(modelo=nombre)
            )
        if getattr(modelo, "importancias", None) is not None:
            reporte.tabla("lightgbm_importancias.csv", modelo.importancias)
            reporte.nota("lightgbm_mejor_iteracion", modelo.mejor_iteracion)
    if parametros:
        reporte.tabla("parametros_por_serie.csv", pd.concat(parametros, ignore_index=True))

    artefactos_modelos = ["parametros_por_serie.csv"] if parametros else []
    if any(getattr(m, "importancias", None) is not None for m in modelos.values()):
        artefactos_modelos.append("lightgbm_importancias.csv")
    reporte.etapa(
        "modelos", "Ajuste y optimización de los modelos", rf="RF-6",
        entrada={"series": panel.shape[1], "meses": panel.shape[0]},
        salida={"modelos_ajustados": len(modelos)},
        decisiones={
            "modelos_activos": [m for m in cfg.modelos_activos if m != "motor"],
            "exp_smooth": cfg.modelos.get("exp_smooth", {}),
            "holt_winters": cfg.modelos.get("holt_winters", {}),
            "croston": cfg.modelos.get("croston", {}),
            "lightgbm": cfg.modelos.get("lightgbm", {}),
        },
        conteos={
            "duracion_por_modelo_s": tiempos,
            "respaldos_en_validacion_y_prueba": respaldos,
        },
        artefactos=artefactos_modelos,
        notas=["Los parámetros se estiman SOLO con entrenamiento; validación la "
               "usa únicamente la parada temprana de LightGBM (RN-2)."],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 8. Motor -----------------------------------------------------------
    inicio = _paso(f"8/{total}", "Motor de selección (decide mirando SOLO validación)")
    # El motor decide comparando los pronósticos contra la serie REAL de
    # validación: la corrección del evento vive solo en el ajuste.
    seleccion_modo = str(cfg.modelos.get("motor_seleccion", "un_paso"))
    criterio_externo = None
    if seleccion_modo == "multihorizonte":
        # Ablación pre-registrada: la ventana de selección es el error
        # multihorizonte de validación (proyección recursiva desde el cierre
        # de entrenamiento) — la misma mecánica con la que el motor será
        # evaluado, y la del backtest del sistema en producción.
        print("      Ventana de selección: MULTIHORIZONTE sobre validación "
              "(ablación pre-registrada)")
        criterio_externo = multihorizonte.criterio_seleccion_multihorizonte(
            cfg, modelos, panel, panel_real, panel_entrenamiento, cohorte_ajustada,
        )
    motor = Motor(cfg, predicciones, panel_real)
    motor.ajustar(panel_entrenamiento, panel_validacion,
                  criterio_externo=criterio_externo)
    predicciones["motor"] = truncar_en_cero(
        enmascarar_fuera_de_vida(motor.predecir(panel), panel)
    )
    for linea in motor.informe.lineas():
        print("      " + linea)
    reporte.tabla("seleccion_motor.csv", motor.informe.seleccion)
    reporte.etapa(
        "motor", "Motor de selección por serie", rf="RF-6",
        entrada={"candidatos": len(cfg.modelos_candidatos)},
        salida={"series_con_eleccion": int(motor.informe.reparto.sum())},
        decisiones={"regla": motor.informe.regla,
                    "ventana_de_decision": (
                        "validación multihorizonte h=1..12 desde el cierre de "
                        "entrenamiento (RN-2; ablación pre-registrada)"
                        if seleccion_modo == "multihorizonte"
                        else "validación a un paso (RN-2)"
                    ),
                    "seleccion": seleccion_modo},
        conteos={
            "reparto_de_elegidos": motor.informe.reparto,
            "empates": motor.informe.empates,
            "sin_candidato_valido": motor.informe.respaldos,
        },
        artefactos=["seleccion_motor.csv"],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 8b. Unidad de análisis ---------------------------------------------
    inicio = _paso(f"8b/{total}", "Unidad de análisis: la evidencia de la serie "
                                  "SKU-canal-regional")
    # El maestro de costos ya se cargó en el paso 4; acá se reusa: el
    # contrafactual y la suite valorizada comparten la misma valorización.
    eleccion_motor = motor.informe.seleccion.set_index("serie")["modelo_elegido"]
    informe_unidad = unidad_analisis.evaluar(
        cfg, panel_ajuste_largo, cohorte_ajustada, panel_real, eleccion_motor,
        maestro.costo_por_serie if maestro.disponible else None,
    )
    for linea in informe_unidad.lineas():
        print("      " + linea)
    reporte.tabla("unidad_estructura_sku.csv", informe_unidad.tabla_sku)
    artefactos_unidad = ["unidad_estructura_sku.csv"]
    if not informe_unidad.contrafactual.empty:
        reporte.tabla("unidad_contrafactual.csv", informe_unidad.contrafactual)
        artefactos_unidad.append("unidad_contrafactual.csv")
    reporte.nota("unidad_analisis", {
        **informe_unidad.resumen,
        "contrafactual": informe_unidad.contrafactual.to_dict("records"),
    })
    reporte.etapa(
        "unidad_analisis", "Unidad de análisis: evidencia de la serie elegida",
        entrada={"skus": informe_unidad.resumen["skus_totales"]},
        salida={"skus_multicombo": informe_unidad.resumen["skus_multicombo"]},
        decisiones={
            "unidad": "SKU-canal-regional",
            "estructura_medida_sobre": "bloque de entrenamiento, panel de ajuste",
            "ventana_evento_excluida_de": "deriva y correlaciones — con la "
                "copia de gestión previa el interanual daba 0 exacto (pares "
                "fabricados); el régimen ADI-CV² sí usa el entrenamiento "
                "completo (la copia es demanda real del año previo)",
            "contrafactual_evaluado_sobre": "validación, un paso, valorizado, "
                "máscara COMÚN a ambos enfoques (comparación pareada; "
                "exclusiones contadas) — prueba jamás",
            "solape_minimo_correlacion_meses": unidad_analisis.SOLAPE_MINIMO,
            "umbrales_adi_cv2": [unidad_analisis.ADI_CORTE, unidad_analisis.CV2_CORTE],
        },
        conteos={
            k: v for k, v in informe_unidad.resumen.items()
            if isinstance(v, (int, float))
        },
        artefactos=artefactos_unidad,
        notas=[
            "Tres patas: (1) la unidad la impone la decisión — demanda "
            "independiente por combinación y unidades NO agregables entre "
            "SKUs; (2) independencia estadística dentro del SKU (deriva, "
            "correlación, regímenes, ganadores distintos); (3) el mismo "
            "modelo repartiendo desde el SKU agregado no domina (contrafactual "
            "con participaciones rezagadas: sin fuga).",
            "Declarables del contrafactual: las participaciones de meses "
            "tardíos de validación usan reales de meses de validación "
            "ANTERIORES — es el protocolo a un paso (solo información previa "
            "al mes predicho), no fuga. El costo unitario es el del último "
            "corte (posterior a validación); al aplicarse idéntico a ambos "
            "enfoques no puede favorecer a ninguno.",
        ],
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 9. Métricas --------------------------------------------------------
    inicio = _paso(f"9/{total}", "Métricas por serie sobre el bloque de prueba")
    escala = mae_naive_entrenamiento(panel_entrenamiento)
    orden = [m for m in cfg.modelos_activos if m in predicciones]
    errores = {
        nombre: metricas.metricas_por_serie(
            panel_prueba, predicciones[nombre].reindex(index=panel_prueba.index), escala
        )
        for nombre in orden
    }

    largo = pd.concat(
        [tabla_errores.assign(modelo=nombre).reset_index()
         for nombre, tabla_errores in errores.items()],
        ignore_index=True,
    )
    reporte.tabla("errores_por_serie.csv", largo)

    resumen_completo = metricas.resumen(errores, orden=orden)
    reporte.tabla("resumen_metricas.csv", resumen_completo)
    reporte.tabla("tabla8_resultados.csv", metricas.tabla8(resumen_completo))
    print(metricas.tabla8(resumen_completo).round(3).to_string(index=False))

    # --- Suite valorizada: la agregación entre SKUs y la métrica de decisión D.
    # Sin maestro de costos el experimento sigue (las métricas por serie y el
    # contraste pareado no lo necesitan), pero se declara la omisión. El
    # maestro ya se cargó en el paso 4.
    artefactos_metricas = ["errores_por_serie.csv", "resumen_metricas.csv",
                           "tabla8_resultados.csv"]
    notas_metricas = []
    conteos_metricas: dict = {
        "series_sin_mape": {
            fila["modelo"]: int(fila["series_sin_mape"])
            for _, fila in resumen_completo.iterrows()
        },
    }
    suite_val = None
    if maestro.disponible:
        suite_val = metricas.suite_valorizada(
            errores, maestro.costo_por_serie, orden=orden
        )
        reporte.tabla("tabla_valorizada.csv", metricas.tabla_valorizada(suite_val))
        reporte.nota("valorizado", suite_val)
        artefactos_metricas.append("tabla_valorizada.csv")
        conteos_metricas["cobertura_de_costo"] = round(
            float(suite_val["cobertura_series"].iloc[0]), 4
        )
        conteos_metricas["D_por_modelo_pct"] = {
            fila["modelo"]: round(float(fila["D"]), 1)
            for _, fila in suite_val.iterrows()
        }
        print("\n      MÉTRICA DE DECISIÓN  D = WMAPE_val + |Bias_val|  (menor es mejor):")
        print(metricas.tabla_valorizada(suite_val).round(1).to_string(index=False))
        notas_metricas.append(
            "Las unidades difieren entre SKUs: la única agregación con sentido "
            "físico entre productos es la valorizada (|error| × costo unitario). "
            "Los promedios en 'unidades' de la Tabla 8 son referenciales."
        )
    else:
        notas_metricas.append(
            "SIN maestro de costos (datos/crudo/costos.csv): la suite valorizada "
            "y la métrica de decisión D no se calcularon en esta corrida."
        )

    reporte.etapa(
        "metricas", "Métricas por serie y suite valorizada", rf="RF-7",
        entrada={"modelos": len(orden)},
        salida={"series_evaluadas": int(resumen_completo["series"].iloc[0])},
        decisiones={
            "metricas_por_serie": ["MAE", "RMSE", "MAPE", "Bias", "MASE",
                                    "error compuesto (mae+|bias|)"],
            "metrica_de_decision": "D = WMAPE_val + |Bias_val| (valorizada por costo)",
            "resumen": "media y mediana, sobre la intersección de series",
        },
        conteos=conteos_metricas,
        artefactos=artefactos_metricas,
        notas=notas_metricas,
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 10. Pruebas estadísticas ------------------------------------------
    inicio = _paso(f"10/{total}", "Contraste estadístico")
    metrica = cfg.pruebas.metrica_contraste
    matriz = metricas.matriz_de_errores(errores, metrica=metrica)
    propuestos = [m for m in ("motor", "lightgbm") if m in matriz.columns]
    referencias = [
        m for m in (cfg.modelos["benchmark_promedio_movil"], cfg.modelos["benchmark_naive"])
        if m in matriz.columns
    ]
    resultados = pruebas.bateria_completa(
        matriz, propuestos, referencias,
        alfa=cfg.pruebas.alfa, alternativa=cfg.pruebas.alternativa,
        n_maximo_shapiro=cfg.pruebas.shapiro_n_maximo, semilla=cfg.pruebas.semilla,
    )
    acierto = motor.tasa_de_acierto(matriz)

    reporte.texto(
        "pruebas_estadisticas.txt",
        informe_pruebas(resultados, cfg, metrica, acierto),
    )
    reporte.tabla("victorias_por_modelo.csv", resultados["victorias"])
    if resultados["friedman"].nemenyi is not None:
        reporte.tabla(
            "nemenyi.csv",
            resultados["friedman"].nemenyi.rename_axis("modelo").reset_index(),
        )
    reporte.tabla(
        "rangos_friedman.csv",
        (resultados["friedman"].rangos_medios
         .rename("rango_medio").rename_axis("modelo").reset_index()),
    )
    print(f"      Friedman: chi2 = {resultados['friedman'].chi2:,.2f}, "
          f"p = {resultados['friedman'].p:.3e}, W de Kendall = "
          f"{resultados['friedman'].kendall_w:.4f}")
    for r in resultados["wilcoxon"]:
        print(f"      Wilcoxon {r.propuesto} vs {r.referencia}: "
              f"r = {r.r:+.3f}, p = {r.p:.3e}, "
              f"gana en {100 * r.gana_propuesto / max(r.n_pares, 1):.1f} % de las series")
    print(f"      Tasa de acierto del motor: {100 * acierto['tasa_acierto']:.2f} % "
          f"(azar: {100 * acierto['azar_esperado']:.2f} %)")
    artefactos_contraste = ["pruebas_estadisticas.txt", "victorias_por_modelo.csv",
                            "rangos_friedman.csv"]
    if resultados["friedman"].nemenyi is not None:
        artefactos_contraste.append("nemenyi.csv")
    reporte.etapa(
        "contraste", "Contraste estadístico de la hipótesis", rf="RF-8",
        entrada={"bloques_completos": resultados["n_bloques"]},
        decisiones={
            "metrica_contraste": metrica,
            "alfa": cfg.pruebas.alfa,
            "alternativa": cfg.pruebas.alternativa,
            "esquema": "Wilcoxon pareado + Friedman con post hoc de Nemenyi "
                       "(Demšar 2006)",
        },
        conteos={
            "friedman_chi2": resultados["friedman"].chi2,
            "friedman_p": resultados["friedman"].p,
            "kendall_w": resultados["friedman"].kendall_w,
            "comparaciones_wilcoxon": len(resultados["wilcoxon"]),
            "tasa_acierto_motor": acierto["tasa_acierto"],
        },
        artefactos=artefactos_contraste,
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 10b. Evaluación multihorizonte -------------------------------------
    inicio = _paso(f"10b/{total}", "Evaluación multihorizonte (orígenes rodantes)")
    resultado_mh = None
    if cfg.multihorizonte is None or not cfg.multihorizonte.activo:
        print("      Desactivada por configuración.")
        reporte.etapa(
            "evaluacion_multihorizonte", "Evaluación multihorizonte", rf="RF-7",
            decisiones={"activo": False},
            notas=["Desactivada por configuración (multihorizonte.activo)."],
            duracion_s=time.perf_counter() - inicio,
        )
    elif not maestro.disponible:
        print("      SIN maestro de costos: la curva D(h) no puede calcularse.")
        reporte.etapa(
            "evaluacion_multihorizonte", "Evaluación multihorizonte", rf="RF-7",
            decisiones={"activo": True},
            notas=[
                "SIN maestro de costos (datos/crudo/costos.csv): las unidades "
                "difieren entre SKUs y sin valorizar no hay agregación válida "
                "por horizonte. La etapa se saltó y se declara.",
            ],
            duracion_s=time.perf_counter() - inicio,
        )
    else:
        elegido = motor.informe.seleccion.set_index("serie")["modelo_elegido"]
        resultado_mh = multihorizonte.evaluar(
            cfg, modelos, elegido, panel, panel_real, panel_entrenamiento,
            cohorte_ajustada, maestro.costo_por_serie,
        )
        for linea in resultado_mh.informe.lineas():
            print("      " + linea)
        reporte.tabla("multihorizonte_horizonte.csv", resultado_mh.resumen_horizonte)
        reporte.tabla("multihorizonte_origen.csv", resultado_mh.resumen_origen)
        reporte.nota("multihorizonte", {
            "origenes": resultado_mh.informe.origenes,
            "horizonte_maximo": resultado_mh.informe.horizonte_maximo,
            "reentrenos_lightgbm": resultado_mh.informe.reentrenos_lightgbm,
            "respaldos": resultado_mh.informe.respaldos,
            "observaciones": resultado_mh.informe.observaciones,
            "observaciones_con_costo": resultado_mh.informe.observaciones_con_costo,
            "D_global_pct": resultado_mh.informe.d_global,
        })
        reporte.etapa(
            "evaluacion_multihorizonte",
            "Evaluación multihorizonte (orígenes rodantes)", rf="RF-7",
            entrada={"origenes": len(resultado_mh.informe.origenes),
                     "modelos": len(resultado_mh.informe.modelos)},
            salida={"observaciones": resultado_mh.informe.observaciones},
            decisiones={
                "origenes": cfg.multihorizonte.origenes,
                "horizonte_maximo": cfg.multihorizonte.horizonte_maximo,
                "reentrenar_lightgbm": cfg.multihorizonte.reentrenar_lightgbm,
                "seleccion_motor": "congelada de validación (RN-2)",
                "proyeccion": "recursiva, la mecánica del sistema en producción; "
                              "Croston constante (sus estados solo se actualizan "
                              "con demanda observada)",
            },
            conteos={
                "respaldos_por_modelo": resultado_mh.informe.respaldos,
                "reentrenos_lightgbm": resultado_mh.informe.reentrenos_lightgbm,
                "sin_pronostico": resultado_mh.informe.sin_pronostico,
                "D_global_pct": {
                    m: round(d, 1)
                    for m, d in resultado_mh.informe.d_global.items()
                },
            },
            artefactos=["multihorizonte_horizonte.csv", "multihorizonte_origen.csv"],
            notas=[
                "Protocolo SEPARADO del principal (RN-4): la Tabla 8 y el "
                "contraste siguen siendo a un paso; esta etapa mide quién "
                "sostiene el horizonte de planificación. El objetivo operativo "
                "es 18 meses; el histórico permite medir hasta h="
                f"{cfg.multihorizonte.horizonte_maximo}.",
            ],
            duracion_s=time.perf_counter() - inicio,
        )
    _fin(inicio)

    # --- 11. Figuras --------------------------------------------------------
    inicio = _paso(f"11/{total}", "Figuras")
    dpi = int(cfg.figuras.get("dpi", 300))
    reporte.figura(figuras.figura2_error_compuesto(
        resumen_completo, reporte.directorio / "figura2_error.png", dpi=dpi
    ))
    reporte.figura(figuras.figura3_dispersion(
        errores, reporte.directorio / "figura3_dispersion.png", dpi=dpi
    ))
    reporte.figura(figuras.figura4_diferencia_critica(
        resultados["friedman"].rangos_medios,
        resultados["friedman"].diferencia_critica,
        resultados["friedman"].n_bloques,
        reporte.directorio / "figura4_diferencia_critica.png", dpi=dpi,
    ))
    artefactos_figuras = ["figura2_error.png", "figura3_dispersion.png",
                          "figura4_diferencia_critica.png"]
    if resultado_mh is not None:
        reporte.figura(figuras.figura5_horizonte(
            resultado_mh.resumen_horizonte,
            reporte.directorio / "figura5_horizonte.png",
            benchmarks=(str(cfg.modelos["benchmark_promedio_movil"]),
                        str(cfg.modelos["benchmark_naive"])),
            dpi=dpi,
        ))
        artefactos_figuras.append("figura5_horizonte.png")
    reporte.etapa(
        "figuras", "Figuras del documento", rf="RF-9",
        decisiones={"dpi": dpi, "escala_grises": True},
        artefactos=artefactos_figuras,
        duracion_s=time.perf_counter() - inicio,
    )
    _fin(inicio)

    # --- 12. Manifiesto -----------------------------------------------------
    inicio = _paso(f"12/{total}", "Manifiesto de trazabilidad")
    reporte.nota("particion", resumen_particion)
    reporte.nota("cohorte", informe_cohorte)
    reporte.nota("respaldos_por_modelo", respaldos)
    reporte.nota("motor", {
        "regla": motor.informe.regla,
        "empates": motor.informe.empates,
        "respaldos": motor.informe.respaldos,
        "reparto": motor.informe.reparto,
        "acierto": acierto,
    })
    reporte.nota("tabla8", metricas.tabla8(resumen_completo))
    reporte.nota("friedman", {
        "chi2": resultados["friedman"].chi2,
        "p": resultados["friedman"].p,
        "kendall_w": resultados["friedman"].kendall_w,
        "n_bloques": resultados["friedman"].n_bloques,
        "k": resultados["friedman"].k,
        "diferencia_critica": resultados["friedman"].diferencia_critica,
        "rangos_medios": resultados["friedman"].rangos_medios,
    })
    reporte.nota("wilcoxon", [
        {"propuesto": r.propuesto, "referencia": r.referencia, "r": r.r, "z": r.z,
         "p": r.p, "significativo": r.significativo,
         "gana_propuesto": r.gana_propuesto, "n_pares": r.n_pares}
        for r in resultados["wilcoxon"]
    ])
    ruta = reporte.escribir_manifiesto(cfg.ruta_datos)
    _fin(inicio)

    print(f"\nSalidas en {reporte.directorio}")
    print(f"Manifiesto: {ruta}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Motor de pronóstico de demanda — NOVAPACK S.A.",
    )
    parser.add_argument(
        "comando", choices=["inspeccionar", "ejecutar"],
        help="inspeccionar: carga e informe. ejecutar: pipeline completo.",
    )
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    parser.add_argument(
        "--anular", action="append", default=[], metavar="CLAVE=VALOR",
        help="Anula una clave de la configuración para esta corrida, por ejemplo "
             "--anular modelos.motor_regla=mae_mas_bias. Repetible. Sirve para "
             "correr ablaciones sin duplicar el archivo de configuración; la "
             "anulación queda registrada en el manifiesto.",
    )
    args = parser.parse_args(argv)

    cfg = cargar_config(args.config, anulaciones=args.anular)
    if args.anular:
        print(f"Anulaciones   : {', '.join(args.anular)}")
    print("=" * 78)
    print("MOTOR DE PRONÓSTICO DE DEMANDA — NOVAPACK S.A.")
    print("=" * 78)
    print(f"Configuración : {cfg.ruta_config}")
    print(f"Datos         : {cfg.ruta_datos}")
    print(f"Período       : {cfg.periodo.inicio} .. {cfg.periodo.fin}")
    print(f"Bloques       : entrenamiento <= {cfg.particion.fin_entrenamiento} · "
          f"validación <= {cfg.particion.fin_validacion} · prueba <= {cfg.periodo.fin}")
    print(f"Modelos       : {', '.join(cfg.modelos_activos)}")

    if args.comando == "inspeccionar":
        return comando_inspeccionar(cfg)
    return comando_ejecutar(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
