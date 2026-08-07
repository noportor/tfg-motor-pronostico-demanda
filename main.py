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

from src import carga, features, figuras, inspeccion, metricas, particion, pruebas, series
from src.config import cargar_config
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
    _fin(inicio)

    # --- 2. Series ----------------------------------------------------------
    inicio = _paso(f"2/{total}", "Construyendo las series mensuales")
    panel_largo, informe_series = series.construir_series(df, cfg)
    print(f"      {informe_series.series_construidas:,} series · "
          f"{informe_series.filas_panel:,} filas · {informe_series.rango[0]}"
          f" .. {informe_series.rango[1]}")
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
    _fin(inicio)

    # --- 4. Partición y cohorte --------------------------------------------
    inicio = _paso(f"4/{total}", "Partición temporal y criterios de inclusión")
    corte = particion.particionar(cfg)
    resumen_particion = corte.resumen(cfg.periodo.mes_inicio_gestion)
    for linea in resumen_particion.to_string(index=False).split("\n"):
        print("      " + linea)

    cohorte_larga, informe_cohorte = particion.aplicar_criterios_inclusion(
        panel_largo, cfg, corte
    )
    for linea in informe_cohorte.lineas():
        print("      " + linea)
    reporte.tabla("cohorte_flujo.csv", informe_cohorte.tabla_flujo())
    reporte.tabla("particion.csv", resumen_particion)
    _fin(inicio)

    # --- 5. Paneles ---------------------------------------------------------
    inicio = _paso(f"5/{total}", "Panel ancho y bloques")
    panel = series.a_panel_ancho(cohorte_larga)
    panel_entrenamiento = panel.loc[panel.index <= corte.entrenamiento[-1]]
    panel_validacion = panel.loc[
        (panel.index >= corte.validacion[0]) & (panel.index <= corte.validacion[-1])
    ]
    panel_prueba = panel.loc[panel.index >= corte.prueba[0]]
    print(f"      panel {panel.shape[0]} meses × {panel.shape[1]:,} series  "
          f"(train {len(panel_entrenamiento)} · val {len(panel_validacion)} · "
          f"test {len(panel_prueba)})")
    _fin(inicio)

    # --- 6. Features --------------------------------------------------------
    inicio = _paso(f"6/{total}", "Construyendo features (sin fuga temporal)")
    tabla = features.construir_features(cohorte_larga, cfg)
    print(f"      {len(tabla):,} filas × {len(features.nombres_de_features(cfg))} features")
    _fin(inicio)

    # --- 7. Modelos ---------------------------------------------------------
    inicio = _paso(f"7/{total}", "Ajustando modelos y pronosticando a un paso")
    modelos = construir_modelos(cfg, tabla)
    predicciones: dict[str, pd.DataFrame] = {}
    respaldos: dict[str, int] = {}
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
    _fin(inicio)

    if parametros:
        reporte.tabla("parametros_por_serie.csv", pd.concat(parametros, ignore_index=True))

    # --- 8. Motor -----------------------------------------------------------
    inicio = _paso(f"8/{total}", "Motor de selección (decide mirando SOLO validación)")
    motor = Motor(cfg, predicciones, panel)
    motor.ajustar(panel_entrenamiento, panel_validacion)
    predicciones["motor"] = truncar_en_cero(
        enmascarar_fuera_de_vida(motor.predecir(panel), panel)
    )
    for linea in motor.informe.lineas():
        print("      " + linea)
    reporte.tabla("seleccion_motor.csv", motor.informe.seleccion)
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
