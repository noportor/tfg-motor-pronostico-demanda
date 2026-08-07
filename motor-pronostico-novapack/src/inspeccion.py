"""RF-2 — Inspección de los datos (fase de comprensión de datos de CRISP-DM).

El informe que produce este módulo tiene un destinatario concreto: el apartado
«Criterios de inclusión y exclusión» de la tesis. Su criterio de aceptación es
que permita decidir esos criterios **sin abrir los datos manualmente**, y por eso
incluye la distribución completa de meses con venta por combinación y no solo un
promedio.

Este módulo no escribe archivos (§9: sin efectos secundarios ocultos). Devuelve
el informe como texto y como estructura; escribirlo es tarea de ``reporte.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, gestion_de


@dataclass
class InformeInspeccion:
    lineas: list[str] = field(default_factory=list)
    datos: dict = field(default_factory=dict)

    def texto(self) -> str:
        return "\n".join(self.lineas)


def _titulo(texto: str) -> list[str]:
    return ["", "=" * 78, texto.upper(), "=" * 78]


def _subtitulo(texto: str) -> list[str]:
    return ["", texto, "-" * len(texto)]


def _tabla(df: pd.DataFrame, indice: bool = False) -> list[str]:
    return df.to_string(index=indice).split("\n")


def inspeccionar(
    df: pd.DataFrame,
    cfg: Config,
    panel: pd.DataFrame | None = None,
) -> InformeInspeccion:
    """Informe descriptivo de los datos crudos y, si se pasa, del panel mensual.

    Args:
        df: salida de ``carga.cargar`` — filas de venta con tipos ya convertidos.
        panel: salida de ``series.construir_series`` — panel mensual.
    """
    informe = InformeInspeccion()
    L = informe.lineas
    D = informe.datos

    L += _titulo("Informe de inspección de datos — NOVAPACK S.A.")

    # -- 1. Cobertura temporal ----------------------------------------------
    periodo = df["fecha"].dt.to_period("M")
    meses_presentes = pd.PeriodIndex(periodo.unique()).sort_values()
    esperados = pd.period_range(cfg.periodo.inicio, cfg.periodo.fin, freq="M")
    faltantes = esperados.difference(meses_presentes)

    D["rango_fechas"] = (str(df["fecha"].min().date()), str(df["fecha"].max().date()))
    D["meses_con_registro"] = int(len(meses_presentes))
    D["meses_esperados"] = int(len(esperados))

    L += _subtitulo("1. Cobertura temporal")
    L += [
        f"Rango de fechas observado        : {D['rango_fechas'][0]} .. {D['rango_fechas'][1]}",
        f"Período declarado en config.yaml : {cfg.periodo.inicio} .. {cfg.periodo.fin}"
        f"  ({len(esperados)} meses)",
        f"Meses distintos con registro     : {len(meses_presentes)}",
        f"Meses del período sin ningún registro: {len(faltantes)}"
        + (f"  -> {', '.join(map(str, faltantes[:12]))}" if len(faltantes) else ""),
        "",
        f"Gestiones fiscales cubiertas (cierre en mes {cfg.periodo.mes_inicio_gestion - 1 or 12}): "
        f"{gestion_de(cfg.periodo.inicio, cfg.periodo.mes_inicio_gestion)}"
        f"–{gestion_de(cfg.periodo.fin, cfg.periodo.mes_inicio_gestion)}",
    ]

    # -- 2. Dimensiones ------------------------------------------------------
    combinaciones = df[["sku", "canal", "regional"]].drop_duplicates()
    D["n_sku"] = int(df["sku"].nunique())
    D["n_canales"] = int(df["canal"].nunique())
    D["n_regionales"] = int(df["regional"].nunique())
    D["n_combinaciones"] = int(len(combinaciones))

    L += _subtitulo("2. Dimensiones del portafolio")
    L += [
        f"SKU distintos                    : {D['n_sku']:,}",
        f"Canales distintos                : {D['n_canales']:,}"
        f"   -> {', '.join(sorted(map(str, df['canal'].dropna().unique())))}",
        f"Regionales distintas             : {D['n_regionales']:,}"
        f"   -> {', '.join(sorted(map(str, df['regional'].dropna().unique())))}",
        f"Combinaciones SKU–canal–regional : {D['n_combinaciones']:,}",
        "",
        "Cobertura teórica del cruce: "
        f"{D['n_sku']:,} × {D['n_canales']} × {D['n_regionales']} = "
        f"{D['n_sku'] * D['n_canales'] * D['n_regionales']:,} combinaciones posibles; "
        f"con registro hay {D['n_combinaciones']:,} "
        f"({100 * D['n_combinaciones'] / max(D['n_sku'] * D['n_canales'] * D['n_regionales'], 1):.1f} %).",
    ]

    # -- 3. Calidad de los registros ----------------------------------------
    duplicados = df.duplicated(subset=["fecha", "sku", "canal", "regional"], keep=False)
    nulos = df.isna().sum()
    negativos = int((df["cantidad"] < 0).sum())
    ceros = int((df["cantidad"] == 0).sum())

    D["registros_duplicados"] = int(duplicados.sum())
    D["nulos_por_columna"] = {c: int(v) for c, v in nulos.items()}
    D["registros_negativos"] = negativos
    D["unidades_negativas"] = float(df.loc[df["cantidad"] < 0, "cantidad"].sum())
    D["registros_en_cero"] = ceros

    L += _subtitulo("3. Calidad de los registros")
    L += [
        f"Filas totales                    : {len(df):,}",
        f"Filas duplicadas por (fecha, sku, canal, regional): {D['registros_duplicados']:,}",
        "   (se agregan por suma en la construcción de series; no se descartan)",
        f"Registros con cantidad negativa  : {negativos:,} "
        f"({D['unidades_negativas']:,.0f} unidades) — devoluciones",
        f"   Tratamiento configurado       : {cfg.series.devoluciones}",
        f"Registros con cantidad exactamente cero: {ceros:,}",
        "",
        "Nulos por columna:",
    ]
    L += [f"   {c:<12} {v:,}" for c, v in D["nulos_por_columna"].items()]

    # -- 4. Volumen por gestión ---------------------------------------------
    trabajo = df.copy()
    trabajo["periodo"] = periodo
    trabajo["gestion"] = [
        gestion_de(p, cfg.periodo.mes_inicio_gestion) for p in trabajo["periodo"]
    ]
    trabajo["combinacion"] = (
        trabajo["sku"].astype(str) + "|"
        + trabajo["canal"].astype(str) + "|"
        + trabajo["regional"].astype(str)
    )
    por_gestion = (
        trabajo.groupby("gestion")
        .agg(
            registros=("cantidad", "size"),
            unidades=("cantidad", "sum"),
            skus=("sku", "nunique"),
            combinaciones=("combinacion", "nunique"),
        )
        .reset_index()
    )
    por_gestion["unidades"] = por_gestion["unidades"].round(0)
    D["volumen_por_gestion"] = por_gestion.to_dict(orient="records")

    L += _subtitulo("4. Volumen por gestión fiscal (abril–marzo)")
    L += _tabla(por_gestion)

    # -- 5. Volumen por año calendario --------------------------------------
    trabajo["anio"] = trabajo["fecha"].dt.year
    por_anio = (
        trabajo.groupby("anio")
        .agg(registros=("cantidad", "size"), unidades=("cantidad", "sum"))
        .reset_index()
    )
    por_anio["unidades"] = por_anio["unidades"].round(0)
    D["volumen_por_anio"] = por_anio.to_dict(orient="records")
    L += _subtitulo("5. Volumen por año calendario")
    L += _tabla(por_anio)

    # -- 6. Estacionalidad agregada -----------------------------------------
    por_mes = (
        trabajo.assign(mes=trabajo["fecha"].dt.month)
        .groupby("mes")["cantidad"].sum()
        .reset_index()
    )
    total = por_mes["cantidad"].sum()
    por_mes["porcentaje"] = (100 * por_mes["cantidad"] / total).round(2)
    por_mes["cantidad"] = por_mes["cantidad"].round(0)
    D["estacionalidad_mensual"] = por_mes.to_dict(orient="records")
    L += _subtitulo("6. Estacionalidad agregada (participación de cada mes)")
    L += _tabla(por_mes)
    L += [
        "",
        "Si el reparto fuera plano cada mes tendría 8,33 %. La desviación respecto "
        "de ese valor es la estacionalidad del portafolio agregado.",
    ]

    # -- 7. Reparto por canal y regional ------------------------------------
    por_canal = (
        trabajo.groupby(["regional", "canal"])
        .agg(registros=("cantidad", "size"), unidades=("cantidad", "sum"),
             skus=("sku", "nunique"), combinaciones=("combinacion", "nunique"))
        .reset_index()
    )
    por_canal["unidades"] = por_canal["unidades"].round(0)
    D["reparto_canal_regional"] = por_canal.to_dict(orient="records")
    L += _subtitulo("7. Reparto por regional y canal")
    L += _tabla(por_canal)

    # -- 8. Distribución de registros por combinación ------------------------
    registros_por_comb = (
        trabajo.groupby(["sku", "canal", "regional"], observed=True)
        .size()
    )
    cuantiles = registros_por_comb.quantile([0, .05, .10, .25, .50, .75, .90, .95, 1.0])
    D["registros_por_combinacion"] = {
        f"p{int(q * 100)}": int(v) for q, v in cuantiles.items()
    }

    L += _subtitulo("8. Meses con venta por combinación (decide el historial mínimo)")
    L += [f"   {k:<6} {v:>6,}" for k, v in D["registros_por_combinacion"].items()]

    cortes = [12, 24, 36, 48, 60, 72, 84, 96, 108]
    supervivencia = pd.DataFrame({
        "meses_con_venta_>=": cortes,
        "combinaciones": [int((registros_por_comb >= c).sum()) for c in cortes],
    })
    supervivencia["porcentaje"] = (
        100 * supervivencia["combinaciones"] / len(registros_por_comb)
    ).round(1)
    D["supervivencia_historial"] = supervivencia.to_dict(orient="records")
    L += ["", "Cuántas combinaciones sobreviven a cada umbral de historial:"]
    L += _tabla(supervivencia)

    # -- 9. Volumen acumulado por combinación --------------------------------
    volumen_por_comb = trabajo.groupby(["sku", "canal", "regional"], observed=True)["cantidad"].sum()
    cuantiles_vol = volumen_por_comb.quantile([0, .05, .25, .50, .75, .95, 1.0]).round(1)
    D["volumen_por_combinacion"] = {
        f"p{int(q * 100)}": float(v) for q, v in cuantiles_vol.items()
    }
    L += _subtitulo("9. Volumen acumulado por combinación (decide el volumen mínimo)")
    L += [f"   {k:<6} {v:>12,.1f}" for k, v in D["volumen_por_combinacion"].items()]

    cortes_vol = [10, 50, 100, 500, 1000, 5000]
    superv_vol = pd.DataFrame({
        "unidades_>=": cortes_vol,
        "combinaciones": [int((volumen_por_comb >= c).sum()) for c in cortes_vol],
    })
    superv_vol["porcentaje"] = (
        100 * superv_vol["combinaciones"] / len(volumen_por_comb)
    ).round(1)
    L += ["", "Cuántas combinaciones sobreviven a cada umbral de volumen:"]
    L += _tabla(superv_vol)

    # -- 10. Panel mensual ---------------------------------------------------
    if panel is not None:
        vida = panel.groupby("serie", observed=True)["y"]
        meses_vida = vida.size()
        ceros_vida = panel.assign(c=(panel["y"] <= 0).astype(int)).groupby("serie")["c"].sum()
        proporcion = (ceros_vida / meses_vida).round(3)

        D["panel_series"] = int(len(meses_vida))
        D["panel_filas"] = int(len(panel))
        D["proporcion_ceros"] = {
            f"p{int(q * 100)}": float(v)
            for q, v in proporcion.quantile([0, .25, .50, .75, .90, .95, 1.0]).items()
        }

        L += _subtitulo("10. Panel mensual reconstruido (con ceros intercalados)")
        L += [
            f"Series en el panel               : {D['panel_series']:,}",
            f"Filas del panel                  : {D['panel_filas']:,}",
            f"Vida media de una serie          : {meses_vida.mean():.1f} meses "
            f"(mediana {meses_vida.median():.0f})",
            "",
            "Proporción de meses en cero dentro de la vida de cada serie",
            "(decide el umbral proporcion_maxima_ceros):",
        ]
        L += [f"   {k:<6} {v:>8.3f}" for k, v in D["proporcion_ceros"].items()]

        cortes_ceros = [0.30, 0.50, 0.60, 0.70, 0.80, 0.90]
        superv_ceros = pd.DataFrame({
            "ceros_<=": cortes_ceros,
            "series": [int((proporcion <= c).sum()) for c in cortes_ceros],
        })
        superv_ceros["porcentaje"] = (100 * superv_ceros["series"] / len(proporcion)).round(1)
        L += ["", "Cuántas series sobreviven a cada umbral de ceros:"]
        L += _tabla(superv_ceros)

        # Intermitencia: cuánto del panel es demanda esporádica. No filtra nada,
        # pero explica por qué se incorpora Croston como brazo del experimento.
        adi = (meses_vida / (panel.assign(p=(panel["y"] > 0).astype(int))
                             .groupby("serie")["p"].sum().clip(lower=1)))
        D["series_intermitentes_adi_1_32"] = int((adi >= 1.32).sum())
        L += [
            "",
            f"Series con ADI >= 1,32 (demanda intermitente, Syntetos–Boylan): "
            f"{D['series_intermitentes_adi_1_32']:,} de {len(adi):,} "
            f"({100 * D['series_intermitentes_adi_1_32'] / len(adi):.1f} %).",
            "Es la razón de incorporar Croston al conjunto de modelos comparados.",
        ]

    L += ["", "=" * 78, "Fin del informe.", ""]
    return informe
