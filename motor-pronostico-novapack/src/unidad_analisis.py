"""Etapa `unidad_analisis` — por qué la serie es SKU–canal–regional.

La unidad de análisis es la decisión que sostiene a todas las demás: qué es
«una serie», sobre qué se pronostica, qué se agrega y qué no. El documento la
declara temprano y este módulo produce la evidencia REPRODUCIBLE que la
justifica, con tres patas:

1. **La impone la decisión.** La demanda es independiente al nivel
   SKU–canal–regional (quién compra y dónde), y las unidades de medida
   difieren ENTRE SKUs: no existe agregación física por encima del SKU. Esa
   pata es de dominio, no de datos — acá solo se documenta.
2. **Independencia estadística dentro del SKU.** Si las combinaciones de un
   mismo SKU se movieran juntas, pronosticar el SKU agregado y repartir sería
   suficiente. Se mide: cuántos SKUs viven en varias combinaciones, cuánto
   deriva la participación de cada combinación mes a mes, cuánto correlacionan
   entre sí (en niveles y en variación interanual) y si conviven regímenes de
   demanda distintos (matriz ADI–CV² de Syntetos–Boylan) bajo el mismo SKU.
   Además: en cuántos SKUs el motor elige modelos DISTINTOS por combinación.
3. **La alternativa ensayada no domina.** Contrafactual justo: el MISMO modelo
   pronosticando arriba (SKU agregado, prorrateado por participaciones
   históricas rezagadas) contra abajo (por combinación), evaluado valorizado.
   Si repartir desde arriba no gana, la unidad elegida no está pagando un
   costo oculto.

Disciplina temporal de la etapa (ninguna RN se toca):

- La **estructura** (pata 2) se mide SOLO sobre el bloque de entrenamiento y
  sobre el panel de AJUSTE — el mismo que alimenta a los modelos.
- La evidencia de **ganadores** usa la selección del motor, que decide mirando
  solo validación (RN-2).
- El **contrafactual** se ajusta con historia y se evalúa SOLO sobre los meses
  de validación, contra la serie observada. El bloque de prueba no se toca:
  esta etapa justifica un supuesto de diseño y un supuesto de diseño no puede
  calibrarse mirando la respuesta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .modelos.base import truncar_en_cero
from .modelos.naive import Naive
from .modelos.promedio_movil import PromedioMovil

# Umbrales de Syntetos–Boylan (2005), los mismos del informe de inspección.
ADI_CORTE = 1.32
CV2_CORTE = 0.49

# Meses de solape mínimo para que un par de combinaciones aporte correlación:
# con menos historia común, el coeficiente es ruido con nombre de número.
SOLAPE_MINIMO = 24
# Observaciones positivas mínimas para clasificar el régimen de una serie.
POSITIVOS_MINIMOS = 4


# ---------------------------------------------------------------------------
# Pata 2 — estructura dentro del SKU (solo entrenamiento)
# ---------------------------------------------------------------------------

def clasificar_regimen(largo: pd.DataFrame) -> pd.DataFrame:
    """ADI y CV² por serie, con su cuadrante de la matriz de Syntetos–Boylan.

    ``ADI`` = vida observada / meses con demanda positiva (≥ 1; alto = esporádica).
    ``CV²`` = (desvío muestral / media)² de las demandas POSITIVAS.
    Series con menos de ``POSITIVOS_MINIMOS`` meses positivos quedan sin
    cuadrante (contadas, no inventadas).
    """
    positivos = largo.loc[largo["y"] > 0].groupby("serie")["y"]
    n_positivos = positivos.count()
    media = positivos.mean()
    desvio = positivos.std(ddof=1)

    vida = largo.groupby("serie")["periodo"].agg(["min", "max"])
    meses_vida = (vida["max"] - vida["min"]).apply(lambda d: d.n) + 1

    tabla = pd.DataFrame({
        "sku": largo.drop_duplicates("serie").set_index("serie")["sku"],
        "meses_vida": meses_vida,
        "meses_con_demanda": n_positivos.reindex(meses_vida.index).fillna(0).astype(int),
    })
    tabla["adi"] = np.where(
        tabla["meses_con_demanda"] > 0,
        tabla["meses_vida"] / tabla["meses_con_demanda"].clip(lower=1),
        np.nan,
    )
    cv2 = (desvio / media) ** 2
    tabla["cv2"] = cv2.reindex(tabla.index)

    clasificable = (
        (tabla["meses_con_demanda"] >= POSITIVOS_MINIMOS) & tabla["cv2"].notna()
    )
    esporadica = tabla["adi"] >= ADI_CORTE
    variable = tabla["cv2"] >= CV2_CORTE
    cuadrante = np.select(
        [
            clasificable & ~esporadica & ~variable,
            clasificable & ~esporadica & variable,
            clasificable & esporadica & ~variable,
            clasificable & esporadica & variable,
        ],
        ["suave", "erratica", "intermitente", "lumpy"],
        default="sin_clasificar",
    )
    tabla["cuadrante"] = cuadrante
    return tabla


def _deriva_participacion(largo: pd.DataFrame, skus_multi: pd.Index) -> pd.Series:
    """Deriva media de la participación mensual, en puntos porcentuales por SKU.

    Para cada mes con venta del SKU, la participación de una combinación es su
    parte del total del SKU ese mes. La deriva es la media de |Δ participación|
    entre meses CONSECUTIVOS con total positivo en ambos. Un SKU cuyas
    combinaciones mantienen participaciones estables sería prorrateable desde
    arriba; una deriva alta dice que el reparto es una diana móvil.

    El llamador ya quitó los meses de la ventana del evento exógeno: como acá
    los meses de la frontera dejan de ser consecutivos, sus pares no aportan —
    exactamente lo que corresponde.
    """
    multi = largo.loc[largo["sku"].isin(skus_multi)].copy()
    total = multi.groupby(["sku", "periodo"], observed=True)["y"].transform("sum")
    multi = multi.loc[total > 0].assign(participacion=multi["y"] / total[total > 0])

    multi = multi.sort_values(["serie", "periodo"], kind="stable")
    multi["mes_entero"] = pd.PeriodIndex(multi["periodo"]).asi8
    por_serie = multi.groupby("serie", observed=True)
    consecutivo = por_serie["mes_entero"].diff().eq(1)
    delta = por_serie["participacion"].diff().abs()

    validas = multi.assign(delta=delta.where(consecutivo)).dropna(subset=["delta"])
    return 100.0 * validas.groupby("sku", observed=True)["delta"].mean()


def _correlaciones_por_sku(
    largo: pd.DataFrame, skus_multi: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlación de a pares entre combinaciones del mismo SKU.

    Devuelve (resumen_por_sku, pares): en niveles y en variación interanual
    (diferencia a 12 meses, que descuenta la estacionalidad común — dos
    combinaciones pueden «correlacionar» solo porque ambas venden en campaña).
    Pares con menos de ``SOLAPE_MINIMO`` meses comunes no aportan (contados).
    """
    filas_sku: list[dict] = []
    filas_pares: list[dict] = []

    multi = largo.loc[largo["sku"].isin(skus_multi)]
    for sku, grupo in multi.groupby("sku", observed=True, sort=True):
        ancho = grupo.pivot(index="periodo", columns="serie", values="y").sort_index()
        if ancho.shape[1] < 2:
            continue
        # Reindexar a TODOS los meses calendario del rango: el llamador quitó
        # la ventana del evento y sin este reindex el diff(12) —que es
        # posicional— cruzaría 24 meses reales creyendo que son 12.
        ancho = ancho.reindex(
            pd.period_range(ancho.index.min(), ancho.index.max(), freq="M")
        )
        niveles = ancho.corr(min_periods=SOLAPE_MINIMO)
        yoy = ancho.diff(12).corr(min_periods=max(12, SOLAPE_MINIMO - 12))

        series = list(ancho.columns)
        pares_nivel, pares_yoy, sin_solape = [], [], 0
        for i in range(len(series)):
            for j in range(i + 1, len(series)):
                r = niveles.iloc[i, j]
                r_yoy = yoy.iloc[i, j]
                if np.isnan(r):
                    sin_solape += 1
                    continue
                pares_nivel.append(float(r))
                if not np.isnan(r_yoy):
                    pares_yoy.append(float(r_yoy))
                filas_pares.append({
                    "sku": sku, "serie_a": series[i], "serie_b": series[j],
                    "r_niveles": float(r),
                    "r_yoy": float(r_yoy) if not np.isnan(r_yoy) else np.nan,
                })

        filas_sku.append({
            "sku": sku,
            "pares_validos": len(pares_nivel),
            "pares_sin_solape": sin_solape,
            "correlacion_mediana": float(np.median(pares_nivel)) if pares_nivel else np.nan,
            "correlacion_yoy_mediana": float(np.median(pares_yoy)) if pares_yoy else np.nan,
        })

    # Esqueletos explícitos: un DataFrame vacío SIN columnas rompería los
    # joins y selecciones aguas abajo (panel sin SKUs multi-combinación).
    columnas_sku = ["sku", "pares_validos", "pares_sin_solape",
                    "correlacion_mediana", "correlacion_yoy_mediana"]
    columnas_pares = ["sku", "serie_a", "serie_b", "r_niveles", "r_yoy"]
    return (
        pd.DataFrame(filas_sku, columns=columnas_sku),
        pd.DataFrame(filas_pares, columns=columnas_pares),
    )


def estructura_por_sku(
    largo_entrenamiento: pd.DataFrame,
    excluir_meses: pd.PeriodIndex | None = None,
) -> tuple[pd.DataFrame, dict]:
    """La pata 2 completa, medida SOLO sobre el bloque de entrenamiento.

    ``excluir_meses`` es la ventana del evento exógeno, y se quita de la
    DERIVA y de las CORRELACIONES. La razón la encontró la auditoría del
    experimento: con el tratamiento ``copiar_gestion_previa`` el panel de
    ajuste cumple y(t) = y(t−12) dentro de la ventana, así que la diferencia
    interanual es 0 EXACTO — doce pares (0,0) fabricados por par de
    combinaciones que inflan la correlación interanual (verificado: r pasaba
    de +0,07 a +0,27 en series independientes); y el año siguiente el
    «interanual» sería en realidad una diferencia a 24 meses. Con la ventana
    fuera, cada estadístico mide lo que declara. El régimen ADI–CV² sí se
    calcula sobre el entrenamiento completo (la copia es demanda real del año
    previo y no fabrica estructura de régimen; se declara).
    """
    regimen = clasificar_regimen(largo_entrenamiento)

    combos = regimen.groupby("sku").size().rename("n_combos")
    skus_multi = combos.index[combos >= 2]

    sin_evento = largo_entrenamiento
    if excluir_meses is not None and len(excluir_meses):
        sin_evento = largo_entrenamiento.loc[
            ~largo_entrenamiento["periodo"].isin(excluir_meses)
        ]

    deriva = _deriva_participacion(sin_evento, skus_multi)
    correl_sku, pares = _correlaciones_por_sku(sin_evento, skus_multi)

    cuadrantes = (
        regimen.loc[regimen["cuadrante"] != "sin_clasificar"]
        .groupby("sku")["cuadrante"].agg(["nunique", lambda c: "|".join(sorted(set(c)))])
    )
    cuadrantes.columns = ["n_cuadrantes", "cuadrantes"]

    tabla = pd.DataFrame({"n_combos": combos})
    tabla["deriva_participacion_pp"] = deriva
    tabla = tabla.join(correl_sku.set_index("sku"), how="left")
    tabla = tabla.join(cuadrantes, how="left")
    tabla.index.name = "sku"

    multi = tabla.loc[tabla["n_combos"] >= 2]
    con_cuadrantes = multi.loc[multi["n_cuadrantes"].notna()]
    resumen = {
        "skus_totales": int(len(tabla)),
        "skus_multicombo": int(len(multi)),
        "proporcion_multicombo": float(len(multi) / len(tabla)) if len(tabla) else np.nan,
        "mediana_combos_en_multi": float(multi["n_combos"].median()) if len(multi) else np.nan,
        "deriva_participacion_pp_media": float(multi["deriva_participacion_pp"].mean()),
        "pares_correlacion": int(len(pares)),
        "correlacion_mediana": float(pares["r_niveles"].median()) if len(pares) else np.nan,
        "correlacion_yoy_mediana": float(pares["r_yoy"].median()) if len(pares) else np.nan,
        "proporcion_pares_bajo_05": (
            float((pares["r_niveles"] < 0.5).mean()) if len(pares) else np.nan
        ),
        "skus_con_regimenes_mixtos": int((con_cuadrantes["n_cuadrantes"] >= 2).sum()),
        "skus_multi_clasificables": int(len(con_cuadrantes)),
    }
    return tabla.reset_index(), resumen


# ---------------------------------------------------------------------------
# Ganadores distintos bajo el mismo SKU (selección del motor, validación)
# ---------------------------------------------------------------------------

def ganadores_por_sku(
    eleccion: pd.Series, serie_a_sku: pd.Series
) -> tuple[pd.DataFrame, dict]:
    """¿En cuántos SKUs el motor eligió modelos DISTINTOS por combinación?

    Si el mejor modelo fuera propiedad del SKU, elegir por combinación sería
    burocracia. Se mide sobre la cohorte (el motor solo decide ahí) y con la
    selección de validación (RN-2).
    """
    tabla = pd.DataFrame({
        "modelo": eleccion,
        "sku": serie_a_sku.reindex(eleccion.index),
    }).dropna(subset=["sku"])

    if tabla.empty:
        # groupby sobre vacío no propaga el nombre del índice: se devuelve el
        # esqueleto explícito para que el merge posterior no reviente.
        vacio = pd.DataFrame(
            columns=["sku", "n_combos_cohorte", "n_modelos_ganadores"]
        )
        return vacio, {
            "skus_cohorte_multi": 0,
            "skus_multiganador": 0,
            "proporcion_multiganador": float("nan"),
        }

    por_sku = tabla.groupby("sku")["modelo"].agg(
        n_combos_cohorte="size", n_modelos_ganadores="nunique"
    ).reset_index()

    multi = por_sku.loc[por_sku["n_combos_cohorte"] >= 2]
    resumen = {
        "skus_cohorte_multi": int(len(multi)),
        "skus_multiganador": int((multi["n_modelos_ganadores"] >= 2).sum()),
        "proporcion_multiganador": (
            float((multi["n_modelos_ganadores"] >= 2).mean()) if len(multi) else np.nan
        ),
    }
    return por_sku, resumen


# ---------------------------------------------------------------------------
# Pata 3 — contrafactual: mismo modelo, arriba contra abajo (validación)
# ---------------------------------------------------------------------------

def contrafactual_topdown(
    cfg: Config,
    cohorte_largo: pd.DataFrame,
    panel_real: pd.DataFrame,
    costo_por_serie: pd.Series,
) -> pd.DataFrame:
    """El MISMO modelo pronosticando el SKU agregado (y repartiendo) contra
    pronosticar cada combinación. Un paso, solo validación, valorizado.

    - **Agregar dentro del SKU es legítimo**: todas sus combinaciones comparten
      unidad de medida (la razón por la que por ENCIMA del SKU no se puede).
    - **El reparto no mira el futuro**: la participación de cada combinación es
      su parte de los últimos doce meses CERRADOS (rezago de un mes).
    - **Comparación justa**: mismos modelos (los dos benchmarks declarados),
      mismas combinaciones, mismos meses, misma valorización. Solo cambia el
      nivel al que se pronostica.
    """
    fin_validacion = cfg.particion.fin_validacion
    meses_validacion = pd.period_range(
        cfg.particion.fin_entrenamiento + 1, fin_validacion, freq="M"
    )

    # Nada posterior a validación entra a esta función — verificación activa.
    largo = cohorte_largo.loc[cohorte_largo["periodo"] <= fin_validacion]

    combos_por_sku = largo.drop_duplicates("serie").groupby("sku")["serie"].agg(list)
    skus_multi = combos_por_sku.index[combos_por_sku.str.len() >= 2]
    series_multi = [s for sku in skus_multi for s in combos_por_sku[sku]]
    if not series_multi:
        return pd.DataFrame()

    ancho = (
        largo.loc[largo["serie"].isin(series_multi)]
        .pivot(index="periodo", columns="serie", values="y").sort_index()
    )
    mapa_sku = largo.drop_duplicates("serie").set_index("serie")["sku"]
    ancho_sku = ancho.T.groupby(mapa_sku.reindex(ancho.columns)).sum(min_count=1).T

    # Participación de los últimos 12 meses cerrados (shift primero: el mes que
    # se predice jamás entra a su propio reparto).
    suma_combo = ancho.fillna(0.0).shift(1).rolling(12, min_periods=12).sum()
    suma_sku = ancho_sku.fillna(0.0).shift(1).rolling(12, min_periods=12).sum()
    participacion = suma_combo / suma_sku[mapa_sku.reindex(ancho.columns)].to_numpy()

    real = panel_real.reindex(index=meses_validacion, columns=ancho.columns)
    costo = costo_por_serie.reindex(ancho.columns)

    # La ventana del promedio móvil se DERIVA del nombre del benchmark: fijarla
    # aparte dejaría una etiqueta configurable sobre un cálculo hardcodeado
    # (una ablación que cambiara el benchmark reportaría un modelo calculando
    # otro, en silencio).
    nombre_ma = str(cfg.modelos["benchmark_promedio_movil"])
    try:
        ventana_ma = int(nombre_ma.split("_")[1])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"No se puede derivar la ventana del benchmark '{nombre_ma}' "
            "(se espera el formato ma_<ventana>)."
        ) from error
    modelos = {
        nombre_ma: PromedioMovil(ventana_ma),
        str(cfg.modelos["benchmark_naive"]): Naive(),
    }

    filas: list[dict] = []
    for nombre, modelo in modelos.items():
        abajo = truncar_en_cero(modelo.predecir(ancho)).reindex(index=meses_validacion)
        arriba_sku = truncar_en_cero(modelo.predecir(ancho_sku))
        arriba = (
            arriba_sku[mapa_sku.reindex(ancho.columns)].to_numpy() * participacion
        )
        arriba = pd.DataFrame(
            arriba, index=ancho.index, columns=ancho.columns
        ).reindex(index=meses_validacion)

        # Máscara COMÚN a los dos enfoques: la comparación es pareada o no es
        # (hallazgo de la auditoría: la participación da 0/0 = NaN cuando el
        # SKU acumula doce meses cerrados en cero, y evaluar cada enfoque solo
        # donde puede pronosticar le regalaba al top-down justo los meses de
        # resurrección donde el bottom-up come su peor error — un delta
        # fabricado de hasta 66 puntos en el escenario reproducido).
        comparable = (
            real.notna() & abajo.notna() & arriba.notna()
            & costo.notna().to_numpy()[None, :]
        )
        con_real = real.notna() & costo.notna().to_numpy()[None, :]
        excluidas = int((con_real & ~comparable).to_numpy().sum())

        for enfoque, p in (("bottom_up", abajo), ("top_down", arriba)):
            y = real.where(comparable)
            demanda = float((y * costo).sum().sum())
            error = float(((p.where(comparable) - y).abs() * costo).sum().sum())
            predicho = float((p.where(comparable) * costo).sum().sum())
            wmape = 100.0 * error / demanda if demanda > 0 else np.nan
            bias = 100.0 * (predicho - demanda) / demanda if demanda > 0 else np.nan
            filas.append({
                "modelo": nombre,
                "enfoque": enfoque,
                "wmape_val": wmape,
                "bias_val": bias,
                "D": wmape + abs(bias) if demanda > 0 else np.nan,
                "series": int(comparable.any(axis=0).sum()),
                "skus": int(len(skus_multi)),
                "observaciones": int(comparable.to_numpy().sum()),
                "observaciones_excluidas_del_pareo": excluidas,
            })

    tabla = pd.DataFrame(filas)
    # La lectura directa: Δ = top_down − bottom_up en D (positivo = repartir
    # desde arriba es PEOR).
    pivote = tabla.pivot(index="modelo", columns="enfoque", values="D")
    tabla["delta_D_topdown"] = tabla["modelo"].map(
        pivote["top_down"] - pivote["bottom_up"]
    )
    return tabla


# ---------------------------------------------------------------------------
# Orquestación de la etapa
# ---------------------------------------------------------------------------

@dataclass
class InformeUnidad:
    resumen: dict = field(default_factory=dict)
    tabla_sku: pd.DataFrame = field(default_factory=pd.DataFrame)
    contrafactual: pd.DataFrame = field(default_factory=pd.DataFrame)

    def lineas(self) -> list[str]:
        r = self.resumen
        salida = [
            f"SKUs (todas las series, entrenamiento) : {r['skus_totales']:,}",
            f"SKUs en varias combinaciones            : {r['skus_multicombo']:,} "
            f"({100 * r['proporcion_multicombo']:.1f} %) · mediana "
            f"{r['mediana_combos_en_multi']:.0f} combinaciones",
            f"Deriva de participación (media)         : "
            f"{r['deriva_participacion_pp_media']:.1f} pp/mes",
            f"Correlación intra-SKU (mediana)         : "
            f"{r['correlacion_mediana']:.2f} niveles · "
            f"{r['correlacion_yoy_mediana']:.2f} interanual · "
            f"{100 * r['proporcion_pares_bajo_05']:.0f} % de pares < 0,5 "
            f"({r['pares_correlacion']:,} pares)",
            f"SKUs con regímenes ADI–CV² mixtos       : "
            f"{r['skus_con_regimenes_mixtos']:,} de "
            f"{r['skus_multi_clasificables']:,} clasificables",
        ]
        if "skus_multiganador" in r:
            salida.append(
                f"SKUs con ganadores DISTINTOS (motor)    : "
                f"{r['skus_multiganador']:,} de {r['skus_cohorte_multi']:,} "
                f"({100 * r['proporcion_multiganador']:.1f} %)"
            )
        for _, fila in self.contrafactual.iterrows():
            salida.append(
                f"Contrafactual {fila['modelo']:<9} {fila['enfoque']:<10}: "
                f"D = {fila['D']:6.1f} %"
            )
        return salida


def evaluar(
    cfg: Config,
    panel_ajuste_largo: pd.DataFrame,
    cohorte_largo: pd.DataFrame,
    panel_real: pd.DataFrame,
    eleccion_motor: pd.Series,
    costo_por_serie: pd.Series | None,
) -> InformeUnidad:
    """Corre las tres patas y devuelve el informe con sus tablas."""
    largo_entrenamiento = panel_ajuste_largo.loc[
        panel_ajuste_largo["periodo"] <= cfg.particion.fin_entrenamiento
    ]
    # La ventana del evento se excluye de deriva y correlaciones SIEMPRE que
    # esté declarada, sin importar el tratamiento: con la copia fabrica ceros
    # interanuales, y cruda es un shock común exógeno que infla la correlación
    # con co-movimiento que no es demanda estructural.
    ventana_evento = cfg.evento.ventana() if cfg.evento is not None else None
    tabla_sku, resumen = estructura_por_sku(
        largo_entrenamiento, excluir_meses=ventana_evento
    )

    serie_a_sku = cohorte_largo.drop_duplicates("serie").set_index("serie")["sku"]
    por_sku_ganadores, resumen_ganadores = ganadores_por_sku(
        eleccion_motor, serie_a_sku
    )
    resumen.update(resumen_ganadores)
    tabla_sku = tabla_sku.merge(
        por_sku_ganadores[["sku", "n_combos_cohorte", "n_modelos_ganadores"]],
        on="sku", how="left",
    )

    contrafactual = pd.DataFrame()
    if costo_por_serie is not None and not costo_por_serie.empty:
        contrafactual = contrafactual_topdown(
            cfg, cohorte_largo, panel_real, costo_por_serie
        )

    return InformeUnidad(
        resumen=resumen, tabla_sku=tabla_sku, contrafactual=contrafactual
    )
