"""RF-4 — Partición temporal en tres bloques y definición de la muestra.

Partición (RN-2)
----------------
::

    entrenamiento  ->  ajusta los modelos
    validación     ->  el motor decide qué modelo gana en cada serie
    prueba         ->  se reporta el desempeño final

La selección por serie que hace el motor solo puede mirar validación. Si esa
decisión se tomara con datos de prueba el resultado reportado estaría
contaminado y la comparación sería inválida.

Los cortes caen en frontera de **gestión fiscal** (INNOVA cierra en marzo), no
en frontera de año calendario, para que ningún bloque parta una campaña escolar
por la mitad y cada bloque contenga un número entero de ciclos comerciales.

Criterios de inclusión y exclusión
----------------------------------
Se miden **exclusivamente sobre el bloque de entrenamiento**. Medirlos sobre la
historia completa seleccionaría las series usando información de validación y
prueba: una serie que se apaga durante el período de prueba quedaría excluida
por «pocos meses con venta», y el resultado reportado se calcularía sobre una
muestra elegida a posteriori. Es la forma más silenciosa de contaminar la
comparación, y por eso la ventana de medición está fijada aquí y no es
configurable.

El desglose de cuántas series excluyó cada criterio por separado va literalmente
al apartado «Criterios de inclusión y exclusión» de la tesis, y el N resultante
es el tamaño de la muestra.

Interruptores de cohorte (sensibilidad «cohorte al origen»)
-----------------------------------------------------------
La cascada publicada exige presencia en validación (paso 5) y en prueba
(paso 6). El paso 6 usa información del bloque de prueba: una serie que dejó de
vender antes de abril de 2025 no entra. Tres interruptores permiten medir cuánto
pesa esa decisión; sus valores por defecto reproducen la cascada publicada bit a
bit, y el informe de la cohorte declara siempre el valor de cada uno:

- ``inclusion.exigir_presencia_prueba`` (por defecto ``true``): con ``false``
  el paso 6 no se aplica y queda en la cascada con cero excluidas, declarado.
- ``series.extender_vida_hasta_fin_periodo`` (por defecto ``false``, en
  ``src/series.py``): con ``true`` cada serie vive hasta el fin del período,
  con ceros después de su última venta. La presencia en un bloque se mide
  entonces sobre la VIDA OBSERVADA (primera a última venta), no sobre los ceros
  agregados: si no, toda serie quedaría «presente» por construcción.
- ``inclusion.estado_al_origen`` (por defecto ``todas``): con ``activa`` se
  agrega el paso 7, que conserva sólo las series activas al origen de la prueba
  (venta en los seis meses previos), con la regla de ``clasificar_estados``
  medida al cierre de validación sobre la actividad observada.

Variante A «cohorte al origen»: ``exigir_presencia_prueba: false`` y
``extender_vida_hasta_fin_periodo: true``. Variante B «cohorte al origen, sólo
activas»: la A más ``estado_al_origen: activa``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, gestion_de

BLOQUES = ("entrenamiento", "validacion", "prueba")


@dataclass(frozen=True)
class Particion:
    """Los tres bloques, ya resueltos a rangos de meses concretos."""

    entrenamiento: pd.PeriodIndex
    validacion: pd.PeriodIndex
    prueba: pd.PeriodIndex

    def bloque(self, nombre: str) -> pd.PeriodIndex:
        if nombre not in BLOQUES:
            raise ValueError(f"Bloque desconocido '{nombre}'. Válidos: {BLOQUES}")
        return getattr(self, nombre)

    def resumen(self, mes_inicio_gestion: int) -> pd.DataFrame:
        filas = []
        for nombre in BLOQUES:
            meses = self.bloque(nombre)
            filas.append({
                "bloque": nombre,
                "desde": str(meses[0]),
                "hasta": str(meses[-1]),
                "meses": len(meses),
                "gestiones": (
                    f"{gestion_de(meses[0], mes_inicio_gestion)}"
                    f"–{gestion_de(meses[-1], mes_inicio_gestion)}"
                ),
            })
        return pd.DataFrame(filas)


def particionar(cfg: Config) -> Particion:
    """Resuelve los tres bloques a rangos de meses y verifica su orden."""
    entrenamiento = pd.period_range(
        cfg.periodo.inicio, cfg.particion.fin_entrenamiento, freq="M"
    )
    validacion = pd.period_range(
        cfg.inicio_validacion, cfg.particion.fin_validacion, freq="M"
    )
    prueba = pd.period_range(cfg.inicio_prueba, cfg.periodo.fin, freq="M")

    for nombre, meses in zip(BLOQUES, (entrenamiento, validacion, prueba)):
        if len(meses) == 0:
            raise ValueError(f"El bloque '{nombre}' quedó vacío con los cortes declarados.")

    # RN-2 verificada aquí además de en config.py: es la restricción más fácil
    # de romper por accidente y la más grave.
    if entrenamiento[-1] >= validacion[0]:
        raise ValueError(
            f"Solapamiento entrenamiento/validación: {entrenamiento[-1]} >= {validacion[0]}"
        )
    if validacion[-1] >= prueba[0]:
        raise ValueError(
            f"Solapamiento validación/prueba: {validacion[-1]} >= {prueba[0]}"
        )

    return Particion(entrenamiento=entrenamiento, validacion=validacion, prueba=prueba)


def etiquetar_bloques(panel: pd.DataFrame, particion: Particion) -> pd.DataFrame:
    """Agrega la columna ``bloque`` al panel largo."""
    resultado = panel.copy()
    condiciones = [
        resultado["periodo"] <= particion.entrenamiento[-1],
        resultado["periodo"] <= particion.validacion[-1],
    ]
    resultado["bloque"] = np.select(
        condiciones, ["entrenamiento", "validacion"], default="prueba"
    )
    return resultado


# ---------------------------------------------------------------------------
# Criterios de inclusión y exclusión
# ---------------------------------------------------------------------------

def interruptores_cohorte(cfg: Config) -> dict:
    """Valor de cada interruptor de cohorte, con su clave de configuración."""
    return {
        "inclusion.exigir_presencia_prueba": cfg.inclusion.exigir_presencia_prueba,
        "series.extender_vida_hasta_fin_periodo":
            cfg.series.extender_vida_hasta_fin_periodo,
        "inclusion.estado_al_origen": cfg.inclusion.estado_al_origen,
    }


def cohorte_por_defecto(cfg: Config) -> bool:
    """True si los tres interruptores están en el valor de la corrida publicada."""
    return (
        cfg.inclusion.exigir_presencia_prueba
        and not cfg.series.extender_vida_hasta_fin_periodo
        and cfg.inclusion.estado_al_origen == "todas"
    )


def _presencia_en_bloques(
    panel: pd.DataFrame, particion: Particion, cfg: Config
) -> tuple[set, set]:
    """Series presentes en validación y en prueba.

    Con la vida acotada (por defecto) «presente» es tener filas en el bloque,
    exactamente como en la corrida publicada. Con la vida extendida hasta el fin
    del período todas las series tienen filas en todos los bloques posteriores
    a su nacimiento (ceros), así que la presencia se mide sobre la vida
    OBSERVADA: que el tramo entre la primera y la última venta toque el bloque.
    Con relleno de ceros intermedios las dos lecturas coinciden cuando la vida
    no está extendida.
    """
    if not cfg.series.extender_vida_hasta_fin_periodo:
        en_validacion = set(
            panel.loc[panel["periodo"].isin(particion.validacion), "serie"].unique()
        )
        en_prueba = set(
            panel.loc[panel["periodo"].isin(particion.prueba), "serie"].unique()
        )
        return en_validacion, en_prueba

    positivos = panel.loc[panel["y"] > 0].groupby("serie", observed=True)["periodo"]
    primera = positivos.min()
    ultima = positivos.max()

    def _vida_toca(bloque: pd.PeriodIndex) -> set:
        toca = (primera <= bloque[-1]) & (ultima >= bloque[0])
        return set(primera.index[toca.to_numpy()])

    return _vida_toca(particion.validacion), _vida_toca(particion.prueba)


def activas_al_origen(panel_estado: pd.DataFrame, particion: Particion) -> set:
    """Series activas al origen de la prueba (venta en los seis meses previos).

    Usa la regla de ``clasificar_estados`` medida al cierre de validación, el
    último mes conocido en el origen, sobre la actividad observada.
    """
    estados = clasificar_estados(panel_estado, particion.validacion[-1])
    return set(estados.index[(estados == "activa").to_numpy()])


@dataclass
class InformeCohorte:
    """Diagrama de flujo de la muestra: cuántas series cayó cada criterio."""

    series_iniciales: int = 0
    excluidas_historial: int = 0
    excluidas_ceros: int = 0
    excluidas_volumen: int = 0
    excluidas_sin_entrenamiento: int = 0
    excluidas_sin_validacion: int = 0
    excluidas_sin_prueba: int = 0
    excluidas_no_activas_al_origen: int = 0
    series_finales: int = 0
    umbrales: dict = field(default_factory=dict)
    interruptores: dict = field(default_factory=dict)
    flujo: list[dict] = field(default_factory=list)
    excluidas_aislado: dict = field(default_factory=dict)

    def tabla_flujo(self) -> pd.DataFrame:
        return pd.DataFrame(self.flujo)

    def lineas(self) -> list[str]:
        salida = [
            "Los criterios se miden SOBRE EL BLOQUE DE ENTRENAMIENTO exclusivamente.",
            "",
            "Interruptores de cohorte:",
            *(f"  {clave} = {valor}" for clave, valor in self.interruptores.items()),
            "",
            "Cascada (cada paso cuenta las series que sobrevivieron a los anteriores):",
        ]
        for paso in self.flujo:
            salida.append(
                f"  {paso['paso']:<52} {paso['series']:>7,}"
                + (f"   (−{paso['excluidas']:,})" if paso["excluidas"] else "")
            )
        if self.excluidas_aislado:
            salida += ["", "Efecto aislado de cada criterio sobre el total de series:"]
            for criterio, cuantas in self.excluidas_aislado.items():
                salida.append(f"  {criterio:<52} {cuantas:>7,}")
        salida += [
            "",
            f"  N de la muestra: {self.series_finales:,} series SKU–canal–regional",
        ]
        return salida


def aplicar_criterios_inclusion(
    panel: pd.DataFrame,
    cfg: Config,
    particion: Particion,
    panel_estado: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, InformeCohorte]:
    """Filtra el panel a la cohorte de análisis y documenta cada exclusión.

    Los criterios se aplican en cascada y en orden fijo, de modo que el conteo
    de cada paso es «series que sobrevivieron a los anteriores y cayó este». Se
    reporta además, en columna aparte, cuántas series habría excluido cada
    criterio de forma aislada: el orden de aplicación cambia el reparto de las
    exclusiones pero no el conjunto final, y el lector tiene derecho a ver ambos.

    Args:
        panel: panel largo sobre el que se miden los criterios (en el pipeline,
            el de AJUSTE).
        panel_estado: panel de actividad OBSERVADA para el estado al origen
            (paso 7, sólo con ``inclusion.estado_al_origen: activa``). Si no se
            da, se usa ``panel``.
    """
    informe = InformeCohorte(
        umbrales={
            "historial_minimo_meses": cfg.inclusion.historial_minimo_meses,
            "proporcion_maxima_ceros": cfg.inclusion.proporcion_maxima_ceros,
            "volumen_minimo_unidades": cfg.inclusion.volumen_minimo_unidades,
        },
        interruptores=interruptores_cohorte(cfg),
    )

    entrenamiento = panel.loc[panel["periodo"] <= particion.entrenamiento[-1]]

    todas = pd.Index(sorted(panel["serie"].unique()), name="serie")
    informe.series_iniciales = len(todas)

    # --- Estadísticos por serie, calculados solo con entrenamiento ----------
    agrupado = entrenamiento.groupby("serie", observed=True)["y"]
    meses_train = agrupado.size().reindex(todas, fill_value=0)
    volumen_train = agrupado.sum().reindex(todas, fill_value=0.0)
    ceros_train = (
        entrenamiento.assign(es_cero=lambda d: (d["y"] <= 0).astype(int))
        .groupby("serie", observed=True)["es_cero"].sum()
        .reindex(todas, fill_value=0)
    )
    proporcion_ceros = np.where(meses_train > 0, ceros_train / meses_train.clip(lower=1), 1.0)
    proporcion_ceros = pd.Series(proporcion_ceros, index=todas)

    # Presencia en cada bloque: una serie sin filas en prueba no puede
    # evaluarse, y una sin filas en validación no permite que el motor elija.
    # Con la vida extendida se mide sobre la vida observada (ver el módulo).
    en_validacion, en_prueba = _presencia_en_bloques(panel, particion, cfg)
    vida_extendida = cfg.series.extender_vida_hasta_fin_periodo

    def _paso(nombre: str, sobreviven: pd.Index, antes: pd.Index) -> pd.Index:
        informe.flujo.append({
            "paso": nombre,
            "series": len(sobreviven),
            "excluidas": len(antes) - len(sobreviven),
        })
        return sobreviven

    vivas = todas
    informe.flujo.append({"paso": "0. series construidas", "series": len(vivas),
                          "excluidas": 0})

    antes = vivas
    vivas = vivas[(meses_train.reindex(vivas) > 0).to_numpy()]
    informe.excluidas_sin_entrenamiento = len(antes) - len(vivas)
    _paso("1. con historia en entrenamiento", vivas, antes)

    antes = vivas
    vivas = vivas[
        (meses_train.reindex(vivas) >= cfg.inclusion.historial_minimo_meses).to_numpy()
    ]
    informe.excluidas_historial = len(antes) - len(vivas)
    _paso(
        f"2. historial >= {cfg.inclusion.historial_minimo_meses} meses",
        vivas, antes,
    )

    antes = vivas
    vivas = vivas[
        (proporcion_ceros.reindex(vivas)
         <= cfg.inclusion.proporcion_maxima_ceros).to_numpy()
    ]
    informe.excluidas_ceros = len(antes) - len(vivas)
    _paso(
        f"3. proporción de meses en cero <= {cfg.inclusion.proporcion_maxima_ceros:.2f}",
        vivas, antes,
    )

    antes = vivas
    vivas = vivas[
        (volumen_train.reindex(vivas)
         >= cfg.inclusion.volumen_minimo_unidades).to_numpy()
    ]
    informe.excluidas_volumen = len(antes) - len(vivas)
    _paso(
        f"4. volumen acumulado >= {cfg.inclusion.volumen_minimo_unidades:.0f} unidades",
        vivas, antes,
    )

    antes = vivas
    vivas = pd.Index([s for s in vivas if s in en_validacion], name="serie")
    informe.excluidas_sin_validacion = len(antes) - len(vivas)
    _paso(
        "5. presente en validación (vida observada)" if vida_extendida
        else "5. presente en validación",
        vivas, antes,
    )

    antes = vivas
    if cfg.inclusion.exigir_presencia_prueba:
        vivas = pd.Index([s for s in vivas if s in en_prueba], name="serie")
        informe.excluidas_sin_prueba = len(antes) - len(vivas)
        _paso(
            "6. presente en prueba (vida observada)" if vida_extendida
            else "6. presente en prueba",
            vivas, antes,
        )
    else:
        # El paso queda en la cascada, declarado y sin excluir: quien lea la
        # tabla ve que el criterio existe y que en esta corrida se apagó.
        _paso("6. presente en prueba: no se aplica "
              "(inclusion.exigir_presencia_prueba = false)", vivas, antes)

    if cfg.inclusion.estado_al_origen == "activa":
        activas = activas_al_origen(
            panel if panel_estado is None else panel_estado, particion
        )
        antes = vivas
        vivas = pd.Index([s for s in vivas if s in activas], name="serie")
        informe.excluidas_no_activas_al_origen = len(antes) - len(vivas)
        _paso(
            f"7. activa al origen (venta entre {particion.validacion[-1] - 5} "
            f"y {particion.validacion[-1]})",
            vivas, antes,
        )

    informe.series_finales = len(vivas)
    if informe.series_finales == 0:
        raise ValueError(
            "Los criterios de inclusión dejaron la muestra vacía. Revisá la "
            "sección `inclusion` de config.yaml contra el informe de inspección."
        )

    # Efecto AISLADO de cada criterio sobre el total de series. El orden de
    # aplicación reparte distinto las exclusiones de la cascada pero no cambia
    # el conjunto final; el lector tiene derecho a ver las dos lecturas.
    informe.excluidas_aislado = {
        f"historial < {cfg.inclusion.historial_minimo_meses} meses":
            int((meses_train < cfg.inclusion.historial_minimo_meses).sum()),
        f"ceros > {cfg.inclusion.proporcion_maxima_ceros:.2f}":
            int((proporcion_ceros > cfg.inclusion.proporcion_maxima_ceros).sum()),
        f"volumen < {cfg.inclusion.volumen_minimo_unidades:.0f} unidades":
            int((volumen_train < cfg.inclusion.volumen_minimo_unidades).sum()),
    }

    cohorte = panel.loc[panel["serie"].isin(set(vivas))].reset_index(drop=True)
    return cohorte, informe


# ---------------------------------------------------------------------------
# Clasificación por estados: clasificar y declarar, no descartar en silencio
# ---------------------------------------------------------------------------

def clasificar_estados(
    panel_real: pd.DataFrame, fin_entrenamiento: pd.Period
) -> pd.Series:
    """Estado de actividad de cada serie al cierre de entrenamiento.

    Es el molde de la práctica de planificación de la empresa, medido sobre la
    actividad OBSERVADA (la corrección del evento vive solo en el ajuste):

    - ``insuficiente``: sin ninguna venta positiva en entrenamiento.
    - ``nueva`` / ``reciente``: menos de 6 / 12 meses desde la primera venta.
    - ``descontinuada`` / ``dormida``: 12+ / 6–11 meses sin vender al cierre.
    - ``activa``: vendió dentro de los últimos 6 meses.
    """
    train = panel_real.loc[panel_real["periodo"] <= fin_entrenamiento]
    positivos = train.loc[train["y"] > 0].groupby("serie")["periodo"]
    primera = positivos.min()
    ultima = positivos.max()

    series = pd.Index(sorted(train["serie"].unique()), name="serie")
    vida_meses = pd.Series(
        [(fin_entrenamiento - primera[s]).n + 1 if s in primera.index else 0
         for s in series], index=series,
    )
    sin_venta = pd.Series(
        [(fin_entrenamiento - ultima[s]).n if s in ultima.index else np.inf
         for s in series], index=series,
    )

    estado = np.select(
        [
            vida_meses == 0,
            vida_meses < 6,
            vida_meses < 12,
            sin_venta >= 12,
            sin_venta >= 6,
        ],
        ["insuficiente", "nueva", "reciente", "descontinuada", "dormida"],
        default="activa",
    )
    return pd.Series(estado, index=series, name="estado")


def tabla_fuera_de_muestra(
    panel_real: pd.DataFrame,
    panel_ajuste: pd.DataFrame,
    series_cohorte: set[str],
    cfg: Config,
    particion: Particion,
    valor_por_serie: pd.Series | None = None,
) -> pd.DataFrame:
    """Una fila por serie del panel: estado, si quedó en la muestra y por qué no.

    Es la contracara declarativa de la cascada: ninguna serie sale del estudio
    sin quedar clasificada (estado de actividad) y atribuida (el primer
    criterio de la cascada que la excluyó, medido sobre el panel de AJUSTE,
    exactamente como lo hace ``aplicar_criterios_inclusion``). El peso
    valorizado de cada serie (demanda observada de entrenamiento × costo) hace
    visible cuánto negocio queda fuera y dónde.
    """
    fin_train = particion.entrenamiento[-1]
    entrenamiento = panel_ajuste.loc[panel_ajuste["periodo"] <= fin_train]
    todas = pd.Index(sorted(panel_ajuste["serie"].unique()), name="serie")

    agrupado = entrenamiento.groupby("serie", observed=True)["y"]
    meses = agrupado.size().reindex(todas, fill_value=0)
    volumen = agrupado.sum().reindex(todas, fill_value=0.0)
    ceros = (
        entrenamiento.assign(es_cero=lambda d: (d["y"] <= 0).astype(int))
        .groupby("serie", observed=True)["es_cero"].sum()
        .reindex(todas, fill_value=0)
    )
    proporcion_ceros = pd.Series(
        np.where(meses > 0, ceros / meses.clip(lower=1), 1.0), index=todas
    )
    en_validacion, en_prueba = _presencia_en_bloques(panel_ajuste, particion, cfg)

    # El motivo sigue el ORDEN de la cascada: el primero que falla, ese es.
    condiciones = [
        (meses == 0).to_numpy(),
        (meses < cfg.inclusion.historial_minimo_meses).to_numpy(),
        (proporcion_ceros > cfg.inclusion.proporcion_maxima_ceros).to_numpy(),
        (volumen < cfg.inclusion.volumen_minimo_unidades).to_numpy(),
        np.array([s not in en_validacion for s in todas]),
    ]
    motivos = [
        "sin_historia_en_entrenamiento",
        f"historial<{cfg.inclusion.historial_minimo_meses}m",
        f"ceros>{cfg.inclusion.proporcion_maxima_ceros:.2f}",
        f"volumen<{cfg.inclusion.volumen_minimo_unidades:.0f}u",
        "ausente_en_validacion",
    ]
    if cfg.inclusion.exigir_presencia_prueba:
        condiciones.append(np.array([s not in en_prueba for s in todas]))
        motivos.append("ausente_en_prueba")
    if cfg.inclusion.estado_al_origen == "activa":
        activas = activas_al_origen(panel_real, particion)
        condiciones.append(np.array([s not in activas for s in todas]))
        motivos.append("no_activa_al_origen")
    motivo = pd.Series(
        np.select(condiciones, motivos, default="incluida"), index=todas
    )

    tabla = pd.DataFrame({
        "serie": todas,
        "estado": clasificar_estados(panel_real, fin_train).reindex(todas)
        .fillna("insuficiente"),
        "incluida": [s in series_cohorte for s in todas],
        "motivo_exclusion": motivo.to_numpy(),
        "meses_entrenamiento": meses.to_numpy(),
        "proporcion_ceros": proporcion_ceros.round(4).to_numpy(),
        "volumen_entrenamiento": volumen.round(2).to_numpy(),
    }).reset_index(drop=True)

    if not cohorte_por_defecto(cfg):
        # Sólo en las corridas de sensibilidad (la tabla publicada no cambia):
        # el estado al ORIGEN de la prueba, para leer el resultado por estado.
        tabla.insert(
            2, "estado_al_origen",
            clasificar_estados(panel_real, particion.validacion[-1])
            .reindex(todas).fillna("insuficiente").to_numpy(),
        )

    if valor_por_serie is not None and not valor_por_serie.empty:
        tabla["valor_entrenamiento"] = (
            valor_por_serie.reindex(tabla["serie"]).round(2).to_numpy()
        )
    return tabla


def sensibilidad_umbrales(
    panel_ajuste: pd.DataFrame,
    particion: Particion,
    valor_por_serie: pd.Series | None = None,
    historial: tuple[int, ...] = (24, 36, 48),
    ceros: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    volumen: tuple[float, ...] = (0.0, 50.0, 100.0),
    cfg: Config | None = None,
    panel_estado: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """N de la muestra (y cobertura valorizada) para cada combinación de umbrales.

    Muestra que la muestra declarada no es un punto arbitrario: el lector ve
    cómo se movería el N —y cuánta demanda valorizada de ENTRENAMIENTO
    cubriría— bajo umbrales alternativos. Los estadísticos por serie se miden
    UNA vez con la misma semántica de ``aplicar_criterios_inclusion`` (la
    consistencia la garantiza una prueba: la fila de los umbrales base debe
    reproducir exactamente el N de la cascada).

    Con ``cfg`` se respetan los interruptores de cohorte (presencia sobre la
    vida observada, paso 6 apagado, estado al origen con ``panel_estado``);
    sin ``cfg`` rige la cascada publicada.
    """
    fin_train = particion.entrenamiento[-1]
    entrenamiento = panel_ajuste.loc[panel_ajuste["periodo"] <= fin_train]
    todas = pd.Index(sorted(panel_ajuste["serie"].unique()), name="serie")

    agrupado = entrenamiento.groupby("serie", observed=True)["y"]
    meses = agrupado.size().reindex(todas, fill_value=0)
    volumen_train = agrupado.sum().reindex(todas, fill_value=0.0)
    ceros_train = (
        entrenamiento.assign(es_cero=lambda d: (d["y"] <= 0).astype(int))
        .groupby("serie", observed=True)["es_cero"].sum()
        .reindex(todas, fill_value=0)
    )
    proporcion = pd.Series(
        np.where(meses > 0, ceros_train / meses.clip(lower=1), 1.0), index=todas
    )
    if cfg is None:
        en_validacion = set(
            panel_ajuste.loc[panel_ajuste["periodo"].isin(particion.validacion), "serie"]
        )
        en_prueba = set(
            panel_ajuste.loc[panel_ajuste["periodo"].isin(particion.prueba), "serie"]
        )
        presencia = pd.Series(
            [s in en_validacion and s in en_prueba for s in todas], index=todas
        )
    else:
        en_validacion, en_prueba = _presencia_en_bloques(panel_ajuste, particion, cfg)
        exigir_prueba = cfg.inclusion.exigir_presencia_prueba
        activas = (
            activas_al_origen(
                panel_ajuste if panel_estado is None else panel_estado, particion
            )
            if cfg.inclusion.estado_al_origen == "activa" else None
        )
        presencia = pd.Series(
            [s in en_validacion
             and (not exigir_prueba or s in en_prueba)
             and (activas is None or s in activas)
             for s in todas],
            index=todas,
        )

    valor_total = (
        float(valor_por_serie.sum())
        if valor_por_serie is not None and not valor_por_serie.empty else None
    )

    filas = []
    for h in historial:
        for c in ceros:
            for v in volumen:
                incluida = (
                    (meses >= max(h, 1)) & (proporcion <= c)
                    & (volumen_train >= v) & presencia
                )
                fila = {
                    "historial_minimo_meses": h,
                    "proporcion_maxima_ceros": c,
                    "volumen_minimo_unidades": v,
                    "n_series": int(incluida.sum()),
                }
                if valor_total:
                    cubierto = float(
                        valor_por_serie.reindex(todas[incluida]).sum()
                    )
                    fila["pct_demanda_train_valorizada"] = round(
                        100.0 * cubierto / valor_total, 2
                    )
                filas.append(fila)
    return pd.DataFrame(filas)
