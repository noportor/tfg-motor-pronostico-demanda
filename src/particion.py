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

Los cortes caen en frontera de **gestión fiscal** (NOVAPACK cierra en marzo), no
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
    series_finales: int = 0
    umbrales: dict = field(default_factory=dict)
    flujo: list[dict] = field(default_factory=list)
    excluidas_aislado: dict = field(default_factory=dict)

    def tabla_flujo(self) -> pd.DataFrame:
        return pd.DataFrame(self.flujo)

    def lineas(self) -> list[str]:
        salida = [
            "Los criterios se miden SOBRE EL BLOQUE DE ENTRENAMIENTO exclusivamente.",
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
    panel: pd.DataFrame, cfg: Config, particion: Particion
) -> tuple[pd.DataFrame, InformeCohorte]:
    """Filtra el panel a la cohorte de análisis y documenta cada exclusión.

    Los criterios se aplican en cascada y en orden fijo, de modo que el conteo
    de cada paso es «series que sobrevivieron a los anteriores y cayó este». Se
    reporta además, en columna aparte, cuántas series habría excluido cada
    criterio de forma aislada: el orden de aplicación cambia el reparto de las
    exclusiones pero no el conjunto final, y el lector tiene derecho a ver ambos.
    """
    informe = InformeCohorte(
        umbrales={
            "historial_minimo_meses": cfg.inclusion.historial_minimo_meses,
            "proporcion_maxima_ceros": cfg.inclusion.proporcion_maxima_ceros,
            "volumen_minimo_unidades": cfg.inclusion.volumen_minimo_unidades,
        }
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
    en_validacion = set(
        panel.loc[panel["periodo"].isin(particion.validacion), "serie"].unique()
    )
    en_prueba = set(
        panel.loc[panel["periodo"].isin(particion.prueba), "serie"].unique()
    )

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
    _paso("5. presente en validación", vivas, antes)

    antes = vivas
    vivas = pd.Index([s for s in vivas if s in en_prueba], name="serie")
    informe.excluidas_sin_prueba = len(antes) - len(vivas)
    _paso("6. presente en prueba", vivas, antes)

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
