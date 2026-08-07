"""Evaluación multihorizonte con orígenes rodantes — la segunda mitad de la RN-4.

Por qué existe
--------------
El protocolo principal del experimento evalúa a UN paso: cada mes se pronostica
con el valor real del mes anterior. Es la comparación justa entre brazos y es el
caso h=1 de este módulo. Pero el objetivo operativo del pronóstico en NOVAPACK
es el horizonte de PLANIFICACIÓN: la corrida mensual del sistema en producción
proyecta 18 meses hacia adelante, y las decisiones de compra y producción se
toman sobre esos meses lejanos.

A un paso, un modelo que proyecta línea recta (Naïve, promedio móvil, suavizado
exponencial simple) nunca queda expuesto: se re-engancha a la realidad cada mes.
A dieciocho, la raya horizontal ignora la campaña escolar. Ese contraste es
invisible para el protocolo a un paso y es exactamente lo que este módulo mide.

La RN-4 lo anticipa: «si en algún momento se agrega evaluación multihorizonte,
debe aplicarse por igual a todos los modelos y reportarse por separado». Ambas
condiciones se cumplen acá: mismos orígenes y misma mecánica de proyección para
todos los brazos, y salidas separadas de la Tabla 8 (que sigue siendo a un paso).

El protocolo
------------
- **Orígenes rodantes.** La operación real re-pronostica TODO el horizonte cada
  mes. El experimento replica eso: cada mes del bloque de prueba es un origen y
  desde cada uno se proyecta hasta ``horizonte_maximo`` o hasta donde haya
  realidad con la que comparar. h=1 agrega tantos orígenes como meses de prueba;
  el horizonte máximo, uno solo (el conteo se reporta).
- **La historia de cada origen es todo lo anterior al origen**, incluidos los
  meses de prueba ya transcurridos. No hay fuga: ningún pronóstico usa el futuro
  del mes que predice — es literalmente lo que el planificador sabe ese día.
- **Parámetros congelados, estado que avanza (RN-2).** Los parámetros de cada
  modelo se estimaron una sola vez con entrenamiento; lo que avanza con el
  origen es el estado (niveles, ventanas), alimentado con los meses observados.
  LightGBM es la excepción declarada: el sistema en producción re-entrena el
  modelo global en cada corrida mensual, así que aquí se re-entrena por origen
  con la historia disponible y el número de árboles CONGELADO del ajuste base
  (la parada temprana no se repite: re-mirar validación en cada origen sería
  re-optimizar, y mirar prueba sería fuga).
- **La selección del motor no se re-decide.** La elección por serie se tomó
  mirando solo validación (RN-2) y se mantiene fija en todos los orígenes.
- **Proyección recursiva** — la mecánica del sistema en producción: el
  pronóstico de cada mes se realimenta como un dato para pronosticar el
  siguiente. Con esa recursión el suavizado exponencial reproduce exactamente
  su forma cerrada (línea plana en el nivel) y Holt-Winters reproduce
  ``ℓ + h·b + s`` — las pruebas lo verifican contra la fórmula a mano. Croston
  es la excepción: sus estados solo se actualizan con demanda observada, de
  modo que su pronóstico multihorizonte es su constante congelada en el origen.
- **Agregación valorizada por horizonte.** Las unidades difieren entre SKUs:
  la única curva de degradación agregable entre productos es la valorizada,
  ``D(h) = WMAPE_val(h) + |Bias_val(h)|``, agrupando todos los orígenes que
  alcanzan ese horizonte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .features import construir_features, nombres_de_features
from .modelos.base import truncar_en_cero


# ---------------------------------------------------------------------------
# Proyección por modelo
# ---------------------------------------------------------------------------

def _extender_indice(historia: pd.DataFrame, hasta: pd.Period) -> pd.DataFrame:
    """Agrega al panel los meses futuros (en NaN) hasta ``hasta`` inclusive."""
    indice = pd.period_range(historia.index[0], hasta, freq="M")
    return historia.reindex(indice)


def proyectar_recursivo(
    modelo, historia: pd.DataFrame, horizonte: int
) -> pd.DataFrame:
    """Proyecta ``horizonte`` meses realimentando las propias predicciones.

    Es la mecánica del sistema en producción: el pronóstico del mes h entra al
    panel como si fuera un dato y alimenta el pronóstico del mes h+1. Se trunca
    en cero ANTES de realimentar: una predicción negativa no puede propagarse
    como demanda.
    """
    origen = historia.index[-1]
    extendido = _extender_indice(historia, origen + horizonte)
    for h in range(1, horizonte + 1):
        mes = origen + h
        paso = modelo.predecir(extendido)
        extendido.loc[mes] = truncar_en_cero(paso.loc[[mes]]).iloc[0]
    return extendido.loc[extendido.index > origen]


def proyectar_constante(
    modelo, historia: pd.DataFrame, horizonte: int
) -> pd.DataFrame:
    """Proyección plana: el pronóstico a un paso en el origen, repetido.

    Es la proyección multihorizonte correcta de Croston: sus estados solo se
    actualizan con demanda observada, así que realimentar predicciones los
    contaminaría (el cociente z/p derivaría sin haber visto ningún pedido).
    """
    origen = historia.index[-1]
    extendido = _extender_indice(historia, origen + 1)
    paso = truncar_en_cero(modelo.predecir(extendido).loc[[origen + 1]])
    meses = pd.period_range(origen + 1, origen + horizonte, freq="M")
    return pd.DataFrame(
        np.repeat(paso.to_numpy(dtype=float), horizonte, axis=0),
        index=meses, columns=historia.columns,
    )


class ProyectorLightGBM:
    """Proyección recursiva del brazo global, con re-entrenamiento por origen.

    Dos decisiones que replican al sistema en producción sin heredar sus fugas:

    - **Re-entrenamiento por origen** con ``num_boost_round`` congelado del
      ajuste base. La tabla de entrenamiento de cada origen es el recorte
      ``periodo <= origen`` de la tabla completa: como toda feature usa solo
      información anterior a su mes (RN-3, garantizado por
      ``tests/test_features.py``), recortar equivale a reconstruir con la
      historia disponible en el origen.
    - **Features del futuro desde la simulación.** Los rezagos de los meses
      proyectados se calculan sobre la historia real MÁS las predicciones ya
      realimentadas, nunca sobre los valores reales del futuro. Para eso la
      tabla de predicción se reconstruye con ``construir_features`` en cada
      paso: usar la tabla del pipeline (construida con la realidad completa)
      sería fuga a partir de h=2.
    """

    def __init__(self, cfg: Config, cohorte_ajustada: pd.DataFrame, modelo_base):
        self.cfg = cfg
        self.base = modelo_base
        self.columnas = nombres_de_features(cfg)
        self.categoricas = [c for c in cfg.features.categoricas if c in self.columnas]
        crudos = cfg.modelos["lightgbm"]
        self.parametros = {
            k: v for k, v in crudos.items()
            if k not in ("num_boost_round", "early_stopping_rounds")
        }
        contexto = ["serie", "sku", "canal", "regional", "periodo", "y"]
        self.largo = cohorte_ajustada[contexto].copy()
        self.identidad = (
            self.largo.drop_duplicates("serie")[["serie", "sku", "canal", "regional"]]
            .reset_index(drop=True)
        )
        self.reentrenos = 0
        self._cache_origen: pd.Period | None = None
        self._cache_booster = None

    def _booster_para(self, origen: pd.Period, reentrenar: bool):
        """Devuelve (booster, num_iteration) para el origen dado."""
        if not reentrenar:
            return self.base.booster, int(self.base.booster.best_iteration)
        if self._cache_origen == origen:
            return self._cache_booster, None
        import lightgbm as lgb

        filas = self.base.tabla.loc[self.base.tabla["periodo"] <= origen]
        conjunto = lgb.Dataset(
            filas[self.columnas], label=filas["y"],
            categorical_feature=self.categoricas, free_raw_data=False,
        )
        booster = lgb.train(
            self.parametros, conjunto,
            num_boost_round=int(self.base.mejor_iteracion),
        )
        self.reentrenos += 1
        self._cache_origen, self._cache_booster = origen, booster
        return booster, None

    def proyectar(
        self, origen: pd.Period, horizonte: int, reentrenar: bool
    ) -> pd.DataFrame:
        booster, num_iteracion = self._booster_para(origen, reentrenar)
        extendido = self.largo.loc[self.largo["periodo"] <= origen]
        salida: dict[pd.Period, pd.Series] = {}

        for h in range(1, horizonte + 1):
            mes = origen + h
            nuevas = self.identidad.assign(periodo=mes, y=np.nan)
            tabla = construir_features(
                pd.concat([extendido, nuevas], ignore_index=True), self.cfg
            )
            filas = tabla.loc[tabla["periodo"] == mes]
            predicho = booster.predict(
                filas[self.columnas], num_iteration=num_iteracion
            )
            predicho = np.maximum(np.asarray(predicho, dtype=float), 0.0)
            valores = pd.Series(
                predicho, index=pd.Index(filas["serie"].astype(str), name="serie")
            )
            salida[mes] = valores
            extendido = pd.concat(
                [extendido, nuevas.assign(y=nuevas["serie"].map(valores).to_numpy())],
                ignore_index=True,
            )

        proyeccion = pd.DataFrame(salida).T
        proyeccion.index = pd.PeriodIndex(proyeccion.index, freq="M")
        return proyeccion


# ---------------------------------------------------------------------------
# Agregación valorizada
# ---------------------------------------------------------------------------

def agregado_valorizado(obs: pd.DataFrame, claves: list[str]) -> pd.DataFrame:
    """Suite valorizada sobre observaciones largas, agrupada por ``claves``.

    ``obs`` debe traer las columnas ``y_real``, ``y_pred``, ``costo``, ``serie``
    y ``origen``. Mismas definiciones que ``metricas.suite_valorizada``:
    WMAPE_val, Bias_val y D sobre demanda y error llevados a dinero. Los grupos
    sin demanda valorizada positiva quedan en NaN (visibles, no descartados).
    """
    tabla = obs.assign(
        demanda_val=obs["costo"] * obs["y_real"],
        error_val=obs["costo"] * (obs["y_pred"] - obs["y_real"]).abs(),
        pred_val=obs["costo"] * obs["y_pred"],
    )
    grupos = tabla.groupby(claves, observed=True).agg(
        demanda=("demanda_val", "sum"),
        error=("error_val", "sum"),
        pred=("pred_val", "sum"),
        n_observaciones=("y_real", "size"),
        n_series=("serie", "nunique"),
        n_origenes=("origen", "nunique"),
    ).reset_index()

    con_demanda = grupos["demanda"] > 0
    grupos["wmape_val"] = np.where(
        con_demanda, 100.0 * grupos["error"] / grupos["demanda"], np.nan
    )
    grupos["bias_val"] = np.where(
        con_demanda,
        100.0 * (grupos["pred"] - grupos["demanda"]) / grupos["demanda"],
        np.nan,
    )
    grupos["D"] = grupos["wmape_val"] + grupos["bias_val"].abs()
    return grupos.drop(columns=["pred"])


# ---------------------------------------------------------------------------
# El protocolo completo
# ---------------------------------------------------------------------------

@dataclass
class InformeMultihorizonte:
    origenes: list[str] = field(default_factory=list)
    horizonte_maximo: int = 0
    modelos: list[str] = field(default_factory=list)
    observaciones: int = 0
    observaciones_con_costo: int = 0
    reentrenos_lightgbm: int = 0
    respaldos: dict[str, int] = field(default_factory=dict)
    sin_pronostico: int = 0
    d_global: dict[str, float] = field(default_factory=dict)

    def lineas(self) -> list[str]:
        cobertura = (
            100.0 * self.observaciones_con_costo / self.observaciones
            if self.observaciones else float("nan")
        )
        salida = [
            f"Orígenes rodantes                : {len(self.origenes)} "
            f"({self.origenes[0]} .. {self.origenes[-1]})" if self.origenes else
            "Orígenes                          : 0",
            f"Horizonte máximo                 : {self.horizonte_maximo} meses",
            f"Observaciones (origen, serie, h) : {self.observaciones:,} "
            f"({cobertura:.1f} % con costo)",
            f"Re-entrenamientos de LightGBM    : {self.reentrenos_lightgbm}",
        ]
        con_respaldo = {m: n for m, n in self.respaldos.items() if n}
        if con_respaldo:
            salida.append(f"Respaldos aplicados               : {con_respaldo}")
        if self.d_global:
            salida.append("D global (todos los horizontes, menor es mejor):")
            for modelo, d in sorted(self.d_global.items(), key=lambda kv: kv[1]):
                salida.append(f"   {modelo:<20} {d:7.1f} %")
        return salida


@dataclass
class ResultadoMultihorizonte:
    resumen_horizonte: pd.DataFrame
    resumen_origen: pd.DataFrame
    informe: InformeMultihorizonte


def _origenes_de(cfg: Config) -> list[pd.Period]:
    """Los orígenes del protocolo. El primero es SIEMPRE el cierre de
    validación: un origen anterior evaluaría sobre meses que el motor usó para
    seleccionar (RN-2)."""
    if cfg.multihorizonte.origenes == "fijo":
        return [cfg.particion.fin_validacion]
    return list(
        pd.period_range(cfg.particion.fin_validacion, cfg.periodo.fin - 1, freq="M")
    )


def evaluar(
    cfg: Config,
    modelos: dict[str, object],
    elegido_por_serie: pd.Series,
    panel_ajuste: pd.DataFrame,
    panel_real: pd.DataFrame,
    panel_entrenamiento: pd.DataFrame,
    cohorte_ajustada: pd.DataFrame,
    costo_por_serie: pd.Series,
) -> ResultadoMultihorizonte:
    """Corre el protocolo multihorizonte completo y agrega la curva D(h).

    Args:
        modelos: los brazos ya AJUSTADOS del paso 7 (parámetros congelados).
        elegido_por_serie: la selección del motor, decidida en validación.
        panel_ajuste: panel ancho de AJUSTE (con el tratamiento del evento);
            es la historia que alimenta las proyecciones, igual que en el
            protocolo a un paso.
        panel_real: panel ancho observado; la evaluación compara SIEMPRE
            contra él (RN-1).
        costo_por_serie: maestro de costos; sin él no hay agregación válida
            entre SKUs y el llamador debe saltarse esta etapa declarándolo.
    """
    mh = cfg.multihorizonte
    origenes = _origenes_de(cfg)
    if origenes[0] < cfg.particion.fin_validacion:
        raise ValueError(
            f"El primer origen ({origenes[0]}) es anterior al cierre de "
            f"validación ({cfg.particion.fin_validacion}): la evaluación "
            "pisaría la ventana de selección del motor (RN-2)."
        )

    media_entrenamiento = panel_entrenamiento.mean(axis=0, skipna=True)
    proyector_lgbm = (
        ProyectorLightGBM(cfg, cohorte_ajustada, modelos["lightgbm"])
        if "lightgbm" in modelos else None
    )

    informe = InformeMultihorizonte(
        origenes=[str(o) for o in origenes],
        horizonte_maximo=mh.horizonte_maximo,
        modelos=sorted(modelos) + ["motor"],
    )
    respaldos: dict[str, int] = {nombre: 0 for nombre in informe.modelos}
    filas_obs: list[pd.DataFrame] = []
    columnas = np.asarray(panel_ajuste.columns, dtype=object)

    for origen in origenes:
        horizonte = min(mh.horizonte_maximo, (cfg.periodo.fin - origen).n)
        historia = panel_ajuste.loc[panel_ajuste.index <= origen]
        meses = pd.period_range(origen + 1, origen + horizonte, freq="M")
        real_futuro = panel_real.reindex(index=meses)
        ultimo_observado = historia.ffill().iloc[-1]

        proyecciones: dict[str, pd.DataFrame] = {}
        for nombre, modelo in modelos.items():
            if nombre == "lightgbm":
                cruda = proyector_lgbm.proyectar(
                    origen, horizonte, reentrenar=mh.reentrenar_lightgbm
                )
            elif nombre == "croston":
                cruda = proyectar_constante(modelo, historia, horizonte)
            else:
                cruda = proyectar_recursivo(modelo, historia, horizonte)

            cruda = cruda.reindex(index=meses, columns=panel_ajuste.columns)
            respaldos[nombre] += int(
                (cruda.isna() & real_futuro.notna()).to_numpy().sum()
            )
            # Misma cascada declarada que el protocolo a un paso, adaptada al
            # horizonte: primero el último valor observado en el origen (el
            # Naïve multihorizonte), después la media de entrenamiento.
            completa = cruda.fillna(ultimo_observado).fillna(media_entrenamiento)
            proyecciones[nombre] = truncar_en_cero(completa)

        # El motor aplica, por serie, la proyección del modelo que ganó en
        # validación. La decisión NO se re-toma en ningún origen.
        motor = pd.DataFrame(
            np.nan, index=meses, columns=panel_ajuste.columns, dtype=float
        )
        for nombre, grupo in elegido_por_serie.groupby(elegido_por_serie, sort=True):
            if nombre not in proyecciones:
                continue
            presentes = [s for s in grupo.index if s in motor.columns]
            motor[presentes] = proyecciones[nombre][presentes]
        proyecciones["motor"] = motor

        y = real_futuro.to_numpy(dtype=float)
        for nombre, proyeccion in proyecciones.items():
            p = proyeccion.to_numpy(dtype=float)
            comparable = ~np.isnan(y) & ~np.isnan(p)
            informe.sin_pronostico += int((~np.isnan(y) & np.isnan(p)).sum())
            fila_mes, fila_serie = np.where(comparable)
            filas_obs.append(pd.DataFrame({
                "modelo": nombre,
                "origen": str(origen),
                "horizonte": fila_mes + 1,   # meses consecutivos desde origen+1
                "serie": columnas[fila_serie],
                "y_real": y[comparable],
                "y_pred": p[comparable],
            }))

    obs = pd.concat(filas_obs, ignore_index=True)
    obs["costo"] = obs["serie"].map(costo_por_serie)
    con_costo = obs.dropna(subset=["costo"])

    informe.observaciones = len(obs)
    informe.observaciones_con_costo = len(con_costo)
    informe.respaldos = respaldos
    informe.reentrenos_lightgbm = (
        proyector_lgbm.reentrenos if proyector_lgbm else 0
    )

    resumen_horizonte = agregado_valorizado(con_costo, ["modelo", "horizonte"])
    resumen_origen = agregado_valorizado(con_costo, ["modelo", "origen"])
    global_ = agregado_valorizado(con_costo, ["modelo"])
    informe.d_global = {
        str(fila["modelo"]): float(fila["D"])
        for _, fila in global_.iterrows()
        if not np.isnan(fila["D"])
    }

    return ResultadoMultihorizonte(
        resumen_horizonte=resumen_horizonte,
        resumen_origen=resumen_origen,
        informe=informe,
    )
