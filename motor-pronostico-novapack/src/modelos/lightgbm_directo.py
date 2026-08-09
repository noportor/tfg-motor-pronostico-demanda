"""LightGBM DIRECTO por horizonte — el ataque a la recursión (ML2-V4).

El brazo global del estudio proyecta el horizonte de forma RECURSIVA: realimenta
sus propias predicciones y arrastra sus errores mes a mes. Este brazo entrena
**un modelo por horizonte** (h = 1 … H): el modelo h aprende a predecir la
demanda de un mes usando las features construidas h−1 meses antes.

La elegancia del enfoque directo en este pipeline: como toda feature usa solo
información ANTERIOR a su mes (RN-3), las features del mes ``origen+1`` son
computables con la realidad disponible en el origen. Cada horizonte se predice
con features REALES — sin simulación, sin realimentación, sin fuga.

Decisiones declaradas
---------------------
- **Hiperparámetros del campeón** (los declarados de la búsqueda de LightGBM),
  con el número de árboles fijado POR HORIZONTE mediante parada temprana contra
  los pares de validación de ese horizonte — el mismo uso legítimo de
  validación que la interfaz común concede al brazo base.
- **A un paso es el mismo modelo que el brazo base** (h=1 comparte pares,
  parámetros y semilla): su valor diferencial vive en la curva D(h), donde
  reemplaza la recursión por predicción directa.
- **Sin re-entrenamiento por origen** (a diferencia del brazo recursivo, que
  replica el re-entrenamiento mensual del sistema en producción): parámetros
  congelados del ajuste único, la asimetría se declara.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..features import categoricas_de, nombres_de_features


class ModeloLightGBMDirecto:
    """Doce boosters, uno por horizonte, sobre las features del pipeline."""

    nombre = "lightgbm_directo"

    def __init__(self, cfg: Config, tabla: pd.DataFrame):
        self.cfg = cfg
        self.tabla = tabla
        self.columnas = nombres_de_features(cfg)
        self.categoricas = [c for c in categoricas_de(cfg) if c in self.columnas]
        mh = getattr(cfg, "multihorizonte", None)
        self.horizonte = int(mh.horizonte_maximo) if mh is not None else 12
        self.boosters: dict[int, object] = {}
        self._y = tabla.set_index(["serie", "periodo"])["y"]

    # -- pares (features en p, objetivo en p+h-1) ----------------------------

    def _pares(self, h: int) -> pd.DataFrame:
        """La tabla de features con el objetivo del horizonte ``h``.

        ``y_objetivo(fila en p) = y(p + h − 1)``: el modelo h aprende a mirar
        h−1 meses más lejos que sus features. h=1 reproduce el brazo base.
        """
        objetivo = self.tabla["periodo"] + (h - 1)
        indice = pd.MultiIndex.from_arrays(
            [self.tabla["serie"], objetivo], names=["serie", "periodo"]
        )
        return self.tabla.assign(
            mes_objetivo=objetivo,
            y_objetivo=self._y.reindex(indice).to_numpy(),
        )

    # -- interfaz común ------------------------------------------------------

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        import lightgbm as lgb

        fin_entrenamiento = entrenamiento.index[-1]
        meses_validacion = set(validacion.index)

        crudos = self.cfg.modelos["lightgbm"]
        parametros = {
            k: v for k, v in crudos.items()
            if k not in ("num_boost_round", "early_stopping_rounds")
        }
        num_boost_round = int(crudos["num_boost_round"])
        early_stopping = int(crudos["early_stopping_rounds"])

        for h in range(1, self.horizonte + 1):
            pares = self._pares(h).dropna(subset=["y_objetivo"])
            entrena = pares.loc[pares["mes_objetivo"] <= fin_entrenamiento]
            valida = pares.loc[pares["mes_objetivo"].isin(meses_validacion)]
            if entrena.empty or valida.empty:
                raise ValueError(
                    f"Horizonte {h}: sin pares de entrenamiento o validación."
                )
            dtrain = lgb.Dataset(
                entrena[self.columnas], label=entrena["y_objetivo"],
                categorical_feature=self.categoricas, free_raw_data=False,
            )
            dval = lgb.Dataset(
                valida[self.columnas], label=valida["y_objetivo"],
                categorical_feature=self.categoricas, reference=dtrain,
                free_raw_data=False,
            )
            self.boosters[h] = lgb.train(
                parametros, dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dval], valid_names=["validacion"],
                callbacks=[
                    lgb.early_stopping(early_stopping, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

    def _predecir_horizonte(self, filas: pd.DataFrame, h: int) -> pd.Series:
        booster = self.boosters[h]
        predicho = booster.predict(
            filas[self.columnas], num_iteration=booster.best_iteration
        )
        return pd.Series(
            np.maximum(np.asarray(predicho, dtype=float), 0.0),
            index=pd.Index(filas["serie"].astype(str), name="serie"),
        )

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Pronóstico a un paso: el booster h=1 (idéntico al brazo base)."""
        if not self.boosters:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")
        meses = set(datos.index)
        series = set(datos.columns)
        filas = self.tabla.loc[
            self.tabla["periodo"].isin(meses) & self.tabla["serie"].isin(series)
        ]
        booster = self.boosters[1]
        predicho = booster.predict(
            filas[self.columnas], num_iteration=booster.best_iteration
        )
        largo = pd.DataFrame({
            "serie": filas["serie"].to_numpy(),
            "periodo": filas["periodo"].to_numpy(),
            "prediccion": np.asarray(predicho, dtype=float),
        })
        ancho = largo.pivot(index="periodo", columns="serie", values="prediccion")
        return ancho.reindex(index=datos.index, columns=datos.columns)

    # -- protocolo multihorizonte -------------------------------------------

    def proyectar_directo(self, historia: pd.DataFrame, horizonte: int) -> pd.DataFrame:
        """El abanico directo: cada horizonte con su booster, features REALES.

        Las features del mes ``origen+1`` usan solo información ≤ origen
        (RN-3): la fila del pipeline sirve tal cual para todos los horizontes.
        """
        if not self.boosters:
            raise RuntimeError("Hay que llamar a ajustar() antes de proyectar.")
        origen = historia.index[-1]
        meses = pd.period_range(origen + 1, origen + horizonte, freq="M")
        filas = self.tabla.loc[self.tabla["periodo"] == origen + 1]
        salida = pd.DataFrame(
            np.nan, index=meses, columns=historia.columns, dtype=float
        )
        if filas.empty:
            return salida
        for h in range(1, min(horizonte, self.horizonte) + 1):
            valores = self._predecir_horizonte(filas, h)
            salida.loc[origen + h] = valores.reindex(historia.columns).to_numpy()
        return salida
