"""LightGBM — modelo global sobre todas las series.

Es el único brazo *global* del experimento: un solo modelo aprende de las miles
de series a la vez, en lugar de ajustar un modelo por serie. Ahí está su ventaja
potencial —una serie con poca historia se beneficia del patrón estacional que el
modelo aprendió en el resto del portafolio— y también su principal limitación a
declarar: sus números no son comparables entre subconjuntos distintos de series,
porque entrenar con 3.000 series no es lo mismo que entrenar con 6.000.

Decisiones que se justifican en la memoria
------------------------------------------
- **Objetivo ``regression_l1``.** Optimiza el error absoluto, que es la métrica
  principal del estudio. Con el objetivo cuadrático por defecto el modelo
  perseguiría los picos y el MAE reportado sería peor que el que puede lograr.
- **Parada temprana contra validación.** Es el único uso legítimo del bloque de
  validación por parte de un modelo, y por eso la interfaz común lo recibe. La
  parada temprana fija el número de árboles; no toca el bloque de prueba.
- **Un solo hilo y modo determinista.** Con varios hilos el orden de reducción
  en punto flotante varía entre corridas y el modelo no es reproducible bit a
  bit, que es lo que exige la RN-5. Se paga en tiempo de entrenamiento.
- **Predicción a un paso.** El modelo se entrena una vez y sus parámetros quedan
  congelados; lo que avanza mes a mes son las features, construidas con los
  valores REALES hasta el mes anterior. Es exactamente el mismo protocolo que
  siguen los demás brazos (RN-4). La proyección multihorizonte (recursiva, con
  re-entrenamiento por origen) vive en ``src/multihorizonte.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..features import categoricas_de, nombres_de_features


class ModeloLightGBM:
    """Modelo global de gradient boosting sobre el panel completo."""

    nombre = "lightgbm"

    def __init__(self, cfg: Config, tabla: pd.DataFrame):
        """
        Args:
            tabla: salida de ``features.construir_features`` para el panel
                completo. Se inyecta en el constructor porque este brazo es el
                único que necesita la representación larga con features; los
                demás operan directamente sobre el panel ancho.
        """
        self.cfg = cfg
        self.tabla = tabla
        self.columnas = nombres_de_features(cfg)
        self.categoricas = [c for c in categoricas_de(cfg) if c in self.columnas]
        self.booster = None
        self.mejor_iteracion: int | None = None
        self.importancias: pd.DataFrame | None = None

    # -- utilidades ---------------------------------------------------------

    def _filas_de(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Sub-tabla con las filas cuyo mes está en el panel indicado."""
        meses = set(panel.index)
        series = set(panel.columns)
        seleccion = self.tabla["periodo"].isin(meses) & self.tabla["serie"].isin(series)
        return self.tabla.loc[seleccion]

    # -- interfaz común -----------------------------------------------------

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        import lightgbm as lgb

        filas_entrenamiento = self._filas_de(entrenamiento)
        filas_validacion = self._filas_de(validacion)

        if filas_entrenamiento.empty:
            raise ValueError("No hay filas de entrenamiento para LightGBM.")
        if filas_validacion.empty:
            raise ValueError(
                "No hay filas de validación: sin ellas no puede haber parada "
                "temprana y el número de árboles quedaría sin justificar."
            )

        parametros = {
            k: v for k, v in self.cfg.modelos["lightgbm"].items()
            if k not in ("num_boost_round", "early_stopping_rounds")
        }
        num_boost_round = int(self.cfg.modelos["lightgbm"]["num_boost_round"])
        early_stopping = int(self.cfg.modelos["lightgbm"]["early_stopping_rounds"])

        conjunto_entrenamiento = lgb.Dataset(
            filas_entrenamiento[self.columnas],
            label=filas_entrenamiento["y"],
            categorical_feature=self.categoricas,
            free_raw_data=False,
        )
        conjunto_validacion = lgb.Dataset(
            filas_validacion[self.columnas],
            label=filas_validacion["y"],
            categorical_feature=self.categoricas,
            reference=conjunto_entrenamiento,
            free_raw_data=False,
        )

        self.booster = lgb.train(
            parametros,
            conjunto_entrenamiento,
            num_boost_round=num_boost_round,
            valid_sets=[conjunto_validacion],
            valid_names=["validacion"],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        self.mejor_iteracion = int(self.booster.best_iteration)

        self.importancias = (
            pd.DataFrame({
                "feature": self.booster.feature_name(),
                "ganancia": self.booster.feature_importance(importance_type="gain"),
                "divisiones": self.booster.feature_importance(importance_type="split"),
            })
            .sort_values("ganancia", ascending=False)
            .reset_index(drop=True)
        )

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")

        filas = self._filas_de(datos)
        predicho = self.booster.predict(
            filas[self.columnas], num_iteration=self.booster.best_iteration
        )

        largo = pd.DataFrame({
            "serie": filas["serie"].to_numpy(),
            "periodo": filas["periodo"].to_numpy(),
            "prediccion": np.asarray(predicho, dtype=float),
        })
        ancho = largo.pivot(index="periodo", columns="serie", values="prediccion")
        return ancho.reindex(index=datos.index, columns=datos.columns)
