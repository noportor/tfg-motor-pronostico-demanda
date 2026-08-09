"""Brazos neuronales globales — corrida exploratoria ML2 (post-documento).

Qué son y por qué entran
------------------------
Cuatro arquitecturas de pronóstico neuronal, todas GLOBALES (un modelo aprende
de las miles de series a la vez, igual que LightGBM). Cada una responde una
pregunta que el estudio documentado dejó abierta:

=============  =============================================================
``dlinear``    Control de higiene: un modelo lineal directo. Si una red no le
               gana a esto, la arquitectura no aportó (Zeng et al., 2023).
``nhits``      Multi-horizonte DIRECTO y liviano: ataca la degradación del
               pronóstico recursivo, que es la debilidad medida de LightGBM
               en la curva D(h).
``deepar``     RNN autorregresiva probabilística: ¿una verosimilitud de
               conteos gana donde Tweedie ganó? (Salinas et al., 2020).
``tft``        Temporal Fusion Transformer: el transformer diseñado para
               covariables; en esta V1 corre univariante y su incorporación
               de exógenas queda declarada como siguiente paso (Lim, 2021).
=============  =============================================================

Decisiones del protocolo, declaradas
------------------------------------
- **Un solo entrenamiento por brazo** sobre el bloque de entrenamiento, con
  parada temprana contra validación — la MISMA excepción que la interfaz común
  ya concede a LightGBM (es el único uso legítimo de validación: fija cuándo
  parar, no qué predecir).
- **Entrenados con horizonte 12 (directo).** El pronóstico a un paso del
  protocolo principal es el paso h=1 del abanico directo emitido desde el mes
  anterior: historia real hasta t−1, predicción de t. Sin recursión.
- **Multihorizonte por proyección DIRECTA** (``proyectar_directo``): desde el
  origen se emite el abanico h=1..H de una sola vez. Parámetros congelados del
  ajuste único; el estado que avanza es la ventana de historia observada
  (RN-2: sin re-entrenamiento por origen — se declara la asimetría con
  LightGBM, que replica el re-entrenamiento del sistema en producción).
- **V1 univariante.** Sin exógenas: la comparación mide arquitectura, no
  ingeniería de variables. La V2 con las exógenas del pipeline queda anotada.
- **Predicciones solo donde se evalúa.** Los brazos emiten pronóstico para
  validación y prueba; los meses de entrenamiento quedan en NaN y la cascada
  de respaldo del pipeline los completa (no participan de ninguna métrica).
- **Semillas fijas.** El entrenamiento en GPU no es reproducible bit a bit
  entre hardware distinto (se declara, RN-5); la semilla fija la secuencia de
  inicialización y muestreo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

_FRECUENCIA = "MS"  # mes calendario, anclado a inicio de mes


def _a_largo(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel ancho (Period × serie) -> formato largo de neuralforecast.

    Los NaN (fuera de la vida de la serie) se descartan: para una red global
    «no existía el producto» no es un cero observado.
    """
    largo = panel.stack().rename("y").reset_index()
    largo.columns = ["ds", "unique_id", "y"]
    # pandas 3: stack() conserva los NaN — el descarte debe ser explícito.
    largo = largo.dropna(subset=["y"])
    largo["ds"] = largo["ds"].dt.to_timestamp()
    return largo[["unique_id", "ds", "y"]].sort_values(
        ["unique_id", "ds"], kind="stable"
    ).reset_index(drop=True)


def _pivotear(pred: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Salida larga de neuralforecast -> panel ancho con PeriodIndex mensual."""
    ancho = pred.pivot(index="ds", columns="unique_id", values=columna)
    ancho.index = pd.PeriodIndex(ancho.index, freq="M")
    ancho.columns.name = None
    return ancho


class ModeloNeural:
    """Envoltorio común de los brazos de ``neuralforecast``.

    Cumple la interfaz ``Modelo`` (ajustar/predecir) y agrega
    ``proyectar_directo`` para el protocolo multihorizonte.
    """

    def __init__(self, cfg: Config, arquitectura: str):
        self.cfg = cfg
        self.arquitectura = arquitectura
        self.nombre = arquitectura
        comun = dict(cfg.modelos.get("neuronales", {}))
        self.parametros = dict(comun.get(arquitectura, {}))
        self.horizonte = int(comun.get("horizonte", 12))
        self.input_size = int(comun.get("input_size", 24))
        self.semilla = int(comun.get("semilla", 20260408))
        self.max_steps = int(self.parametros.pop("max_steps",
                                                 comun.get("max_steps", 1000)))
        self.perdida = str(self.parametros.pop("perdida",
                                               comun.get("perdida", "mae")))
        self.nf = None
        self._historia_ajuste: pd.DataFrame | None = None

    # -- construcción del modelo --------------------------------------------

    def _perdida(self):
        from neuralforecast.losses.pytorch import (
            MAE, MSE, DistributionLoss,
        )
        if self.perdida == "mae":
            return MAE()
        if self.perdida == "mse":
            return MSE()
        if self.perdida == "negbinomial":
            return DistributionLoss(distribution="NegativeBinomial",
                                    level=[80], return_params=False)
        if self.perdida == "studentt":
            return DistributionLoss(distribution="StudentT",
                                    level=[80], return_params=False)
        raise ValueError(f"Pérdida no reconocida: {self.perdida!r}")

    def _construir(self):
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, TFT, DLinear, DeepAR

        clases = {"nhits": NHITS, "tft": TFT, "dlinear": DLinear,
                  "deepar": DeepAR}
        if self.arquitectura not in clases:
            raise ValueError(
                f"Arquitectura desconocida: {self.arquitectura!r}. "
                f"Disponibles: {sorted(clases)}"
            )
        base = dict(
            h=self.horizonte,
            input_size=self.input_size,
            max_steps=self.max_steps,
            random_seed=self.semilla,
            # Parada temprana contra la validación que recibe ajustar():
            early_stop_patience_steps=int(self.parametros.pop(
                "early_stop_patience_steps", 5)),
            val_check_steps=int(self.parametros.pop("val_check_steps", 50)),
            scaler_type=self.parametros.pop("scaler_type", "robust"),
            # Sin barras de progreso ni logger: el pipeline emite miles de
            # predicciones y las barras ahogan el registro de la corrida.
            enable_progress_bar=False,
            logger=False,
            **self.parametros,
        )
        base["loss"] = self._perdida()
        if self.arquitectura == "deepar":
            # DeepAR trae su propio escalado interno; el parámetro genérico de
            # ventanas no aplica en todas las versiones de la librería.
            base.pop("scaler_type", None)
        modelo = clases[self.arquitectura](**base)
        return NeuralForecast(models=[modelo], freq=_FRECUENCIA)

    # -- interfaz común ------------------------------------------------------

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        """Un solo entrenamiento; validación SOLO para la parada temprana."""
        import torch

        torch.manual_seed(self.semilla)
        np.random.seed(self.semilla)

        historia = pd.concat([entrenamiento, validacion])
        largo = _a_largo(historia)
        self.nf = self._construir()
        # val_size en meses: los últimos meses de cada serie dentro del df
        # entregado — coincide con el bloque de validación del estudio.
        self.nf.fit(df=largo, val_size=len(validacion))
        self._historia_ajuste = historia

    def _predecir_desde(self, historia: pd.DataFrame) -> pd.DataFrame:
        """Abanico directo h=1..H desde el último mes de ``historia``."""
        largo = _a_largo(historia)
        pred = self.nf.predict(df=largo)
        pred = pred.reset_index() if "unique_id" not in pred.columns else pred
        columna = next(
            c for c in pred.columns
            if c not in ("unique_id", "ds")
            and not c.endswith(("-lo-80", "-hi-80", "-median"))
        )
        return _pivotear(pred, columna)

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Pronóstico a un paso: para cada mes evaluado, el paso h=1 del
        abanico emitido desde el mes anterior (historia REAL hasta t−1)."""
        if self.nf is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")

        primer_evaluado = self.cfg.particion.fin_entrenamiento + 1
        meses_evaluados = [m for m in datos.index if m >= primer_evaluado]

        salida = pd.DataFrame(np.nan, index=datos.index, columns=datos.columns)
        for mes in meses_evaluados:
            historia = datos.loc[datos.index < mes]
            abanico = self._predecir_desde(historia)
            if mes in abanico.index:
                fila = abanico.loc[mes].reindex(datos.columns)
                salida.loc[mes] = fila.to_numpy(dtype=float)
        return salida

    # -- protocolo multihorizonte -------------------------------------------

    def proyectar_directo(self, historia: pd.DataFrame, horizonte: int) -> pd.DataFrame:
        """Proyección multihorizonte NATIVA: el abanico directo desde el origen.

        Parámetros congelados del ajuste único; solo avanza la ventana de
        historia observada que condiciona la predicción.
        """
        if self.nf is None:
            raise RuntimeError("Hay que llamar a ajustar() antes de proyectar.")
        abanico = self._predecir_desde(historia)
        # La librería pronostica desde el último mes VIVO de cada serie: una
        # serie apagada antes del origen emite su abanico en meses anteriores.
        # El índice esperado se fija explícito; lo que no caiga ahí queda NaN
        # (la evaluación solo compara donde hay realidad).
        origen = historia.index[-1]
        meses = pd.period_range(origen + 1, origen + horizonte, freq="M")
        return abanico.reindex(index=meses)
