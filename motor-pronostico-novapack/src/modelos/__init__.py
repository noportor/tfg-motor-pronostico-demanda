"""RF-6 — Registro de los modelos del experimento.

Los once brazos, con el papel que juega cada uno en la comparación:

===================  ==========================================================
Brazo                Papel
===================  ==========================================================
``naive_m1``         Línea base estándar y método actual de la empresa
``naive_m1_gf``      Variante con corrección de tendencia usada por la empresa
``ma_2``             Promedio móvil bimestral — método actual
``ma_2_gf``          Idem con corrección de tendencia — método actual
``ma_12``            Promedio móvil anual — método actual y benchmark declarado
``ma_12_gf``         Idem con corrección de tendencia — método actual
``exp_smooth_opt``   Clásico estadístico: suavizado exponencial simple
``holt_winters``     Clásico estadístico: nivel + tendencia + estacionalidad
``croston``          Clásico estadístico para demanda intermitente
``lightgbm``         Aprendizaje automático, modelo global
``motor``            La propuesta: selección por serie sobre validación
===================  ==========================================================

Los seis primeros no son una elección del autor: son los modelos que
Planificación aplica hoy, confirmados contra el sistema en producción. §10 lo
exige explícitamente — el benchmark debe ser el método real, no una versión
debilitada.
"""

from __future__ import annotations

import pandas as pd

from ..config import Config
from .base import Modelo, mae_naive_entrenamiento, truncar_en_cero
from .croston import Croston
from .lightgbm_modelo import ModeloLightGBM
from .motor import Motor
from .naive import Naive, NaiveConCrecimiento
from .promedio_movil import PromedioMovil
from .suavizado import HoltWinters, SuavizadoExponencial

__all__ = [
    "Modelo", "Motor", "construir_modelos",
    "mae_naive_entrenamiento", "truncar_en_cero",
]


def construir_modelos(cfg: Config, tabla: pd.DataFrame) -> dict[str, object]:
    """Instancia los brazos declarados en ``modelos.activos``, salvo el motor.

    El motor se construye aparte porque necesita los pronósticos de los demás.
    """
    disponibles = {
        "naive_m1": lambda: Naive(),
        "naive_m1_gf": lambda: NaiveConCrecimiento(),
        "ma_2": lambda: PromedioMovil(2),
        "ma_2_gf": lambda: PromedioMovil(2, con_crecimiento=True),
        "ma_12": lambda: PromedioMovil(12),
        "ma_12_gf": lambda: PromedioMovil(12, con_crecimiento=True),
        "exp_smooth_opt": lambda: SuavizadoExponencial(
            _rejilla_alfa(cfg.modelos.get("exp_smooth", {}))
        ),
        "holt_winters": lambda: HoltWinters(
            periodo_estacional=int(
                cfg.modelos.get("holt_winters", {}).get("periodo_estacional", 12)
            ),
            alfas=cfg.modelos.get("holt_winters", {}).get("alfa_rejilla"),
            betas=cfg.modelos.get("holt_winters", {}).get("beta_rejilla"),
            gammas=cfg.modelos.get("holt_winters", {}).get("gamma_rejilla"),
        ),
        "croston": lambda: Croston(
            alfas=cfg.modelos.get("croston", {}).get("alfa_rejilla")
        ),
        "lightgbm": lambda: ModeloLightGBM(cfg, tabla),
        # --- Brazos ML2 (corrida exploratoria post-documento) ---------------
        # Imports perezosos: el experimento documentado no depende de torch.
        "dlinear": lambda: _neural(cfg, "dlinear", tabla),
        "nhits": lambda: _neural(cfg, "nhits", tabla),
        "deepar": lambda: _neural(cfg, "deepar", tabla),
        "tft": lambda: _neural(cfg, "tft", tabla),
        "chronos": lambda: _fundacional(cfg, tabla),
    }

    modelos: dict[str, object] = {}
    for nombre in cfg.modelos_activos:
        if nombre == "motor":
            continue
        if nombre not in disponibles:
            raise ValueError(
                f"El modelo '{nombre}' está en modelos.activos pero no existe. "
                f"Disponibles: {sorted(disponibles)}"
            )
        modelos[nombre] = disponibles[nombre]()
    return modelos


def _neural(cfg: Config, arquitectura: str, tabla: pd.DataFrame):
    from .neuronales import ModeloNeural
    return ModeloNeural(cfg, arquitectura, tabla)


def _fundacional(cfg: Config, tabla: pd.DataFrame):
    from .fundacional import ModeloChronos
    return ModeloChronos(cfg, tabla)


def _rejilla_alfa(configuracion: dict) -> list[float]:
    """Rejilla de alfa del suavizado exponencial, a partir de min/max/paso."""
    minimo = float(configuracion.get("alfa_min", 0.05))
    maximo = float(configuracion.get("alfa_max", 0.95))
    paso = float(configuracion.get("alfa_paso", 0.05))
    if paso <= 0:
        raise ValueError("exp_smooth.alfa_paso debe ser mayor que cero.")
    valores = []
    actual = minimo
    while actual <= maximo + 1e-9:
        valores.append(round(actual, 6))
        actual += paso
    return valores
