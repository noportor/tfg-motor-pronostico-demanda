"""Tratamiento de eventos exógenos en la serie de AJUSTE.

El caso que motiva este módulo es el COVID-19: entre marzo-2020 y febrero-2021
las ventas cayeron hasta −88 % valorizado por causas ajenas a la demanda
estructural, y 181 series activas se apagaron por completo durante el
confinamiento. Dejar ese año crudo distorsiona todo lo que se ajusta sobre
entrenamiento: los pares (features → objetivo) de LightGBM, las rejillas de α,
los estados estacionales y la escala del MASE.

El tratamiento base NO es una invención del investigador: es la práctica
documentada de Planificación de la empresa durante el evento — **se copió la
historia de la gestión pasada**. Como el benchmark debe ser el método real
(§10), el experimento reproduce esa práctica; la historia cruda queda como
ablación de sensibilidad (``--anular evento_exogeno.tratamiento=nada``).

Tres garantías innegociables:

1. **La serie observada sigue siendo la fuente de verdad.** El panel corregido
   existe solo para el ajuste; la evaluación compara siempre contra lo real.
2. **La ventana del evento cae íntegra en entrenamiento** — lo valida la
   configuración: validación y prueba jamás ven un dato corregido.
3. **Nada es silencioso**: cuántos meses se reemplazaron, en cuántas series, y
   cuántos no pudieron reemplazarse (sin historia previa) queda contado en el
   informe y en ``flujo.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Config


@dataclass
class InformeEvento:
    nombre: str = ""
    tratamiento: str = "nada"
    desde: str = ""
    hasta: str = ""
    meses_ventana: int = 0
    observaciones_en_ventana: int = 0
    observaciones_reemplazadas: int = 0
    sin_historia_previa: int = 0
    series_afectadas: int = 0
    detalle: dict = field(default_factory=dict)

    def lineas(self) -> list[str]:
        if self.tratamiento == "nada":
            return [
                f"Evento '{self.nombre}' ({self.desde}..{self.hasta}): SIN "
                f"tratamiento (historia cruda) — ablación de sensibilidad.",
            ]
        return [
            f"Evento '{self.nombre}' ({self.desde}..{self.hasta}, "
            f"{self.meses_ventana} meses) — tratamiento: {self.tratamiento} "
            f"(práctica documentada de la empresa).",
            f"  Observaciones en la ventana      : {self.observaciones_en_ventana:,}",
            f"  Reemplazadas por el mes del año previo: {self.observaciones_reemplazadas:,} "
            f"en {self.series_afectadas:,} series",
            f"  Sin historia previa (quedan crudas): {self.sin_historia_previa:,}",
            "  La evaluación compara SIEMPRE contra la serie observada.",
        ]


def aplicar_tratamiento(
    panel_largo: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, InformeEvento]:
    """Devuelve el panel de AJUSTE (largo) y el informe del tratamiento.

    ``copiar_gestion_previa``: cada observación cuyo mes cae en la ventana del
    evento se reemplaza por el valor de la MISMA serie doce meses antes. Si la
    serie no tenía vida doce meses antes (nació durante o poco antes del
    evento), el valor observado se conserva y se cuenta: copiar desde la nada
    sería inventar datos.
    """
    informe = InformeEvento()
    if cfg.evento is None:
        return panel_largo, informe

    evento = cfg.evento
    informe.nombre = evento.nombre
    informe.tratamiento = evento.tratamiento
    informe.desde = str(evento.desde)
    informe.hasta = str(evento.hasta)
    informe.meses_ventana = len(evento.ventana())

    en_ventana = panel_largo["periodo"].isin(evento.ventana())
    informe.observaciones_en_ventana = int(en_ventana.sum())

    if evento.tratamiento == "nada":
        return panel_largo, informe

    # --- copiar_gestion_previa ----------------------------------------------
    ajustado = panel_largo.copy()

    # El valor de la misma serie doce meses antes, alineado por (serie, periodo).
    previo = panel_largo[["serie", "periodo", "y"]].copy()
    previo["periodo"] = previo["periodo"] + 12
    previo = previo.rename(columns={"y": "y_gestion_previa"})

    ajustado = ajustado.merge(previo, on=["serie", "periodo"], how="left")

    reemplazable = en_ventana & ajustado["y_gestion_previa"].notna()
    informe.observaciones_reemplazadas = int(reemplazable.sum())
    informe.sin_historia_previa = int((en_ventana & ~reemplazable).sum())
    informe.series_afectadas = int(
        ajustado.loc[reemplazable, "serie"].nunique()
    )

    ajustado.loc[reemplazable, "y"] = ajustado.loc[reemplazable, "y_gestion_previa"]
    ajustado = ajustado.drop(columns="y_gestion_previa")

    return ajustado, informe
