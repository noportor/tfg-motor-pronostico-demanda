"""El mezclador — combinación de pronósticos en vez de selección (ML2-V4).

Por qué existe
--------------
El motor paga la maldición del ganador: elige UN candidato por serie sobre doce
observaciones de validación, y su tasa de acierto medida (10–21 %) muestra
cuánto de esa elección es ruido. El hallazgo más robusto de las competencias de
pronóstico (M4, M5) es que la COMBINACIÓN simple de pocos modelos razonables es
extraordinariamente difícil de batir: promediar cubre los errores no
correlacionados de los candidatos sin apostar todo a uno.

Dos mezcladores, para medir la afirmación en estos datos:

``mezcla_prom``   Promedio simple del menú curado — la hipótesis M4 pura.
``mezcla_pond``   Promedio ponderado por el INVERSO del error compuesto de
                  validación por serie — un punto medio entre promediar y
                  seleccionar: la evidencia de validación inclina los pesos,
                  pero nunca apuesta todo a un candidato.

La restricción que los hace válidos (RN-2)
------------------------------------------
Los pesos se estiman EXCLUSIVAMENTE sobre el bloque de validación (misma
verificación activa que el motor) y quedan congelados: en prueba y en el
multihorizonte se aplican tal cual. El menú es curado y declarado en la
configuración — mezclar brazos rotos no es cobertura, es contaminación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

NOMBRES_MEZCLA = ("mezcla_prom", "mezcla_pond")


def _promedio_ponderado(
    paneles: dict[str, pd.DataFrame],
    pesos: pd.DataFrame,
    indice: pd.Index,
    columnas: pd.Index,
) -> pd.DataFrame:
    """Promedio ponderado consciente de NaN.

    Donde un candidato no tiene pronóstico, su peso se redistribuye entre los
    presentes (renormalización por celda). Sin candidatos presentes: NaN.
    """
    numerador = np.zeros((len(indice), len(columnas)))
    denominador = np.zeros_like(numerador)
    for candidato, panel in paneles.items():
        p = panel.reindex(index=indice, columns=columnas).to_numpy(dtype=float)
        w = pesos[candidato].reindex(columnas).to_numpy(dtype=float)[None, :]
        presente = ~np.isnan(p)
        numerador += np.where(presente, p * w, 0.0)
        denominador += np.where(presente, w, 0.0)
    with np.errstate(invalid="ignore"):
        resultado = np.where(denominador > 0, numerador / denominador, np.nan)
    return pd.DataFrame(resultado, index=indice, columns=columnas)


class Mezclador:
    """Combina los pronósticos de un menú curado con pesos de validación."""

    def __init__(
        self,
        cfg: Config,
        nombre: str,
        predicciones: dict[str, pd.DataFrame],
        panel: pd.DataFrame,
    ):
        if nombre not in NOMBRES_MEZCLA:
            raise ValueError(f"Mezclador desconocido: {nombre!r}")
        self.cfg = cfg
        self.nombre = nombre
        self.predicciones = predicciones
        self.panel = panel
        declarados = list(
            cfg.modelos.get("mezclador", {}).get("candidatos", [])
        )
        if not declarados:
            raise ValueError(
                "modelos.mezclador.candidatos está vacío: el menú del "
                "mezclador es una decisión declarada, no un implícito."
            )
        self.candidatos = sorted(
            m for m in declarados if m in predicciones
        )
        ausentes = sorted(set(declarados) - set(self.candidatos))
        if ausentes:
            raise ValueError(
                f"El menú del mezclador declara candidatos sin pronóstico: "
                f"{ausentes}. El menú se declara sobre brazos activos."
            )
        self.pesos: pd.DataFrame | None = None  # serie × candidato

    # -- interfaz común ------------------------------------------------------

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        """Estima los pesos por serie mirando SOLO validación (RN-2)."""
        meses_validacion = pd.PeriodIndex(validacion.index)
        limite = self.cfg.particion.fin_validacion
        if (meses_validacion > limite).any():
            raise ValueError(
                "La ventana de pesos del mezclador contiene meses posteriores "
                f"al cierre de validación ({limite}): sería ponderar mirando "
                "prueba (RN-2)."
            )

        series = pd.Index(self.panel.columns, name="serie")
        if self.nombre == "mezcla_prom":
            self.pesos = pd.DataFrame(
                1.0, index=series, columns=self.candidatos
            )
            return

        real = self.panel.reindex(index=meses_validacion)
        errores = pd.DataFrame(index=series, columns=self.candidatos, dtype=float)
        for candidato in self.candidatos:
            predicho = self.predicciones[candidato].reindex(
                index=meses_validacion, columns=series
            )
            # El error compuesto por serie (mae + |bias|): el mismo criterio
            # del motor y el análogo por-serie de la métrica de decisión D.
            mae = (predicho - real).abs().mean(axis=0, skipna=True)
            sesgo = (
                predicho.mean(axis=0, skipna=True) - real.mean(axis=0, skipna=True)
            ).abs()
            errores[candidato] = mae + sesgo

        # Inverso del error, normalizado por serie. Un error nulo no puede
        # acaparar el peso por un artefacto numérico: el piso es una fracción
        # del error mediano de la propia serie.
        matriz = errores.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            piso = np.nanmedian(matriz, axis=1, keepdims=True) * 1e-3
        piso = np.where(np.isfinite(piso) & (piso > 0), piso, 1e-9)
        inverso = np.where(np.isnan(matriz), np.nan, 1.0 / (matriz + piso))
        # Series sin ningún error medible: promedio simple (respaldo declarado).
        sin_evidencia = np.isnan(inverso).all(axis=1)
        inverso = np.where(np.isnan(inverso), 0.0, inverso)
        inverso[sin_evidencia] = 1.0
        self.pesos = pd.DataFrame(
            inverso, index=series, columns=self.candidatos
        )

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        if self.pesos is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")
        paneles = {c: self.predicciones[c] for c in self.candidatos}
        return _promedio_ponderado(
            paneles, self.pesos, datos.index, datos.columns
        )

    # -- protocolo multihorizonte -------------------------------------------

    def componer(self, proyecciones: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Mezcla las proyecciones multihorizonte de los candidatos presentes.

        Mismos pesos congelados de validación: en el horizonte no se re-decide
        nada, igual que la selección del motor.
        """
        if self.pesos is None:
            raise RuntimeError("Hay que llamar a ajustar() antes de componer.")
        presentes = {
            c: proyecciones[c] for c in self.candidatos if c in proyecciones
        }
        if not presentes:
            raise ValueError(
                "Ninguno de los candidatos del mezclador tiene proyección "
                "multihorizonte."
            )
        referencia = next(iter(presentes.values()))
        return _promedio_ponderado(
            presentes, self.pesos, referencia.index, referencia.columns
        )

    # -- reporte -------------------------------------------------------------

    def informe_lineas(self) -> list[str]:
        regla = ("promedio simple" if self.nombre == "mezcla_prom"
                 else "inverso del error compuesto de validación, por serie")
        salida = [f"{self.nombre}: {regla} · menú: {', '.join(self.candidatos)}"]
        if self.nombre == "mezcla_pond" and self.pesos is not None:
            reparto = (self.pesos.div(self.pesos.sum(axis=1), axis=0)
                       .mean(axis=0).sort_values(ascending=False))
            reparto_txt = " · ".join(
                f"{m} {100 * w:.0f} %" for m, w in reparto.items()
            )
            salida.append(f"   peso medio: {reparto_txt}")
        return salida
