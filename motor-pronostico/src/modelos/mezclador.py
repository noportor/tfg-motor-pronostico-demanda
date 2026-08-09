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
``mezcla_h``      (V5) Promedio ponderado POR HORIZONTE: la V4 midió que el
                  menú óptimo cambia con la distancia (Chronos brilla cerca,
                  LightGBM sostiene lejos, el suavizado solo sirve corto) y
                  que los pesos POR SERIE son ruidosos (12 observaciones:
                  ``mezcla_pond`` perdió contra el promedio). Acá el peso es
                  por candidato × horizonte, estimado sobre el abanico de
                  validación AGREGADO e VALORIZADO en todas las series —
                  cientos de veces más evidencia por peso que el por-serie.

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
NOMBRE_MEZCLA_H = "mezcla_h"
NOMBRE_MEZCLA_CONMUTADA = "mezcla_conmutada"


def _menu_declarado(
    cfg: Config,
    predicciones: dict[str, pd.DataFrame],
    declarados: list[str] | None = None,
) -> list[str]:
    """El menú declarado (de la config o explícito), verificado contra los
    brazos activos."""
    if declarados is None:
        declarados = list(cfg.modelos.get("mezclador", {}).get("candidatos", []))
    if not declarados:
        raise ValueError(
            "modelos.mezclador.candidatos está vacío: el menú del "
            "mezclador es una decisión declarada, no un implícito."
        )
    candidatos = sorted(m for m in declarados if m in predicciones)
    ausentes = sorted(set(declarados) - set(candidatos))
    if ausentes:
        raise ValueError(
            f"El menú del mezclador declara candidatos sin pronóstico: "
            f"{ausentes}. El menú se declara sobre brazos activos."
        )
    return candidatos


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
        candidatos: list[str] | None = None,
    ):
        if nombre not in NOMBRES_MEZCLA:
            raise ValueError(f"Mezclador desconocido: {nombre!r}")
        self.cfg = cfg
        self.nombre = nombre
        self.predicciones = predicciones
        self.panel = panel
        # `candidatos` explícito permite instancias internas con OTRO menú
        # (la conmutada usa dos); sin él rige el menú de la configuración.
        self.candidatos = _menu_declarado(cfg, predicciones, candidatos)
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


def _promedio_ponderado_por_fila(
    paneles: dict[str, pd.DataFrame],
    pesos_filas: pd.DataFrame,
    indice: pd.Index,
    columnas: pd.Index,
) -> pd.DataFrame:
    """Promedio ponderado con pesos POR FILA (mes), consciente de NaN.

    El gemelo de ``_promedio_ponderado`` con el eje del peso girado: acá el
    peso de un candidato varía por fila del panel (el horizonte), no por
    columna (la serie). Donde un candidato no tiene pronóstico su peso se
    redistribuye entre los presentes; sin candidatos presentes: NaN.
    """
    numerador = np.zeros((len(indice), len(columnas)))
    denominador = np.zeros_like(numerador)
    for candidato, panel in paneles.items():
        p = panel.reindex(index=indice, columns=columnas).to_numpy(dtype=float)
        w = pesos_filas[candidato].to_numpy(dtype=float)[:, None]
        presente = ~np.isnan(p)
        numerador += np.where(presente, p * w, 0.0)
        denominador += np.where(presente, w, 0.0)
    with np.errstate(invalid="ignore"):
        resultado = np.where(denominador > 0, numerador / denominador, np.nan)
    return pd.DataFrame(resultado, index=indice, columns=columnas)


class MezcladorHorizonte:
    """Mezcla con pesos por candidato × horizonte (V5: ``mezcla_h``).

    El peso del candidato ``c`` en el horizonte ``h`` es el inverso de su
    D valorizada — ``WMAPE_val(h) + |Bias_val(h)|`` agregada en TODAS las
    series con costo — medida sobre el abanico de validación (h=1..H desde el
    cierre de entrenamiento, el mismo insumo de la ventana de selección
    multihorizonte del motor). RN-2: el abanico vive íntegro en validación,
    con verificación activa, y los pesos quedan congelados para prueba y para
    el protocolo multihorizonte.

    En el protocolo a un paso cada mes evaluado es un pronóstico h=1 (la
    historia real llega hasta t−1), así que ahí aplican los pesos de h=1 — la
    correspondencia se declara.
    """

    nombre = NOMBRE_MEZCLA_H

    def __init__(
        self,
        cfg: Config,
        predicciones: dict[str, pd.DataFrame],
        panel: pd.DataFrame,
        costo_por_serie: pd.Series,
    ):
        self.cfg = cfg
        self.predicciones = predicciones
        self.panel = panel
        self.costo_por_serie = costo_por_serie
        self.candidatos = _menu_declarado(cfg, predicciones)
        self.pesos_h: pd.DataFrame | None = None   # horizonte × candidato
        self.d_horizonte: pd.DataFrame | None = None

    # -- interfaz común ------------------------------------------------------

    def ajustar(
        self,
        proyecciones_validacion: dict[str, pd.DataFrame],
        real: pd.DataFrame,
    ) -> None:
        """Estima los pesos por horizonte sobre el abanico de validación."""
        indice = pd.PeriodIndex(real.index)
        limite = self.cfg.particion.fin_validacion
        if (indice > limite).any():
            raise ValueError(
                "El abanico de pesos del mezclador por horizonte contiene "
                f"meses posteriores al cierre de validación ({limite}): "
                "sería ponderar mirando prueba (RN-2)."
            )
        ausentes = sorted(
            c for c in self.candidatos if c not in proyecciones_validacion
        )
        if ausentes:
            raise ValueError(
                "El menú de mezcla_h declara candidatos sin abanico de "
                f"validación: {ausentes}."
            )

        costo = self.costo_por_serie.reindex(real.columns).to_numpy(dtype=float)
        y = real.to_numpy(dtype=float)
        horizontes = pd.Index(range(1, len(indice) + 1), name="horizonte")
        d = pd.DataFrame(index=horizontes, columns=self.candidatos, dtype=float)
        for candidato in self.candidatos:
            p = proyecciones_validacion[candidato].reindex(
                index=indice, columns=real.columns
            ).to_numpy(dtype=float)
            # La D valorizada por horizonte, con las mismas definiciones que
            # la agregación del protocolo: solo celdas comparables con costo.
            valida = ~np.isnan(y) & ~np.isnan(p) & ~np.isnan(costo)[None, :]
            demanda = np.where(valida, costo[None, :] * y, 0.0).sum(axis=1)
            error = np.where(
                valida, costo[None, :] * np.abs(p - y), 0.0
            ).sum(axis=1)
            pred = np.where(valida, costo[None, :] * p, 0.0).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                wmape = np.where(demanda > 0, 100.0 * error / demanda, np.nan)
                sesgo = np.where(
                    demanda > 0, 100.0 * (pred - demanda) / demanda, np.nan
                )
            d[candidato] = wmape + np.abs(sesgo)
        self.d_horizonte = d

        # Inverso de la D, con el mismo piso relativo que mezcla_pond: una D
        # casi nula no puede acaparar el peso por un artefacto numérico.
        matriz = d.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            piso = np.nanmedian(matriz, axis=1, keepdims=True) * 1e-3
        piso = np.where(np.isfinite(piso) & (piso > 0), piso, 1e-9)
        inverso = np.where(np.isnan(matriz), np.nan, 1.0 / (matriz + piso))
        # Horizontes sin evidencia valorizada: promedio simple (respaldo).
        sin_evidencia = np.isnan(inverso).all(axis=1)
        inverso = np.where(np.isnan(inverso), 0.0, inverso)
        inverso[sin_evidencia] = 1.0
        self.pesos_h = pd.DataFrame(
            inverso, index=horizontes, columns=self.candidatos
        )

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Mezcla a un paso: pesos de h=1 (cada mes evaluado es un h=1)."""
        if self.pesos_h is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")
        paneles = {c: self.predicciones[c] for c in self.candidatos}
        series = pd.Index(datos.columns, name="serie")
        pesos = pd.DataFrame(
            np.tile(self.pesos_h.iloc[0].to_numpy(dtype=float), (len(series), 1)),
            index=series, columns=self.candidatos,
        )
        return _promedio_ponderado(paneles, pesos, datos.index, datos.columns)

    # -- protocolo multihorizonte -------------------------------------------

    def componer(self, proyecciones: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Mezcla el abanico con el peso de CADA horizonte (fila i → h=i+1).

        Mismos pesos congelados de validación; horizontes más allá de la
        ventana de pesos usan el último peso disponible (declarado).
        """
        if self.pesos_h is None:
            raise RuntimeError("Hay que llamar a ajustar() antes de componer.")
        presentes = {
            c: proyecciones[c] for c in self.candidatos if c in proyecciones
        }
        if not presentes:
            raise ValueError(
                "Ninguno de los candidatos de mezcla_h tiene proyección "
                "multihorizonte."
            )
        referencia = next(iter(presentes.values()))
        posiciones = np.minimum(
            np.arange(len(referencia.index)), len(self.pesos_h) - 1
        )
        pesos_filas = self.pesos_h.iloc[posiciones]
        return _promedio_ponderado_por_fila(
            presentes, pesos_filas, referencia.index, referencia.columns
        )

    # -- reporte -------------------------------------------------------------

    def informe_lineas(self) -> list[str]:
        salida = [
            f"{self.nombre}: inverso de la D valorizada de validación, por "
            f"horizonte · menú: {', '.join(self.candidatos)}"
        ]
        if self.pesos_h is not None:
            reparto = self.pesos_h.div(self.pesos_h.sum(axis=1), axis=0)
            for h in (reparto.index[0], reparto.index[-1]):
                fila = reparto.loc[h].sort_values(ascending=False)
                fila_txt = " · ".join(
                    f"{m} {100 * w:.0f} %" for m, w in fila.items()
                )
                salida.append(f"   h={h}: {fila_txt}")
        return salida


class MezclaConmutada:
    """Conmutación por régimen entre dos menús de mezcla (V8).

    La V7 midió que ningún menú estático domina ambos regímenes: el menú
    curado gana el año normal (67,8 / 72,9) y el menú diverso gana el año de
    quiebre (77,7-80,7 / 70,8-74,2) porque en el quiebre TODO el menú curado
    comparte el signo del sesgo y no queda nada que cancelar. Este brazo
    convierte esa disyuntiva en un sistema: dos mezclas de promedio simple
    (la regla se fija a priori — la lección M4 tres veces confirmada) sobre
    los dos menús declarados, y un DETECTOR causal que transfiere peso hacia
    el menú diverso cuando el sistema base muestra sesgo de régimen.

    El detector (RN-3/RN-2 por construcción):
    - Señal: el sesgo valorizado del sistema BASE sobre los últimos
      ``ventana_meses`` meses YA OBSERVADOS (pronóstico a un paso de la
      mezcla base contra la realidad, valorizado por costo). En el mes t la
      ventana termina en t−1: la señal jamás toca el mes que se predice.
    - λ(t) = rampa declarada entre ``umbral_activacion_pct`` (por debajo:
      régimen normal, λ=0, la conmutada ES la mezcla base) y
      ``umbral_pleno_pct`` (por encima: quiebre declarado, λ=1, la conmutada
      ES la mezcla diversa). Sin ventana completa de evidencia: λ=0
      (respaldo declarado — sin señal se asume normalidad).
    - En el multihorizonte, λ se evalúa UNA vez en el origen (el estado de
      régimen que el planificador conoce ese día) y aplica al abanico entero.

    Los umbrales son constantes DECLARADAS (no se calibran mirando prueba);
    la señal mensual se persiste (``tabla_senal``) para el análisis de
    sensibilidad post-hoc.
    """

    nombre = NOMBRE_MEZCLA_CONMUTADA

    def __init__(
        self,
        cfg: Config,
        predicciones: dict[str, pd.DataFrame],
        panel: pd.DataFrame,
        costo_por_serie: pd.Series,
    ):
        mezclador_cfg = cfg.modelos.get("mezclador", {})
        conmutador = mezclador_cfg.get("conmutador", {}) or {}
        self.ventana = int(conmutador.get("ventana_meses", 3))
        self.umbral_activacion = float(
            conmutador.get("umbral_activacion_pct", 10.0)
        )
        self.umbral_pleno = float(conmutador.get("umbral_pleno_pct", 20.0))
        if self.ventana < 1:
            raise ValueError("conmutador.ventana_meses debe ser >= 1.")
        if self.umbral_pleno <= self.umbral_activacion:
            raise ValueError(
                "conmutador.umbral_pleno_pct debe ser mayor que "
                "umbral_activacion_pct (la rampa necesita ancho)."
            )
        quiebre = list(mezclador_cfg.get("candidatos_quiebre", []))
        if not quiebre:
            raise ValueError(
                "mezcla_conmutada requiere modelos.mezclador."
                "candidatos_quiebre: el menú diverso es una decisión "
                "declarada, no un implícito."
            )
        self.cfg = cfg
        self.panel = panel
        self.costo_por_serie = costo_por_serie
        self.base = Mezclador(cfg, "mezcla_prom", predicciones, panel)
        self.quiebre = Mezclador(
            cfg, "mezcla_prom", predicciones, panel, candidatos=quiebre
        )
        self._pred_base: pd.DataFrame | None = None
        self._real_val: pd.Series | None = None   # Σ costo·real por mes
        self._pred_val: pd.Series | None = None   # Σ costo·pred por mes

    # -- interfaz común ------------------------------------------------------

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        self.base.ajustar(entrenamiento, validacion)
        self.quiebre.ajustar(entrenamiento, validacion)
        # La señal se precalcula sobre TODO el panel: agregados valorizados
        # mensuales de la realidad y del pronóstico base, solo en celdas
        # donde existen ambos y hay costo.
        pred = self.base.predecir(self.panel)
        costo = self.costo_por_serie.reindex(self.panel.columns).to_numpy(dtype=float)
        y = self.panel.to_numpy(dtype=float)
        p = pred.to_numpy(dtype=float)
        valido = ~np.isnan(y) & ~np.isnan(p) & ~np.isnan(costo)[None, :]
        self._pred_base = pred
        self._real_val = pd.Series(
            np.where(valido, costo[None, :] * y, 0.0).sum(axis=1),
            index=self.panel.index,
        )
        self._pred_val = pd.Series(
            np.where(valido, costo[None, :] * p, 0.0).sum(axis=1),
            index=self.panel.index,
        )

    def _senal_en(self, mes: pd.Period) -> tuple[float, float]:
        """``(sesgo_movil_pct, lambda)`` del mes, con ventana que termina en
        ``mes − 1``. Sin ventana completa con demanda observada: λ=0."""
        if self._real_val is None:
            raise RuntimeError("Hay que llamar a ajustar() antes.")
        ventana = pd.period_range(mes - self.ventana, mes - 1, freq="M")
        disponibles = [m for m in ventana if m in self._real_val.index]
        if len(disponibles) < self.ventana:
            return np.nan, 0.0
        real = float(self._real_val.loc[disponibles].sum())
        pred = float(self._pred_val.loc[disponibles].sum())
        if real <= 0:
            return np.nan, 0.0
        sesgo = 100.0 * (pred - real) / real
        rampa = (abs(sesgo) - self.umbral_activacion) / (
            self.umbral_pleno - self.umbral_activacion
        )
        return sesgo, float(np.clip(rampa, 0.0, 1.0))

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        p_base = self.base.predecir(datos)
        p_quiebre = self.quiebre.predecir(datos)
        lambdas = pd.Series(
            [self._senal_en(mes)[1] for mes in datos.index], index=datos.index
        )
        return self._combinar(p_base, p_quiebre, lambdas)

    @staticmethod
    def _combinar(
        p_base: pd.DataFrame, p_quiebre: pd.DataFrame, lambdas: pd.Series
    ) -> pd.DataFrame:
        """Combinación por fila consciente de NaN: donde falta un lado, el
        otro toma todo el peso (nunca se fabrica un hueco)."""
        base = p_base.to_numpy(dtype=float)
        quiebre = p_quiebre.reindex(
            index=p_base.index, columns=p_base.columns
        ).to_numpy(dtype=float)
        lam = lambdas.to_numpy(dtype=float)[:, None]
        peso_base = np.where(~np.isnan(base), 1.0 - lam, 0.0)
        peso_quiebre = np.where(~np.isnan(quiebre), lam, 0.0)
        total = peso_base + peso_quiebre
        with np.errstate(invalid="ignore"):
            resultado = np.where(
                total > 0,
                (np.nan_to_num(base) * peso_base
                 + np.nan_to_num(quiebre) * peso_quiebre) / np.where(total > 0, total, 1.0),
                np.nan,
            )
        return pd.DataFrame(resultado, index=p_base.index, columns=p_base.columns)

    # -- protocolo multihorizonte -------------------------------------------

    def componer(self, proyecciones: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """λ del ORIGEN (meses observados ≤ origen) aplicado al abanico entero."""
        fan_base = self.base.componer(proyecciones)
        fan_quiebre = self.quiebre.componer(proyecciones)
        primer_mes = pd.Period(fan_base.index[0], freq="M")
        _, lam = self._senal_en(primer_mes)
        lambdas = pd.Series(lam, index=fan_base.index)
        return self._combinar(fan_base, fan_quiebre, lambdas)

    # -- reporte -------------------------------------------------------------

    def tabla_senal(self, meses: pd.PeriodIndex) -> pd.DataFrame:
        filas = []
        for mes in meses:
            sesgo, lam = self._senal_en(mes)
            filas.append({"mes": str(mes), "sesgo_movil_pct": sesgo,
                          "lambda": lam})
        return pd.DataFrame(filas)

    def informe_lineas(self) -> list[str]:
        salida = [
            f"{self.nombre}: (1−λ)·prom[{', '.join(self.base.candidatos)}] + "
            f"λ·prom[{', '.join(self.quiebre.candidatos)}]",
            f"   señal: sesgo valorizado móvil {self.ventana} m del sistema "
            f"base · rampa {self.umbral_activacion:.0f} %→"
            f"{self.umbral_pleno:.0f} %",
        ]
        return salida
