"""Brazo fundacional — Chronos-Bolt, pronóstico sin entrenamiento (ML2).

La pregunta que responde
------------------------
¿Un modelo FUNDACIONAL de series temporales —preentrenado por terceros sobre
millones de series públicas, aplicado aquí sin ver un solo dato de la empresa
durante su entrenamiento— compite contra un GBM afinado con variables del
negocio? Si la respuesta es «no», el hallazgo vale tanto como si fuera «sí».

Decisiones declaradas
---------------------
- **Cero-shot estricto.** ``ajustar()`` solo carga los pesos publicados: no hay
  fine-tuning, y el bloque de validación NO se usa (ni siquiera para parar
  temprano — no hay nada que parar). Es el único brazo cuyos parámetros no
  vieron jamás datos de NOVAPACK.
- **Los datos no salen de la máquina.** Los pesos se descargan una vez desde el
  repositorio público y la inferencia corre local (GPU si hay): ninguna serie
  viaja a una API externa (Anexo B).
- **Punto de pronóstico = media.** La salida nativa es probabilística
  (cuantiles); se toma la media, consistente con el objetivo de los demás
  brazos (la demanda esperada, no la mediana — la lección Tweedie del estudio).
- **Contexto = la vida observada de la serie** hasta el mes anterior al que se
  predice, con el mismo protocolo a un paso de la interfaz común, y proyección
  DIRECTA h=1..H en el multihorizonte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


class ModeloChronos:
    """Chronos-Bolt aplicado cero-shot sobre el panel."""

    nombre = "chronos"

    def __init__(self, cfg: Config, tabla: pd.DataFrame | None = None):
        self.cfg = cfg
        parametros = dict(cfg.modelos.get("chronos", {}))
        self.checkpoint = str(parametros.get("checkpoint", "amazon/chronos-bolt-base"))
        self.contexto_maximo = int(parametros.get("contexto_maximo", 96))
        self.lote = int(parametros.get("lote", 256))
        # calibrar: false | "global" | "categoria" (true equivale a "global").
        crudo = parametros.get("calibrar", False)
        self.calibrar = "global" if crudo is True else (str(crudo) if crudo else "")
        if self.calibrar == "categoria" and tabla is None:
            raise ValueError(
                "chronos: calibración por categoría declarada pero el brazo "
                "no recibió la tabla de features."
            )
        self._categoria_por_serie = (
            tabla.drop_duplicates("serie").set_index("serie")["categoria"]
            .astype(str)
            if (tabla is not None and "categoria" in tabla.columns) else None
        )
        self.pipeline = None
        self.factor = 1.0
        self.factores: pd.Series | None = None  # por serie (V3, por categoría)

    # -- interfaz común ------------------------------------------------------

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        """Carga los pesos publicados y, si se pide, calibra el sesgo.

        La calibración (V2) es un único factor multiplicativo global estimado
        SOBRE VALIDACIÓN con el mismo protocolo a un paso: la V1 midió que el
        modelo fundacional acierta en magnitud (WMAPE competitivo) pero
        subestima sistemáticamente; corregir ese nivel con un escalar es la
        intervención mínima que no toca los pesos ni re-entrena nada. Es un
        uso de validación análogo a la parada temprana: fija UNA constante,
        no qué predecir; y la prueba queda intacta (RN-2).
        """
        import torch
        from chronos import BaseChronosPipeline

        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.checkpoint,
            device_map=dispositivo,
            torch_dtype=torch.bfloat16 if dispositivo == "cuda" else torch.float32,
        )
        if self.calibrar:
            historia_total = pd.concat([entrenamiento, validacion])
            piezas: list[pd.DataFrame] = []
            for mes in validacion.index:
                historia = historia_total.loc[historia_total.index < mes]
                abanico = self._abanico(historia, horizonte=1)
                if mes not in abanico.index:
                    continue
                y = validacion.loc[mes]
                p = abanico.loc[mes].reindex(y.index)
                comparable = y.notna() & p.notna()
                piezas.append(pd.DataFrame({
                    "serie": y.index[comparable],
                    "real": y[comparable].to_numpy(dtype=float),
                    "pred": p[comparable].to_numpy(dtype=float),
                }))
            celdas = (pd.concat(piezas, ignore_index=True) if piezas
                      else pd.DataFrame(columns=["serie", "real", "pred"]))
            reales, predichos = float(celdas["real"].sum()), float(celdas["pred"].sum())
            self.factor = reales / predichos if predichos > 0 else 1.0

            if self.calibrar == "categoria" and self._categoria_por_serie is not None:
                # Un factor por categoría (masa suficiente: son 5), con el
                # global como respaldo donde no haya con qué estimar.
                celdas["categoria"] = celdas["serie"].map(self._categoria_por_serie)
                por_cat = celdas.groupby("categoria", observed=True).agg(
                    real=("real", "sum"), pred=("pred", "sum")
                )
                factor_cat = (por_cat["real"] / por_cat["pred"]).where(
                    por_cat["pred"] > 0, self.factor
                )
                self.factores = (
                    self._categoria_por_serie.map(factor_cat)
                    .fillna(self.factor)
                )

    def _abanico(self, historia: pd.DataFrame, horizonte: int) -> pd.DataFrame:
        """Media pronosticada h=1..``horizonte`` para cada serie del panel."""
        import torch

        contextos: list[torch.Tensor] = []
        series: list[str] = []
        for serie in historia.columns:
            vida = historia[serie].dropna().to_numpy(dtype=np.float32)
            if len(vida) == 0:
                continue
            contextos.append(torch.tensor(vida[-self.contexto_maximo:]))
            series.append(serie)

        medias = np.full((horizonte, len(series)), np.nan)
        for inicio in range(0, len(contextos), self.lote):
            bloque = contextos[inicio:inicio + self.lote]
            # Posicional a propósito: la 1.x lo llama ``context`` y la 2.x
            # ``inputs`` — el primer argumento funciona en ambas.
            _, media = self.pipeline.predict_quantiles(
                bloque,
                prediction_length=horizonte,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            medias[:, inicio:inicio + len(bloque)] = (
                media.to(torch.float32).cpu().numpy().T
            )

        origen = historia.index[-1]
        meses = pd.period_range(origen + 1, origen + horizonte, freq="M")
        abanico = pd.DataFrame(medias, index=meses, columns=series)
        if self.factores is not None:
            abanico = abanico * self.factores.reindex(abanico.columns).fillna(self.factor)
        else:
            abanico = abanico * self.factor
        return abanico.reindex(columns=historia.columns)

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Pronóstico a un paso para los meses evaluados (validación y prueba)."""
        if self.pipeline is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")

        primer_evaluado = self.cfg.particion.fin_entrenamiento + 1
        meses_evaluados = [m for m in datos.index if m >= primer_evaluado]

        salida = pd.DataFrame(np.nan, index=datos.index, columns=datos.columns)
        for mes in meses_evaluados:
            historia = datos.loc[datos.index < mes]
            abanico = self._abanico(historia, horizonte=1)
            if mes in abanico.index:
                salida.loc[mes] = abanico.loc[mes].reindex(datos.columns).to_numpy()
        return salida

    # -- protocolo multihorizonte -------------------------------------------

    def proyectar_directo(self, historia: pd.DataFrame, horizonte: int) -> pd.DataFrame:
        """Abanico directo h=1..H desde el origen, con la misma inferencia."""
        if self.pipeline is None:
            raise RuntimeError("Hay que llamar a ajustar() antes de proyectar.")
        return self._abanico(historia, horizonte)
