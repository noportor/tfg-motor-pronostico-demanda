"""El motor de selección — la propuesta de la tesis.

Para cada serie SKU–canal–regional el motor elige el modelo con menor error **en
validación** y lo aplica en prueba. No es un modelo más: es un meta-modelo que
arbitra entre los demás.

La restricción que lo hace válido (RN-2)
----------------------------------------
La decisión mira **exclusivamente el bloque de validación**. Si la selección se
tomara con datos de prueba, el motor estaría eligiendo con la respuesta a la
vista y el resultado reportado no sería un pronóstico sino un ajuste posterior.
Esa contaminación no dejaría rastro en los números —se vería como un motor
brillante— y por eso el módulo verifica activamente que ningún mes de prueba
entre en la ventana de decisión, en lugar de confiar en que quien lo llame lo
haga bien.

Winner's curse
--------------
Elegir el mínimo de diez candidatos sobre doce observaciones sobreestima siempre
lo bien que le irá al ganador fuera de la muestra. Por eso el reporte incluye la
**tasa de acierto**: en cuántas series el modelo elegido en validación resultó
efectivamente el mejor en prueba. Es un resultado por sí solo, y si sale baja hay
que decirlo (§12).

Asimetría a declarar en la memoria
----------------------------------
LightGBM fija su número de árboles por parada temprana **contra validación**, que
es la misma ventana sobre la que el motor decide. Los demás candidatos estiman
sus parámetros solo con entrenamiento y llegan a validación completamente a
ciegas. Es decir: en el momento de la selección LightGBM juega con una ventaja
que los otros no tienen, y por tanto el motor lo elegirá algo más a menudo de lo
que merecería con un protocolo perfectamente simétrico.

Corregirlo exigiría una partición anidada —un cuarto bloque solo para la parada
temprana— que dejaría menos historia para todo lo demás. Se opta por declarar la
asimetría en lugar de disimularla; el diagnóstico que la acota es precisamente la
tasa de acierto, que se calcula sobre prueba y donde esa ventaja desaparece.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config


@dataclass
class InformeMotor:
    seleccion: pd.DataFrame = field(default_factory=pd.DataFrame)
    regla: str = "mae"
    empates: int = 0
    respaldos: int = 0
    reparto: pd.Series = field(default_factory=pd.Series)

    def lineas(self) -> list[str]:
        salida = [
            f"Regla de selección               : {self.regla}",
            f"Series con empate en validación  : {self.empates:,} "
            f"(desempate alfabético, declarado)",
            f"Series sin candidato válido      : {self.respaldos:,} "
            f"(se asigna el respaldo declarado)",
            "",
            "Reparto de modelos elegidos:",
        ]
        total = int(self.reparto.sum()) or 1
        for modelo, cuantas in self.reparto.sort_values(ascending=False).items():
            salida.append(f"   {modelo:<20} {cuantas:>7,}   {100 * cuantas / total:5.1f} %")
        return salida


class Motor:
    """Elige por serie el modelo con menor error en validación."""

    nombre = "motor"

    def __init__(
        self,
        cfg: Config,
        predicciones: dict[str, pd.DataFrame],
        panel: pd.DataFrame,
    ):
        """
        Args:
            predicciones: nombre de modelo -> panel ancho de pronósticos a un
                paso, ya truncados en cero.
            panel: panel ancho de valores reales.
        """
        self.cfg = cfg
        self.predicciones = predicciones
        self.panel = panel
        self.informe = InformeMotor(regla=str(cfg.modelos.get("motor_regla", "mae")))
        self._elegido: pd.Series | None = None

    # -- interfaz común -----------------------------------------------------

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> None:
        """Decide el modelo ganador por serie mirando SOLO validación."""
        meses_validacion = pd.PeriodIndex(validacion.index)

        # Verificación activa de la RN-2: la ventana de decisión no puede
        # contener ningún mes posterior al cierre de validación.
        limite = self.cfg.particion.fin_validacion
        if (meses_validacion > limite).any():
            fuera = meses_validacion[meses_validacion > limite]
            raise ValueError(
                "La ventana de selección del motor contiene meses posteriores al "
                f"cierre de validación ({limite}): {list(fuera)[:5]}. "
                "Eso sería seleccionar mirando el bloque de prueba (RN-2)."
            )

        candidatos = [
            m for m in self.cfg.modelos_candidatos if m in self.predicciones
        ]
        if not candidatos:
            raise ValueError("El motor no recibió ningún modelo candidato.")
        # Orden alfabético fijo: es también el criterio de desempate, y tiene que
        # ser reproducible (RN-5).
        candidatos = sorted(candidatos)

        real = self.panel.reindex(index=meses_validacion)
        series = list(self.panel.columns)

        errores = pd.DataFrame(index=pd.Index(series, name="serie"), columns=candidatos,
                               dtype=float)
        for modelo in candidatos:
            predicho = self.predicciones[modelo].reindex(
                index=meses_validacion, columns=series
            )
            diferencia = (predicho - real).abs()
            criterio = diferencia.mean(axis=0, skipna=True)
            if self.informe.regla == "mae_mas_bias":
                # Regla nativa del motor en producción: penaliza además el sesgo
                # sistemático. Se deja parametrizada porque es la que la empresa
                # aplica hoy, pero la principal del estudio es el MAE puro.
                media_real = real.mean(axis=0, skipna=True)
                media_predicha = predicho.mean(axis=0, skipna=True)
                sesgo = (media_predicha - media_real).abs()
                criterio = criterio + sesgo
            errores[modelo] = criterio

        matriz = errores.to_numpy(dtype=float)
        todo_nan = np.isnan(matriz).all(axis=1)
        minimo = np.nanmin(np.where(np.isnan(matriz), np.inf, matriz), axis=1)
        es_minimo = np.isclose(matriz, minimo[:, None], rtol=0, atol=1e-12)
        n_optimos = np.where(np.isnan(matriz), False, es_minimo).sum(axis=1)

        posicion = np.where(np.isnan(matriz), np.inf, matriz).argmin(axis=1)
        elegido = np.array(candidatos, dtype=object)[posicion]

        respaldo = str(self.cfg.modelos["benchmark_naive"])
        elegido = np.where(todo_nan, respaldo, elegido)

        self.informe.empates = int(((n_optimos > 1) & ~todo_nan).sum())
        self.informe.respaldos = int(todo_nan.sum())
        self._elegido = pd.Series(elegido, index=errores.index, name="modelo_elegido")

        self.informe.seleccion = (
            errores.assign(
                modelo_elegido=self._elegido,
                criterio_validacion=np.where(todo_nan, np.nan, minimo),
                candidatos_validos=(~np.isnan(matriz)).sum(axis=1),
                empate=(n_optimos > 1) & ~todo_nan,
            )
            .reset_index()
        )
        self.informe.reparto = self._elegido.value_counts()

    def predecir(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Aplica, en cada serie, el pronóstico del modelo que ganó en validación."""
        if self._elegido is None:
            raise RuntimeError("Hay que llamar a ajustar() antes que a predecir().")

        salida = pd.DataFrame(
            np.nan, index=datos.index, columns=datos.columns, dtype=float
        )
        elegido = self._elegido.reindex(datos.columns)

        for modelo, grupo in elegido.groupby(elegido, sort=True):
            if modelo not in self.predicciones:
                continue
            columnas = list(grupo.index)
            salida[columnas] = self.predicciones[modelo].reindex(
                index=datos.index, columns=columnas
            )
        return salida

    # -- diagnóstico --------------------------------------------------------

    def tasa_de_acierto(self, errores_prueba: pd.DataFrame) -> dict:
        """¿Cuántas veces el elegido en validación fue el mejor en prueba?

        Args:
            errores_prueba: DataFrame índice = serie, columnas = modelo,
                valores = MAE en el bloque de prueba.

        Cuantifica el *winner's curse*. Una tasa cercana a 1/k (k = número de
        candidatos) significaría que la selección no transfiere nada; una tasa
        alta, que el desempeño relativo de los modelos es estable entre bloques.
        """
        if self._elegido is None:
            raise RuntimeError("Hay que llamar a ajustar() antes.")

        candidatos = [c for c in sorted(self.cfg.modelos_candidatos)
                      if c in errores_prueba.columns]
        tabla = errores_prueba[candidatos]
        mejor_real = tabla.idxmin(axis=1)
        elegido = self._elegido.reindex(tabla.index)

        comparables = mejor_real.notna() & elegido.notna()
        aciertos = int((mejor_real[comparables] == elegido[comparables]).sum())
        total = int(comparables.sum())

        # Coste del error de selección: cuánto MAE de más se paga por no haber
        # elegido el óptimo de prueba. Es la lectura honesta del winner's curse.
        mae_elegido = pd.Series(
            [tabla.loc[s, elegido[s]] if elegido.get(s) in tabla.columns else np.nan
             for s in tabla.index],
            index=tabla.index,
        )
        mae_optimo = tabla.min(axis=1)
        exceso = (mae_elegido - mae_optimo)

        return {
            "series_comparadas": total,
            "aciertos": aciertos,
            "tasa_acierto": aciertos / total if total else float("nan"),
            "azar_esperado": 1 / len(candidatos) if candidatos else float("nan"),
            "exceso_mae_medio": float(exceso.mean(skipna=True)),
            "exceso_mae_mediano": float(exceso.median(skipna=True)),
        }
