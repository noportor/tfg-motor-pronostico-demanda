"""RF-7 — Métricas de error, calculadas por serie sobre el bloque de prueba.

Las reglas que siguen no pueden alterarse; cada una responde a un problema
concreto que aparece con datos de demanda reales.

**MAPE.** Está indefinido cuando la demanda real es cero, y en este portafolio
hay meses en cero por todas partes. Se calcula sobre los meses con demanda
positiva y **se reporta qué porcentaje de observaciones quedó fuera**. Un MAPE
excelente calculado sobre el 40 % de los meses no es un MAPE excelente.

**RMSE.** Depende de la escala. Se calcula por serie y luego se promedia; nunca
sobre el conjunto agrupado, porque ahí las series de gran volumen dominarían el
resultado sin que se note.

**Bias.** Porcentual: ``(media(pred) − media(real)) / media(real) × 100``.

**MASE.** Escalado con el error del Naïve en entrenamiento. Es la métrica
robusta a los ceros y la que conviene reportar cuando el MAPE excluye demasiado.

Media y mediana
---------------
De cada métrica se reportan **las dos**. En las pruebas del prototipo apareció
un caso donde la media del MAE era casi idéntica entre dos modelos mientras la
mediana difería en más de la mitad y el contraste daba altamente significativo:
unas pocas series de gran volumen dominaban el promedio. Reportar solo la media
lleva a conclusiones equivocadas.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

COLUMNAS_METRICAS = (
    "mae", "rmse", "mape", "bias", "mase",
    "error_compuesto_pct", "error_compuesto_unidades",
)


def metricas_por_serie(
    real: pd.DataFrame,
    predicho: pd.DataFrame,
    escala_mase: pd.Series,
) -> pd.DataFrame:
    """Calcula todas las métricas para cada serie del bloque recibido.

    Args:
        real: panel ancho de valores observados, ya recortado al bloque.
        predicho: panel ancho de pronósticos, mismo índice y columnas.
        escala_mase: MAE del Naïve en entrenamiento, por serie.

    Returns:
        Una fila por serie con las métricas y los conteos que las acompañan.
        Las series sin ninguna observación en el bloque quedan fuera.
    """
    predicho = predicho.reindex(index=real.index, columns=real.columns)

    y = real.to_numpy(dtype=float)
    p = predicho.to_numpy(dtype=float)

    observado = ~np.isnan(y) & ~np.isnan(p)
    n_obs = observado.sum(axis=0)

    error = np.where(observado, p - y, np.nan)
    absoluto = np.abs(error)

    # Una serie puede no tener ninguna observación válida en el bloque; NumPy
    # avisa al promediar sobre vacío. El aviso se silencia PORQUE el caso está
    # tratado —la métrica queda en NaN y se cuenta—, no para taparlo (§10).
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mae = np.nanmean(np.where(observado, absoluto, np.nan), axis=0)
        rmse = np.sqrt(np.nanmean(np.where(observado, error ** 2, np.nan), axis=0))

        # --- MAPE: solo meses con demanda REAL positiva ---------------------
        positivo = observado & (y > 0)
        n_positivo = positivo.sum(axis=0)
        porcentual = np.where(positivo, absoluto / np.where(y != 0, y, np.nan), np.nan)
        mape = 100.0 * np.nanmean(np.where(positivo, porcentual, np.nan), axis=0)
        mape = np.where(n_positivo > 0, mape, np.nan)

        # --- Bias porcentual sobre las medias -------------------------------
        media_real = np.nanmean(np.where(observado, y, np.nan), axis=0)
        media_predicha = np.nanmean(np.where(observado, p, np.nan), axis=0)
        bias_unidades = media_predicha - media_real
        bias = np.where(
            media_real != 0, 100.0 * bias_unidades / media_real, np.nan
        )

    denominador = escala_mase.reindex(real.columns).to_numpy(dtype=float)
    # El MASE queda indefinido si el Naïve no se equivoca nunca en entrenamiento
    # (serie constante). Se deja en NaN y se cuenta; dividir por cero no puede
    # pasar en silencio.
    with np.errstate(divide="ignore", invalid="ignore"):
        mase = np.where(
            (denominador > 0) & ~np.isnan(denominador), mae / denominador, np.nan
        )

    tabla = pd.DataFrame(
        {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "bias": bias,
            "mase": mase,
            "bias_unidades": bias_unidades,
            "media_real": media_real,
            "media_predicha": media_predicha,
            "n_observaciones": n_obs,
            "n_meses_con_demanda": n_positivo,
            "meses_excluidos_del_mape": n_obs - n_positivo,
            "pct_excluido_del_mape": np.where(
                n_obs > 0, 100.0 * (n_obs - n_positivo) / n_obs, np.nan
            ),
            "escala_mase": denominador,
        },
        index=pd.Index(real.columns, name="serie"),
    )

    # --- Error compuesto -----------------------------------------------------
    # El documento describe un «error compuesto (MAE + |Bias|)» expresado en
    # porcentaje. Sumar un MAE en unidades con un Bias en porcentaje no es una
    # operación válida: son magnitudes de distinta dimensión. Se emiten por
    # tanto las DOS versiones dimensionalmente coherentes, y el documento debe
    # declarar cuál usa:
    #
    #   error_compuesto_pct       = MAPE + |Bias|              (todo en %)
    #   error_compuesto_unidades  = MAE  + |media(p) − media(y)|  (todo en unidades)
    #
    # La primera es la que reproduce la lectura «el modelo se equivoca un N %»;
    # la segunda es la regla nativa del motor en producción.
    tabla["error_compuesto_pct"] = tabla["mape"] + tabla["bias"].abs()
    tabla["error_compuesto_unidades"] = tabla["mae"] + tabla["bias_unidades"].abs()

    return tabla.loc[tabla["n_observaciones"] > 0]


def resumen(errores: dict[str, pd.DataFrame], orden: list[str] | None = None) -> pd.DataFrame:
    """Media y mediana de cada métrica, por modelo. Es la Tabla 8 del documento.

    Todos los modelos se resumen sobre **el mismo conjunto de series**: la
    intersección de las series evaluables. Sin esa intersección, un modelo que
    falla en las series difíciles parecería mejor por haberse medido sobre un
    subconjunto más fácil.
    """
    if not errores:
        raise ValueError("No hay errores que resumir.")

    comunes = None
    for tabla in errores.values():
        indice = set(tabla.index)
        comunes = indice if comunes is None else (comunes & indice)
    comunes = sorted(comunes or set())
    if not comunes:
        raise ValueError(
            "Ningún modelo comparte series evaluables: la comparación pareada "
            "sería imposible."
        )

    nombres = orden or list(errores)
    filas = []
    for nombre in nombres:
        if nombre not in errores:
            continue
        tabla = errores[nombre].loc[comunes]
        fila: dict[str, object] = {"modelo": nombre, "series": len(tabla)}
        for metrica in COLUMNAS_METRICAS:
            fila[f"{metrica}_media"] = float(tabla[metrica].mean(skipna=True))
            fila[f"{metrica}_mediana"] = float(tabla[metrica].median(skipna=True))
        fila["pct_excluido_del_mape_medio"] = float(
            tabla["pct_excluido_del_mape"].mean(skipna=True)
        )
        # Cuántas series apoyan su MAPE en muy pocos meses. Un MAPE calculado
        # sobre uno o dos meses con demanda no es un porcentaje de error: es una
        # anécdota. El conteo va al informe para que el lector pueda descontarlo.
        fila["meses_con_demanda_mediana"] = float(
            tabla["n_meses_con_demanda"].median()
        )
        fila["series_mape_con_menos_de_3_meses"] = int(
            (tabla["n_meses_con_demanda"] < 3).sum()
        )
        fila["series_sin_mape"] = int(tabla["mape"].isna().sum())
        fila["series_sin_mase"] = int(tabla["mase"].isna().sum())
        fila["series_sin_bias"] = int(tabla["bias"].isna().sum())
        filas.append(fila)

    return pd.DataFrame(filas)


def tabla8(resumen_completo: pd.DataFrame) -> pd.DataFrame:
    """Recorte de la Tabla 8 del documento: MAE, MAPE, RMSE y Bias por modelo."""
    columnas = {
        "modelo": "Modelo",
        "series": "Series",
        "mae_media": "MAE medio (unidades)",
        "mae_mediana": "MAE mediano (unidades)",
        "mape_media": "MAPE medio (%)",
        "mape_mediana": "MAPE mediano (%)",
        "rmse_media": "RMSE medio (unidades)",
        "rmse_mediana": "RMSE mediano (unidades)",
        "bias_media": "Bias medio (%)",
        "bias_mediana": "Bias mediano (%)",
        "mase_media": "MASE medio",
        "mase_mediana": "MASE mediano",
        "error_compuesto_pct_media": "Error compuesto medio (MAPE + |Bias|, %)",
        "pct_excluido_del_mape_medio": "Observaciones excluidas del MAPE (%)",
        "series_mape_con_menos_de_3_meses": "Series con MAPE sobre <3 meses",
    }
    presentes = [c for c in columnas if c in resumen_completo.columns]
    return resumen_completo[presentes].rename(columns=columnas)


def suite_valorizada(
    errores: dict[str, pd.DataFrame],
    costo_por_serie: pd.Series,
    orden: list[str] | None = None,
) -> pd.DataFrame:
    """Métricas VALORIZADAS por costo unitario — la agregación entre SKUs.

    Las unidades difieren entre SKUs: sumar o promediar errores en unidades
    entre productos no tiene sentido físico. Llevado a dinero
    (``|error| × costo``) todo queda en la misma unidad y la agregación es
    legítima. Por serie:

    - ``demanda_val = costo × media_real × n_obs``  (demanda del bloque, en Bs)
    - ``error_val   = costo × mae × n_obs``          (error absoluto total, Bs)
    - ``pred_val    = costo × media_pred × n_obs``

    Y por modelo:

    - **WMAPE_val** = Σ error_val / Σ demanda_val — el % de la demanda
      valorizada que se erra (es el WMAE normalizado).
    - **Bias_val**  = (Σ pred_val − Σ demanda_val) / Σ demanda_val — sobre o
      sub-pronóstico sistemático en dinero.
    - **RMSE_val**  = raíz del MSE agrupado de los errores valorizados,
      reconstruido exacto desde los RMSE por serie
      (MSE agrupado = Σ n·(costo·rmse)² / Σ n). Agrupar aquí SÍ es válido:
      misma unidad.
    - **D = WMAPE_val + |Bias_val|** — la MÉTRICA DE DECISIÓN del estudio:
      ambos términos son porcentajes del mismo denominador (la suma es
      dimensionalmente coherente, a diferencia del MAE+|Bias| en unidades) y
      captura el objetivo del pronóstico: errar poco Y sin sesgo sistemático.
      Es además la regla de selección declarada por la empresa.

    Las series sin costo se excluyen de esta suite y su peso se declara en
    ``cobertura`` (regla de outliers visibles). El contraste estadístico NO
    depende de esto: al ser pareado por serie, es invariante a multiplicar
    cada serie por una constante positiva.
    """
    if costo_por_serie.empty:
        raise ValueError(
            "No hay maestro de costos: la suite valorizada no puede calcularse."
        )

    nombres = orden or list(errores)
    filas = []
    for nombre in nombres:
        if nombre not in errores:
            continue
        tabla = errores[nombre].copy()
        tabla["costo"] = costo_por_serie.reindex(tabla.index)
        con_costo = tabla.dropna(subset=["costo"])

        demanda = float(
            (con_costo["costo"] * con_costo["media_real"]
             * con_costo["n_observaciones"]).sum()
        )
        error = float(
            (con_costo["costo"] * con_costo["mae"]
             * con_costo["n_observaciones"]).sum()
        )
        pred = float(
            (con_costo["costo"] * con_costo["media_predicha"]
             * con_costo["n_observaciones"]).sum()
        )
        n_total = float(con_costo["n_observaciones"].sum())
        mse_val = float(
            (con_costo["n_observaciones"]
             * (con_costo["costo"] * con_costo["rmse"]) ** 2).sum()
        ) / n_total if n_total else float("nan")

        if demanda <= 0:
            raise ValueError(
                f"La demanda valorizada del modelo '{nombre}' no es positiva; "
                f"revisá el maestro de costos."
            )

        wmape = 100.0 * error / demanda
        bias = 100.0 * (pred - demanda) / demanda
        filas.append({
            "modelo": nombre,
            "series_con_costo": int(len(con_costo)),
            "cobertura_series": float(len(con_costo) / len(tabla)) if len(tabla) else 0.0,
            "demanda_valorizada": demanda,
            "error_valorizado": error,
            "wmape_val": wmape,
            "bias_val": bias,
            "rmse_val": mse_val ** 0.5,
            "D": wmape + abs(bias),
        })

    return pd.DataFrame(filas).sort_values("D").reset_index(drop=True)


def tabla_valorizada(suite: pd.DataFrame) -> pd.DataFrame:
    """La compañera valorizada de la Tabla 8, ordenada por la métrica de decisión."""
    columnas = {
        "modelo": "Modelo",
        "wmape_val": "WMAPE valorizado (%)",
        "bias_val": "Bias valorizado (%)",
        "D": "D = WMAPE + |Bias| (%)",
        "error_valorizado": "Error absoluto (Bs)",
        "rmse_val": "RMSE valorizado (Bs)",
        "series_con_costo": "Series con costo",
    }
    presentes = [c for c in columnas if c in suite.columns]
    return suite[presentes].rename(columns=columnas)


def matriz_de_errores(
    errores: dict[str, pd.DataFrame], metrica: str = "mae"
) -> pd.DataFrame:
    """Matriz serie × modelo con una métrica. Es la entrada de las pruebas pareadas.

    Se restringe a las series donde **todos** los modelos tienen valor: las
    pruebas de Wilcoxon y Friedman exigen bloques completos, y rellenar los
    huecos con imputaciones cambiaría los rangos sin que se note.
    """
    columnas = {
        nombre: tabla[metrica] for nombre, tabla in errores.items()
        if metrica in tabla.columns
    }
    if not columnas:
        raise ValueError(f"Ningún modelo reportó la métrica '{metrica}'.")
    matriz = pd.DataFrame(columnas)
    return matriz.dropna(axis=0, how="any")
