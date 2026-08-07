"""RF-8 — Contraste estadístico de la hipótesis.

Esquema metodológico
--------------------
Demšar (2006) es la referencia canónica para comparar clasificadores —y por
extensión modelos de pronóstico— sobre múltiples conjuntos de datos, y recomienda
exactamente este esquema: **Wilcoxon** de rangos con signo para comparar dos
modelos, y **Friedman con post hoc de Nemenyi** cuando son varios. Se sigue al
pie de la letra.

La unidad de análisis es la **serie**: cada serie SKU–canal–regional es un
bloque, y dentro de cada bloque los once modelos compiten sobre exactamente las
mismas observaciones. Por eso las pruebas son pareadas y por eso la matriz de
errores se restringe a bloques completos.

Sobre Shapiro-Wilk
------------------
Se aplica a las **diferencias pareadas entre los errores de dos modelos**, no a
las ventas, y sirve para justificar por qué se usan pruebas no paramétricas.
Conviene leerlo como **diagnóstico y no como puerta de decisión**: con miles de
series la prueba rechaza la normalidad casi siempre, de modo que como criterio
de decisión no informaría nada. La elección de Wilcoxon se justifica *a priori*
por la asimetría y los valores extremos característicos de los errores de
pronóstico en demanda.

Sobre el tamaño del efecto
--------------------------
Con n de miles, cualquier diferencia sale significativa. Los resultados se
encabezan con el **tamaño del efecto**, no con el p-valor: ``r = Z / √N`` como
pide el requerimiento, y además la correlación rango-biserial de Kerby, que se
interpreta directamente como «en qué proporción de las series gana un modelo
sobre el otro».
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Normalidad
# ---------------------------------------------------------------------------

def shapiro_wilk(
    diferencias: np.ndarray, n_maximo: int = 5000, semilla: int = 0
) -> dict:
    """Shapiro-Wilk sobre las diferencias de error entre dos modelos.

    Por encima de unos miles de observaciones la implementación pierde precisión
    y la prueba satura, así que se submuestrea con semilla fija y se declara.
    """
    limpio = np.asarray(diferencias, dtype=float)
    limpio = limpio[~np.isnan(limpio)]
    n_total = int(limpio.size)

    if n_total < 3:
        return {"n": n_total, "n_usado": 0, "submuestreado": False,
                "estadistico": float("nan"), "p": float("nan"),
                "nota": "menos de 3 observaciones"}

    if np.allclose(limpio, limpio[0]):
        return {"n": n_total, "n_usado": n_total, "submuestreado": False,
                "estadistico": float("nan"), "p": float("nan"),
                "nota": "todas las diferencias son idénticas: la prueba no aplica"}

    submuestreado = n_total > n_maximo
    if submuestreado:
        generador = np.random.default_rng(semilla)
        muestra = generador.choice(limpio, size=n_maximo, replace=False)
    else:
        muestra = limpio

    estadistico, p = stats.shapiro(muestra)
    return {
        "n": n_total,
        "n_usado": int(muestra.size),
        "submuestreado": submuestreado,
        "estadistico": float(estadistico),
        "p": float(p),
        "nota": ("submuestreado con semilla fija" if submuestreado else ""),
    }


# ---------------------------------------------------------------------------
# Wilcoxon pareado
# ---------------------------------------------------------------------------

@dataclass
class ResultadoWilcoxon:
    propuesto: str
    referencia: str
    n_pares: int
    n_no_nulos: int
    estadistico: float
    p: float
    z: float
    r: float
    rango_biserial: float
    gana_propuesto: int
    gana_referencia: int
    empates: int
    mediana_diferencia: float
    alternativa: str
    significativo: bool
    normalidad: dict = field(default_factory=dict)


def wilcoxon_pareado(
    error_propuesto: pd.Series,
    error_referencia: pd.Series,
    alternativa: str = "less",
    alfa: float = 0.05,
    n_maximo_shapiro: int = 5000,
    semilla: int = 0,
) -> ResultadoWilcoxon:
    """Wilcoxon de rangos con signo, pareado y unilateral.

    ``alternativa='less'`` contrasta que el error del modelo propuesto es
    estocásticamente MENOR que el de la referencia, que es la dirección de la
    hipótesis de la tesis.

    El estadístico Z se calcula con la aproximación normal **con corrección por
    empates**, porque hace falta para el tamaño del efecto ``r = Z / √N`` y
    SciPy no lo expone.
    """
    pareado = pd.concat(
        {"propuesto": error_propuesto, "referencia": error_referencia}, axis=1
    ).dropna()
    a = pareado["propuesto"].to_numpy(dtype=float)
    b = pareado["referencia"].to_numpy(dtype=float)
    n_pares = int(a.size)

    diferencia = a - b
    no_nulas = diferencia[diferencia != 0]
    n = int(no_nulas.size)

    gana_propuesto = int((diferencia < 0).sum())   # menor error = gana
    gana_referencia = int((diferencia > 0).sum())
    empates = int((diferencia == 0).sum())

    normalidad = shapiro_wilk(diferencia, n_maximo_shapiro, semilla)

    # Caso degenerado: dos vectores idénticos. No hay evidencia de diferencia y
    # la prueba no puede reportar significancia (criterio de aceptación de RF-8).
    if n == 0:
        return ResultadoWilcoxon(
            propuesto=str(error_propuesto.name), referencia=str(error_referencia.name),
            n_pares=n_pares, n_no_nulos=0,
            estadistico=float("nan"), p=1.0, z=0.0, r=0.0, rango_biserial=0.0,
            gana_propuesto=gana_propuesto, gana_referencia=gana_referencia,
            empates=empates, mediana_diferencia=0.0,
            alternativa=alternativa, significativo=False, normalidad=normalidad,
        )

    estadistico, p = stats.wilcoxon(
        a, b, alternative=alternativa, zero_method="wilcox", correction=False,
        method="approx" if n > 25 else "auto",
    )

    rangos = stats.rankdata(np.abs(no_nulas))
    w_positivo = float(rangos[no_nulas > 0].sum())
    w_negativo = float(rangos[no_nulas < 0].sum())

    media = n * (n + 1) / 4.0
    _, cuentas = np.unique(np.abs(no_nulas), return_counts=True)
    correccion_empates = float(((cuentas ** 3) - cuentas).sum())
    varianza = n * (n + 1) * (2 * n + 1) / 24.0 - correccion_empates / 48.0
    z = (w_positivo - media) / np.sqrt(varianza) if varianza > 0 else 0.0

    # r = Z / √N con N = número de pares (no de diferencias no nulas): es la
    # convención del requerimiento y la más conservadora de las dos.
    r = float(z / np.sqrt(n_pares)) if n_pares else 0.0

    # Kerby (2014): proporción neta de series en que gana cada modelo.
    total_rangos = w_positivo + w_negativo
    rango_biserial = float((w_positivo - w_negativo) / total_rangos) if total_rangos else 0.0

    return ResultadoWilcoxon(
        propuesto=str(error_propuesto.name),
        referencia=str(error_referencia.name),
        n_pares=n_pares,
        n_no_nulos=n,
        estadistico=float(estadistico),
        p=float(p),
        z=float(z),
        r=r,
        rango_biserial=rango_biserial,
        gana_propuesto=gana_propuesto,
        gana_referencia=gana_referencia,
        empates=empates,
        mediana_diferencia=float(np.median(diferencia)),
        alternativa=alternativa,
        significativo=bool(p < alfa),
        normalidad=normalidad,
    )


# ---------------------------------------------------------------------------
# Friedman + Nemenyi
# ---------------------------------------------------------------------------

@dataclass
class ResultadoFriedman:
    modelos: list[str]
    n_bloques: int
    k: int
    chi2: float
    p: float
    kendall_w: float
    significativo: bool
    rangos_medios: pd.Series
    nemenyi: pd.DataFrame | None
    diferencia_critica: float
    contraste_scikit_posthocs: bool


def friedman_con_nemenyi(
    matriz: pd.DataFrame, alfa: float = 0.05
) -> ResultadoFriedman:
    """Friedman omnibus y, si rechaza, post hoc de Nemenyi todos-contra-todos.

    El post hoc es **protegido**: solo tiene sentido leerlo si el omnibus
    rechaza la hipótesis de que todos los modelos rinden igual.

    No se aplica una corrección de Holm encima de Nemenyi: el rango
    studentizado ya controla la tasa de error por familia de las k(k−1)/2
    comparaciones, y corregir dos veces sería innecesariamente conservador.
    """
    limpio = matriz.dropna(axis=0, how="any")
    modelos = list(limpio.columns)
    k = len(modelos)
    n = len(limpio)
    if k < 3:
        raise ValueError("Friedman necesita al menos tres modelos.")
    if n < 2:
        raise ValueError("Friedman necesita al menos dos bloques completos.")

    columnas = [limpio[c].to_numpy(dtype=float) for c in modelos]

    # Rangos medios: 1 = mejor. Es la lectura que acompaña al diagrama de
    # diferencia crítica.
    rangos = limpio.rank(axis=1, method="average")
    rangos_medios = rangos.mean(axis=0)

    # Caso degenerado: TODOS los bloques empatan por completo (por ejemplo, dos
    # modelos que devuelven exactamente lo mismo). La corrección por empates de
    # Friedman se anula y SciPy devuelve NaN por un 0/0. Un NaN silencioso aquí
    # sería peligroso: se leería como «no se pudo calcular» cuando el dato es
    # inequívoco —no hay ninguna evidencia de diferencia—, así que se resuelve
    # explícitamente a chi2 = 0 y p = 1 (criterio de aceptación de la RF-8).
    todo_empatado = bool((rangos.nunique(axis=1) == 1).all())
    if todo_empatado:
        chi2, p = 0.0, 1.0
    else:
        chi2, p = stats.friedmanchisquare(*columnas)
        if np.isnan(chi2) or np.isnan(p):
            raise ValueError(
                "Friedman devolvió NaN sin que todos los bloques estén empatados. "
                "No se reporta un resultado que no se entiende: revisá la matriz "
                "de errores antes de continuar."
            )

    # Kendall W: tamaño del efecto del omnibus. No es comparable entre análisis
    # con distinto k.
    kendall_w = float(chi2 / (n * (k - 1))) if n * (k - 1) else float("nan")

    # Diferencia crítica de Nemenyi (Demšar 2006, ec. 4).
    q_alfa = float(stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2))
    diferencia_critica = q_alfa * np.sqrt(k * (k + 1) / (6.0 * n))

    nemenyi = None
    contraste = False
    if p < alfa:
        nemenyi = _nemenyi(rangos_medios, n, k)
        try:
            import scikit_posthocs as sp

            alterno = sp.posthoc_nemenyi_friedman(limpio.to_numpy(dtype=float))
            alterno.index = modelos
            alterno.columns = modelos
            # Contraste cruzado: si las dos implementaciones no coinciden hay un
            # error en alguna de las dos y hay que saberlo antes de publicar.
            contraste = bool(
                np.allclose(
                    nemenyi.to_numpy(dtype=float),
                    alterno.reindex(index=modelos, columns=modelos).to_numpy(dtype=float),
                    atol=1e-6, equal_nan=True,
                )
            )
        except ImportError:
            contraste = False

    return ResultadoFriedman(
        modelos=modelos,
        n_bloques=n,
        k=k,
        chi2=float(chi2),
        p=float(p),
        kendall_w=kendall_w,
        significativo=bool(p < alfa),
        rangos_medios=rangos_medios.sort_values(),
        nemenyi=nemenyi,
        diferencia_critica=float(diferencia_critica),
        contraste_scikit_posthocs=contraste,
    )


def _nemenyi(rangos_medios: pd.Series, n: int, k: int) -> pd.DataFrame:
    """Matriz de p-valores de Nemenyi a partir de los rangos medios."""
    nombres = list(rangos_medios.index)
    valores = rangos_medios.to_numpy(dtype=float)
    diferencia = np.abs(valores[:, None] - valores[None, :])
    escala = np.sqrt(k * (k + 1) / (6.0 * n))
    q = diferencia / escala
    p = stats.studentized_range.sf(q * np.sqrt(2), k, np.inf)
    p = np.clip(p, 0.0, 1.0)
    np.fill_diagonal(p, 1.0)
    return pd.DataFrame(p, index=nombres, columns=nombres)


# ---------------------------------------------------------------------------
# Reparto de victorias
# ---------------------------------------------------------------------------

def porcentaje_de_victorias(matriz: pd.DataFrame) -> pd.DataFrame:
    """En qué porcentaje de series gana cada modelo (menor error).

    Los empates se reparten: si dos modelos empatan en el mínimo, cada uno se
    lleva media victoria. Adjudicárselas todas al primero por orden de columna
    sería un artefacto del orden de las columnas.
    """
    limpio = matriz.dropna(axis=0, how="any")
    minimos = limpio.min(axis=1)
    es_minimo = limpio.eq(minimos, axis=0)
    reparto = es_minimo.div(es_minimo.sum(axis=1), axis=0)

    victorias = reparto.sum(axis=0)
    return pd.DataFrame({
        "modelo": victorias.index,
        "victorias": victorias.to_numpy(),
        "porcentaje": (100.0 * victorias / len(limpio)).to_numpy(),
        "victorias_sin_empate": es_minimo[es_minimo.sum(axis=1) == 1].sum(axis=0).to_numpy(),
    }).sort_values("victorias", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batería completa
# ---------------------------------------------------------------------------

def bateria_completa(
    matriz: pd.DataFrame,
    propuestos: list[str],
    referencias: list[str],
    alfa: float = 0.05,
    alternativa: str = "less",
    n_maximo_shapiro: int = 5000,
    semilla: int = 0,
) -> dict:
    """Ejecuta todo el contraste y devuelve los resultados en bruto.

    Args:
        matriz: serie × modelo con la métrica de contraste (bloques completos).
        propuestos: modelos cuya superioridad se contrasta (motor y LightGBM).
        referencias: modelos de referencia (promedio móvil y Naïve).
    """
    limpio = matriz.dropna(axis=0, how="any")

    comparaciones = []
    for propuesto in propuestos:
        for referencia in referencias:
            if propuesto == referencia:
                continue
            if propuesto not in limpio.columns or referencia not in limpio.columns:
                continue
            comparaciones.append(
                wilcoxon_pareado(
                    limpio[propuesto], limpio[referencia],
                    alternativa=alternativa, alfa=alfa,
                    n_maximo_shapiro=n_maximo_shapiro, semilla=semilla,
                )
            )

    return {
        "n_bloques": len(limpio),
        "wilcoxon": comparaciones,
        "friedman": friedman_con_nemenyi(limpio, alfa=alfa),
        "victorias": porcentaje_de_victorias(limpio),
    }
