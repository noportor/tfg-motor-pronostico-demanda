"""ANEXO H — Script de contraste estadístico de la hipótesis.

Reproduce, desde el artefacto ``errores_por_serie.csv`` de la corrida, los
valores reportados en el apartado «Contraste estadístico de la hipótesis»:
Shapiro–Wilk sobre las diferencias, Wilcoxon de rangos con signo (pareado,
unilateral) del motor frente a los dos métodos de referencia, y Friedman con
post hoc de Nemenyi sobre los once modelos evaluados.

La métrica contrastada es el error compuesto por serie (MAE + |Bias| en
unidades), la misma del cuerpo del documento. El insumo es la matriz
serie x modelo que la corrida persiste (``matriz_contraste.csv``): usar la
matriz exacta garantiza reproducir los conteos y estadísticos al bit.
Requiere: pandas, numpy, scipy y scikit-posthocs.

Uso:  python anexo_h_contraste.py [ruta/matriz_contraste.csv]
"""

import sys

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

RUTA = sys.argv[1] if len(sys.argv) > 1 else "salidas/matriz_contraste.csv"
PROPUESTO = "motor"
REFERENCIAS = ["ma_12", "naive_m1"]       # método vigente y línea base ingenua
ALFA = 0.05

# --- Matriz serie x modelo (error compuesto MAE + |Bias|, en unidades) ------
matriz = pd.read_csv(RUTA).set_index("serie").dropna()
print(f"series con bloque completo: {len(matriz)}  |  modelos: {matriz.shape[1]}")

# --- Wilcoxon pareado unilateral (menor error), con Z corregido por empates -
for referencia in REFERENCIAS:
    a = matriz[PROPUESTO].to_numpy(float)
    b = matriz[referencia].to_numpy(float)
    diferencia = a - b
    no_nulas = diferencia[diferencia != 0]
    n = no_nulas.size

    # Normalidad de las diferencias (muestra de 5000 si N excede el límite)
    muestra = diferencia
    if muestra.size > 5000:
        muestra = np.random.default_rng(0).choice(muestra, 5000, replace=False)
    w_shapiro, p_shapiro = stats.shapiro(muestra)

    estadistico, p = stats.wilcoxon(
        a, b, alternative="less", zero_method="wilcox", method="approx"
    )
    rangos = stats.rankdata(np.abs(no_nulas))
    w_pos = rangos[no_nulas > 0].sum()
    media = n * (n + 1) / 4.0
    _, cuentas = np.unique(np.abs(no_nulas), return_counts=True)
    varianza = n * (n + 1) * (2 * n + 1) / 24.0 - ((cuentas**3 - cuentas).sum()) / 48.0
    z = (w_pos - media) / np.sqrt(varianza)
    r = z / np.sqrt(len(matriz))          # r = Z / raiz(N pares)

    print(f"\n{PROPUESTO} vs {referencia}")
    print(f"  Shapiro-Wilk W = {w_shapiro:.2f}  (p = {p_shapiro:.3g}) -> no normal")
    print(f"  gana {PROPUESTO}: {(diferencia < 0).sum()}  |  gana {referencia}: "
          f"{(diferencia > 0).sum()}  |  empates: {(diferencia == 0).sum()}")
    print(f"  W = {estadistico:,.1f}  Z = {z:.2f}  p = {p:.3g}  r = {r:.2f}"
          f"  ->  {'significativo' if p < ALFA else 'NO significativo'} (alfa {ALFA})")

# --- Friedman omnibus + Nemenyi (todos contra todos) ------------------------
bloques = matriz.rank(axis=1)             # rango 1 = menor error en la serie
chi2, p_friedman = stats.friedmanchisquare(*[matriz[c] for c in matriz.columns])
n_series, k = matriz.shape
kendall_w = chi2 / (n_series * (k - 1))
q_alfa = stats.studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2)
cd = q_alfa * np.sqrt(k * (k + 1) / (6.0 * n_series))

print(f"\nFriedman: chi2 = {chi2:,.1f}  p = {p_friedman:.3g}  "
      f"W de Kendall = {kendall_w:.3f}")
print(f"Diferencia critica de Nemenyi (alfa {ALFA}): CD = {cd:.3f}")
print("\nrangos medios (1 = mejor):")
print(bloques.mean().sort_values().round(2).to_string())

nemenyi = sp.posthoc_nemenyi_friedman(matriz.to_numpy())
nemenyi.index = nemenyi.columns = matriz.columns
print(f"\nNemenyi, p-valores de {PROPUESTO} contra cada modelo:")
print(nemenyi[PROPUESTO].drop(PROPUESTO).round(4).to_string())
