"""Anexo 8 (Anexo H en la versión evaluada), Tabla H3 (Tabla 20 del documento final): incertidumbre
agrupada de las diferencias del contraste.

Propósito
---------
El contraste de Wilcoxon del apartado 4.3.2 trata cada serie SKU–canal–regional
como un bloque independiente. Las series de un mismo SKU comparten producto,
precio y campañas, y todas las series se evalúan sobre el mismo calendario
del bloque de prueba: sus errores no son independientes y los valores p resultan
optimistas. Este script mide la incertidumbre de las diferencias reportadas con
dos remuestreos que respetan esa dependencia, sin volver a ajustar ningún
modelo, sólo desde las salidas publicadas de la corrida principal:

1. **Conglomerados de SKU** (bootstrap por conglomerados). Se remuestrean SKU
   completos, con reposición, y cada SKU sorteado entra con TODAS sus series.
   La correlación entre las series de un mismo producto queda dentro de cada
   réplica en lugar de contarse como información independiente.
2. **Bloques móviles de meses** (bootstrap por bloques móviles circulares). Se
   remuestrean los doce meses del bloque de prueba en bloques de tres meses
   consecutivos (circular: el bloque que empieza en febrero de 2026 toma
   febrero y marzo de 2026 y abril de 2025), cuatro bloques por réplica. Cada
   réplica recalcula, con los meses sorteados, el error de cada serie y la D
   del portafolio. Captura la dependencia temporal común: un mes atípico
   afecta a todas las series a la vez.

Magnitudes, para cada comparación del contraste publicado (propuesto menos
referencia; negativo = el propuesto erra menos):

* diferencia media por serie del error compuesto en unidades
  (``error_compuesto_unidades`` = MAE + |sesgo| por serie, la métrica del
  contraste);
* diferencia mediana por serie de esa misma métrica;
* diferencia de la D valorizada del portafolio, en puntos porcentuales
  (D = WMAPE valorizado + |sesgo valorizado agregado|).

Intervalos percentiles al 95 % (percentiles 2,5 y 97,5 de las réplicas,
interpolación lineal de NumPy). B = 2.000 réplicas, semilla 20260408. Todas las
comparaciones usan las MISMAS réplicas (números aleatorios comunes), de modo
que las filas son comparables entre sí.

Las comparaciones son las cinco del contraste del documento: el motor frente
al promedio móvil de 12 meses, a su variante con factor de crecimiento y al
modelo ingenuo, y LightGBM frente al promedio móvil de 12 meses y al ingenuo.
El valor p y el tamaño del efecto que se listan al lado son los publicados: se
recalculan con la misma función de la corrida (``src.pruebas.wilcoxon_pareado``)
sobre la misma matriz publicada, y el script se detiene si las cuatro
comparaciones de ``pruebas_estadisticas.txt`` no se reproducen.

Insumos (todos de ``salidas/``; el script verifica su SHA-256 contra
``manifiesto.json`` y no abre ningún archivo de ``datos/``):

* ``matriz_contraste.csv``: métrica de contraste por serie y modelo.
* ``errores_por_serie.csv``: MAE, medias real y predicha y meses observados
  por serie y modelo (los componentes de la D).
* ``series_y_predicciones.csv``: real y pronóstico por serie y mes del bloque
  de prueba (variante temporal) y demanda observada de entrenamiento.
* ``fuera_de_muestra.csv``: valor de entrenamiento valorizado a costo por
  serie. El costo unitario de cada serie se deduce como ese valor sobre la
  demanda observada de entrenamiento, igual que el script del Anexo 14; el
  script se detiene si la D así reconstruida no reproduce la Tabla 8
  valorizada (``tabla_valorizada.csv``) de los once modelos.
* ``tabla_valorizada.csv``, ``error_por_mes.csv``: controles.

Salidas (sólo agregados; ningún identificador de serie ni de producto):
``tabla_H3_incertidumbre_agrupada.csv`` y ``.md`` en ``--destino``.

Uso
---
    python -X utf8 scripts/incertidumbre_agrupada.py [--salidas salidas]
        [--destino resultados/tablas] [--replicas 2000] [--semilla 20260408]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.pruebas import wilcoxon_pareado  # noqa: E402

REPLICAS = 2000
SEMILLA = 20260408
NIVEL = 0.95
LARGO_BLOQUE = 3
METRICA = "error_compuesto_unidades"

# (propuesto, referencia): las cinco comparaciones del contraste del documento.
COMPARACIONES: list[tuple[str, str]] = [
    ("motor", "ma_12"),
    ("motor", "ma_12_gf"),
    ("motor", "naive_m1"),
    ("lightgbm", "ma_12"),
    ("lightgbm", "naive_m1"),
]
# Las que publica pruebas_estadisticas.txt (la quinta, motor frente a ma_12_gf,
# la reproduce el script del Anexo 8): control de reproducción del p y de r.
PUBLICADAS_EN_TXT: dict[tuple[str, str], tuple[float, float]] = {
    # (r, rango biserial) tal como los imprime pruebas_estadisticas.txt
    ("motor", "ma_12"): (-0.3071, -0.3726),
    ("motor", "naive_m1"): (-0.0866, -0.1154),
    ("lightgbm", "ma_12"): (-0.0934, -0.1078),
    ("lightgbm", "naive_m1"): (0.0867, 0.1001),
}

NOMBRES = {
    "motor": "motor",
    "lightgbm": "LightGBM",
    "ma_12": "promedio móvil de 12 meses",
    "ma_12_gf": "promedio móvil de 12 meses con factor de crecimiento",
    "naive_m1": "ingenuo",
}
MAGNITUDES = {
    "media": "Diferencia media por serie (unidades)",
    "mediana": "Diferencia mediana por serie (unidades)",
    "d_portafolio": "Diferencia de D valorizada del portafolio (puntos porcentuales)",
}
REMUESTREOS = {
    "sku": "conglomerados de SKU",
    "bloques": "bloques móviles circulares de 3 meses",
}
TOLERANCIA_D = 1e-5          # puntos porcentuales
TOLERANCIA_SERIE = 1e-3      # unidades (el archivo mensual se publica con 4 decimales)
TOLERANCIA_D_MENSUAL = 1e-4  # puntos porcentuales, por el mismo redondeo


# ---------------------------------------------------------------------------
# Piezas puras (las prueba tests/test_incertidumbre_agrupada.py)
# ---------------------------------------------------------------------------

def sku_de_serie(series: pd.Index | pd.Series) -> np.ndarray:
    """El SKU es el primer campo de la clave «SKU|CANAL|REGIONAL»."""
    claves = pd.Index(series).astype(str)
    if not (claves.str.count(r"\|") == 2).all():
        raise ValueError("Toda clave de serie debe tener la forma SKU|CANAL|REGIONAL.")
    return claves.str.split("|").str[0].to_numpy()


def pesos_conglomerados(
    codigos: np.ndarray, replicas: int, rng: np.random.Generator
) -> np.ndarray:
    """Matriz réplicas × series con la multiplicidad de cada serie.

    En cada réplica se sortean, con reposición, tantos SKU como hay; cada serie
    hereda el número de veces que salió su SKU (cero si no salió).
    """
    n_conglomerados = int(codigos.max()) + 1
    sorteo = rng.integers(0, n_conglomerados, size=(replicas, n_conglomerados))
    conteo = np.zeros((replicas, n_conglomerados), dtype=np.int64)
    filas = np.repeat(np.arange(replicas), n_conglomerados)
    np.add.at(conteo, (filas, sorteo.ravel()), 1)
    return conteo[:, codigos]


def pesos_bloques_moviles(
    n_meses: int, largo: int, replicas: int, rng: np.random.Generator
) -> np.ndarray:
    """Matriz réplicas × meses con la multiplicidad de cada mes.

    Bootstrap por bloques móviles circulares: ceil(n_meses / largo) bloques de
    ``largo`` meses consecutivos, cada uno con inicio uniforme en los n_meses
    posibles (el calendario se cierra sobre sí mismo); se recorta a n_meses.
    """
    n_bloques = -(-n_meses // largo)
    inicios = rng.integers(0, n_meses, size=(replicas, n_bloques))
    meses = (inicios[:, :, None] + np.arange(largo)[None, None, :]) % n_meses
    meses = meses.reshape(replicas, -1)[:, :n_meses]
    conteo = np.zeros((replicas, n_meses), dtype=np.int64)
    filas = np.repeat(np.arange(replicas), n_meses)
    np.add.at(conteo, (filas, meses.ravel()), 1)
    return conteo


def mediana_ponderada(valores: np.ndarray, pesos: np.ndarray) -> float:
    """Mediana del multiconjunto en que cada valor aparece ``pesos`` veces.

    Igual a ``np.median(np.repeat(valores, pesos))`` sin construir el arreglo.
    """
    pesos = np.asarray(pesos, dtype=np.int64)
    validos = (pesos > 0) & ~np.isnan(valores)
    v = np.asarray(valores, dtype=float)[validos]
    w = pesos[validos]
    if w.sum() == 0:
        return float("nan")
    orden = np.argsort(v, kind="mergesort")
    v, w = v[orden], w[orden]
    acumulado = np.cumsum(w)
    total = int(acumulado[-1])
    bajo = v[np.searchsorted(acumulado, (total - 1) // 2, side="right")]
    alto = v[np.searchsorted(acumulado, total // 2, side="right")]
    return float((bajo + alto) / 2.0)


def d_portafolio(error: np.ndarray, real: np.ndarray, predicho: np.ndarray) -> np.ndarray:
    """D = 100·Σerror/Σreal + 100·|Σpredicho − Σreal|/Σreal, ya sumados.

    Recibe las sumas valorizadas (por réplica) y devuelve la D de cada una.
    """
    return 100.0 * error / real + 100.0 * np.abs(predicho - real) / real


def compuesto_por_serie(
    absoluto: np.ndarray, firmado: np.ndarray, observado: np.ndarray, pesos_mes: np.ndarray
) -> np.ndarray:
    """Error compuesto por serie (MAE + |sesgo|) con los meses ponderados.

    ``absoluto``, ``firmado`` y ``observado`` son series × meses (ceros donde no
    hay observación); ``pesos_mes`` es réplicas × meses. Devuelve series ×
    réplicas, con NaN donde la réplica no dejó ningún mes observado.
    """
    n = observado @ pesos_mes.T
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = (absoluto @ pesos_mes.T) / n
        sesgo = (firmado @ pesos_mes.T) / n
    salida = mae + np.abs(sesgo)
    salida[n == 0] = np.nan
    return salida


def intervalo(replicas: np.ndarray, nivel: float = NIVEL) -> tuple[float, float]:
    cola = 100.0 * (1.0 - nivel) / 2.0
    finitas = replicas[np.isfinite(replicas)]
    bajo, alto = np.percentile(finitas, [cola, 100.0 - cola])
    return float(bajo), float(alto)


def excluye_cero(bajo: float, alto: float) -> bool:
    return bool(bajo > 0 or alto < 0)


@dataclass
class Fila:
    propuesto: str
    referencia: str
    magnitud: str
    remuestreo: str
    estimacion: float
    ic_inferior: float
    ic_superior: float


def bootstrap_por_sku(
    diferencias: dict[tuple[str, str], np.ndarray],
    componentes_d: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
    codigos: np.ndarray,
    replicas: int,
    semilla: int,
) -> list[Fila]:
    """Media, mediana y D por conglomerados de SKU, para cada comparación.

    ``diferencias``: por comparación, la diferencia por serie (propuesto menos
    referencia). ``componentes_d``: por modelo, (error, real, predicho)
    valorizados por serie (ceros en las series sin costo).
    """
    rng = np.random.default_rng(semilla)
    pesos = pesos_conglomerados(codigos, replicas, rng).astype(float)
    filas: list[Fila] = []
    for (prop, ref), d in diferencias.items():
        medias = (pesos @ d) / pesos.sum(axis=1)
        medianas = np.array([mediana_ponderada(d, w) for w in pesos.astype(np.int64)])
        bajo, alto = intervalo(medias)
        filas.append(Fila(prop, ref, "media", "sku", float(d.mean()), bajo, alto))
        bajo, alto = intervalo(medianas)
        filas.append(Fila(prop, ref, "mediana", "sku", float(np.median(d)), bajo, alto))
        if componentes_d is not None:
            dp = _d_de(componentes_d[prop], pesos)
            dr = _d_de(componentes_d[ref], pesos)
            uno = np.ones((1, len(d)))
            est = float(_d_de(componentes_d[prop], uno)[0] - _d_de(componentes_d[ref], uno)[0])
            bajo, alto = intervalo(dp - dr)
            filas.append(Fila(prop, ref, "d_portafolio", "sku", est, bajo, alto))
    return filas


def _d_de(componentes: tuple[np.ndarray, np.ndarray, np.ndarray], pesos: np.ndarray) -> np.ndarray:
    error, real, predicho = componentes
    return d_portafolio(pesos @ error, pesos @ real, pesos @ predicho)


def bootstrap_por_bloques(
    mensual: dict[str, dict[str, np.ndarray]],
    comparaciones: list[tuple[str, str]],
    estimaciones: dict[tuple[str, str, str], float],
    replicas: int,
    semilla: int,
    largo: int = LARGO_BLOQUE,
) -> list[Fila]:
    """Media, mediana y D por bloques móviles circulares de meses.

    ``mensual[modelo]`` trae, series × meses: ``absoluto``, ``firmado``,
    ``observado`` (ceros donde no hay dato) y, meses: ``error_val``,
    ``real_val``, ``pred_val`` (sumas valorizadas del portafolio por mes).
    ``estimaciones`` lleva la estimación puntual publicada de cada magnitud.
    """
    cualquiera = next(iter(mensual.values()))
    n_meses = cualquiera["absoluto"].shape[1]
    rng = np.random.default_rng(semilla)
    pesos = pesos_bloques_moviles(n_meses, largo, replicas, rng).astype(float)

    compuesto = {
        m: compuesto_por_serie(v["absoluto"], v["firmado"], v["observado"], pesos)
        for m, v in mensual.items()
    }
    d = {}
    for m, v in mensual.items():
        if "error_val" in v:
            d[m] = d_portafolio(pesos @ v["error_val"], pesos @ v["real_val"], pesos @ v["pred_val"])

    filas: list[Fila] = []
    for prop, ref in comparaciones:
        dif = compuesto[prop] - compuesto[ref]           # series × réplicas
        with np.errstate(invalid="ignore"):
            medias = np.nanmean(dif, axis=0)
            medianas = np.nanmedian(dif, axis=0)
        for clave, valores in (("media", medias), ("mediana", medianas)):
            bajo, alto = intervalo(valores)
            filas.append(Fila(prop, ref, clave, "bloques",
                              estimaciones[(prop, ref, clave)], bajo, alto))
        if prop in d and ref in d:
            bajo, alto = intervalo(d[prop] - d[ref])
            filas.append(Fila(prop, ref, "d_portafolio", "bloques",
                              estimaciones[(prop, ref, "d_portafolio")], bajo, alto))
    return filas


# ---------------------------------------------------------------------------
# Lectura y controles sobre las salidas publicadas
# ---------------------------------------------------------------------------

INSUMOS = ("matriz_contraste.csv", "errores_por_serie.csv", "series_y_predicciones.csv",
           "fuera_de_muestra.csv", "tabla_valorizada.csv", "error_por_mes.csv",
           "pruebas_estadisticas.txt")


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def verificar_insumos(salidas: Path) -> dict:
    """Compara el SHA-256 de cada insumo con el del manifiesto de la corrida."""
    manifiesto = json.loads((salidas / "manifiesto.json").read_text(encoding="utf-8"))
    sellos = {s["archivo"]: s["sha256"] for s in manifiesto["salidas"]}
    for nombre in INSUMOS:
        if nombre not in sellos:
            raise SystemExit(f"{nombre} no figura en manifiesto.json")
        if _sha256(salidas / nombre) != sellos[nombre]:
            raise SystemExit(f"{nombre} no coincide con el hash del manifiesto de la corrida")
    return manifiesto


def costo_por_serie(salidas: Path, mensual: pd.DataFrame) -> pd.Series:
    """Costo unitario deducido: valor de entrenamiento / demanda observada."""
    fm = pd.read_csv(salidas / "fuera_de_muestra.csv").set_index("serie")
    entrenamiento = mensual[mensual["bloque"] == "entrenamiento"]
    observada = entrenamiento.groupby("serie")["y_real"].sum()
    valor = fm["valor_entrenamiento"].reindex(observada.index)
    costo = (valor / observada).where(observada > 0)
    return costo[costo > 0]


def construir(
    salidas: Path,
    replicas: int,
    semilla: int,
    comparaciones: list[tuple[str, str]] | None = None,
    publicadas: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Remuestreos de una corrida publicada.

    Por omisión, las cinco comparaciones de la corrida principal y su control de
    reproducción (Tabla H3). ``comparaciones`` y ``publicadas`` permiten aplicar
    el mismo método a otra corrida con otras comparaciones (Tabla H4,
    ``scripts/incertidumbre_extension.py``): ``publicadas`` lleva, para cada
    comparación que la corrida imprime en ``pruebas_estadisticas.txt``, el par
    (r, rango biserial) a cuatro decimales, y el script se detiene si no se
    reproduce.
    """
    if comparaciones is None:
        comparaciones = COMPARACIONES
    if publicadas is None:
        publicadas = PUBLICADAS_EN_TXT
    manifiesto = verificar_insumos(salidas)

    matriz = pd.read_csv(salidas / "matriz_contraste.csv").set_index("serie").dropna()
    modelos = sorted({m for par in comparaciones for m in par})
    faltan = [m for m in modelos if m not in matriz.columns]
    if faltan:
        raise SystemExit(f"La matriz publicada no tiene los modelos {faltan}")

    skus = sku_de_serie(matriz.index)
    codigos = pd.factorize(skus)[0]

    # --- Valor p y tamaño del efecto publicados (misma función de la corrida)
    publicados = {}
    for prop, ref in comparaciones:
        w = wilcoxon_pareado(matriz[prop].rename(prop), matriz[ref].rename(ref))
        publicados[(prop, ref)] = w
        if (prop, ref) in publicadas:
            r_txt, rb_txt = publicadas[(prop, ref)]
            if round(w.r, 4) != r_txt or round(w.rango_biserial, 4) != rb_txt:
                raise SystemExit(f"No se reproduce el Wilcoxon publicado de {prop} frente a {ref}")

    diferencias = {
        (p, r): (matriz[p] - matriz[r]).to_numpy(dtype=float) for p, r in comparaciones
    }

    # --- Componentes valorizados por serie (D del portafolio) ---------------
    mensual = pd.read_csv(salidas / "series_y_predicciones.csv",
                          usecols=["periodo", "serie", "y_real", "bloque", *modelos])
    costo = costo_por_serie(salidas, mensual).reindex(matriz.index)
    con_costo = costo.notna().to_numpy()
    c = costo.fillna(0.0).to_numpy()

    errores = pd.read_csv(salidas / "errores_por_serie.csv")
    componentes = {}
    tabla_val = pd.read_csv(salidas / "tabla_valorizada.csv").set_index("Modelo")
    control_d = {}
    for m in modelos:
        e = errores[errores["modelo"] == m].set_index("serie").reindex(matriz.index)
        n = e["n_observaciones"].to_numpy(dtype=float)
        componentes[m] = (c * e["mae"].to_numpy() * n,
                          c * e["media_real"].to_numpy() * n,
                          c * e["media_predicha"].to_numpy() * n)
        d = float(_d_de(componentes[m], np.ones((1, len(n))))[0])
        publicada = float(tabla_val.loc[m, "D = WMAPE + |Bias| (%)"])
        if abs(d - publicada) > TOLERANCIA_D:
            raise SystemExit(f"La D reconstruida de {m} no reproduce la Tabla 8 valorizada")
        control_d[m] = (d, publicada)

    # --- Paneles mensuales del bloque de prueba (variante temporal) ----------
    prueba = mensual[mensual["bloque"] == "prueba"]
    meses = sorted(prueba["periodo"].unique())
    real = prueba.pivot(index="serie", columns="periodo", values="y_real").reindex(
        index=matriz.index, columns=meses)
    y = real.to_numpy(dtype=float)
    por_mes = pd.read_csv(salidas / "error_por_mes.csv")
    paneles = {}
    control_mensual = 0.0
    control_d_mensual = 0.0
    control_serie = 0.0
    modelos_por_mes: list[str] = []
    for m in modelos:
        p = prueba.pivot(index="serie", columns="periodo", values=m).reindex(
            index=matriz.index, columns=meses).to_numpy(dtype=float)
        ok = ~np.isnan(y) & ~np.isnan(p)
        firmado = np.where(ok, p - y, 0.0)
        absoluto = np.abs(firmado)
        valor = con_costo[:, None] & ok
        error_val = (np.where(valor, absoluto, 0.0) * c[:, None]).sum(axis=0)
        real_val = (np.where(valor, y, 0.0) * c[:, None]).sum(axis=0)
        pred_val = (np.where(valor, p, 0.0) * c[:, None]).sum(axis=0)
        paneles[m] = {"absoluto": absoluto, "firmado": firmado, "observado": ok.astype(float),
                      "error_val": error_val, "real_val": real_val, "pred_val": pred_val}
        # Controles: el panel mensual reproduce la métrica por serie y la D por mes.
        identidad = compuesto_por_serie(absoluto, firmado, ok.astype(float), np.ones((1, len(meses))))[:, 0]
        control_serie = max(control_serie, float(np.nanmax(np.abs(identidad - matriz[m].to_numpy()))))
        d_mensual = float(d_portafolio(error_val.sum(), real_val.sum(), pred_val.sum()))
        control_d_mensual = max(control_d_mensual, abs(d_mensual - control_d[m][1]))
        # error_por_mes.csv publica sólo algunos modelos: se controla con los que trae.
        pm = por_mes[por_mes["modelo"] == m].set_index("periodo").reindex(meses)
        if pm["wmape_val"].notna().all():
            modelos_por_mes.append(m)
            control_mensual = max(control_mensual, float(np.max(np.abs(
                100 * error_val / real_val - pm["wmape_val"].to_numpy()))))
            control_mensual = max(control_mensual, float(np.max(np.abs(
                100 * (pred_val - real_val) / real_val - pm["bias_val"].to_numpy()))))
    if control_serie > TOLERANCIA_SERIE:
        raise SystemExit("El panel mensual no reproduce la matriz de contraste publicada")
    if control_d_mensual > TOLERANCIA_D_MENSUAL:
        raise SystemExit("El panel mensual no reproduce la D de la Tabla 8 valorizada")
    if not modelos_por_mes or control_mensual > TOLERANCIA_D_MENSUAL:
        raise SystemExit("El panel mensual no reproduce error_por_mes.csv")

    # --- Remuestreos -----------------------------------------------------
    filas = bootstrap_por_sku(diferencias, componentes, codigos, replicas, semilla)
    estimaciones = {(f.propuesto, f.referencia, f.magnitud): f.estimacion for f in filas}
    filas += bootstrap_por_bloques(paneles, comparaciones, estimaciones, replicas, semilla)

    registros = []
    for f in filas:
        w = publicados[(f.propuesto, f.referencia)]
        registros.append({
            "comparacion": f"{f.propuesto} - {f.referencia}",
            "propuesto": f.propuesto,
            "referencia": f.referencia,
            "magnitud": f.magnitud,
            "remuestreo": f.remuestreo,
            "estimacion": f.estimacion,
            "ic95_inferior": f.ic_inferior,
            "ic95_superior": f.ic_superior,
            "excluye_cero": "sí" if excluye_cero(f.ic_inferior, f.ic_superior) else "no",
            "p_wilcoxon_publicado": w.p,
            "r_publicado": w.r,
            "rango_biserial_publicado": w.rango_biserial,
            "series_a_favor": w.gana_propuesto,
            "series_en_contra": w.gana_referencia,
            "empates": w.empates,
        })
    tabla = pd.DataFrame(registros)
    orden_m = {k: i for i, k in enumerate(MAGNITUDES)}
    orden_r = {k: i for i, k in enumerate(REMUESTREOS)}
    orden_c = {f"{p} - {r}": i for i, (p, r) in enumerate(comparaciones)}
    tabla = tabla.sort_values(
        by=["magnitud", "remuestreo", "comparacion"],
        key=lambda s: s.map({**orden_m, **orden_r, **orden_c}),
    ).reset_index(drop=True)

    resumen = {
        "replicas": replicas,
        "semilla": semilla,
        "nivel": NIVEL,
        "largo_bloque": LARGO_BLOQUE,
        "series": int(len(matriz)),
        "conglomerados_sku": int(codigos.max() + 1),
        "series_con_costo": int(con_costo.sum()),
        "meses_prueba": len(meses),
        "control_d_max_pp": max(abs(a - b) for a, b in control_d.values()),
        "control_serie_max_unidades": control_serie,
        "control_d_mensual_max_pp": control_d_mensual,
        "control_por_mes_max_pp": control_mensual,
        "modelos_control_por_mes": modelos_por_mes,
        "corrida": manifiesto["generado_en"],
        "commit": manifiesto["codigo"]["commit"],
    }
    return tabla, resumen


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def _num(x: float, decimales: int = 2) -> str:
    """Coma decimal, punto de miles y signo menos tipográfico, como el documento."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    texto = f"{x:,.{decimales}f}"
    if float(texto.replace(",", "")) == 0:
        texto = texto.lstrip("-")              # sin «−0,00»
    texto = texto.replace(",", "§").replace(".", ",").replace("§", ".")
    return texto.replace("-", "−")


def _cientifico(x: float) -> str:
    """3.9e-08 -> «3,9 × 10⁻⁸», como escribe el documento."""
    if x == 0:
        return "0"
    mantisa, exponente = f"{x:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantisa.replace('.', ',')} × 10{str(int(exponente)).translate(sup)}"


def _p(p: float) -> str:
    if p < 0.001:
        return "< 0,001"
    return _num(p, 4 if p > 0.99 else 3)


def escribir_md(tabla: pd.DataFrame, resumen: dict, ruta: Path) -> None:
    lineas = [
        "Tabla H3. Incertidumbre agrupada de las diferencias del contraste "
        "(propuesto menos referencia; negativo = el propuesto erra menos)",
        "",
    ]
    for clave, titulo in MAGNITUDES.items():
        decimales = 1 if clave == "d_portafolio" else 2   # D en puntos, como la Tabla 8
        lineas += [f"**{titulo}**", "",
                   "| Comparación | Estimación | IC 95 % por SKU | Excluye el cero (SKU) | "
                   "IC 95 % por bloques de 3 meses | Excluye el cero (bloques) | "
                   "p publicado | r publicado |",
                   "|---|---|---|---|---|---|---|---|"]
        for prop, ref in COMPARACIONES:
            sel = tabla[(tabla["propuesto"] == prop) & (tabla["referencia"] == ref)
                        & (tabla["magnitud"] == clave)].set_index("remuestreo")
            s, b = sel.loc["sku"], sel.loc["bloques"]
            lineas.append(
                f"| {NOMBRES[prop]} frente a {NOMBRES[ref]} | {_num(s['estimacion'], decimales)} | "
                f"[{_num(s['ic95_inferior'], decimales)}; {_num(s['ic95_superior'], decimales)}] | "
                f"{s['excluye_cero']} | "
                f"[{_num(b['ic95_inferior'], decimales)}; {_num(b['ic95_superior'], decimales)}] | "
                f"{b['excluye_cero']} | {_p(s['p_wilcoxon_publicado'])} | "
                f"{_num(s['r_publicado'], 2)} |"
            )
        lineas.append("")
    lineas += [
        f"Remuestreo: B = {resumen['replicas']:,} réplicas".replace(",", ".")
        + f", semilla {resumen['semilla']}, intervalos percentiles al 95 %. "
        f"Conglomerados: {resumen['conglomerados_sku']} SKU que agrupan "
        f"{_num(resumen['series'], 0)} series (cada SKU sorteado entra con todas sus series). "
        f"Bloques: {resumen['meses_prueba']} meses del bloque de prueba en bloques móviles "
        f"circulares de {resumen['largo_bloque']} meses. D valorizada sobre "
        f"{_num(resumen['series_con_costo'], 0)} series con costo. El p y el r son los del "
        "Wilcoxon unilateral publicado, que trata cada serie como independiente.",
        "",
        f"Controles: la D reconstruida reproduce la Tabla 8 valorizada con una diferencia "
        f"máxima de {_cientifico(resumen['control_d_max_pp'])} puntos; el panel mensual reproduce la "
        f"métrica por serie con una diferencia máxima de {_cientifico(resumen['control_serie_max_unidades'])} "
        f"unidades, la D de cada modelo con {_cientifico(resumen['control_d_mensual_max_pp'])} puntos y el "
        f"WMAPE y el sesgo valorizados de cada mes de error_por_mes.csv "
        f"({', '.join(resumen['modelos_control_por_mes'])}) con "
        f"{_cientifico(resumen['control_por_mes_max_pp'])} puntos (redondeo del archivo mensual). "
        "La estimación es la de la muestra completa, sin remuestrear, en ambas variantes.",
        "",
        f"Corrida: {resumen['corrida']} (commit {resumen['commit']}).",
        "",
    ]
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salidas", default=str(RAIZ / "salidas"))
    parser.add_argument("--destino", default=str(RAIZ / "resultados" / "tablas"))
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    args = parser.parse_args(argv)

    tabla, resumen = construir(Path(args.salidas), args.replicas, args.semilla)
    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino / "tabla_H3_incertidumbre_agrupada.csv", index=False,
                 float_format="%.6g")
    escribir_md(tabla, resumen, destino / "tabla_H3_incertidumbre_agrupada.md")
    print((destino / "tabla_H3_incertidumbre_agrupada.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
