"""Anexo 11 (Anexo K en la versión evaluada), Tabla K1 (Tabla 25 del documento final): sensibilidad de
la cohorte de evaluación.

Propósito
---------
La cascada publicada (apartado 2.6.3) exige que cada serie tenga venta en el
bloque de prueba (paso 6) y acota cada serie entre su primera y su última venta:
los meses posteriores a la última venta no se evalúan. Las dos condiciones usan
información posterior al origen de la prueba (abril de 2025). Este script
compara, desde las salidas de tres corridas completas del motor, la evaluación
publicada con dos cohortes fijadas con la información disponible al origen:

* **Original** (``salidas``): la corrida publicada, configuración por defecto.
* **Variante A** (``salidas_sensibilidad_a``), «cohorte al origen»: sin el paso 6
  y con la vida de cada serie extendida con ceros hasta el fin del período
  (``inclusion.exigir_presencia_prueba: false``,
  ``series.extender_vida_hasta_fin_periodo: true``).
* **Variante B** (``salidas_sensibilidad_b``), «cohorte al origen, sólo activas»:
  la A más ``inclusion.estado_al_origen: activa`` (venta en los seis meses
  previos al origen, regla de ``clasificar_estados``).

No vuelve a ajustar ningún modelo: sólo lee las salidas de cada corrida, verifica
el SHA-256 de cada insumo contra el ``manifiesto.json`` de esa corrida y no abre
ningún archivo de ``datos/`` salvo, si se pide y existe, el de quiebres de stock,
del que escribe sólo conteos agregados.

Salidas (sólo agregados; ningún identificador de serie ni de producto), en
``--destino``:

* ``tabla_K1_sensibilidad_cohorte.csv``: una fila por cohorte (series, series con
  costo, cobertura del valor de entrenamiento de la población objetivo, D
  valorizada del motor, de ``ma_12_gf``, de ``ma_12``, del ingenuo y de LightGBM,
  diferencia motor menos ``ma_12_gf`` y posición del motor entre los modelos).
* ``tabla_K1_desglose_variante_a.csv``: dentro de la variante A, por estado al
  origen (activa, dormida, descontinuada) y por venta en el bloque de prueba (con
  venta / sin venta): series, cobertura y D del motor y de ``ma_12_gf``; en los
  grupos sin venta la D no existe (la demanda valorizada es cero) y se informa el
  error valorizado y los puntos de WMAPE que el grupo agrega a la D de la A.
* ``tabla_K1_aporte_sin_venta.csv``: por modelo, el error valorizado de las series
  de la A sin venta en el bloque de prueba y los puntos de WMAPE que agregan.
* ``tabla_K1_resumen.json``: controles y conteos (commit, estado del árbol de
  trabajo y hash de configuración de cada corrida; estados al origen; cambios de
  modelo elegido; meses comunes entre la original y la A, con los modelos que
  pronostican igual y la D de la A en esos meses; D multihorizonte; quiebres).
* ``tabla_K1_sensibilidad_cohorte.md``: las tres tablas, con los controles y, si
  se leyó, el cruce con los quiebres registrados.

Definiciones (las mismas del motor):

* Costo unitario de cada serie: valor de entrenamiento (``fuera_de_muestra.csv``)
  sobre su demanda observada de entrenamiento (``series_y_predicciones.csv``),
  como los scripts de los Anexos 8 y 14. El script se detiene si la D así
  reconstruida no reproduce ``tabla_valorizada.csv`` de cada corrida.
* D = WMAPE valorizado + |sesgo valorizado agregado|, sobre las series con costo,
  a un paso en el bloque de prueba.
* Cobertura: valor de entrenamiento de las series de la cohorte sobre el de toda
  la población objetivo (la misma medida de la cascada del Anexo 11 y de la
  Tabla N2).

Uso
---
    python -X utf8 scripts/sensibilidad_cohorte.py [--original salidas]
        [--variante-a salidas_sensibilidad_a] [--variante-b salidas_sensibilidad_b]
        [--destino resultados/tablas] [--quiebres datos/crudo/quiebres.csv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

PREFIJO = "tabla_K1"
PRACTICA = "ma_12_gf"
MODELOS_TABLA = ("motor", "ma_12_gf", "ma_12", "naive_m1", "lightgbm")
NOMBRES = {
    "motor": "motor",
    "ma_12_gf": "práctica vigente (promedio móvil de 12 meses con factor de crecimiento)",
    "ma_12": "promedio móvil de 12 meses",
    "naive_m1": "ingenuo",
    "lightgbm": "LightGBM",
}
ESTADOS = ("activa", "dormida", "descontinuada")
TOLERANCIA_D = 1e-5   # puntos porcentuales
INSUMOS = ("fuera_de_muestra.csv", "errores_por_serie.csv", "series_y_predicciones.csv",
           "tabla_valorizada.csv", "cohorte_flujo.csv", "particion.csv", "seleccion_motor.csv")


# ---------------------------------------------------------------------------
# Piezas puras (las prueba tests/test_sensibilidad_cohorte.py)
# ---------------------------------------------------------------------------

def costo_por_serie(fm: pd.DataFrame, mensual: pd.DataFrame) -> pd.Series:
    """Costo unitario deducido: valor de entrenamiento / demanda observada de entrenamiento."""
    entrenamiento = mensual[mensual["bloque"] == "entrenamiento"]
    observada = entrenamiento.groupby("serie")["y_real"].sum()
    valor = fm.set_index("serie")["valor_entrenamiento"].reindex(observada.index)
    costo = (valor / observada).where(observada > 0)
    return costo[costo > 0]


def componentes_valorizados(errores: pd.DataFrame, costo: pd.Series) -> pd.DataFrame:
    """Por serie y modelo, el error, la demanda y el pronóstico valorizados del bloque de prueba.

    Sólo las series con costo; ``n`` es el número de meses evaluados. La suma de
    las columnas sobre un grupo de series da los términos de su D.
    """
    t = errores[["serie", "modelo", "mae", "media_real", "media_predicha",
                 "n_observaciones"]].copy()
    t["costo"] = t["serie"].map(costo)
    t = t.dropna(subset=["costo"])
    n = t["n_observaciones"].astype(float)
    t["error"] = t["costo"] * t["mae"] * n
    t["real"] = t["costo"] * t["media_real"] * n
    t["pred"] = t["costo"] * t["media_predicha"] * n
    return t[["serie", "modelo", "n_observaciones", "error", "real", "pred"]]


def d_valorizada(error: float, real: float, pred: float) -> float:
    """D = 100·error/real + 100·|pred − real|/real; NaN si no hay demanda."""
    if real <= 0:
        return float("nan")
    return 100.0 * error / real + 100.0 * abs(pred - real) / real


def d_por_modelo(comp: pd.DataFrame, series: pd.Index | None = None) -> pd.Series:
    """D valorizada de cada modelo sobre ``series`` (todas si es None)."""
    if series is not None:
        comp = comp[comp["serie"].isin(series)]
    sumas = comp.groupby("modelo")[["error", "real", "pred"]].sum()
    return pd.Series({m: d_valorizada(r.error, r.real, r.pred) for m, r in sumas.iterrows()},
                     dtype=float)


def cobertura(fm: pd.DataFrame, series: pd.Index) -> float:
    """Valor de entrenamiento de ``series`` sobre el de toda la población (%)."""
    valor = fm.set_index("serie")["valor_entrenamiento"].fillna(0.0)
    total = float(valor.sum())
    return 100.0 * float(valor.reindex(series).fillna(0.0).sum()) / total if total else float("nan")


def series_sin_venta_en_prueba(errores: pd.DataFrame) -> pd.Index:
    """Series cuya venta observada en el bloque de prueba suma cero (columna media_real)."""
    uno = errores[errores["modelo"] == errores["modelo"].iloc[0]]
    total = uno["media_real"] * uno["n_observaciones"]
    return pd.Index(uno.loc[total <= 0, "serie"].unique(), name="serie")


def cruce_quiebres(series: pd.Index, quiebres: pd.DataFrame, desde: str, hasta: str) -> dict:
    """Conteos agregados de quiebres registrados en el bloque de prueba para ``series``.

    ``quiebres`` trae sku, regional, fecha, semanas_medidas y semanas_en_cero
    (sin canal: se cruza por SKU y regional, los dos extremos de la clave
    SKU|CANAL|REGIONAL). Devuelve cuántas series tienen al menos un mes medido,
    al menos un mes con alguna semana en cero y todos sus meses medidos con todas
    las semanas en cero. Sólo conteos: ningún identificador.
    """
    q = quiebres.copy()
    q["periodo"] = pd.to_datetime(q["fecha"]).dt.to_period("M")
    p_desde, p_hasta = pd.Period(desde, "M"), pd.Period(hasta, "M")
    q = q[(q["periodo"] >= p_desde) & (q["periodo"] <= p_hasta)]
    q["sku"] = q["sku"].astype(str).str.strip()
    q["regional"] = q["regional"].astype(str).str.strip()
    q["con_quiebre"] = q["semanas_en_cero"] > 0
    q["quiebre_total"] = (q["semanas_medidas"] > 0) & (q["semanas_en_cero"] >= q["semanas_medidas"])
    por_par = q.groupby(["sku", "regional"]).agg(
        meses=("periodo", "nunique"), con_quiebre=("con_quiebre", "any"),
        todo_en_quiebre=("quiebre_total", "all"), meses_con_quiebre=("con_quiebre", "sum"))
    partes = pd.Index(series).astype(str).str.split("|")
    claves = pd.MultiIndex.from_arrays([partes.str[0], partes.str[-1]], names=["sku", "regional"])
    cruz = por_par.reindex(claves)
    medidas = cruz["meses"].notna()
    return {
        "series": int(len(series)),
        "con_medicion_en_prueba": int(medidas.sum()),
        "con_algun_quiebre_en_prueba": int(cruz.loc[medidas, "con_quiebre"].astype(bool).sum()),
        "con_quiebre_en_todos_los_meses_medidos": int(
            cruz.loc[medidas, "todo_en_quiebre"].astype(bool).sum()),
        "meses_con_quiebre_mediana": (float(cruz.loc[medidas, "meses_con_quiebre"].median())
                                      if medidas.any() else float("nan")),
    }


def comparar_meses_comunes(mensual_o: pd.DataFrame, mensual_a: pd.DataFrame,
                           costo_a: pd.Series, modelos: list[str]) -> dict:
    """Meses-serie del bloque de prueba que evalúa la corrida original, vistos en la variante.

    Devuelve cuántos son, qué modelos pronostican exactamente lo mismo en ellos
    en las dos corridas y la D de cada modelo en la variante restringida a esos
    meses (con el costo de la variante). Si un modelo con sesgo agregado
    negativo pronostica igual, los meses en cero que agrega la variante no
    mueven su D: lo que suman al WMAPE lo restan del |sesgo|.
    """
    clave = ["serie", "periodo"]
    prueba_o = mensual_o[(mensual_o["bloque"] == "prueba") & mensual_o["y_real"].notna()]
    unido = prueba_o.merge(mensual_a[mensual_a["bloque"] == "prueba"], on=clave,
                           suffixes=("_o", "_a"))
    if len(unido) != len(prueba_o):
        raise SystemExit("La variante no tiene todos los meses-serie que evalúa la corrida original")
    if not np.allclose(unido["y_real_o"], unido["y_real_a"], rtol=0, atol=1e-9):
        raise SystemExit("La venta observada de los meses comunes difiere entre las corridas")
    identicos = sorted(m for m in modelos if np.allclose(
        unido[f"{m}_o"].astype(float), unido[f"{m}_a"].astype(float),
        rtol=0, atol=1e-9, equal_nan=True))
    t = unido.assign(c=unido["serie"].map(costo_a)).dropna(subset=["c"])
    real = float((t["c"] * t["y_real_a"]).sum())
    d = {m: d_valorizada(float((t["c"] * (t[f"{m}_a"] - t["y_real_a"]).abs()).sum()), real,
                         float((t["c"] * t[f"{m}_a"]).sum()))
         for m in modelos}
    return {"meses_serie": int(len(prueba_o)), "modelos_con_pronostico_identico": identicos,
            "D_variante_en_meses_de_la_original": d}


# ---------------------------------------------------------------------------
# Lectura de una corrida, con sus controles
# ---------------------------------------------------------------------------

def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


@dataclass
class Corrida:
    nombre: str
    carpeta: Path
    manifiesto: dict
    fm: pd.DataFrame
    errores: pd.DataFrame
    comp: pd.DataFrame
    series: pd.Index
    flujo: pd.DataFrame
    prueba: tuple[str, str]
    d_publicada: pd.Series
    con_venta_validacion: pd.Index = field(default_factory=lambda: pd.Index([]))
    control_d: float = 0.0
    controles: list[str] = field(default_factory=list)


def leer_corrida(nombre: str, carpeta: Path, verificar_hash: bool = True) -> Corrida:
    manifiesto = json.loads((carpeta / "manifiesto.json").read_text(encoding="utf-8"))
    if verificar_hash:
        sellos = {s["archivo"]: s["sha256"] for s in manifiesto.get("salidas", [])}
        for archivo in INSUMOS:
            if archivo not in sellos:
                raise SystemExit(f"{carpeta.name}/{archivo} no figura en su manifiesto.json")
            if _sha256(carpeta / archivo) != sellos[archivo]:
                raise SystemExit(f"{carpeta.name}/{archivo} no coincide con el hash de su manifiesto")

    fm = pd.read_csv(carpeta / "fuera_de_muestra.csv")
    errores = pd.read_csv(carpeta / "errores_por_serie.csv")
    mensual = pd.read_csv(carpeta / "series_y_predicciones.csv",
                          usecols=["serie", "y_real", "bloque"])
    flujo = pd.read_csv(carpeta / "cohorte_flujo.csv")
    particion = pd.read_csv(carpeta / "particion.csv").set_index("bloque")
    prueba = (str(particion.loc["prueba", "desde"]), str(particion.loc["prueba", "hasta"]))
    tabla_val = pd.read_csv(carpeta / "tabla_valorizada.csv").set_index("Modelo")
    d_publicada = tabla_val["D = WMAPE + |Bias| (%)"].astype(float)

    series = pd.Index(sorted(errores["serie"].unique()), name="serie")
    incluidas = pd.Index(sorted(fm.loc[fm["incluida"].astype(bool), "serie"]), name="serie")
    final = int(flujo["series"].iloc[-1])
    if not (len(series) == len(incluidas) == final) or not series.equals(incluidas):
        raise SystemExit(f"{carpeta.name}: la cohorte de errores_por_serie, fuera_de_muestra y "
                         f"cohorte_flujo no coincide ({len(series)}, {len(incluidas)}, {final})")
    manif_final = manifiesto.get("resultados", {}).get("cohorte", {}).get("series_finales")
    if manif_final is not None and int(manif_final) != final:
        raise SystemExit(f"{carpeta.name}: el manifiesto declara {manif_final} series y la cascada {final}")

    costo = costo_por_serie(fm, mensual)
    comp = componentes_valorizados(errores, costo)
    d = d_por_modelo(comp)
    diferencia = float((d - d_publicada.reindex(d.index)).abs().max())
    if not np.isfinite(diferencia) or diferencia > TOLERANCIA_D:
        raise SystemExit(f"{carpeta.name}: la D reconstruida no reproduce tabla_valorizada.csv "
                         f"(diferencia máxima {diferencia:.2e} puntos)")
    validacion = mensual[(mensual["bloque"] == "validacion") & (mensual["y_real"] > 0)]
    con_venta_validacion = pd.Index(sorted(validacion["serie"].unique()), name="serie")
    return Corrida(nombre, carpeta, manifiesto, fm, errores, comp, series, flujo, prueba,
                   d_publicada, con_venta_validacion=con_venta_validacion, control_d=diferencia)


def fila_cohorte(c: Corrida) -> dict:
    d = c.d_publicada
    orden = d.rank(method="min").astype(int)
    con_costo = int(c.comp.loc[c.comp["modelo"] == "motor", "serie"].nunique())
    fila = {
        "cohorte": c.nombre,
        "series": int(len(c.series)),
        "series_con_costo": con_costo,
        "cobertura_valor_entrenamiento_pct": cobertura(c.fm, c.series),
    }
    for m in MODELOS_TABLA:
        fila[f"D_{m}"] = float(d[m])
    fila[f"motor_menos_{PRACTICA}_pp"] = float(d["motor"] - d[PRACTICA])
    fila["posicion_motor"] = int(orden["motor"])
    fila["modelos"] = int(len(d))
    fila["modelo_de_menor_D"] = str(d.idxmin())
    fila["duracion_segundos"] = c.manifiesto.get("duracion_segundos")
    fila["commit"] = c.manifiesto.get("codigo", {}).get("commit")
    return fila


def desglose(c: Corrida, original: Corrida | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Desglose de la variante A por estado al origen y venta en el bloque de prueba."""
    if "estado_al_origen" not in c.fm.columns:
        raise SystemExit(f"{c.carpeta.name}/fuera_de_muestra.csv no trae estado_al_origen")
    estado = c.fm.set_index("serie")["estado_al_origen"].reindex(c.series)
    sin_venta = series_sin_venta_en_prueba(c.errores)
    venta = pd.Series(np.where(c.series.isin(sin_venta), "sin venta en prueba",
                               "con venta en prueba"), index=c.series)
    sumas_total = c.comp.groupby("modelo")[["error", "real", "pred"]].sum()
    demanda_total = float(sumas_total.loc["motor", "real"])

    grupos: list[tuple[str, str, pd.Index]] = [("todas", "todas", c.series)]
    for v in ("con venta en prueba", "sin venta en prueba"):
        grupos.append(("todas", v, c.series[venta.to_numpy() == v]))
    # Activas o dormidas: la parte de la A que pasa el paso 5 con venta observada
    # en validación, sin depender de ventas posteriores al origen.
    al_origen = np.isin(estado.to_numpy(), ("activa", "dormida"))
    grupos.append(("activa o dormida", "todas", c.series[al_origen]))
    for e in ESTADOS:
        en_estado = estado.to_numpy() == e
        grupos.append((e, "todas", c.series[en_estado]))
        for v in ("con venta en prueba", "sin venta en prueba"):
            grupos.append((e, v, c.series[en_estado & (venta.to_numpy() == v)]))
    otros = sorted(set(estado.dropna()) - set(ESTADOS))
    if otros:
        raise SystemExit(f"Estados al origen no previstos en la cohorte: {otros}")

    filas = []
    for e, v, s in grupos:
        sub = c.comp[c.comp["serie"].isin(s)]
        sumas = sub.groupby("modelo")[["error", "real", "pred"]].sum()
        fila = {
            "estado_al_origen": e, "venta_en_prueba": v,
            "series": int(len(s)),
            "series_con_costo": int(sub.loc[sub["modelo"] == "motor", "serie"].nunique()),
            "cobertura_valor_entrenamiento_pct": cobertura(c.fm, s),
            "demanda_valorizada_prueba_bs": float(sumas["real"].get("motor", 0.0)),
        }
        for m in ("motor", PRACTICA):
            r = sumas.loc[m] if m in sumas.index else pd.Series({"error": 0.0, "real": 0.0, "pred": 0.0})
            fila[f"D_{m}"] = d_valorizada(r["error"], r["real"], r["pred"])
            fila[f"error_valorizado_{m}_bs"] = float(r["error"])
            fila[f"puntos_wmape_en_A_{m}"] = 100.0 * float(r["error"]) / demanda_total
        fila[f"motor_menos_{PRACTICA}_pp"] = fila["D_motor"] - fila[f"D_{PRACTICA}"]
        filas.append(fila)
    tabla = pd.DataFrame(filas)

    sub = c.comp[c.comp["serie"].isin(sin_venta)]
    aporte = sub.groupby("modelo")[["error", "pred"]].sum()
    aporte = pd.DataFrame({
        "modelo": aporte.index,
        "error_valorizado_sin_venta_bs": aporte["error"].to_numpy(),
        "puntos_wmape_en_A": 100.0 * aporte["error"].to_numpy() / demanda_total,
        "D_variante_A": c.d_publicada.reindex(aporte.index).to_numpy(),
    })
    componentes = terminos_por_modelo(c.comp).add_suffix("_A")
    if original is not None:
        componentes = componentes.join(terminos_por_modelo(original.comp).add_suffix("_original"))
    aporte = aporte.join(componentes, on="modelo")
    aporte = aporte.sort_values("puntos_wmape_en_A").reset_index(drop=True)
    return tabla, aporte, sin_venta


def terminos_por_modelo(comp: pd.DataFrame) -> pd.DataFrame:
    """WMAPE, sesgo y D valorizados de cada modelo (columnas wmape, sesgo, D)."""
    s = comp.groupby("modelo")[["error", "real", "pred"]].sum()
    t = pd.DataFrame({"wmape": 100.0 * s["error"] / s["real"],
                      "sesgo": 100.0 * (s["pred"] - s["real"]) / s["real"]})
    t["D"] = t["wmape"] + t["sesgo"].abs()
    return t


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def _num(x, decimales: int = 1) -> str:
    """Coma decimal, punto de miles y signo menos tipográfico, como el documento."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    texto = f"{x:,.{decimales}f}"
    if float(texto.replace(",", "")) == 0:
        texto = texto.lstrip("-")
    texto = texto.replace(",", "§").replace(".", ",").replace("§", ".")
    return texto.replace("-", "−")


def escribir_md(cohortes: pd.DataFrame, tabla_a: pd.DataFrame | None, aporte: pd.DataFrame | None,
                quiebres: dict | None, notas: list[str], ruta: Path) -> None:
    lineas = ["Tabla K1. Sensibilidad de la cohorte de evaluación: tamaño, cobertura y D valorizada "
              "a un paso en el bloque de prueba", "",
              "| Cohorte | Series | Series con costo | Cobertura del valor de entrenamiento (%) | "
              "D motor (%) | D práctica vigente (%) | D promedio móvil de 12 meses (%) | "
              "D ingenuo (%) | D LightGBM (%) | Motor menos práctica vigente (pp) | "
              "Posición del motor |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in cohortes.iterrows():
        lineas.append(
            f"| {r['cohorte']} | {_num(r['series'], 0)} | {_num(r['series_con_costo'], 0)} | "
            f"{_num(r['cobertura_valor_entrenamiento_pct'], 2)} | {_num(r['D_motor'])} | "
            f"{_num(r['D_ma_12_gf'])} | {_num(r['D_ma_12'])} | {_num(r['D_naive_m1'])} | "
            f"{_num(r['D_lightgbm'])} | {_num(r['motor_menos_ma_12_gf_pp'])} | "
            f"{int(r['posicion_motor'])} de {int(r['modelos'])} |")
    lineas.append("")
    if tabla_a is not None:
        lineas += ["**Variante A por estado al origen y venta en el bloque de prueba**", "",
                   "| Estado al origen | Venta en prueba | Series | Series con costo | "
                   "Cobertura (%) | D motor (%) | D práctica vigente (%) | "
                   "Motor menos práctica vigente (pp) | Puntos de WMAPE en la A: motor | "
                   "Puntos de WMAPE en la A: práctica vigente |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in tabla_a.iterrows():
            lineas.append(
                f"| {r['estado_al_origen']} | {r['venta_en_prueba']} | {_num(r['series'], 0)} | "
                f"{_num(r['series_con_costo'], 0)} | {_num(r['cobertura_valor_entrenamiento_pct'], 2)} | "
                f"{_num(r['D_motor'])} | {_num(r['D_ma_12_gf'])} | "
                f"{_num(r['motor_menos_ma_12_gf_pp'])} | {_num(r['puntos_wmape_en_A_motor'], 2)} | "
                f"{_num(r['puntos_wmape_en_A_ma_12_gf'], 2)} |")
        lineas.append("")
    if aporte is not None:
        lineas += ["**Series de la variante A sin venta en el bloque de prueba: error valorizado "
                   "por modelo**", "",
                   "| Modelo | Error valorizado de las series sin venta (Bs) | Puntos de WMAPE que "
                   "agregan a la A | WMAPE original (%) | Sesgo original (%) | D original (%) | "
                   "WMAPE en la A (%) | Sesgo en la A (%) | D en la A (%) |",
                   "|---|---|---|---|---|---|---|---|---|"]
        for _, r in aporte.iterrows():
            lineas.append(f"| {r['modelo']} | {_num(r['error_valorizado_sin_venta_bs'], 0)} | "
                          f"{_num(r['puntos_wmape_en_A'], 2)} | {_num(r.get('wmape_original'), 2)} | "
                          f"{_num(r.get('sesgo_original'), 2)} | {_num(r.get('D_original'), 2)} | "
                          f"{_num(r['wmape_A'], 2)} | {_num(r['sesgo_A'], 2)} | "
                          f"{_num(r['D_variante_A'], 2)} |")
        lineas.append("")
    if quiebres is not None:
        lineas += ["**Quiebres registrados en el bloque de prueba (cruce por SKU y regional)**", ""]
        for clave, q in quiebres.items():
            lineas.append(
                f"- {clave}: {_num(q['series'], 0)} series; {_num(q['con_medicion_en_prueba'], 0)} "
                f"con medición de existencias en el bloque; {_num(q['con_algun_quiebre_en_prueba'], 0)} "
                f"con al menos un mes con semanas de existencia en cero; "
                f"{_num(q['con_quiebre_en_todos_los_meses_medidos'], 0)} con existencia en cero en "
                f"todas las semanas medidas del bloque.")
        lineas += ["", "Fuente de los quiebres: archivo de quiebres de la captura (semanas con "
                   "existencia en cero por SKU, regional y mes; sin canal). Sólo conteos agregados.",
                   ""]
        lineas.append("")
    lineas += notas
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _estado_arbol(manifiesto: dict) -> str:
    """Declara si la corrida se lanzó con el árbol de trabajo limpio y, si no, qué archivos difieren."""
    codigo = manifiesto.get("codigo", {})
    if codigo.get("arbol_limpio") is None or codigo.get("arbol_limpio"):
        return ""
    archivos = codigo.get("archivos_modificados") or []
    return ("; árbol de trabajo con cambios sin commit"
            + (": " + ", ".join(str(a) for a in archivos) if archivos else ""))


def construir(original: Path, variante_a: Path, variante_b: Path | None,
              quiebres: Path | None, verificar_hash: bool = True) -> dict:
    corridas = [leer_corrida("original", original, verificar_hash),
                leer_corrida("A: cohorte al origen", variante_a, verificar_hash)]
    if variante_b is not None and (variante_b / "manifiesto.json").exists():
        corridas.append(leer_corrida("B: cohorte al origen, sólo activas", variante_b, verificar_hash))
    orig, a = corridas[0], corridas[1]

    # Controles de contención: la A contiene la cohorte publicada; la B está dentro de la A.
    if not orig.series.isin(a.series).all():
        raise SystemExit("La variante A no contiene la cohorte publicada")
    if len(corridas) == 3:
        b = corridas[2]
        if not b.series.isin(a.series).all():
            raise SystemExit("La variante B no está contenida en la A")
        activas_a = a.fm.set_index("serie")["estado_al_origen"].reindex(a.series) == "activa"
        if not b.series.equals(pd.Index(sorted(a.series[activas_a.to_numpy()]), name="serie")):
            raise SystemExit("La variante B no es la A restringida a las activas al origen")

    cohortes = pd.DataFrame([fila_cohorte(c) for c in corridas])
    tabla_a, aporte, sin_venta = desglose(a, orig)
    agregadas = a.series.difference(orig.series)
    if not sin_venta.isin(agregadas).all():
        raise SystemExit("Hay series de la cohorte publicada sin venta en el bloque de prueba")

    quiebres_res = None
    quiebres_sello = None
    if quiebres is not None and quiebres.exists():
        # El archivo de quiebres es un insumo más: su hash debe ser el que la
        # corrida publicada selló en datos.auxiliares (si lo selló).
        auxiliares = {s.get("archivo"): s.get("sha256")
                      for s in orig.manifiesto.get("datos", {}).get("auxiliares", []) or []}
        if quiebres.name in auxiliares:
            if verificar_hash and _sha256(quiebres) != auxiliares[quiebres.name]:
                raise SystemExit(f"{quiebres.name} no coincide con el hash que selló la corrida "
                                 "publicada (datos.auxiliares)")
            quiebres_sello = "verificado contra datos.auxiliares de la corrida publicada"
        else:
            quiebres_sello = "sin sello en el manifiesto de la corrida publicada"
        q = pd.read_csv(quiebres)
        quiebres_res = {
            "A, sin venta en prueba": cruce_quiebres(sin_venta, q, *a.prueba),
            "A, con venta en prueba": cruce_quiebres(a.series.difference(sin_venta), q, *a.prueba),
        }

    estado_a = a.fm.set_index("serie")["estado_al_origen"].reindex(a.series)
    validacion_por_estado = {
        e: {"series": int((estado_a == e).sum()),
            "con_venta_en_validacion": int(((estado_a == e)
                                            & a.series.isin(a.con_venta_validacion)).sum()),
            "con_venta_en_prueba": int(((estado_a == e) & ~a.series.isin(sin_venta)).sum()),
            "en_cohorte_publicada": int(((estado_a == e) & a.series.isin(orig.series)).sum())}
        for e in ESTADOS}
    eleccion = {c.nombre: pd.read_csv(c.carpeta / "seleccion_motor.csv",
                                      usecols=["serie", "modelo_elegido"]).set_index("serie")["modelo_elegido"]
                for c in corridas}
    base = eleccion[orig.nombre]
    # Sólo sobre las series de la cohorte publicada que también están en la variante.
    cambios_eleccion = {}
    for n, e in eleccion.items():
        if n == orig.nombre:
            continue
        comunes = base.index.intersection(e.index)
        cambios_eleccion[n] = {"series_comunes": int(len(comunes)),
                               "cambian": int((e.loc[comunes] != base.loc[comunes]).sum())}
    multihorizonte = {c.nombre: c.manifiesto.get("resultados", {}).get("multihorizonte", {})
                      .get("D_global_pct") for c in corridas}
    obs_orig = int(orig.errores.loc[orig.errores["modelo"] == "motor", "n_observaciones"].sum())
    obs_a_con = int(a.errores.loc[(a.errores["modelo"] == "motor")
                                  & a.errores["serie"].isin(orig.series), "n_observaciones"].sum())

    # Meses comunes: qué modelos pronostican igual en la original y en la A, y la D
    # de la A en los meses que evalúa la original (separa el efecto de los ceros
    # agregados del efecto de reentrenar el modelo global y de la selección).
    modelos = sorted(orig.errores["modelo"].unique())
    m_o = pd.read_csv(orig.carpeta / "series_y_predicciones.csv")
    m_a = pd.read_csv(a.carpeta / "series_y_predicciones.csv")
    meses_comunes = None
    if all(m in m_o.columns and m in m_a.columns for m in modelos):
        meses_comunes = comparar_meses_comunes(m_o, m_a, costo_por_serie(a.fm, m_a), modelos)
    # La B, sobre las series que comparte con la original (sus activas con venta en prueba).
    meses_comunes_b = None
    if len(corridas) == 3:
        m_b = pd.read_csv(corridas[2].carpeta / "series_y_predicciones.csv")
        if all(m in m_b.columns for m in modelos) and meses_comunes is not None:
            m_ob = m_o[m_o["serie"].isin(set(m_b["serie"]))]
            meses_comunes_b = comparar_meses_comunes(
                m_ob, m_b, costo_por_serie(corridas[2].fm, m_b), modelos)
    resumen = {
        "corridas": {c.nombre: {"carpeta": c.carpeta.name,
                                "generado_en": c.manifiesto.get("generado_en"),
                                "commit": c.manifiesto.get("codigo", {}).get("commit"),
                                "arbol_limpio": c.manifiesto.get("codigo", {}).get("arbol_limpio"),
                                "archivos_modificados": c.manifiesto.get("codigo", {}).get(
                                    "archivos_modificados", []),
                                "configuracion_sha256": c.manifiesto.get("configuracion", {}).get("sha256"),
                                "duracion_segundos": c.manifiesto.get("duracion_segundos"),
                                "control_d_max_pp": c.control_d,
                                "flujo": c.flujo.to_dict(orient="records")}
                     for c in corridas},
        "variante_a": {"series_agregadas": int(len(agregadas)),
                       "sin_venta_en_prueba": int(len(sin_venta)),
                       "meses_evaluados_cohorte_publicada_original": obs_orig,
                       "meses_evaluados_mismas_series_en_A": obs_a_con,
                       "por_estado_al_origen": validacion_por_estado,
                       "meses_de_la_original": meses_comunes},
        "cambios_de_modelo_elegido_en_la_cohorte_publicada": cambios_eleccion,
        "multihorizonte_D_global_pct": multihorizonte,
        "quiebres": quiebres_res,
        "quiebres_sello": quiebres_sello,
        "variante_b_disponible": len(corridas) == 3,
        "variante_b": {"meses_de_la_original_en_series_comunes": meses_comunes_b},
    }

    notas = [
        "D valorizada = WMAPE valorizado + |sesgo valorizado agregado|, a un paso en el bloque de "
        "prueba, sobre las series con costo de cada cohorte. Cada D se compara con la de los otros "
        "modelos de la MISMA cohorte. Cobertura: valor de entrenamiento de la cohorte sobre el de la "
        "población objetivo. «Práctica vigente»: réplica de la fórmula de planificación con factor de "
        "crecimiento (ma_12_gf), sin los ajustes de consenso del plan final.",
        "",
        "Variante A: sin el paso 6 de la cascada (presencia en prueba) y con la vida de cada serie "
        "extendida con ceros hasta el fin del período; agrega "
        f"{len(agregadas)} series a la cohorte publicada, de las que {len(sin_venta)} no tienen venta "
        "en el bloque de prueba. En la A las series de la cohorte publicada se evalúan en los doce "
        f"meses del bloque ({_num(obs_a_con, 0)} meses-serie, frente a {_num(obs_orig, 0)} en la "
        "corrida publicada, que no evalúa los meses posteriores a la última venta).",
        "Variante B: la A restringida a las series activas al origen (venta en los seis meses previos "
        "a abril de 2025, regla de clasificar_estados).",
        "Estado al origen (columna estado_al_origen de fuera_de_muestra.csv de la A): activa, venta en "
        "los seis meses previos; dormida, de seis a once meses sin venta; descontinuada, doce o más, "
        "medido al cierre de validación sobre la actividad observada. En los grupos sin venta en "
        "prueba la D no existe (demanda valorizada nula): se informan los puntos de WMAPE que su "
        "error agrega a la D de la variante A (error valorizado del grupo sobre la demanda valorizada "
        "de toda la A).",
        "Paso 5 en la A (presencia en validación sobre la vida observada, de la primera a la última "
        "venta del período): de las " + _num(validacion_por_estado["descontinuada"]["series"], 0)
        + " series descontinuadas al origen, " + _num(validacion_por_estado["descontinuada"][
            "con_venta_en_validacion"], 0)
        + " tienen venta en validación y " + _num(validacion_por_estado["descontinuada"][
            "con_venta_en_prueba"], 0)
        + " tienen venta en prueba: pasan el paso 5 porque su vida observada se prolonga hasta el "
        "bloque de prueba, información posterior al origen. Por la misma razón "
        + _num(validacion_por_estado["descontinuada"]["en_cohorte_publicada"], 0)
        + " de ellas ya están en la cohorte publicada (su paso 5 mira las filas del bloque de "
        "validación, que existen porque la vida acotada llega hasta la última venta). La fila "
        "«activa o dormida» es la A sin ellas; la B, que exige actividad al origen, no las contiene.",
        "",
        "Controles: en cada corrida, la cohorte de errores_por_serie, fuera_de_muestra y cohorte_flujo "
        "coincide; la D reconstruida desde los archivos por serie reproduce tabla_valorizada.csv "
        "(diferencia máxima "
        + ", ".join(f"{c.nombre}: {c.control_d:.1e} pp" for c in corridas)
        + "); la A contiene la cohorte publicada"
        + ("; la B es la A restringida a las activas al origen" if len(corridas) == 3 else "")
        + "; todas las series sin venta en prueba son series agregadas por la A.",
        "",
        "Corridas: " + "; ".join(
            f"{c.nombre} = {c.carpeta.name} ({c.manifiesto.get('generado_en')}, commit "
            f"{str(c.manifiesto.get('codigo', {}).get('commit'))[:7]}, "
            f"{_num(c.manifiesto.get('duracion_segundos'), 1)} s"
            + _estado_arbol(c.manifiesto) + ")" for c in corridas) + ".",
    ]
    if quiebres_sello is not None:
        notas.append(f"Archivo de quiebres: {quiebres_sello}.")
    if meses_comunes is not None:
        iguales = meses_comunes["modelos_con_pronostico_identico"]
        d_mc = meses_comunes["D_variante_en_meses_de_la_original"]
        notas.append(
            "Meses comunes: en los " + _num(meses_comunes["meses_serie"], 0) + " meses-serie del "
            "bloque de prueba que evalúa la corrida original, " + _num(len(iguales), 0) + " de los "
            + _num(len(d_mc), 0) + " modelos pronostican exactamente lo mismo en la A ("
            + ", ".join(iguales) + "); restringida a esos meses, la D del motor en la A es "
            + _num(d_mc.get("motor"), 2) + " % y la de " + PRACTICA + ", "
            + _num(d_mc.get(PRACTICA), 2) + " %.")
    if len(corridas) < 3:
        notas.append("La variante B no está disponible (falta su manifiesto.json): la tabla no la incluye.")
    if quiebres is not None and quiebres_res is None:
        notas.append("No se encontró el archivo de quiebres: el cruce se omite.")
    return {"cohortes": cohortes, "desglose_a": tabla_a, "aporte": aporte,
            "quiebres": quiebres_res, "resumen": resumen, "notas": notas}


def escribir(resultado: dict, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    resultado["cohortes"].to_csv(destino / f"{PREFIJO}_sensibilidad_cohorte.csv",
                                 index=False, float_format="%.6f")
    resultado["desglose_a"].to_csv(destino / f"{PREFIJO}_desglose_variante_a.csv",
                                   index=False, float_format="%.6f")
    resultado["aporte"].to_csv(destino / f"{PREFIJO}_aporte_sin_venta.csv",
                               index=False, float_format="%.6f")
    (destino / f"{PREFIJO}_resumen.json").write_text(
        json.dumps(resultado["resumen"], ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    escribir_md(resultado["cohortes"], resultado["desglose_a"], resultado["aporte"],
                resultado["quiebres"], resultado["notas"],
                destino / f"{PREFIJO}_sensibilidad_cohorte.md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--original", default=str(RAIZ / "salidas"))
    parser.add_argument("--variante-a", default=str(RAIZ / "salidas_sensibilidad_a"))
    parser.add_argument("--variante-b", default=str(RAIZ / "salidas_sensibilidad_b"))
    parser.add_argument("--destino", default=str(RAIZ / "resultados" / "tablas"))
    parser.add_argument("--quiebres", default=str(RAIZ / "datos" / "crudo" / "quiebres.csv"),
                        help="archivo de quiebres de la captura (opcional; sólo conteos agregados)")
    parser.add_argument("--sin-quiebres", action="store_true", help="no cruza con los quiebres")
    args = parser.parse_args(argv)

    resultado = construir(Path(args.original), Path(args.variante_a), Path(args.variante_b),
                          None if args.sin_quiebres else Path(args.quiebres))
    destino = Path(args.destino)
    escribir(resultado, destino)
    print((destino / f"{PREFIJO}_sensibilidad_cohorte.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
