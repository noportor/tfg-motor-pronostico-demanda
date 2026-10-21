"""Anexo 16 (Anexo P en la versión evaluada): Tablas P2 y P3 (Tablas 43 y 44 del documento final) de la
evaluación comparativa de referencia (benchmarks).

Propósito
---------
Reúne, sin correr ningún modelo, el costo y la precisión de cada brazo que ya
registran las salidas publicadas del motor, con las filas, columnas y rótulos
del documento:

* Tabla P2, corrida principal (gestión de prueba 2026): segundos de ajuste y
  pronóstico a un paso de cada brazo (registro de flujo, etapa de modelos; en
  el motor, la suma de los diez brazos), los mismos segundos con la
  preparación de variables donde corresponde (sólo el modelo global la
  consume; en el motor se suma además la selección por serie), milisegundos
  por serie amortizados sobre la muestra, D valorizada a un paso, D valorizada
  global en el horizonte (orígenes rodantes), rango medio de Friedman y series
  que el motor eligió (en la fila del motor, «n. d.»: reparte la muestra). Agrega, rotuladas «fuera del
  conjunto», dos referencias calculadas desde las mismas salidas: el ingenuo
  estacional (la venta del mismo mes del año anterior) y la aproximación de
  Syntetos y Boylan (SBA), que multiplica el pronóstico de Croston por
  (1 menos alfa/2) con el alfa que Croston eligió en cada serie.
* Tabla P3, extensión exploratoria en las dos gestiones (2026 y 2025): segundos,
  D a un paso y D en el horizonte de los pronosticadores de la extensión y de
  dos anclas medidas en el mismo entorno (LightGBM y Holt-Winters), con el
  costo de la búsqueda de hiperparámetros de cada uno. Los valores del
  ingenuo en ese entorno, que la nota de la tabla cita, van al resumen.

Tiempos. Son de una sola medición, sin repeticiones, y cambian de una corrida
a otra aunque los resultados sean idénticos bit a bit. Por omisión se toman
del registro de flujo de la carpeta ``--principal``; con ``--tiempos`` se
toman de otro registro de flujo (por ejemplo, el de la corrida que cita el
documento), y la salida declara de qué corrida provienen. Los tiempos de la
corrida principal y los de la extensión provienen de entornos distintos y no
se comparan uno a uno. El registro de flujo no guarda procesador ni memoria.

El costo unitario de cada serie, que valoriza las métricas de las referencias
fuera del conjunto, se deduce de las salidas publicadas: valor de entrenamiento de
``fuera_de_muestra.csv`` sobre la demanda OBSERVADA de entrenamiento de
``series_y_predicciones.csv``. El script no abre ningún archivo de ``datos/``.
El cálculo se controla reproduciendo la D a un paso publicada del motor, del
ingenuo y de Croston y la D publicada en el horizonte del ingenuo y de
Croston, y la cobertura del horizonte, antes de calcular las dos referencias
nuevas; si un control no se cumple, el script se detiene.

Supuestos declarados de las referencias fuera del conjunto:

* Ingenuo estacional: el pronóstico del mes t es la venta observada en t menos
  12. Con orígenes rodantes y horizonte de hasta 12 meses ese valor ya es
  conocido en todo origen, de modo que el pronóstico no depende del origen.
* SBA: el motor proyecta Croston en el horizonte repitiendo su pronóstico a un
  paso desde el origen, y el factor de SBA es constante por serie, así que la
  proyección de SBA en el horizonte se obtiene escalando la de Croston. Es una
  aproximación respecto de un SBA con su propio alfa.

Redondeo: mitad hacia arriba sobre el valor decimal, como en las tablas del
documento; coma decimal y punto de miles; «n. d.» cuando no hay dato.

Uso
---
    python scripts/anexo_p_benchmarks.py --motor . --destino resultados/tablas \\
        [--tiempos ruta/al/flujo.json]

Escribe en ``--destino`` (por omisión ``resultados/tablas`` dentro del motor):
``anexo_p.md``, ``anexo_p_p2.csv``, ``anexo_p_p3.csv`` y
``anexo_p_resumen.json``. Determinista; sólo agregados.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

DESTINO = "resultados/tablas"
NO_DISPONIBLE = "n. d."
# Rótulos de la Tabla 8 del documento (Holt-Winters con guion).
NOMBRES_P2 = {
    "motor": "Motor de selección automática",
    "lightgbm": "LightGBM",
    "exp_smooth_opt": "Suavizado exponencial simple",
    "holt_winters": "Holt-Winters",
    "ma_2": "Promedio móvil 2 meses",
    "ma_12": "Promedio móvil 12 meses",
    "naive_m1": "Ingenuo",
    "croston": "Croston",
    "naive_m1_gf": "Ingenuo con factor de crecimiento",
    "ma_2_gf": "Prom. móvil 2 m con factor de crecimiento",
    "ma_12_gf": "Prom. móvil 12 m con factor de crecimiento",
    "ingenuo_estacional": "Ingenuo estacional (fuera del conjunto)",
    "sba": "SBA aproximado (fuera del conjunto)",
}
FUERA_DEL_MENU = ["ingenuo_estacional", "sba"]
ESTADISTICOS = ["naive_m1", "naive_m1_gf", "ma_2", "ma_2_gf", "ma_12", "ma_12_gf",
                "exp_smooth_opt", "holt_winters", "croston"]
# Filas de la Tabla P3: clave, rótulo y celda de calibración. La celda de las
# redes y de LightGBM se completa con su tabla de ensayos («s (ensayos)»).
EXTENSION = [
    ("chronos", "Chronos-Bolt (fundacional)", "Sin búsqueda propia"),
    ("dlinear", "DLinear", "Sin búsqueda propia"),
    ("nhits", "N-HiTS", None),
    ("tft", "TFT", None),
    ("lightgbm_directo", "LightGBM con proyección directa", "Sin búsqueda propia"),
    ("lightgbm", "LightGBM", None),
    ("holt_winters", "Holt-Winters", "Sin búsqueda propia"),
]
# Ancla que la nota de la Tabla P3 cita sin fila propia.
ANCLA_EN_NOTA = "naive_m1"
CALIBRACION = {"lightgbm": "lightgbm_tuning_ensayos.csv", "nhits": "nhits_tuning_ensayos.csv",
               "tft": "tft_tuning_ensayos.csv"}
TOLERANCIA = 0.05  # puntos de D: los controles deben reproducir lo publicado


# ---------------------------------------------------------------- utilidades
def _num(x, decimales: int = 1) -> str:
    """Número con coma decimal, punto de miles y redondeo mitad hacia arriba."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NO_DISPONIBLE
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimales), rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    texto = f"{abs(d):,.{decimales}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    # Signo menos tipográfico, como en las tablas del documento.
    return ("−" + texto) if d < 0 else texto


def _etapa(flujo: dict, ident: str) -> dict:
    """Etapa del registro de flujo con ese identificador (vacía si no está)."""
    return next((e for e in flujo["etapas"] if e["id"] == ident), {})


def _leer_flujo(ruta: Path) -> dict:
    """Registro de flujo de una corrida."""
    return json.loads(ruta.read_text(encoding="utf-8"))


def _metricas(y, p, c) -> dict:
    """WMAPE, sesgo y D valorizados (mismas definiciones del motor)."""
    demanda = float((c * y).sum())
    wmape = 100 * float((c * np.abs(p - y)).sum()) / demanda
    sesgo = 100 * float((c * (p - y)).sum()) / demanda
    return {"wmape": wmape, "sesgo": sesgo, "D": wmape + abs(sesgo), "demanda": demanda}


def _d_horizonte_publicada(carpeta: Path) -> pd.Series:
    """D global en el horizonte por modelo, recalculada desde multihorizonte_horizonte.csv."""
    h = pd.read_csv(carpeta / "multihorizonte_horizonte.csv")
    h["sesgo_abs"] = h["bias_val"] / 100 * h["demanda"]
    g = h.groupby("modelo").agg(dem=("demanda", "sum"), err=("error", "sum"), ses=("sesgo_abs", "sum"))
    return (g["err"] / g["dem"] + (g["ses"] / g["dem"]).abs()) * 100


def _costo_por_serie(carpeta: Path, datos: pd.DataFrame) -> pd.Series:
    """Costo unitario por serie deducido de las salidas publicadas.

    La tabla de fuera de muestra guarda el valor de entrenamiento de cada serie
    (demanda OBSERVADA de entrenamiento por costo unitario). Su volumen, en
    cambio, es el del panel de ajuste, que lleva corregido el evento exógeno;
    por eso el costo se obtiene dividiendo el valor por la demanda observada de
    entrenamiento, que trae ``series_y_predicciones.csv``. Una serie sin costo
    queda con valor cero y se trata como «sin costo», igual que en el motor.
    """
    fm = pd.read_csv(carpeta / "fuera_de_muestra.csv").set_index("serie")
    observada = datos[datos["bloque"] == "entrenamiento"].groupby("serie")["y_real"].sum()
    valor = fm["valor_entrenamiento"].reindex(observada.index)
    costo = (valor / observada).where(observada > 0)
    return costo[costo > 0]


# ------------------------------------------------- referencias fuera del conjunto
def referencias_fuera_del_menu(carpeta: Path) -> tuple[dict, dict]:
    """Ingenuo estacional y SBA a un paso y en el horizonte, con sus controles."""
    datos = pd.read_csv(carpeta / "series_y_predicciones.csv")
    datos["periodo"] = pd.PeriodIndex(datos["periodo"], freq="M")
    costo = _costo_por_serie(carpeta, datos)
    datos["costo"] = datos["serie"].map(costo)
    datos = datos.sort_values(["serie", "periodo"])
    datos["ingenuo_estacional"] = datos.groupby("serie")["y_real"].shift(12)
    par = pd.read_csv(carpeta / "parametros_por_serie.csv")
    alfa = par[(par["modelo"] == "croston") & (par["parametro"] == "alfa")].set_index("serie")["valor"]
    datos["sba"] = datos["croston"] * (1 - datos["serie"].map(alfa) / 2)

    prueba = datos[(datos["bloque"] == "prueba") & datos["costo"].notna() & datos["y_real"].notna()]
    publicada = pd.read_csv(carpeta / "tabla_valorizada.csv").set_index("Modelo")
    controles = {}
    for m in ("motor", "croston", "naive_m1"):
        calc = _metricas(prueba["y_real"], prueba[m], prueba["costo"])["D"]
        pub = float(publicada.loc[m, "D = WMAPE + |Bias| (%)"])
        controles[f"un_paso_{m}"] = {"calculada": calc, "publicada": pub}
        if abs(calc - pub) > TOLERANCIA:
            raise SystemExit(f"Control a un paso no reproducido para {m}: {calc:.3f} frente a {pub:.3f}")

    un_paso = {}
    for m in FUERA_DEL_MENU:
        sub = prueba.dropna(subset=[m])
        un_paso[m] = _metricas(sub["y_real"], sub[m], sub["costo"])
        un_paso[m]["series"] = int(sub["serie"].nunique())
        un_paso[m]["celdas_sin_valor"] = int(prueba[m].isna().sum())

    # Horizonte: orígenes rodantes desde el cierre de validación hasta el
    # penúltimo mes de prueba, horizonte máximo el del registro multihorizonte.
    meses_prueba = sorted(prueba["periodo"].unique())
    origenes = [meses_prueba[0] - 1 + k for k in range(len(meses_prueba))]
    tabla_h = pd.read_csv(carpeta / "multihorizonte_horizonte.csv")
    hmax = int(tabla_h["horizonte"].max())
    celdas = [(o, h, o + h) for o in origenes for h in range(1, hmax + 1) if o + h <= meses_prueba[-1]]
    celdas = pd.DataFrame(celdas, columns=["origen", "horizonte", "periodo"])
    base = prueba[["serie", "periodo", "y_real", "costo", "ingenuo_estacional"]]
    obs = celdas.merge(base, on="periodo", how="inner")
    obs["mes_siguiente"] = obs["origen"] + 1
    # Los brazos constantes en el horizonte repiten su pronóstico a un paso del origen.
    constantes = datos.set_index(["serie", "periodo"])[["naive_m1", "croston", "sba"]]
    obs = obs.join(constantes.rename_axis(["serie", "mes_siguiente"]), on=["serie", "mes_siguiente"])

    dem_h = (obs["costo"] * obs["y_real"]).groupby(obs["horizonte"]).sum()
    ref = tabla_h[tabla_h["modelo"] == "naive_m1"].set_index("horizonte")
    desvio = float((dem_h / ref["demanda"] - 1).abs().max())
    controles["cobertura_horizonte_desvio_relativo_max"] = desvio
    controles["celdas_horizonte"] = {"calculadas": int(len(obs)),
                                     "publicadas": int(ref["n_observaciones"].sum())}
    if desvio > 1e-6 or len(obs) != int(ref["n_observaciones"].sum()):
        raise SystemExit(f"La cobertura del horizonte no coincide: {controles['celdas_horizonte']}")
    d_h_pub = _d_horizonte_publicada(carpeta)
    for m in ("naive_m1", "croston"):
        calc = _metricas(obs["y_real"], obs[m], obs["costo"])["D"]
        controles[f"horizonte_{m}"] = {"calculada": calc, "publicada": float(d_h_pub[m])}
        if abs(calc - float(d_h_pub[m])) > TOLERANCIA:
            raise SystemExit(f"Control en el horizonte no reproducido para {m}: {calc:.3f}")
    horizonte = {}
    for m in FUERA_DEL_MENU:
        sub = obs.dropna(subset=[m])
        horizonte[m] = _metricas(sub["y_real"], sub[m], sub["costo"])
        horizonte[m]["celdas_sin_valor"] = int(obs[m].isna().sum())
    resultado = {"un_paso": un_paso, "horizonte": horizonte,
                 "series_con_costo": int(prueba["serie"].nunique()),
                 "celdas_horizonte": int(len(obs)), "origenes": [str(o) for o in origenes]}
    return resultado, controles


# ----------------------------------------------------------------- tablas
def tabla_corrida(carpeta: Path, flujo_tiempos: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Costo y precisión por brazo de una corrida (tiempos de ``flujo_tiempos`` si se da)."""
    flujo = _leer_flujo(carpeta / "flujo.json")
    flujo_t = _leer_flujo(flujo_tiempos) if flujo_tiempos else flujo
    modelos = _etapa(flujo_t, "modelos")
    seg_variables = float(_etapa(flujo_t, "features").get("duracion_s", np.nan))
    seg_motor = float(_etapa(flujo_t, "motor").get("duracion_s", np.nan))
    tiempos = pd.Series(modelos["conteos"]["duracion_por_modelo_s"], dtype=float)
    multih = _etapa(flujo, "evaluacion_multihorizonte")
    publicada = pd.read_csv(carpeta / "tabla_valorizada.csv").set_index("Modelo")
    ruta_friedman = carpeta / "rangos_friedman.csv"
    friedman = (pd.read_csv(ruta_friedman).set_index("modelo")["rango_medio"]
                if ruta_friedman.exists() else pd.Series(dtype=float))
    elegidos = pd.Series(_etapa(flujo, "motor").get("conteos", {}).get("reparto_de_elegidos", {}),
                         dtype=float)
    n_muestra = int(pd.read_csv(carpeta / "cohorte_flujo.csv")["series"].iloc[-1])
    d_h = _d_horizonte_publicada(carpeta)
    d_flujo = multih.get("conteos", {}).get("D_global_pct", {})
    for m, v in d_flujo.items():
        if m in d_h.index and abs(float(d_h[m]) - float(v)) > 0.0501:
            raise SystemExit(f"La D en el horizonte recalculada no coincide con el registro para {m}")

    filas = []
    for m in publicada.index:
        seg = float(tiempos.get(m, np.nan))
        if m == "motor":
            # El motor corre los diez brazos; con la preparación de variables
            # de LightGBM se le suma también la selección por serie.
            seg = float(tiempos.sum())
            seg_con = seg + seg_variables + seg_motor
        elif m.startswith("lightgbm"):
            seg_con = seg + seg_variables
        else:
            seg_con = seg
        filas.append({
            "brazo": m, "nombre": NOMBRES_P2.get(m, m),
            "segundos_ajuste_y_pronostico": seg,
            "segundos_con_preparacion_de_variables": seg_con,
            "ms_por_serie_amortizado": 1000 * seg_con / n_muestra,
            "D_un_paso": float(publicada.loc[m, "D = WMAPE + |Bias| (%)"]),
            "wmape_un_paso": float(publicada.loc[m, "WMAPE valorizado (%)"]),
            "sesgo_un_paso": float(publicada.loc[m, "Bias valorizado (%)"]),
            "D_horizonte": float(d_h.get(m, np.nan)),
            "rango_friedman": float(friedman.get(m, np.nan)),
            "series_elegidas": float(elegidos.get(m, np.nan)),
            "series_con_costo": int(publicada.loc[m, "Series con costo"]),
        })
    meta = {
        "resultados": {k: flujo.get("corrida", {}).get(k) for k in ("generado_en", "commit")},
        "tiempos": {k: flujo_t.get("corrida", {}).get(k) for k in ("generado_en", "commit")},
        "series_muestra": n_muestra,
        "segundos_preparacion_variables": seg_variables,
        "segundos_seleccion_motor": seg_motor,
        "segundos_explicacion": float(_etapa(flujo_t, "explicacion").get("duracion_s", np.nan)),
        "segundos_etapa_modelos": float(modelos.get("duracion_s", np.nan)),
        "segundos_etapa_multihorizonte": float(
            _etapa(flujo_t, "evaluacion_multihorizonte").get("duracion_s", np.nan)),
        "segundos_suma_estadisticos": float(tiempos.reindex(ESTADISTICOS).sum()),
    }
    return pd.DataFrame(filas), meta


def calibracion(motor: Path) -> dict:
    """Ensayos, ensayos fallidos y segundos de cada búsqueda de hiperparámetros publicada."""
    salida = {}
    for brazo, archivo in CALIBRACION.items():
        ruta = motor / "resultados" / "ensayos" / archivo
        if not ruta.exists():
            continue
        t = pd.read_csv(ruta)
        fallidos = int(t["fallo"].notna().sum()) if "fallo" in t.columns else 0
        salida[brazo] = {"ensayos": int(len(t)), "segundos": float(t["segundos"].sum()),
                         "fallidos": fallidos}
    return salida


def _texto_busqueda(brazo: str, fijo: str | None, calib: dict) -> str:
    """Celda «Calibración: s (ensayos)» de la Tabla P3 (los fallidos van al resumen y a la nota)."""
    if fijo is not None:
        return fijo
    c = calib.get(brazo)
    if c is None:
        return NO_DISPONIBLE
    return f"{_num(c['segundos'])} ({c['ensayos']})"


def _mejores(t: pd.DataFrame, columna: str) -> dict:
    """Brazos individuales (sin mezclas) con menor D que LightGBM."""
    individuales = t[~t.index.str.startswith("mezcla")]
    ref = individuales.loc["lightgbm", columna]
    sel = individuales[individuales[columna] < ref].sort_values(columna)
    return {m: round(float(v), 2) for m, v in sel[columna].items()}


def construir(motor: Path, principal: str, ext26: str, ext25: str, tiempos: Path | None):
    """Arma las Tablas P2 y P3 y el resumen."""
    p2, meta2 = tabla_corrida(motor / principal, tiempos)
    fuera, controles = referencias_fuera_del_menu(motor / principal)
    for m in FUERA_DEL_MENU:
        u, h = fuera["un_paso"][m], fuera["horizonte"][m]
        p2 = pd.concat([p2, pd.DataFrame([{
            "brazo": m, "nombre": NOMBRES_P2[m], "D_un_paso": u["D"], "wmape_un_paso": u["wmape"],
            "sesgo_un_paso": u["sesgo"], "D_horizonte": h["D"], "series_con_costo": u["series"]}])],
            ignore_index=True)
    calib = calibracion(motor)

    t26, meta26 = tabla_corrida(motor / ext26)
    t25, meta25 = tabla_corrida(motor / ext25)
    a = t26.set_index("brazo")
    b = t25.set_index("brazo")
    filas = []
    for m, nombre, fijo in EXTENSION:
        filas.append({
            "brazo": m, "nombre": nombre,
            "segundos_2026": a.loc[m, "segundos_ajuste_y_pronostico"],
            "D_un_paso_2026": a.loc[m, "D_un_paso"], "D_horizonte_2026": a.loc[m, "D_horizonte"],
            "segundos_2025": b.loc[m, "segundos_ajuste_y_pronostico"],
            "D_un_paso_2025": b.loc[m, "D_un_paso"], "D_horizonte_2025": b.loc[m, "D_horizonte"],
            "busqueda": _texto_busqueda(m, fijo, calib),
        })
    p3 = pd.DataFrame(filas)
    ancla = {g: {"segundos": float(t.loc[ANCLA_EN_NOTA, "segundos_ajuste_y_pronostico"]),
                 "D_un_paso": float(t.loc[ANCLA_EN_NOTA, "D_un_paso"]),
                 "D_horizonte": float(t.loc[ANCLA_EN_NOTA, "D_horizonte"])}
             for g, t in (("2026", a), ("2025", b))}
    i2 = p2.set_index("brazo")
    resumen = {
        "principal": meta2, "extension_2026": meta26, "extension_2025": meta25,
        "cohortes_con_costo": {"2026": int(a["series_con_costo"].iloc[0]),
                               "2025": int(b["series_con_costo"].iloc[0])},
        "calibracion": calib, "fuera_del_menu": fuera, "controles": controles,
        "ingenuo_en_la_extension": ancla,
        "mejora_relativa_motor": {
            "frente_al_ingenuo": float(i2.loc["motor", "D_un_paso"] / i2.loc["naive_m1", "D_un_paso"]),
            "frente_al_ingenuo_estacional": float(i2.loc["motor", "D_un_paso"]
                                                  / i2.loc["ingenuo_estacional", "D_un_paso"]),
        },
        "orden_en_el_horizonte": i2["D_horizonte"].sort_values().round(2).to_dict(),
        "brazos_mejores_que_lightgbm_un_paso": {
            "2026": _mejores(a, "D_un_paso"), "2025": _mejores(b, "D_un_paso")},
        "brazos_mejores_que_lightgbm_horizonte": {
            "2026": _mejores(a, "D_horizonte"), "2025": _mejores(b, "D_horizonte")},
    }
    return p2, p3, resumen


def markdown(p2: pd.DataFrame, p3: pd.DataFrame, r: dict) -> str:
    """Tablas P2 y P3 en markdown, con los rótulos del documento."""
    l = ["Tabla P2. Costo de cómputo y precisión por brazo en la corrida principal", "",
         "| Brazo | Ajuste y pronóstico a un paso (s) | Con preparación de variables (s) "
         "| Por serie, amortizado (ms) | D a un paso (%) | D en el horizonte (%) "
         "| Rango medio de Friedman (Figura 10) | Series elegidas por el motor |",
         "|---|---|---|---|---|---|---|---|"]
    menu = p2[~p2["brazo"].isin(FUERA_DEL_MENU)].sort_values("D_un_paso")
    fuera = p2[p2["brazo"].isin(FUERA_DEL_MENU)].set_index("brazo").loc[FUERA_DEL_MENU].reset_index()
    for _, f in menu.iterrows():
        elegidas = _num(f["series_elegidas"], 0)
        l.append("| " + " | ".join([
            f["nombre"], _num(f["segundos_ajuste_y_pronostico"], 2),
            _num(f["segundos_con_preparacion_de_variables"], 2), _num(f["ms_por_serie_amortizado"], 2),
            _num(f["D_un_paso"]), _num(f["D_horizonte"]), _num(f["rango_friedman"], 2), elegidas]) + " |")
    for _, f in fuera.iterrows():
        l.append("| " + " | ".join([f["nombre"], NO_DISPONIBLE, NO_DISPONIBLE, NO_DISPONIBLE,
                                    _num(f["D_un_paso"]), _num(f["D_horizonte"]), NO_DISPONIBLE,
                                    NO_DISPONIBLE]) + " |")
    t, res = r["principal"]["tiempos"], r["principal"]["resultados"]
    l += ["", f"Tiempos de la corrida generada el {t['generado_en']} (commit {t['commit']}); "
          f"resultados de la corrida generada el {res['generado_en']} (commit {res['commit']}).",
          "", "Tabla P3. Costo de cómputo y precisión de los pronosticadores de la extensión exploratoria "
          "en las dos gestiones de prueba", "",
          "| Brazo | Gestión 2026: ajuste y pronóstico a un paso (s) "
          "| Gestión 2025: ajuste y pronóstico a un paso (s) | Gestión 2026: D a un paso (%) "
          "| Gestión 2025: D a un paso (%) | Gestión 2026: D en el horizonte (%) "
          "| Gestión 2025: D en el horizonte (%) | Calibración: s (ensayos) |",
          "|---|---|---|---|---|---|---|---|"]
    for _, f in p3.iterrows():
        l.append("| " + " | ".join([
            f["nombre"], _num(f["segundos_2026"], 2), _num(f["segundos_2025"], 2),
            _num(f["D_un_paso_2026"]), _num(f["D_un_paso_2025"]), _num(f["D_horizonte_2026"]),
            _num(f["D_horizonte_2025"]), f["busqueda"]]) + " |")
    n = r["ingenuo_en_la_extension"]
    c = r["calibracion"]
    l += ["", "Para la nota de la Tabla P3: ingenuo en el entorno de la extensión, "
          f"{_num(n['2026']['segundos'], 2)} s, {_num(n['2026']['D_un_paso'])} % a un paso y "
          f"{_num(n['2026']['D_horizonte'])} % en el horizonte (2026); {_num(n['2025']['segundos'], 2)} s, "
          f"{_num(n['2025']['D_un_paso'])} % y {_num(n['2025']['D_horizonte'])} % (2025). Ensayos fallidos: "
          + "; ".join(f"{k} {v['fallidos']} de {v['ensayos']}" for k, v in c.items()) + "."]
    return "\n".join(l) + "\n"


def main() -> None:
    """Punto de entrada: lee los argumentos, construye y escribe las salidas."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motor", type=Path, default=Path("."), help="Raíz del motor.")
    parser.add_argument("--destino", type=Path, default=None,
                        help="Carpeta de salida; por omisión, resultados/tablas dentro del motor.")
    parser.add_argument("--principal", default="salidas", help="Carpeta de la corrida principal.")
    parser.add_argument("--tiempos", type=Path, default=None,
                        help="Registro de flujo del que se toman los tiempos de la Tabla P2 "
                             "(por omisión, el de la corrida principal).")
    parser.add_argument("--extension-2026", default="salidas_ml2_v9",
                        help="Carpeta de la extensión con prueba en la gestión 2026.")
    parser.add_argument("--extension-2025", default="salidas_ml2_v9_blindaje",
                        help="Carpeta de la extensión con prueba en la gestión 2025.")
    args = parser.parse_args()
    destino = args.destino or (args.motor / DESTINO)

    p2, p3, resumen = construir(args.motor, args.principal, args.extension_2026,
                                args.extension_2025, args.tiempos)
    destino.mkdir(parents=True, exist_ok=True)
    p2.to_csv(destino / "anexo_p_p2.csv", index=False)
    p3.to_csv(destino / "anexo_p_p3.csv", index=False)
    md = markdown(p2, p3, resumen)
    (destino / "anexo_p.md").write_text(md, encoding="utf-8")
    (destino / "anexo_p_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
