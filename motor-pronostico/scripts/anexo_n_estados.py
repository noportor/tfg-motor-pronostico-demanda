"""Anexo N: conteos por estado de actividad y meses de historia de las series.

Propósito
---------
Reproduce, sólo desde salidas publicadas de la corrida principal, las Tablas
N2 y N3 del Anexo N, los conteos de costo faltante del párrafo que las sigue y
la evidencia de datos de las Tablas N1 y N4:

* Tabla N2: series por estado de actividad al cierre del entrenamiento y por
  inclusión en la muestra, con las series con costo, el peso en el valor de
  entrenamiento valorizado a costo, la mediana de meses de historia y la
  mediana de la proporción de meses en cero.
* Tabla N3: series por el primer criterio de la cascada que las excluye,
  cruzado con el estado, con su peso en el valor; el script se detiene si los
  totales no reproducen ``cohorte_flujo.csv``.
* Control de la Tabla 8 (nota de la Tabla N1): D valorizada a un paso del
  ingenuo y del promedio móvil de dos meses sobre la muestra, con sus series
  y su demanda valorizada; el script se detiene si no reproduce la Tabla 8.
* Molde en dos ejes (evidencia de la Tabla N1): estado de actividad por tramo
  de meses de historia en entrenamiento, con series y peso en el valor.
* Costo faltante: productos de la población, productos y series sin costo
  unitario (en las series con historia) y en la muestra, si la falta es por
  producto completo y el peso de esas series en el volumen de entrenamiento
  de la muestra.
* Evidencia de la Tabla N4 y de los casos de datos de la Tabla N1, tomada del
  registro de flujo y de otras corridas publicadas: tratamiento del evento
  exógeno, filas duplicadas, series intermitentes, celdas con pérdidas, filas
  con quiebres, uso del tipo de cambio por el modelo global (divisiones),
  meses con el detector de régimen activo en cada gestión de la extensión y la
  D de la variante sin tratamiento del evento.

Lo que este script NO calcula, porque exige la captura de ventas (Anexo B) y no
está en ninguna salida publicada: las series que venden en validación y en
prueba y el valor de prueba de las series excluidas, la línea base de respaldo
fuera de la muestra, las seis series sin historia y sin costo (317 en total),
la categoría de las series sin costo, la razón entre costo y precio y el
detalle de los duplicados (pares, meses, combinaciones y exceso de volumen).
El script no abre ningún archivo de ``datos/``.

Definiciones: «estado» es el estado de actividad al cierre de entrenamiento
que la corrida asigna a cada serie («insuficiente» es una serie sin historia
en entrenamiento); «meses de historia» es el número de meses de la serie en el
bloque de entrenamiento; el peso es el valor de entrenamiento (demanda
observada por costo unitario) sobre el total de la población. Una serie con
historia y sin costo tiene valor cero y se cuenta como «sin costo». El costo
unitario de cada serie de la muestra se deduce como valor de entrenamiento
sobre la demanda OBSERVADA de entrenamiento de ``series_y_predicciones.csv``
(el volumen de ``fuera_de_muestra.csv`` es el del panel de ajuste, con el
evento corregido, y daría un costo levemente distinto).

Redondeo: mitad hacia arriba sobre el valor decimal, como en las tablas del
documento; coma decimal y punto de miles.

Uso
---
    python scripts/anexo_n_estados.py --motor . --destino resultados/tablas

Escribe en ``--destino`` (por omisión ``resultados/tablas`` dentro del motor)
sólo agregados, ningún identificador de serie ni de producto:
``anexo_n.md``, ``anexo_n_n2.csv``, ``anexo_n_n3.csv``, ``anexo_n_molde.csv``
y ``anexo_n_resumen.json``. Determinista.
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
ESTADOS = ["activa", "nueva", "reciente", "dormida", "descontinuada", "insuficiente"]
ROTULO_ESTADO = {"activa": "Activa", "nueva": "Nueva", "reciente": "Reciente",
                 "dormida": "Dormida", "descontinuada": "Descontinuada",
                 "insuficiente": "Sin historia en entrenamiento"}
TRAMOS = [(-1, 0, "sin historia"), (0, 11, "1 a 11"), (11, 23, "12 a 23"),
          (23, 35, "24 a 35"), (35, 10_000, "36 o más")]
ROTULO_MOTIVO = {
    "sin_historia_en_entrenamiento": "Sin historia en entrenamiento",
    "historial<36m": "Historial menor que 36 meses",
    "ceros>0.70": "Más del 70 % de meses en cero",
    "volumen<50u": "Volumen acumulado menor que 50 unidades",
    "ausente_en_validacion": "Ausente en validación",
    "ausente_en_prueba": "Ausente en prueba",
    "incluida": "En la muestra",
}
ORDEN_MOTIVOS = ["sin_historia_en_entrenamiento", "historial<36m", "ceros>0.70", "volumen<50u",
                 "ausente_en_validacion", "ausente_en_prueba", "incluida"]
TOLERANCIA = 0.05  # puntos de D: el control debe reproducir la Tabla 8


def _num(x, decimales: int = 1) -> str:
    """Número con coma decimal, punto de miles y redondeo mitad hacia arriba."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n. d."
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimales), rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    texto = f"{abs(d):,.{decimales}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    # Signo menos tipográfico, como en las tablas del documento.
    return ("−" + texto) if d < 0 else texto


def _etapa(flujo: dict, ident: str) -> dict:
    """Etapa del registro de flujo con ese identificador (vacía si no está)."""
    return next((e for e in flujo["etapas"] if e["id"] == ident), {})


def _tramo(meses: pd.Series) -> pd.Series:
    """Tramo de meses de historia en entrenamiento."""
    cortes = [t[0] for t in TRAMOS] + [TRAMOS[-1][1]]
    return pd.cut(meses, bins=cortes, labels=[t[2] for t in TRAMOS])


def _metricas(y, p, c) -> dict:
    """WMAPE, sesgo y D valorizados (mismas definiciones del motor)."""
    demanda = float((c * y).sum())
    wmape = 100 * float((c * np.abs(p - y)).sum()) / demanda
    sesgo = 100 * float((c * (p - y)).sum()) / demanda
    return {"wmape": wmape, "sesgo": sesgo, "D": wmape + abs(sesgo), "demanda": demanda}


def _d_motor(carpeta: Path) -> dict:
    """D a un paso publicada y D global en el horizonte del motor de una corrida."""
    tv = pd.read_csv(carpeta / "tabla_valorizada.csv").set_index("Modelo")
    h = pd.read_csv(carpeta / "multihorizonte_horizonte.csv")
    h = h[h["modelo"] == "motor"]
    ses = (h["bias_val"] / 100 * h["demanda"]).sum()
    dh = 100 * (h["error"].sum() / h["demanda"].sum() + abs(ses / h["demanda"].sum()))
    return {"D_un_paso": float(tv.loc["motor", "D = WMAPE + |Bias| (%)"]), "D_horizonte": float(dh)}


def construir(motor: Path, principal: str, ablacion_evento: str,
              ext26: str, ext25: str) -> tuple[dict, dict]:
    """Calcula las tablas y el resumen desde las salidas publicadas."""
    sal = motor / principal
    fm = pd.read_csv(sal / "fuera_de_muestra.csv")
    cohorte = pd.read_csv(sal / "cohorte_flujo.csv")
    errores = pd.read_csv(sal / "errores_por_serie.csv")
    flujo = json.loads((sal / "flujo.json").read_text(encoding="utf-8"))

    fm["valor"] = fm["valor_entrenamiento"].fillna(0.0)
    fm["con_costo"] = fm["valor"] > 0
    fm["sku"] = fm["serie"].str.split("|").str[0]
    fm["tramo"] = _tramo(fm["meses_entrenamiento"])
    total = float(fm["valor"].sum())

    def agrega(df: pd.DataFrame, claves: list[str]) -> pd.DataFrame:
        """Series, series con costo, peso en el valor y medianas por grupo."""
        t = df.groupby(claves, observed=True).agg(
            series=("serie", "size"),
            series_con_costo=("con_costo", "sum"),
            valor=("valor", "sum"),
            meses_mediana=("meses_entrenamiento", "median"),
            ceros_mediana=("proporcion_ceros", "median"),
        ).reset_index()
        t["pct_valor"] = 100 * t["valor"] / total
        return t.drop(columns="valor")

    n2 = agrega(fm, ["estado", "incluida"])
    molde = agrega(fm, ["estado", "tramo"])
    motivo = agrega(fm, ["motivo_exclusion", "estado"])

    # Control de la cascada: excluidas por primer criterio = cohorte_flujo.
    por_motivo = fm["motivo_exclusion"].value_counts()
    pasos = cohorte["excluidas"].iloc[1:].tolist()
    claves = ["sin_historia_en_entrenamiento", "historial<36m", "ceros>0.70", "volumen<50u",
              "ausente_en_validacion", "ausente_en_prueba"]
    control_cascada = {k: {"fuera_de_muestra": int(por_motivo.get(k, 0)), "cohorte_flujo": int(v)}
                       for k, v in zip(claves, pasos)}
    cascada_ok = all(c["fuera_de_muestra"] == c["cohorte_flujo"] for c in control_cascada.values())
    muestra_ok = int(fm["incluida"].sum()) == int(cohorte["series"].iloc[-1])
    if not (cascada_ok and muestra_ok):
        raise SystemExit(f"La tabla de fuera de muestra no reproduce la cascada: {control_cascada}")

    # Costo faltante. Sólo se puede leer en las series con historia de
    # entrenamiento: las que no la tienen carecen de valor con o sin costo.
    con_historia = fm[fm["meses_entrenamiento"] > 0]
    por_sku = con_historia.groupby("sku")["con_costo"].agg(["all", "any"])
    muestra = fm[fm["incluida"]]
    costo = {
        "alcance": "series con historia de entrenamiento",
        "productos_poblacion": int(fm["sku"].nunique()),
        "productos_con_historia": int(con_historia["sku"].nunique()),
        "productos_sin_costo_con_historia": int((~por_sku["any"]).sum()),
        "productos_con_costo_parcial": int((por_sku["any"] & ~por_sku["all"]).sum()),
        "series_con_historia": int(len(con_historia)),
        "series_sin_costo_con_historia": int((~con_historia["con_costo"]).sum()),
        "productos_muestra": int(muestra["sku"].nunique()),
        "productos_sin_costo_muestra": int(muestra.loc[~muestra["con_costo"], "sku"].nunique()),
        "series_sin_costo_muestra": int((~muestra["con_costo"]).sum()),
        "series_con_costo_muestra": int(muestra["con_costo"].sum()),
        "pct_volumen_entrenamiento_muestra_sin_costo": float(
            100 * muestra.loc[~muestra["con_costo"], "volumen_entrenamiento"].sum()
            / muestra["volumen_entrenamiento"].sum()),
    }

    corto = fm[(fm["meses_entrenamiento"] >= 12) & (fm["meses_entrenamiento"] <= 35)]
    activas_fuera = fm[(fm["estado"] == "activa") & ~fm["incluida"]]

    # Evidencia del registro de flujo.
    evento = _etapa(flujo, "evento_exogeno")
    inspeccion = _etapa(flujo, "inspeccion")
    variables = _etapa(flujo, "features")
    en_ventana = evento.get("entrada", {}).get("observaciones_en_ventana")
    importancias = pd.read_csv(sal / "lightgbm_importancias.csv").drop_duplicates("feature")
    divisiones = importancias.set_index("feature")["divisiones"]
    tipo_cambio = {f: int(d) for f, d in divisiones.items() if str(f).startswith("tc_")}

    detector = {}
    for etiqueta, carpeta in (("gestion_2026", ext26), ("gestion_2025", ext25)):
        ruta = motor / carpeta / "conmutador_senal.csv"
        if ruta.exists():
            s = pd.read_csv(ruta)
            detector[etiqueta] = {"meses_activos": int((s["lambda"] > 0).sum()), "meses": int(len(s))}

    variante_evento = None
    if (motor / ablacion_evento / "tabla_valorizada.csv").exists():
        variante_evento = {"sin_tratamiento": _d_motor(motor / ablacion_evento),
                           "principal": _d_motor(sal)}

    # Fila de control de la Tabla N3 y error del motor por estado en la muestra.
    datos = pd.read_csv(sal / "series_y_predicciones.csv",
                        usecols=["serie", "bloque", "y_real", "naive_m1", "ma_2", "motor"])
    observada = datos[datos["bloque"] == "entrenamiento"].groupby("serie")["y_real"].sum()
    valor = fm.set_index("serie")["valor"].reindex(observada.index)
    costo_serie = (valor / observada).where(observada > 0)
    costo_serie = costo_serie[costo_serie > 0]
    prueba = datos[datos["bloque"] == "prueba"].copy()
    prueba["costo"] = prueba["serie"].map(costo_serie)
    prueba = prueba.dropna(subset=["costo", "y_real"])
    publicada = pd.read_csv(sal / "tabla_valorizada.csv").set_index("Modelo")
    control_n3 = {"series": int(prueba["serie"].nunique())}
    for m in ("naive_m1", "ma_2"):
        control_n3[m] = _metricas(prueba["y_real"], prueba[m], prueba["costo"])
        control_n3[m]["publicada"] = float(publicada.loc[m, "D = WMAPE + |Bias| (%)"])
        if abs(control_n3[m]["D"] - control_n3[m]["publicada"]) > TOLERANCIA:
            raise SystemExit(f"La fila de control no reproduce la Tabla 8 para {m}")
    control_n3["demanda_millones"] = control_n3["naive_m1"]["demanda"] / 1e6

    e_motor = errores[errores["modelo"] == "motor"].merge(fm[["serie", "estado"]], on="serie", how="left")
    error_por_estado = e_motor.groupby("estado").agg(
        series=("serie", "size"), error_compuesto_mediano_pct=("error_compuesto_pct", "median")
    ).reset_index().to_dict("records")

    resumen = {
        "poblacion": int(len(fm)),
        "muestra": int(fm["incluida"].sum()),
        "excluidas": int((~fm["incluida"]).sum()),
        "pct_valor_muestra": float(100 * fm.loc[fm["incluida"], "valor"].sum() / total),
        "pct_valor_excluidas": float(100 * fm.loc[~fm["incluida"], "valor"].sum() / total),
        "por_estado": {e: int(n) for e, n in fm["estado"].value_counts().sort_index().items()},
        "control_cascada": control_cascada,
        "activas_excluidas": {"series": int(len(activas_fuera)),
                              "pct_valor": float(100 * activas_fuera["valor"].sum() / total),
                              "ceros_mediana": float(activas_fuera["proporcion_ceros"].median())},
        "historia_12_a_35": {"series": int(len(corto)),
                             "por_estado": {e: int(n) for e, n in
                                            corto["estado"].value_counts().sort_index().items()}},
        "historia_1_a_11": int(((fm["meses_entrenamiento"] >= 1) & (fm["meses_entrenamiento"] <= 11)).sum()),
        "historia_36_o_mas": int((fm["meses_entrenamiento"] >= 36).sum()),
        "costo_faltante": costo,
        "evento_exogeno": {"observaciones_en_ventana": en_ventana,
                           "observaciones_reemplazadas": evento.get("salida", {}).get(
                               "observaciones_reemplazadas"),
                           "series_afectadas": evento.get("conteos", {}).get("series_afectadas"),
                           "sin_historia_previa": evento.get("conteos", {}).get("sin_historia_previa"),
                           "pct_sin_historia_previa": (100 * evento["conteos"]["sin_historia_previa"]
                                                       / en_ventana if en_ventana else None),
                           "ventana": evento.get("decisiones", {}).get("ventana"),
                           "tratamiento": evento.get("decisiones", {}).get("tratamiento")},
        "inspeccion": {"filas_duplicadas": inspeccion.get("conteos", {}).get("duplicados"),
                       "series_intermitentes_adi_1_32": inspeccion.get("conteos", {}).get(
                           "series_intermitentes_adi_1_32")},
        "variables": {"celdas_con_perdidas": variables.get("conteos", {}).get("celdas_con_perdidas"),
                      "filas_quiebres": variables.get("conteos", {}).get("filas_quiebres"),
                      "series_con_precio": variables.get("conteos", {}).get("series_con_precio")},
        "divisiones_tipo_cambio": tipo_cambio,
        "divisiones_rezago_1": int(divisiones.get("rezago_1", 0)),
        "detector_regimen": detector,
        "variante_sin_tratamiento_del_evento": variante_evento,
        "control_tabla_n3": control_n3,
        "error_motor_por_estado_en_muestra": error_por_estado,
        "corrida": {k: flujo.get("corrida", {}).get(k) for k in ("generado_en", "commit")},
    }
    return {"n2": n2, "molde": molde, "motivo": motivo}, resumen


def markdown(t: dict, r: dict) -> str:
    """Tablas N2 y N3, control de la Tabla 8 y evidencia, en markdown, con los rótulos del anexo."""
    l = ["Tabla N2. Estado de actividad e inclusión en la muestra de las combinaciones de la "
         "población objetivo", "",
         "| Estado | En la muestra | Series | Con costo | % del valor de entrenamiento "
         "| Meses de historia (mediana) | Meses en cero (mediana) |",
         "|---|---|---|---|---|---|---|"]
    n2 = t["n2"].copy()
    n2["orden"] = n2["estado"].map({e: i for i, e in enumerate(ESTADOS)})
    for _, f in n2.sort_values(["orden", "incluida"], ascending=[True, False]).iterrows():
        ceros = "n/d" if f["estado"] == "insuficiente" else _num(100 * f["ceros_mediana"], 1) + " %"
        l.append("| " + " | ".join([
            ROTULO_ESTADO[f["estado"]], "sí" if f["incluida"] else "no", _num(f["series"], 0),
            _num(f["series_con_costo"], 0), _num(f["pct_valor"], 2), _num(f["meses_mediana"], 0),
            ceros]) + " |")

    l += ["", "Tabla N3. Combinaciones de la población objetivo por primer criterio de la cascada "
          "que las excluye y estado de actividad", ""]
    mot = t["motivo"].pivot_table(index="motivo_exclusion", columns="estado", values="series",
                                  aggfunc="sum", fill_value=0)
    peso = t["motivo"].groupby("motivo_exclusion")["pct_valor"].sum()
    cols = [e for e in ESTADOS if e in mot.columns]
    l.append("| Primer criterio que excluye | " + " | ".join(ROTULO_ESTADO[c] for c in cols)
             + " | Total | % del valor |")
    l.append("|---|" + "---|" * (len(cols) + 2))
    for m in [x for x in ORDEN_MOTIVOS if x in mot.index]:
        l.append(f"| {ROTULO_MOTIVO.get(m, m)} | " + " | ".join(_num(mot.loc[m, c], 0) for c in cols)
                 + f" | {_num(mot.loc[m].sum(), 0)} | {_num(peso[m], 2)} |")

    c = r["control_tabla_n3"]
    l += ["", "Control de la Tabla 8 (nota de la Tabla N1): D a un paso sobre la muestra", "",
          "| Grupo | Series evaluadas | Demanda valorizada (millones de bolivianos) | D ingenuo (%) "
          "| D promedio móvil de dos meses (%) |", "|---|---|---|---|---|",
          f"| Muestra (control: reproduce la Tabla 8) | {_num(c['series'], 0)} | "
          f"{_num(c['demanda_millones'], 2)} | {_num(c['naive_m1']['D'])} | {_num(c['ma_2']['D'])} |"]

    l += ["", "Molde en dos ejes (evidencia de la Tabla N1). Series (y % del valor de entrenamiento) "
          "por estado y meses de historia", ""]
    molde = t["molde"]
    tramos = [x[2] for x in TRAMOS]
    l.append("| Estado | " + " | ".join(tramos) + " | Total |")
    l.append("|---|" + "---|" * (len(tramos) + 1))
    for e in ESTADOS:
        sub = molde[molde["estado"] == e].set_index("tramo")
        celdas = []
        for tr in tramos:
            if tr in sub.index and sub.loc[tr, "series"] > 0:
                celdas.append(f"{_num(sub.loc[tr, 'series'], 0)} ({_num(sub.loc[tr, 'pct_valor'], 2)})")
            else:
                celdas.append("0")
        l.append(f"| {ROTULO_ESTADO[e]} | " + " | ".join(celdas)
                 + f" | {_num(sub['series'].sum(), 0)} ({_num(sub['pct_valor'].sum(), 2)}) |")

    k = r["costo_faltante"]
    l += ["", "Costo faltante (series con historia de entrenamiento)", "",
          f"- Productos en la población: {_num(k['productos_poblacion'], 0)}; con historia: "
          f"{_num(k['productos_con_historia'], 0)}; sin costo: "
          f"{_num(k['productos_sin_costo_con_historia'], 0)}; con costo parcial: "
          f"{_num(k['productos_con_costo_parcial'], 0)}.",
          f"- Series con historia sin costo: {_num(k['series_sin_costo_con_historia'], 0)} de "
          f"{_num(k['series_con_historia'], 0)}.",
          f"- Muestra: {_num(k['series_sin_costo_muestra'], 0)} series y "
          f"{_num(k['productos_sin_costo_muestra'], 0)} productos sin costo; "
          f"{_num(k['pct_volumen_entrenamiento_muestra_sin_costo'], 2)} % del volumen de entrenamiento "
          f"de la muestra.",
          "", f"Corrida: {r['corrida']['generado_en']} (commit {r['corrida']['commit']})."]
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
    parser.add_argument("--ablacion-evento", default="salidas_ablacion_covid_crudo",
                        help="Corrida de la variante sin tratamiento del evento exógeno.")
    parser.add_argument("--extension-2026", default="salidas_ml2_v9")
    parser.add_argument("--extension-2025", default="salidas_ml2_v9_blindaje")
    args = parser.parse_args()
    destino = args.destino or (args.motor / DESTINO)

    tablas, resumen = construir(args.motor, args.principal, args.ablacion_evento,
                                args.extension_2026, args.extension_2025)
    destino.mkdir(parents=True, exist_ok=True)
    tablas["n2"].to_csv(destino / "anexo_n_n2.csv", index=False)
    tablas["motivo"].to_csv(destino / "anexo_n_n3.csv", index=False)
    tablas["molde"].to_csv(destino / "anexo_n_molde.csv", index=False)
    md = markdown(tablas, resumen)
    (destino / "anexo_n.md").write_text(md, encoding="utf-8")
    (destino / "anexo_n_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
