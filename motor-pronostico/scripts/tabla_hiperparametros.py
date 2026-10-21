"""Tabla L3 del Anexo L (Tabla 28 del Anexo 12 en el documento final): hiperparámetros del modelo global y lectura de la búsqueda.

Propósito
---------
Reproduce, desde la tabla de ensayos publicada de la búsqueda de hiperparámetros
(``resultados/ensayos/lightgbm_tuning_ensayos.csv``) y desde la configuración
declarada (``config/config.yaml``), la Tabla L3 (Tabla 28 del documento final) con las mismas
filas, columnas y rótulos que el texto del anexo: para cada hiperparámetro, el
valor elegido, el rango explorado con su escala, la función que cumple, el
rango que ocupa en la meseta (los ensayos a menos de dos puntos de la mejor D
interna), la importancia en la búsqueda (evaluador fANOVA de Optuna, el que la
biblioteca usa por omisión) y la correlación de rangos de Spearman con la D
interna dentro de los ensayos Tweedie. La importancia de la función de
pérdida se calcula sobre los 60 ensayos, con la pérdida como variable
categórica. Agrega un resumen por función de pérdida y un resumen de la
búsqueda (meseta, convergencia y costo).

Las correlaciones y las importancias son descriptivas: los ensayos de un
optimizador bayesiano no son una muestra independiente, por eso no se
informan p-valores.

El número de árboles de la corrida principal se lee del manifiesto de esa
corrida (``salidas/manifiesto.json``, campo ``lightgbm_mejor_iteracion``); si
el manifiesto no está, la celda lo dice. Antes de escribir, el script controla
que los valores declarados en la configuración sean los del mejor ensayo; si
no lo son, se detiene.

Entradas: sólo la tabla de ensayos, la configuración y el manifiesto. No lee
los datos crudos ni identificadores. Es determinista: la importancia fANOVA se
promedia sobre cinco semillas fijas. Requiere Optuna para la columna fANOVA;
sin Optuna, esa columna sale «n. d.» y el resto no cambia.

Redondeo: mitad hacia arriba sobre el valor decimal (81,25 da 81,3), como en
las tablas del documento; coma decimal y punto de miles.

Uso
---
    python scripts/tabla_hiperparametros.py --motor . --destino resultados/tablas

Escribe en ``--destino`` (por omisión ``resultados/tablas`` dentro del motor):
``tabla_l3.md`` (la tabla lista para el anexo), ``tabla_l3.csv`` (valores sin
redondeo de presentación), ``tabla_l3_perdidas.csv`` y
``tabla_l3_resumen.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ENSAYOS = "resultados/ensayos/lightgbm_tuning_ensayos.csv"
CONFIGURACION = "config/config.yaml"
MANIFIESTO = "salidas/manifiesto.json"
DESTINO = "resultados/tablas"

# Ancho de la meseta, en puntos de D interna, y semillas del promedio fANOVA.
MESETA_PUNTOS = 2.0
MESETA_ESTRECHA = 1.5
SEMILLAS_FANOVA = (0, 1, 2, 3, 4)

PERDIDAS = {
    "tweedie": "Tweedie",
    "regression_l1": "absoluta",
    "regression_l2": "cuadrática",
}

# Filas de la tabla, en el orden del documento: clave en la tabla de ensayos,
# rótulo, función, decimales del valor elegido, decimales del rango en la
# meseta y escala del rango explorado.
FILAS = [
    ("tweedie_variance_power", "Potencia de la varianza Tweedie ('tweedie_variance_power')",
     "Forma de la pérdida: cuánto crece la varianza con el nivel", 3, 2,
     "lineal; sólo con pérdida Tweedie"),
    ("learning_rate", "Tasa de aprendizaje ('learning_rate')",
     "Ritmo: contrae el aporte de cada árbol", 3, 3, "logarítmica"),
    ("mejor_iteracion", "Número de árboles (parada temprana)",
     "Ritmo: se compensa con la tasa", 0, 0, None),
    ("num_leaves", "Número máximo de hojas ('num_leaves')",
     "Capacidad de cada árbol", 0, 0, "logarítmica, entera"),
    ("min_data_in_leaf", "Mínimo de observaciones por hoja ('min_data_in_leaf')",
     "Capacidad: evidencia mínima de cada hoja", 0, 0, "logarítmica, entera"),
    ("feature_fraction", "Fracción de variables por árbol ('feature_fraction')",
     "Regularización: submuestreo de columnas", 3, 2, "lineal"),
    ("bagging_fraction", "Fracción de filas por iteración ('bagging_fraction')",
     "Regularización: submuestreo de filas", 3, 2, "lineal"),
    ("lambda_l2", "Penalización L2 ('lambda_l2')",
     "Regularización: encoge el valor de cada hoja", 2, 1, "lineal"),
]
NO_APLICA = "No aplica"
NO_DISPONIBLE = "n. d."


def _num(x, decimales: int = 1, signo: bool = False) -> str:
    """Número con coma decimal, punto de miles y redondeo mitad hacia arriba.

    El signo menos es el tipográfico, como en las tablas del documento; con
    ``signo`` verdadero los positivos llevan «+».
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NO_DISPONIBLE
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimales), rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    texto = f"{abs(d):,.{decimales}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    if d < 0:
        return "−" + texto
    return ("+" + texto) if signo and d > 0 else texto


def _compacto(x: float) -> str:
    """Extremo de un rango sin ceros sobrantes (1,0 se escribe 1; 30,0, 30)."""
    d = Decimal(repr(float(x))).normalize()
    decimales = max(0, -d.as_tuple().exponent)
    return _num(float(d), decimales)


def _rango(a: float, b: float, decimales: int) -> str:
    """Rango «a a b» con los decimales pedidos."""
    return f"{_num(a, decimales)} a {_num(b, decimales)}"


def _buscar(objeto, clave):
    """Busca una clave en un JSON anidado (el manifiesto no fija su profundidad)."""
    if isinstance(objeto, dict):
        if clave in objeto:
            return objeto[clave]
        for v in objeto.values():
            r = _buscar(v, clave)
            if r is not None:
                return r
    elif isinstance(objeto, list):
        for v in objeto:
            r = _buscar(v, clave)
            if r is not None:
                return r
    return None


def _fanova(ensayos: pd.DataFrame, espacio: dict, con_perdida: bool) -> dict | None:
    """Importancia fANOVA del evaluador de Optuna, promedio de cinco semillas.

    Con ``con_perdida`` falso se calcula dentro de los ensayos Tweedie e incluye
    la potencia; con ``con_perdida`` verdadero se calcula sobre todos los
    ensayos, con la pérdida como variable categórica y sin la potencia, que
    sólo existe con pérdida Tweedie.
    """
    try:
        import optuna
        from optuna.distributions import CategoricalDistribution as C
        from optuna.distributions import FloatDistribution as F
        from optuna.distributions import IntDistribution as I
    except ImportError:
        return None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    distribuciones = {
        "learning_rate": F(*espacio["learning_rate"], log=True),
        "num_leaves": I(*espacio["num_leaves"], log=True),
        "min_data_in_leaf": I(*espacio["min_data_in_leaf"], log=True),
        "feature_fraction": F(*espacio["feature_fraction"]),
        "bagging_fraction": F(*espacio["bagging_fraction"]),
        "lambda_l2": F(*espacio["lambda_l2"]),
    }
    if con_perdida:
        distribuciones = {"objective": C(list(espacio["objective"])), **distribuciones}
    else:
        distribuciones = {"tweedie_variance_power": F(*espacio["tweedie_variance_power"]),
                          **distribuciones}
    estudio = optuna.create_study(direction="minimize")
    for _, r in ensayos.iterrows():
        parametros = {}
        for k, d in distribuciones.items():
            if isinstance(d, C):
                parametros[k] = r[k]
            elif isinstance(d, I):
                parametros[k] = int(r[k])
            else:
                parametros[k] = float(r[k])
        estudio.add_trial(optuna.trial.create_trial(
            params=parametros, distributions=distribuciones, value=float(r["D_interna"])))
    valores = []
    for semilla in SEMILLAS_FANOVA:
        evaluador = optuna.importance.FanovaImportanceEvaluator(seed=semilla)
        valores.append(optuna.importance.get_param_importances(estudio, evaluator=evaluador))
    return {k: float(v) for k, v in pd.DataFrame(valores).mean().items()}


def _controlar_configuracion(elegido: dict, mejor: pd.Series) -> dict:
    """Controla que los valores declarados sean los del mejor ensayo; si no, se detiene."""
    control = {}
    for clave, *_ in FILAS:
        if clave == "mejor_iteracion":
            continue
        declarado, ensayo = float(elegido[clave]), float(mejor[clave])
        control[clave] = bool(np.isclose(declarado, ensayo, rtol=1e-9, atol=0))
    control["objective"] = elegido["objective"] == mejor["objective"]
    if not all(control.values()):
        raise SystemExit(f"La configuración declarada no es la del mejor ensayo: {control}")
    return control


def construir(motor: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """Arma la Tabla L3, el resumen por pérdida y el resumen de la búsqueda."""
    ensayos = pd.read_csv(motor / ENSAYOS)
    cfg = yaml.safe_load((motor / CONFIGURACION).read_text(encoding="utf-8"))
    elegido = cfg["modelos"]["lightgbm"]
    busqueda = cfg["lightgbm_tuning"]
    espacio = busqueda["espacio"]

    arboles_principal = None
    ruta_manifiesto = motor / MANIFIESTO
    if ruta_manifiesto.exists():
        manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
        arboles_principal = _buscar(manifiesto, "lightgbm_mejor_iteracion")

    mejor_d = float(ensayos["D_interna"].min())
    mejor = ensayos.loc[ensayos["D_interna"].idxmin()]
    control = _controlar_configuracion(elegido, mejor)
    tw = ensayos[ensayos["objective"] == "tweedie"].copy()
    meseta = ensayos[ensayos["D_interna"] <= mejor_d + MESETA_PUNTOS]
    fanova_tw = _fanova(tw, espacio, con_perdida=False)
    fanova_todos = _fanova(ensayos, espacio, con_perdida=True)

    # Resumen por pérdida: mejor y peor ensayo, mediana y costo.
    perdidas = []
    for clave, rotulo in PERDIDAS.items():
        g = ensayos[ensayos["objective"] == clave]
        if g.empty:
            continue
        b = g.loc[g["D_interna"].idxmin()]
        w = g.loc[g["D_interna"].idxmax()]
        perdidas.append({
            "clave": clave, "perdida": rotulo, "ensayos": int(len(g)),
            "mejor_ensayo": int(b["ensayo"]), "mejor_D": float(b["D_interna"]),
            "mejor_wmape": float(b["wmape_interna"]), "mejor_sesgo": float(b["bias_interna"]),
            "mediana_D": float(g["D_interna"].median()),
            "mediana_sesgo": float(g["bias_interna"].median()),
            "peor_ensayo": int(w["ensayo"]), "peor_D": float(w["D_interna"]),
            "arboles_min": int(g["mejor_iteracion"].min()), "arboles_max": int(g["mejor_iteracion"].max()),
            "segundos": float(g["segundos"].sum()),
        })
    perdidas = pd.DataFrame(perdidas)
    p = perdidas.set_index("clave")

    def tramo(clave: str, primero: bool) -> str:
        """Texto «Absoluta, 8 ensayos (D interna de 72,6 a 81,3)» de la fila de la pérdida."""
        r = p.loc[clave]
        rotulo = r["perdida"].capitalize() if primero else r["perdida"]
        if primero:
            return (f"{rotulo}, {r['ensayos']} ensayos (D interna de {_num(r['mejor_D'])} "
                    f"a {_num(r['peor_D'])})")
        return f"{rotulo}, {r['ensayos']} ({_num(r['mejor_D'])} a {_num(r['peor_D'])})"

    orden_perdidas = [k for k in ("regression_l1", "regression_l2", "tweedie") if k in p.index]
    texto_perdida = "; ".join(tramo(k, i == 0) for i, k in enumerate(orden_perdidas)) + " (categórica)"
    solo_meseta = meseta["objective"].map(PERDIDAS).unique()
    filas = [{
        "hiperparametro": "Función de pérdida ('objective')", "clave": "objective",
        "funcion": "Estructural: decide qué estadístico de la demanda aprende el modelo",
        "valor_elegido": PERDIDAS[mejor["objective"]], "valor_elegido_num": np.nan,
        "rango_explorado": texto_perdida,
        "meseta_min": np.nan, "meseta_max": np.nan,
        "meseta_texto": (f"{solo_meseta[0]} en los {len(meseta)}" if len(solo_meseta) == 1
                         else ", ".join(solo_meseta)),
        "fanova": (fanova_todos or {}).get("objective", np.nan),
        "fanova_texto": (f"{_num((fanova_todos or {}).get('objective'), 2)} "
                         f"(sobre los {len(ensayos)} ensayos)"),
        "spearman_tweedie": np.nan, "spearman_texto": NO_APLICA,
    }]
    for clave, rotulo, funcion, dec_valor, dec_meseta, escala in FILAS:
        rho = float(spearmanr(tw[clave], tw["D_interna"])[0])
        if clave == "mejor_iteracion":
            valor_num = float(mejor["mejor_iteracion"])
            valor_txt = f"{int(valor_num)} en el pliegue interno"
            valor_txt += ("; " + (f"{int(arboles_principal)} en la corrida principal"
                                  if arboles_principal is not None
                                  else "corrida principal sin manifiesto"))
            rango_txt = (f"No se busca: parada a las {busqueda['early_stopping_rounds']} rondas sin "
                         f"mejora, con techo de {_num(busqueda['num_boost_round'], 0)} "
                         f"(de {_rango(p.loc['tweedie', 'arboles_min'], p.loc['tweedie', 'arboles_max'], 0)} "
                         f"en los ensayos Tweedie)")
            fanova_txt, fanova_num = NO_APLICA, np.nan
        else:
            valor_num = float(elegido[clave])
            valor_txt = _num(valor_num, dec_valor)
            lo, hi = espacio[clave]
            rango_txt = f"{_compacto(lo)} a {_compacto(hi)} ({escala})"
            fanova_num = (fanova_tw or {}).get(clave, np.nan)
            fanova_txt = _num(fanova_num, 2)
        filas.append({
            "hiperparametro": rotulo, "clave": clave, "funcion": funcion,
            "valor_elegido": valor_txt, "valor_elegido_num": valor_num,
            "rango_explorado": rango_txt,
            "meseta_min": float(meseta[clave].min()), "meseta_max": float(meseta[clave].max()),
            "meseta_texto": _rango(meseta[clave].min(), meseta[clave].max(), dec_meseta),
            "fanova": fanova_num, "fanova_texto": fanova_txt,
            "spearman_tweedie": rho, "spearman_texto": _num(rho, 2, signo=True),
        })
    hilos = elegido.get("num_threads")
    frecuencia = elegido.get("bagging_freq")
    fijos_txt = "; ".join([
        f"Semilla del modelo {elegido.get('seed')}",
        f"semilla de la búsqueda {busqueda['semilla']}",
        "un hilo" if hilos == 1 else f"{hilos} hilos",
        "modo determinista" if elegido.get("deterministic") else "modo no determinista",
        ("submuestreo de filas en cada iteración" if frecuencia == 1
         else f"submuestreo de filas cada {frecuencia} iteraciones"),
    ])
    filas.append({
        "hiperparametro": "Valores fijos (no buscados)", "clave": "fijos", "funcion": "Reproducibilidad",
        "valor_elegido": fijos_txt, "valor_elegido_num": np.nan, "rango_explorado": NO_APLICA,
        "meseta_min": np.nan, "meseta_max": np.nan, "meseta_texto": NO_APLICA,
        "fanova": np.nan, "fanova_texto": NO_APLICA, "spearman_tweedie": np.nan,
        "spearman_texto": NO_APLICA,
    })
    tabla = pd.DataFrame(filas)

    # Los ensayos se numeran desde 0 (numeración de Optuna).
    orden = ensayos.sort_values("ensayo")
    acumulado = pd.Series(orden["D_interna"].cummin().to_numpy(), index=orden["ensayo"].astype(int))
    top10 = ensayos.nsmallest(10, "D_interna")
    primeros10 = orden.head(10)
    tw10 = primeros10[primeros10["objective"] == "tweedie"]["D_interna"]
    otras10 = primeros10[primeros10["objective"] != "tweedie"]["D_interna"]
    resumen = {
        "ensayos": int(len(ensayos)),
        "ensayos_por_perdida": ensayos["objective"].map(PERDIDAS).value_counts().to_dict(),
        "numeracion_desde": int(orden["ensayo"].min()),
        "mejor_ensayo": int(mejor["ensayo"]),
        "mejor_D": mejor_d,
        "mejor_wmape": float(mejor["wmape_interna"]),
        "mejor_sesgo": float(mejor["bias_interna"]),
        "peor_ensayo_global": int(ensayos.loc[ensayos["D_interna"].idxmax(), "ensayo"]),
        "peor_D_global": float(ensayos["D_interna"].max()),
        "meseta_2_puntos": int(len(meseta)),
        "meseta_2_puntos_por_perdida": meseta["objective"].map(PERDIDAS).value_counts().to_dict(),
        "meseta_1_5_puntos": int((ensayos["D_interna"] <= mejor_d + MESETA_ESTRECHA).sum()),
        "mejor_acumulado_hasta_ensayo": {int(n): float(acumulado.loc[n])
                                         for n in (9, 19, 29, 30, 39, 47, 48, 59)
                                         if n in acumulado.index},
        "ensayos_posteriores_al_mejor": int((ensayos["ensayo"] > mejor["ensayo"]).sum()),
        "mejora_despues_del_ensayo_30": (float(acumulado.loc[30]) - mejor_d
                                         if 30 in acumulado.index else None),
        "primeros_10_por_perdida": primeros10["objective"].map(PERDIDAS).value_counts().to_dict(),
        "primeros_10_peor_tweedie": float(tw10.max()) if len(tw10) else None,
        "primeros_10_mejor_otra_perdida": float(otras10.min()) if len(otras10) else None,
        "spearman_tasa_arboles_tweedie": float(spearmanr(tw["learning_rate"], tw["mejor_iteracion"])[0]),
        "tasa_por_arboles_meseta": [float((meseta["learning_rate"] * meseta["mejor_iteracion"]).min()),
                                    float((meseta["learning_rate"] * meseta["mejor_iteracion"]).max())],
        "tasa_por_arboles_diez_mejores": [float((top10["learning_rate"] * top10["mejor_iteracion"]).min()),
                                          float((top10["learning_rate"] * top10["mejor_iteracion"]).max())],
        "segundos_total": float(ensayos["segundos"].sum()),
        "segundos_mediana": float(ensayos["segundos"].median()),
        "l2_elegida_sobre_techo": float(elegido["lambda_l2"]) / float(espacio["lambda_l2"][1]),
        "fanova_tweedie": fanova_tw,
        "fanova_todos_los_ensayos": fanova_todos,
        "fanova_disponible": fanova_tw is not None,
        "arboles_corrida_principal": arboles_principal,
        "control_configuracion_igual_al_mejor_ensayo": control,
        "valores_fijos": {
            "semilla_modelo": elegido.get("seed"), "hilos": elegido.get("num_threads"),
            "determinista": elegido.get("deterministic"),
            "frecuencia_submuestreo_filas": elegido.get("bagging_freq"),
            "semilla_busqueda": busqueda["semilla"],
            "pliegue_interno": [busqueda["fold_desde"], busqueda["fold_hasta"]],
        },
    }
    return tabla, perdidas, resumen, _markdown(tabla, perdidas, resumen)


def _markdown(tabla: pd.DataFrame, perdidas: pd.DataFrame, resumen: dict) -> str:
    """Tabla L3 y resumen por pérdida en markdown, con los rótulos del documento."""
    cab = ["Hiperparámetro", "Valor elegido", "Rango explorado (escala)", "Función",
           f"Rango en la meseta ({resumen['meseta_2_puntos']} ensayos)",
           "Importancia en la búsqueda (Optuna, Tweedie)",
           "Correlación de rangos con la D interna (Tweedie)"]
    lineas = ["Tabla L3. Hiperparámetros del modelo global: valor elegido, rango explorado, "
              "función y lectura de la búsqueda", "",
              "| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for _, r in tabla.iterrows():
        celdas = [r["hiperparametro"], r["valor_elegido"], r["rango_explorado"], r["funcion"],
                  r["meseta_texto"], r["fanova_texto"], r["spearman_texto"]]
        lineas.append("| " + " | ".join(str(c) for c in celdas) + " |")
    lineas += ["", "Resumen por función de pérdida (apoyo de la nota y del texto; "
               "no se publica como tabla)", "",
               "| Pérdida | Ensayos | Mejor ensayo | Mejor D | WMAPE del mejor | Sesgo del mejor "
               "| Mediana de D | Peor ensayo | Peor D | Árboles (mín. a máx.) | Segundos |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, p in perdidas.iterrows():
        lineas.append("| " + " | ".join([
            p["perdida"], str(p["ensayos"]), str(p["mejor_ensayo"]), _num(p["mejor_D"], 1),
            _num(p["mejor_wmape"], 1), _num(p["mejor_sesgo"], 1, signo=True), _num(p["mediana_D"], 1),
            str(p["peor_ensayo"]), _num(p["peor_D"], 1),
            _rango(p["arboles_min"], p["arboles_max"], 0), _num(p["segundos"], 1)]) + " |")
    return "\n".join(lineas) + "\n"


def main() -> None:
    """Punto de entrada: lee los argumentos, construye y escribe las salidas."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motor", type=Path, default=Path("."),
                        help="Raíz del motor (la carpeta que contiene config/ y resultados/).")
    parser.add_argument("--destino", type=Path, default=None,
                        help="Carpeta de salida; por omisión, resultados/tablas dentro del motor.")
    args = parser.parse_args()
    destino = args.destino or (args.motor / DESTINO)

    tabla, perdidas, resumen, md = construir(args.motor)
    destino.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino / "tabla_l3.csv", index=False)
    perdidas.to_csv(destino / "tabla_l3_perdidas.csv", index=False)
    (destino / "tabla_l3.md").write_text(md, encoding="utf-8")
    (destino / "tabla_l3_resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
