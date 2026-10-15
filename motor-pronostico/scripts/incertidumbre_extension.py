"""Anexo H, Tabla H4: incertidumbre agrupada de la extensión exploratoria.

Propósito
---------
La extensión exploratoria (apartado 4.3.3) contrasta con Wilcoxon, en sus dos
gestiones de prueba, las cuatro mezclas, el brazo motor de la extensión y
LightGBM frente al promedio móvil de 12 meses y frente al modelo ingenuo. Esos
contrastes tratan cada serie SKU–canal–regional como independiente, igual que
el de la validación. Este script les aplica el MISMO remuestreo que la Tabla H3
(``scripts/incertidumbre_agrupada.py``, del que reutiliza todas las funciones):

1. **Conglomerados de SKU**: se remuestrean SKU completos, con reposición, y
   cada SKU sorteado entra con todas sus series.
2. **Bloques móviles circulares de 3 meses** del bloque de prueba de cada
   gestión (los pronósticos por serie y mes están publicados en
   ``series_y_predicciones.csv`` de cada corrida).

Magnitudes (propuesto menos referencia; negativo = el propuesto erra menos):
diferencia media y mediana por serie de ``error_compuesto_unidades`` (MAE +
|sesgo|, la métrica del contraste) y diferencia de la D valorizada del
portafolio, en puntos porcentuales. B = 2.000 réplicas, semilla 20260408,
intervalos percentiles al 95 %; dentro de cada gestión todas las comparaciones
usan las mismas réplicas.

Gestiones (cada una es una corrida publicada con su manifiesto):

* ``normal``: la gestión 2026 (abril de 2025 a marzo de 2026), bloque de
  prueba del estudio (``salidas_ml2_v9/``).
* ``quiebre``: la gestión 2025 (abril de 2024 a marzo de 2025), inicio de la
  crisis cambiaria, con el protocolo desplazado un año
  (``salidas_ml2_v9_blindaje/``).

Controles (el script se detiene si alguno falla): el SHA-256 de cada insumo
coincide con el manifiesto de su corrida; la D reconstruida reproduce la
``tabla_valorizada.csv`` de la corrida; el panel mensual reproduce la matriz de
contraste, la D y ``error_por_mes.csv`` (todo esto lo hace
``incertidumbre_agrupada.construir``); y el Wilcoxon recalculado con la función
de la corrida (``src.pruebas.wilcoxon_pareado``) reproduce, para las doce
comparaciones de cada gestión, el r y el rango biserial a cuatro decimales y
el p con el mismo formato con que los imprime ``pruebas_estadisticas.txt``.

Combinación frente a la selección validada (gestión 2026): el documento
afirma además que las mezclas del menú curado superan al motor validado de la
corrida principal (D de 74,9 % y 74,0 % frente a 75,4 %). Esa comparación cruza
dos corridas: las mezclas salen de ``salidas_ml2_v9/`` y el motor validado de
``salidas/``. El script la remuestrea con el mismo método y las mismas réplicas
de la gestión 2026 (filas con referencia ``motor_validado``, sin p ni r porque
ninguna corrida publica ese Wilcoxon), y se detiene si las dos corridas no
evalúan las mismas series, en el mismo orden, con los mismos meses, la misma
venta real y el mismo costo, o si algún modelo común a las dos (todos menos el
motor) no da la misma métrica de contraste.

Salidas (sólo agregados; ningún identificador de serie ni de producto):
``tabla_H4_incertidumbre_extension.csv`` y ``.md`` en ``--destino``.

Uso
---
    python -X utf8 scripts/incertidumbre_extension.py
        [--normal salidas_ml2_v9] [--quiebre salidas_ml2_v9_blindaje]
        [--propuestos mezcla_prom mezcla_pond ...] [--referencias ma_12 naive_m1]
        [--base salidas] [--sin-cruce] [--propuestos-cruce mezcla_prom mezcla_pond]
        [--destino resultados/tablas] [--replicas 2000] [--semilla 20260408]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.reporte import _formato_p  # noqa: E402  (el formato con que la corrida imprime p)


def _cargar_agrupada():
    """Importa scripts/incertidumbre_agrupada.py (scripts/ no es un paquete)."""
    nombre = "incertidumbre_agrupada"
    if nombre in sys.modules:
        return sys.modules[nombre]
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo          # dataclasses resuelve el módulo por nombre
    spec.loader.exec_module(modulo)
    return modulo


ia = _cargar_agrupada()

PROPUESTOS = ["mezcla_prom", "mezcla_pond", "mezcla_h", "mezcla_conmutada", "motor", "lightgbm"]
REFERENCIAS = ["ma_12", "naive_m1"]

GESTIONES = {
    # clave: (directorio por omisión, rótulo)
    "normal": ("salidas_ml2_v9", "gestión 2026 (normal)"),
    "quiebre": ("salidas_ml2_v9_blindaje", "gestión 2025 (inicio de la crisis cambiaria)"),
}

NOMBRES = {
    "mezcla_prom": "promedio simple del menú curado",
    "mezcla_pond": "menú curado ponderado por validación",
    "mezcla_h": "menú curado ponderado por horizonte",
    "mezcla_conmutada": "sistema conmutado",
    "motor": "brazo motor de la extensión",
    "lightgbm": "LightGBM",
    "ma_12": "promedio móvil de 12 meses",
    "naive_m1": "ingenuo",
}


def comparaciones_de(propuestos: list[str], referencias: list[str]) -> list[tuple[str, str]]:
    return [(p, r) for p in propuestos for r in referencias]


# ---------------------------------------------------------------------------
# Lectura de pruebas_estadisticas.txt
# ---------------------------------------------------------------------------

_ENCABEZADO = re.compile(r"^\s{2}(\S+)\s{2}vs\s{2}(\S+)\s*$")
_R = re.compile(r"Tamaño del efecto\s+r\s*=\s*([+-]?\d+\.\d+)")
_RB = re.compile(r"Rango-biserial\s*=\s*([+-]?\d+\.\d+)")
_P = re.compile(r"^\s+p\s+=\s+(.+?)\s+->")


def leer_publicados(texto: str) -> dict[tuple[str, str], dict]:
    """Bloques de Wilcoxon de la sección 1: r, rango biserial y p tal como se imprimen.

    Devuelve, por (propuesto, referencia): ``r`` y ``rango_biserial`` (float, cuatro
    decimales) y ``p_texto`` (la cadena impresa, p. ej. «< 1e-16» o «0.3643»).
    """
    seccion = texto.split("2. FRIEDMAN")[0]
    publicados: dict[tuple[str, str], dict] = {}
    actual: tuple[str, str] | None = None
    for linea in seccion.splitlines():
        m = _ENCABEZADO.match(linea)
        if m:
            actual = (m.group(1), m.group(2))
            publicados[actual] = {}
            continue
        if actual is None:
            continue
        for clave, patron in (("r", _R), ("rango_biserial", _RB)):
            m = patron.search(linea)
            if m:
                publicados[actual][clave] = float(m.group(1))
        m = _P.match(linea)
        if m:
            publicados[actual]["p_texto"] = m.group(1).strip()
    incompletos = [k for k, v in publicados.items() if set(v) != {"r", "rango_biserial", "p_texto"}]
    if incompletos:
        raise ValueError(f"Bloques de Wilcoxon incompletos en pruebas_estadisticas.txt: {incompletos}")
    return publicados


# ---------------------------------------------------------------------------
# Construcción
# ---------------------------------------------------------------------------

def construir_gestion(
    salidas: Path, comparaciones: list[tuple[str, str]], replicas: int, semilla: int
) -> tuple[pd.DataFrame, dict]:
    publicados = leer_publicados((salidas / "pruebas_estadisticas.txt").read_text(encoding="utf-8"))
    faltan = [c for c in comparaciones if c not in publicados]
    if faltan:
        raise SystemExit(f"{salidas.name}: pruebas_estadisticas.txt no publica {faltan}")
    control_r = {c: (publicados[c]["r"], publicados[c]["rango_biserial"]) for c in comparaciones}
    tabla, resumen = ia.construir(salidas, replicas, semilla, comparaciones, control_r)

    # El p recalculado debe imprimirse igual que en el archivo de la corrida.
    for (prop, ref), grupo in tabla.groupby(["propuesto", "referencia"]):
        p = float(grupo["p_wilcoxon_publicado"].iloc[0])
        if _formato_p(p) != publicados[(prop, ref)]["p_texto"]:
            raise SystemExit(f"{salidas.name}: no se reproduce el p publicado de {prop} frente a {ref}")
    tabla.insert(0, "p_publicado_texto", tabla.apply(
        lambda f: publicados[(f["propuesto"], f["referencia"])]["p_texto"], axis=1))
    resumen["comparaciones_reproducidas"] = len(comparaciones)
    # La corrida se identifica por su manifiesto: el commit que registra es el
    # del árbol de trabajo de la extensión, no uno del historial publicado.
    resumen["sha256_manifiesto"] = ia._sha256(salidas / "manifiesto.json")
    return tabla, resumen


def construir(
    directorios: dict[str, Path],
    comparaciones: list[tuple[str, str]],
    replicas: int,
    semilla: int,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    tablas, resumenes = [], {}
    for clave, salidas in directorios.items():
        tabla, resumen = construir_gestion(Path(salidas), comparaciones, replicas, semilla)
        tabla.insert(0, "gestion", clave)
        tablas.append(tabla)
        resumenes[clave] = resumen
    tabla = pd.concat(tablas, ignore_index=True)
    columnas = ["gestion", "comparacion", "propuesto", "referencia", "magnitud", "remuestreo",
                "estimacion", "ic95_inferior", "ic95_superior", "excluye_cero",
                "p_wilcoxon_publicado", "p_publicado_texto", "r_publicado",
                "rango_biserial_publicado", "series_a_favor", "series_en_contra", "empates"]
    return tabla[columnas], resumenes


# ---------------------------------------------------------------------------
# Combinación frente a la selección validada (gestión 2026, dos corridas)
# ---------------------------------------------------------------------------

VALIDADO = "motor_validado"
PROPUESTOS_CRUCE = ["mezcla_prom", "mezcla_pond"]
TOLERANCIA_IGUALES = 1e-9


def _insumos_valorizados(salidas: Path, modelos: list[str], matriz: pd.DataFrame) -> dict:
    """Componentes de la D por serie y paneles mensuales de prueba de ``modelos``.

    Mismo cálculo que ``incertidumbre_agrupada.construir``, con sus controles: la
    D reconstruida reproduce ``tabla_valorizada.csv`` y el panel mensual
    reproduce la matriz de contraste.
    """
    indice = matriz.index
    mensual = pd.read_csv(salidas / "series_y_predicciones.csv",
                          usecols=["periodo", "serie", "y_real", "bloque", *modelos])
    costo = ia.costo_por_serie(salidas, mensual).reindex(indice)
    con_costo = costo.notna().to_numpy()
    c = costo.fillna(0.0).to_numpy()
    errores = pd.read_csv(salidas / "errores_por_serie.csv")
    tabla_val = pd.read_csv(salidas / "tabla_valorizada.csv").set_index("Modelo")
    prueba = mensual[mensual["bloque"] == "prueba"]
    meses = sorted(prueba["periodo"].unique())
    y = prueba.pivot(index="serie", columns="periodo", values="y_real").reindex(
        index=indice, columns=meses).to_numpy(dtype=float)
    componentes, paneles, control_d = {}, {}, {}
    for m in modelos:
        e = errores[errores["modelo"] == m].set_index("serie").reindex(indice)
        n = e["n_observaciones"].to_numpy(dtype=float)
        componentes[m] = (c * e["mae"].to_numpy() * n,
                          c * e["media_real"].to_numpy() * n,
                          c * e["media_predicha"].to_numpy() * n)
        d = float(ia._d_de(componentes[m], np.ones((1, len(n))))[0])
        publicada = float(tabla_val.loc[m, "D = WMAPE + |Bias| (%)"])
        if abs(d - publicada) > ia.TOLERANCIA_D:
            raise SystemExit(f"{salidas.name}: la D reconstruida de {m} no reproduce tabla_valorizada.csv")
        control_d[m] = publicada
        p = prueba.pivot(index="serie", columns="periodo", values=m).reindex(
            index=indice, columns=meses).to_numpy(dtype=float)
        ok = ~np.isnan(y) & ~np.isnan(p)
        firmado = np.where(ok, p - y, 0.0)
        absoluto = np.abs(firmado)
        valor = con_costo[:, None] & ok
        paneles[m] = {"absoluto": absoluto, "firmado": firmado, "observado": ok.astype(float),
                      "error_val": (np.where(valor, absoluto, 0.0) * c[:, None]).sum(axis=0),
                      "real_val": (np.where(valor, y, 0.0) * c[:, None]).sum(axis=0),
                      "pred_val": (np.where(valor, p, 0.0) * c[:, None]).sum(axis=0)}
        identidad = ia.compuesto_por_serie(absoluto, firmado, ok.astype(float),
                                           np.ones((1, len(meses))))[:, 0]
        if float(np.nanmax(np.abs(identidad - matriz[m].to_numpy()))) > ia.TOLERANCIA_SERIE:
            raise SystemExit(f"{salidas.name}: el panel mensual no reproduce la matriz de {m}")
    return {"componentes": componentes, "paneles": paneles, "costo": c, "real": y,
            "meses": meses, "d_publicada": control_d}


def construir_cruce(
    base: Path, normal: Path, propuestos: list[str], replicas: int, semilla: int
) -> tuple[pd.DataFrame, dict]:
    """Mezclas de la extensión (``normal``) frente al motor validado (``base``).

    Mismas réplicas que la gestión 2026: los conglomerados salen del mismo
    índice de series y la semilla es la misma.
    """
    manifiesto_base = ia.verificar_insumos(base)
    ia.verificar_insumos(normal)
    mb = pd.read_csv(base / "matriz_contraste.csv").set_index("serie").dropna()
    me = pd.read_csv(normal / "matriz_contraste.csv").set_index("serie").dropna()
    if not mb.index.equals(me.index):
        raise SystemExit("El motor validado y la extensión no evalúan las mismas series en el mismo orden")
    comunes = sorted((set(mb.columns) & set(me.columns)) - {"motor"})
    distintos = [m for m in comunes
                 if float(np.max(np.abs(mb[m].to_numpy() - me[m].to_numpy()))) > TOLERANCIA_IGUALES]
    if not comunes or distintos:
        raise SystemExit(f"Los modelos comunes a las dos corridas no coinciden: {distintos or 'ninguno'}")
    ib = _insumos_valorizados(base, ["motor"], mb)
    ix = _insumos_valorizados(normal, propuestos, me)
    if ib["meses"] != ix["meses"]:
        raise SystemExit("Las dos corridas no comparten el bloque de prueba")
    mismo_real = np.array_equal(np.isnan(ib["real"]), np.isnan(ix["real"])) and np.allclose(
        np.nan_to_num(ib["real"]), np.nan_to_num(ix["real"]), rtol=0, atol=TOLERANCIA_IGUALES)
    if not mismo_real or not np.allclose(ib["costo"], ix["costo"], rtol=1e-12, atol=0):
        raise SystemExit("Las dos corridas no comparten la venta real o el costo de las series")

    componentes = {VALIDADO: ib["componentes"]["motor"], **ix["componentes"]}
    paneles = {VALIDADO: ib["paneles"]["motor"], **ix["paneles"]}
    comparaciones = [(p, VALIDADO) for p in propuestos]
    diferencias = {(p, VALIDADO): (me[p] - mb["motor"]).to_numpy(dtype=float) for p in propuestos}
    codigos = pd.factorize(ia.sku_de_serie(me.index))[0]
    filas = ia.bootstrap_por_sku(diferencias, componentes, codigos, replicas, semilla)
    estimaciones = {(f.propuesto, f.referencia, f.magnitud): f.estimacion for f in filas}
    filas += ia.bootstrap_por_bloques(paneles, comparaciones, estimaciones, replicas, semilla)

    registros = []
    for f in filas:
        d = diferencias[(f.propuesto, f.referencia)]
        registros.append({
            "gestion": "normal", "comparacion": f"{f.propuesto} - {f.referencia}",
            "propuesto": f.propuesto, "referencia": f.referencia, "magnitud": f.magnitud,
            "remuestreo": f.remuestreo, "estimacion": f.estimacion,
            "ic95_inferior": f.ic_inferior, "ic95_superior": f.ic_superior,
            "excluye_cero": "sí" if ia.excluye_cero(f.ic_inferior, f.ic_superior) else "no",
            "p_wilcoxon_publicado": np.nan, "p_publicado_texto": "no publicado",
            "r_publicado": np.nan, "rango_biserial_publicado": np.nan,
            "series_a_favor": int((d < 0).sum()), "series_en_contra": int((d > 0).sum()),
            "empates": int((d == 0).sum()),
        })
    resumen = {
        "series": int(len(me)),
        "modelos_comunes_identicos": comunes,
        "d_motor_validado": ib["d_publicada"]["motor"],
        "d_propuestos": ix["d_publicada"],
        "corrida_base": manifiesto_base["generado_en"],
        "commit_base": manifiesto_base["codigo"]["commit"],
    }
    return pd.DataFrame(registros), resumen


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def _r(r: float) -> str:
    """r con dos decimales; con tres si a dos quedaría «0,00» (conserva el signo)."""
    return ia._num(r, 2) if round(abs(r), 2) > 0 else ia._num(r, 3)


def escribir_md(tabla: pd.DataFrame, resumenes: dict[str, dict],
                comparaciones: list[tuple[str, str]], ruta: Path,
                resumen_cruce: dict | None = None) -> None:
    _num, _p, _cientifico = ia._num, ia._p, ia._cientifico
    lineas = [
        "Tabla H4. Incertidumbre agrupada de las diferencias de la extensión exploratoria, "
        "en las dos gestiones de prueba (propuesto menos referencia; negativo = el propuesto "
        "erra menos)",
        "",
    ]
    for clave, resumen in resumenes.items():
        rotulo = GESTIONES.get(clave, (None, clave))[1]
        lineas += [f"### {rotulo[0].upper() + rotulo[1:]}", ""]
        sub = tabla[tabla["gestion"] == clave]
        for magnitud, titulo in ia.MAGNITUDES.items():
            decimales = 1 if magnitud == "d_portafolio" else 2
            lineas += [f"**{titulo}**", "",
                       "| Comparación | Estimación | IC 95 % por SKU | Excluye el cero (SKU) | "
                       "IC 95 % por bloques de 3 meses | Excluye el cero (bloques) | "
                       "p publicado | r publicado |",
                       "|---|---|---|---|---|---|---|---|"]
            for prop, ref in comparaciones:
                sel = sub[(sub["propuesto"] == prop) & (sub["referencia"] == ref)
                          & (sub["magnitud"] == magnitud)].set_index("remuestreo")
                s, b = sel.loc["sku"], sel.loc["bloques"]
                lineas.append(
                    f"| {NOMBRES.get(prop, prop)} frente a {NOMBRES.get(ref, ref)} | "
                    f"{_num(s['estimacion'], decimales)} | "
                    f"[{_num(s['ic95_inferior'], decimales)}; {_num(s['ic95_superior'], decimales)}] | "
                    f"{s['excluye_cero']} | "
                    f"[{_num(b['ic95_inferior'], decimales)}; {_num(b['ic95_superior'], decimales)}] | "
                    f"{b['excluye_cero']} | {_p(s['p_wilcoxon_publicado'])} | "
                    f"{_r(s['r_publicado'])} |"
                )
            lineas.append("")
        lineas += [
            f"Remuestreo: {resumen['conglomerados_sku']} SKU que agrupan "
            f"{_num(resumen['series'], 0)} series; {resumen['meses_prueba']} meses del bloque de "
            f"prueba en bloques móviles circulares de {resumen['largo_bloque']} meses; D valorizada "
            f"sobre {_num(resumen['series_con_costo'], 0)} series con costo. Controles: la D "
            f"reconstruida reproduce tabla_valorizada.csv con {_cientifico(resumen['control_d_max_pp'])} "
            f"puntos; el panel mensual reproduce la métrica por serie con "
            f"{_cientifico(resumen['control_serie_max_unidades'])} unidades, la D con "
            f"{_cientifico(resumen['control_d_mensual_max_pp'])} puntos y error_por_mes.csv "
            f"({', '.join(resumen['modelos_control_por_mes'])}) con "
            f"{_cientifico(resumen['control_por_mes_max_pp'])} puntos; el Wilcoxon recalculado "
            f"reproduce el r, el rango biserial y el p de las {resumen['comparaciones_reproducidas']} "
            "comparaciones de pruebas_estadisticas.txt. "
            f"Corrida: {resumen['corrida']} (manifiesto.json con SHA-256 "
            f"{resumen['sha256_manifiesto'][:12]}).",
            "",
        ]
    if resumen_cruce is not None:
        cruce = tabla[tabla["referencia"] == VALIDADO]
        lineas += ["### Gestión 2026: combinación frente a la selección validada", "",
                   "| Comparación | Magnitud | Estimación | IC 95 % por SKU | Excluye el cero (SKU) | "
                   "IC 95 % por bloques de 3 meses | Excluye el cero (bloques) |",
                   "|---|---|---|---|---|---|---|"]
        for prop in cruce["propuesto"].unique():
            for magnitud, titulo in ia.MAGNITUDES.items():
                decimales = 1 if magnitud == "d_portafolio" else 2
                sel = cruce[(cruce["propuesto"] == prop) & (cruce["magnitud"] == magnitud)
                            ].set_index("remuestreo")
                s, b = sel.loc["sku"], sel.loc["bloques"]
                lineas.append(
                    f"| {NOMBRES.get(prop, prop)} frente al motor validado | {titulo} | "
                    f"{_num(s['estimacion'], decimales)} | "
                    f"[{_num(s['ic95_inferior'], decimales)}; {_num(s['ic95_superior'], decimales)}] | "
                    f"{s['excluye_cero']} | "
                    f"[{_num(b['ic95_inferior'], decimales)}; {_num(b['ic95_superior'], decimales)}] | "
                    f"{b['excluye_cero']} |")
        lineas += [
            "",
            f"El motor validado es el de la corrida principal (salidas/; corrida "
            f"{resumen_cruce['corrida_base']}, commit {resumen_cruce['commit_base'][:7]}; D de "
            f"{_num(resumen_cruce['d_motor_validado'], 1)} %); las mezclas, las de la gestión 2026 "
            "de la extensión. Las dos corridas evalúan las mismas "
            f"{_num(resumen_cruce['series'], 0)} series con los mismos meses, la misma venta real y "
            "el mismo costo, y los modelos comunes a las dos dan la misma métrica de contraste "
            f"({', '.join(resumen_cruce['modelos_comunes_identicos'])}). Mismas réplicas que la "
            "gestión 2026. Ninguna corrida publica un Wilcoxon de esta comparación, por eso no "
            "lleva p ni r.",
            "",
        ]
    primera = next(iter(resumenes.values()))
    lineas += [
        f"B = {primera['replicas']:,} réplicas".replace(",", ".")
        + f", semilla {primera['semilla']}, intervalos percentiles al 95 %; dentro de cada gestión "
        "todas las comparaciones usan las mismas réplicas. La estimación es la de la muestra "
        "completa, sin remuestrear. El p y el r son los del Wilcoxon unilateral publicado por cada "
        "corrida, que trata cada serie como independiente. Método idéntico al de la Tabla H3 "
        "(scripts/incertidumbre_agrupada.py).",
        "",
    ]
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--normal", default=str(RAIZ / GESTIONES["normal"][0]))
    parser.add_argument("--quiebre", default=str(RAIZ / GESTIONES["quiebre"][0]))
    parser.add_argument("--propuestos", nargs="+", default=PROPUESTOS)
    parser.add_argument("--referencias", nargs="+", default=REFERENCIAS)
    parser.add_argument("--base", default=str(RAIZ / "salidas"),
                        help="corrida principal, con el motor validado")
    parser.add_argument("--sin-cruce", action="store_true",
                        help="omite la comparación de las mezclas con el motor validado")
    parser.add_argument("--propuestos-cruce", nargs="+", default=PROPUESTOS_CRUCE)
    parser.add_argument("--destino", default=str(RAIZ / "resultados" / "tablas"))
    parser.add_argument("--replicas", type=int, default=ia.REPLICAS)
    parser.add_argument("--semilla", type=int, default=ia.SEMILLA)
    args = parser.parse_args(argv)

    comparaciones = comparaciones_de(args.propuestos, args.referencias)
    directorios = {"normal": Path(args.normal), "quiebre": Path(args.quiebre)}
    tabla, resumenes = construir(directorios, comparaciones, args.replicas, args.semilla)
    resumen_cruce = None
    if not args.sin_cruce:
        cruce, resumen_cruce = construir_cruce(Path(args.base), Path(args.normal), args.propuestos_cruce,
                                               args.replicas, args.semilla)
        tabla = pd.concat([tabla, cruce[tabla.columns]], ignore_index=True)
    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    # Diez cifras significativas: con seis, algún límite cae en el borde del
    # redondeo a dos decimales y el CSV no reproduciría la celda de la tabla
    # (−18,694988… se escribía −18.695 y se leía −18,70).
    tabla.to_csv(destino / "tabla_H4_incertidumbre_extension.csv", index=False,
                 float_format="%.10g")
    escribir_md(tabla, resumenes, comparaciones, destino / "tabla_H4_incertidumbre_extension.md",
                resumen_cruce)
    print((destino / "tabla_H4_incertidumbre_extension.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
