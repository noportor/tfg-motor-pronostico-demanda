"""Pruebas de la sensibilidad de la cohorte (scripts/sensibilidad_cohorte.py, Tabla K1).

Sólo corridas sintéticas, construidas aquí en directorios temporales: ninguna
cifra ni clave de la corrida real.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUTA = Path(__file__).resolve().parent.parent / "scripts" / "sensibilidad_cohorte.py"
_spec = importlib.util.spec_from_file_location("sensibilidad_cohorte", RUTA)
sc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sc          # dataclasses resuelve el módulo por nombre
_spec.loader.exec_module(sc)

MODELOS = ("motor", "ma_12_gf", "ma_12", "naive_m1", "lightgbm")
PRUEBA = [f"2025-{m:02d}" for m in range(4, 13)] + [f"2026-{m:02d}" for m in range(1, 4)]


def _claves(n: int, desde: int = 0) -> list[str]:
    return [f"P{k:03d}|C{k % 2}|R{k % 3}" for k in range(desde, desde + n)]


def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _d(e: pd.DataFrame, costo: pd.Series, modelo: str) -> float:
    t = e[e["modelo"] == modelo].set_index("serie")
    c = costo.reindex(t.index).dropna()
    t = t.loc[c.index]
    real = (c * t["media_real"] * t["n_observaciones"]).sum()
    err = (c * t["mae"] * t["n_observaciones"]).sum()
    pred = (c * t["media_predicha"] * t["n_observaciones"]).sum()
    return 100 * err / real + 100 * abs(pred - real) / real


def _corrida(carpeta: Path, poblacion: list[str], incluidas: list[str], sin_venta: set[str],
             estado_origen: dict[str, str] | None, sin_costo: set[str], semilla: int) -> dict:
    """Escribe una corrida sintética con los archivos que lee el script."""
    rng = np.random.default_rng(semilla)
    carpeta.mkdir(parents=True)
    demanda_train = {s: float(rng.integers(100, 1000)) for s in poblacion}
    costo = pd.Series({s: float(rng.uniform(1, 20)) for s in poblacion if s not in sin_costo})

    fm = pd.DataFrame({
        "serie": poblacion,
        "estado": "activa",
        "incluida": [s in incluidas for s in poblacion],
        "motivo_exclusion": ["incluida" if s in incluidas else "historial<36m" for s in poblacion],
        "meses_entrenamiento": 48,
        "proporcion_ceros": 0.1,
        "volumen_entrenamiento": [demanda_train[s] for s in poblacion],
        "valor_entrenamiento": [demanda_train[s] * costo.get(s, 0.0) if s in costo.index else np.nan
                                for s in poblacion],
    })
    if estado_origen is not None:
        fm.insert(2, "estado_al_origen", [estado_origen.get(s, "descontinuada") for s in poblacion])
    fm.to_csv(carpeta / "fuera_de_muestra.csv", index=False)

    # Demanda de entrenamiento repartida en dos meses (el script la suma).
    mensual = pd.DataFrame(
        [{"periodo": p, "serie": s, "y_real": demanda_train[s] / 2, "bloque": "entrenamiento"}
         for s in incluidas for p in ("2023-01", "2023-02")]
        + [{"periodo": p, "serie": s, "y_real": 0.0, "bloque": "prueba"}
           for s in incluidas for p in PRUEBA])
    mensual.to_csv(carpeta / "series_y_predicciones.csv", index=False)

    filas = []
    for s in incluidas:
        media_real = 0.0 if s in sin_venta else float(rng.uniform(5, 50))
        for k, m in enumerate(MODELOS):
            pred = max(0.0, media_real * (1 + 0.1 * (k - 2)) + float(rng.uniform(0, 2)))
            filas.append({"serie": s, "mae": abs(pred - media_real) + float(rng.uniform(1, 5)),
                          "media_real": media_real, "media_predicha": pred,
                          "n_observaciones": 12, "modelo": m})
    errores = pd.DataFrame(filas)
    errores.to_csv(carpeta / "errores_por_serie.csv", index=False)

    tv = pd.DataFrame({"Modelo": list(MODELOS),
                       "D = WMAPE + |Bias| (%)": [_d(errores, costo, m) for m in MODELOS]})
    tv.to_csv(carpeta / "tabla_valorizada.csv", index=False)
    pd.DataFrame({"paso": ["0. series construidas", "1. final"],
                  "series": [len(poblacion), len(incluidas)],
                  "excluidas": [0, len(poblacion) - len(incluidas)]}).to_csv(
        carpeta / "cohorte_flujo.csv", index=False)
    pd.DataFrame({"bloque": ["entrenamiento", "validacion", "prueba"],
                  "desde": ["2017-04", "2024-04", "2025-04"],
                  "hasta": ["2024-03", "2025-03", "2026-03"]}).to_csv(
        carpeta / "particion.csv", index=False)
    pd.DataFrame({"serie": incluidas,
                  "modelo_elegido": [MODELOS[int(rng.integers(0, len(MODELOS)))] for _ in incluidas]}
                 ).to_csv(carpeta / "seleccion_motor.csv", index=False)
    manifiesto = {
        "generado_en": "2000-01-01T00:00:00+00:00", "duracion_segundos": 1.0,
        "codigo": {"commit": "0" * 40}, "configuracion": {"sha256": "1" * 64},
        "resultados": {"cohorte": {"series_finales": len(incluidas)}},
        "salidas": [{"archivo": a, "sha256": _sha(carpeta / a)} for a in sc.INSUMOS],
    }
    (carpeta / "manifiesto.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    return {"costo": costo, "errores": errores, "fm": fm}


@pytest.fixture()
def tres_corridas(tmp_path):
    poblacion = _claves(40)
    publicada = poblacion[:24]
    agregadas = poblacion[24:32]                 # 8 series que agrega la A
    sin_venta = set(agregadas[:5])               # 5 de ellas sin venta en prueba
    a = publicada + agregadas
    estado = {s: "activa" for s in a}
    for s in agregadas[3:6]:
        estado[s] = "dormida"
    estado[publicada[0]] = "descontinuada"
    b = [s for s in a if estado[s] == "activa"]
    sin_costo = {publicada[1]}
    datos = {
        "original": _corrida(tmp_path / "o", poblacion, publicada, set(), None, sin_costo, 1),
        "a": _corrida(tmp_path / "a", poblacion, a, sin_venta, estado, sin_costo, 2),
        "b": _corrida(tmp_path / "b", poblacion, b, sin_venta & set(b), estado, sin_costo, 3),
    }
    return tmp_path, datos, {"poblacion": poblacion, "publicada": publicada, "a": a, "b": b,
                             "sin_venta": sin_venta, "estado": estado}


def test_tabla_de_cohortes_reproduce_tamano_cobertura_y_d(tres_corridas):
    raiz, datos, conj = tres_corridas
    r = sc.construir(raiz / "o", raiz / "a", raiz / "b", None)
    t = r["cohortes"].set_index("cohorte")
    assert list(t["series"]) == [24, 32, len(conj["b"])]
    assert list(t["series_con_costo"]) == [23, 31, len(conj["b"]) - 1]
    for nombre, clave, cohorte in (("original", "original", conj["publicada"]),
                                   ("A: cohorte al origen", "a", conj["a"])):
        fm = datos[clave]["fm"]
        valor = fm.set_index("serie")["valor_entrenamiento"].fillna(0)
        esperada = 100 * valor.reindex(cohorte).sum() / valor.sum()
        assert t.loc[nombre, "cobertura_valor_entrenamiento_pct"] == pytest.approx(esperada)
        for m in MODELOS:
            assert t.loc[nombre, f"D_{m}"] == pytest.approx(
                _d(datos[clave]["errores"], datos[clave]["costo"], m))
        assert t.loc[nombre, "motor_menos_ma_12_gf_pp"] == pytest.approx(
            t.loc[nombre, "D_motor"] - t.loc[nombre, "D_ma_12_gf"])


def test_desglose_de_la_a_por_estado_y_venta(tres_corridas):
    raiz, datos, conj = tres_corridas
    r = sc.construir(raiz / "o", raiz / "a", raiz / "b", None)
    d = r["desglose_a"].set_index(["estado_al_origen", "venta_en_prueba"])
    assert d.loc[("todas", "todas"), "series"] == 32
    assert d.loc[("todas", "sin venta en prueba"), "series"] == 5
    assert d.loc[("todas", "con venta en prueba"), "series"] == 27
    # Los estados se reparten las 32 series.
    assert sum(d.loc[(e, "todas"), "series"] for e in sc.ESTADOS) == 32
    assert d.loc[("dormida", "sin venta en prueba"), "series"] == 2
    # «Activa o dormida» es la A sin las descontinuadas al origen.
    assert d.loc[("activa o dormida", "todas"), "series"] == 31
    # Sin demanda no hay D; los puntos de WMAPE de los grupos suman los del total.
    assert np.isnan(d.loc[("todas", "sin venta en prueba"), "D_motor"])
    total = d.loc[("todas", "todas"), "puntos_wmape_en_A_motor"]
    partes = (d.loc[("todas", "con venta en prueba"), "puntos_wmape_en_A_motor"]
              + d.loc[("todas", "sin venta en prueba"), "puntos_wmape_en_A_motor"])
    assert partes == pytest.approx(total)
    # El total es el WMAPE valorizado del motor en la A.
    e, c = datos["a"]["errores"], datos["a"]["costo"]
    t = e[e["modelo"] == "motor"].set_index("serie")
    cc = c.reindex(t.index).dropna()
    t = t.loc[cc.index]
    wmape = 100 * (cc * t["mae"] * 12).sum() / (cc * t["media_real"] * 12).sum()
    assert total == pytest.approx(wmape)
    # El aporte por modelo lista los cinco modelos, ordenado de menor a mayor.
    ap = r["aporte"]
    assert set(ap["modelo"]) == set(MODELOS)
    assert ap["puntos_wmape_en_A"].is_monotonic_increasing


def test_sin_variante_b_entrega_dos_filas(tres_corridas):
    raiz, _, _ = tres_corridas
    r = sc.construir(raiz / "o", raiz / "a", raiz / "no_existe", None)
    assert len(r["cohortes"]) == 2
    assert any("variante B no está disponible" in n for n in r["notas"])


def test_se_detiene_si_la_d_no_se_reproduce(tres_corridas):
    raiz, _, _ = tres_corridas
    ruta = raiz / "a" / "tabla_valorizada.csv"
    tv = pd.read_csv(ruta)
    tv.loc[0, "D = WMAPE + |Bias| (%)"] += 0.01
    tv.to_csv(ruta, index=False)
    with pytest.raises(SystemExit):
        sc.construir(raiz / "o", raiz / "a", raiz / "b", None, verificar_hash=False)


def test_se_detiene_si_un_insumo_no_coincide_con_su_manifiesto(tres_corridas):
    raiz, _, _ = tres_corridas
    with open(raiz / "o" / "particion.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(SystemExit):
        sc.construir(raiz / "o", raiz / "a", raiz / "b", None)


def test_se_detiene_si_la_b_no_es_la_a_restringida_a_activas(tres_corridas, tmp_path):
    raiz, _, conj = tres_corridas
    otra = conj["b"][:-1]            # le falta una activa
    _corrida(tmp_path / "b2", conj["poblacion"], otra, set(), conj["estado"], set(), 9)
    with pytest.raises(SystemExit):
        sc.construir(raiz / "o", raiz / "a", tmp_path / "b2", None)


def test_cruce_de_quiebres_cuenta_por_sku_y_regional(tres_corridas):
    raiz, _, conj = tres_corridas
    sin_venta = sorted(conj["sin_venta"])
    filas = []
    # Primera: quiebre total en dos meses de prueba; segunda: un mes con quiebre
    # parcial; tercera: sólo antes de la prueba (no cuenta); el resto sin medición.
    s0, s1, s2 = (s.split("|") for s in sin_venta[:3])
    for p in ("2025-05-01", "2025-06-01"):
        filas.append({"fecha": p, "sku": s0[0], "regional": s0[2],
                      "semanas_medidas": 4, "semanas_en_cero": 4})
    filas.append({"fecha": "2025-07-01", "sku": s1[0], "regional": s1[2],
                  "semanas_medidas": 4, "semanas_en_cero": 1})
    filas.append({"fecha": "2025-08-01", "sku": s1[0], "regional": s1[2],
                  "semanas_medidas": 4, "semanas_en_cero": 0})
    filas.append({"fecha": "2024-07-01", "sku": s2[0], "regional": s2[2],
                  "semanas_medidas": 4, "semanas_en_cero": 4})
    q = pd.DataFrame(filas)
    res = sc.cruce_quiebres(pd.Index(sin_venta), q, "2025-04", "2026-03")
    assert res == {"series": 5, "con_medicion_en_prueba": 2, "con_algun_quiebre_en_prueba": 2,
                   "con_quiebre_en_todos_los_meses_medidos": 1, "meses_con_quiebre_mediana": 1.5}
    ruta = raiz / "quiebres.csv"
    q.to_csv(ruta, index=False)
    r = sc.construir(raiz / "o", raiz / "a", raiz / "b", ruta)
    assert r["quiebres"]["A, sin venta en prueba"]["con_algun_quiebre_en_prueba"] == 2


def test_escritura_solo_agregados(tres_corridas, tmp_path):
    raiz, _, conj = tres_corridas
    destino = tmp_path / "tablas"
    sc.escribir(sc.construir(raiz / "o", raiz / "a", raiz / "b", None), destino)
    for archivo in destino.iterdir():
        texto = archivo.read_text(encoding="utf-8")
        assert not any(s.split("|")[0] in texto for s in conj["poblacion"]), archivo.name
    md = (destino / "tabla_K1_sensibilidad_cohorte.md").read_text(encoding="utf-8")
    assert md.startswith("Tabla K1.")


def test_d_valorizada_sin_demanda_es_nan():
    assert np.isnan(sc.d_valorizada(1.0, 0.0, 1.0))
    assert sc.d_valorizada(10.0, 100.0, 90.0) == pytest.approx(20.0)


def test_declara_arbol_con_cambios_y_descontinuadas_de_la_publicada(tres_corridas):
    raiz, _, _ = tres_corridas
    ruta = raiz / "b" / "manifiesto.json"
    m = json.loads(ruta.read_text(encoding="utf-8"))
    m["codigo"].update({"arbol_limpio": False, "archivos_modificados": ["scripts/x.py"]})
    ruta.write_text(json.dumps(m), encoding="utf-8")
    r = sc.construir(raiz / "o", raiz / "a", raiz / "b", None)
    corr = r["resumen"]["corridas"]
    assert corr["B: cohorte al origen, sólo activas"]["arbol_limpio"] is False
    assert corr["B: cohorte al origen, sólo activas"]["archivos_modificados"] == ["scripts/x.py"]
    assert any("cambios sin commit: scripts/x.py" in n for n in r["notas"])
    # La descontinuada al origen del conjunto sintético está en la cohorte publicada.
    desc = r["resumen"]["variante_a"]["por_estado_al_origen"]["descontinuada"]
    assert desc["series"] == 1 and desc["en_cohorte_publicada"] == 1


def test_se_detiene_si_quiebres_no_coincide_con_el_sello(tres_corridas):
    raiz, _, conj = tres_corridas
    s = sorted(conj["sin_venta"])[0].split("|")
    q = pd.DataFrame([{"fecha": "2025-05-01", "sku": s[0], "regional": s[2],
                       "semanas_medidas": 4, "semanas_en_cero": 4}])
    ruta = raiz / "quiebres.csv"
    q.to_csv(ruta, index=False)
    manif = raiz / "o" / "manifiesto.json"
    m = json.loads(manif.read_text(encoding="utf-8"))
    m["datos"] = {"auxiliares": [{"archivo": "quiebres.csv", "sha256": _sha(ruta)}]}
    manif.write_text(json.dumps(m), encoding="utf-8")
    r = sc.construir(raiz / "o", raiz / "a", raiz / "b", ruta)
    assert r["resumen"]["quiebres_sello"].startswith("verificado")
    with open(ruta, "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(SystemExit):
        sc.construir(raiz / "o", raiz / "a", raiz / "b", ruta)


def test_meses_comunes_separa_ceros_agregados_de_pronosticos_distintos():
    # Dos series, dos meses evaluados en la original; la variante agrega un mes en
    # cero a la primera. «b» pronostica igual (sesgo negativo): su D no cambia; «m» cambia.
    o = pd.DataFrame({"serie": ["s1", "s1", "s2", "s2"], "periodo": ["2025-04", "2025-05"] * 2,
                      "bloque": "prueba", "y_real": [10.0, 8.0, 6.0, 4.0],
                      "b": [5.0, 5.0, 3.0, 3.0], "m": [9.0, 9.0, 5.0, 5.0]})
    a = pd.concat([o.assign(m=[8.0, 8.0, 5.0, 5.0]),
                   pd.DataFrame({"serie": ["s1"], "periodo": ["2025-06"], "bloque": ["prueba"],
                                 "y_real": [0.0], "b": [5.0], "m": [8.0]})], ignore_index=True)
    costo = pd.Series({"s1": 2.0, "s2": 1.0})
    r = sc.comparar_meses_comunes(o, a, costo, ["b", "m"])
    assert r["meses_serie"] == 4
    assert r["modelos_con_pronostico_identico"] == ["b"]
    # D de «b» en los meses comunes = D de «b» con el mes en cero agregado (sesgo negativo).
    real = 2 * 18 + 10
    e_b, p_b = 2 * (5 + 3) + (3 + 1), 2 * 10 + 6
    assert r["D_variante_en_meses_de_la_original"]["b"] == pytest.approx(
        100 * e_b / real + 100 * abs(p_b - real) / real)
    e_b0, p_b0 = e_b + 2 * 5, p_b + 2 * 5
    assert 100 * e_b0 / real + 100 * abs(p_b0 - real) / real == pytest.approx(
        r["D_variante_en_meses_de_la_original"]["b"])
