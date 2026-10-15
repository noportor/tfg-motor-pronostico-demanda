"""Pruebas de la Tabla H4 (scripts/incertidumbre_extension.py).

Sólo paneles sintéticos, construidos aquí: ninguna cifra ni clave de las
corridas reales. Se arma una carpeta de salidas completa (con su manifiesto y
su pruebas_estadisticas.txt) para recorrer el script de punta a punta.
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

RAIZ = Path(__file__).resolve().parent.parent
RUTA = RAIZ / "scripts" / "incertidumbre_extension.py"
_spec = importlib.util.spec_from_file_location("incertidumbre_extension", RUTA)
ie = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ie
_spec.loader.exec_module(ie)

from src.pruebas import wilcoxon_pareado  # noqa: E402
from src.reporte import _formato_p  # noqa: E402

PROPUESTOS = ["mezcla_prom", "motor"]
REFERENCIAS = ["ma_12", "naive_m1"]
MODELOS = PROPUESTOS + REFERENCIAS


# ---------------------------------------------------------------------------
# Carpeta de salidas sintética
# ---------------------------------------------------------------------------

def _sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _sellar(directorio: Path) -> None:
    """Escribe manifiesto.json con el SHA-256 de cada archivo de la carpeta."""
    salidas = [{"archivo": r.name, "sha256": _sha(r)}
               for r in sorted(directorio.iterdir()) if r.name != "manifiesto.json"]
    manifiesto = {"generado_en": "2000-01-01T00:00:00+00:00",
                  "codigo": {"commit": "0" * 40}, "salidas": salidas}
    (directorio / "manifiesto.json").write_text(json.dumps(manifiesto), encoding="utf-8")


def _bloque_wilcoxon(w) -> list[str]:
    """Mismo formato que src/reporte.py::informe_pruebas, sección 1."""
    return [
        f"  {w.propuesto}  vs  {w.referencia}",
        f"    Tamaño del efecto  r = {w.r:+.4f}  (pequeño)",
        f"    Rango-biserial       = {w.rango_biserial:+.4f}  (Kerby: proporción neta)",
        f"    Series a favor       : {w.gana_propuesto:,} de {w.n_pares:,}",
        f"    p                    = {_formato_p(w.p)}   ->  no significativo al 0.05",
        "",
    ]


def _carpeta_sintetica(directorio: Path, semilla: int, n_sku: int = 40, por_sku: int = 3) -> Path:
    rng = np.random.default_rng(semilla)
    directorio.mkdir(parents=True)
    claves = [f"P{k:03d}|C{j}|R{j}" for k in range(n_sku) for j in range(por_sku)]
    n = len(claves)
    sku = np.repeat(np.arange(n_sku), por_sku)
    meses_e = [f"2001-{m:02d}" for m in range(1, 13)]
    meses_p = [f"2002-{m:02d}" for m in range(1, 13)]

    nivel = rng.gamma(3, 10, size=n_sku)[sku] + 5
    y_e = rng.poisson(nivel[:, None], size=(n, 12)).astype(float) + 1
    y_p = rng.poisson(nivel[:, None], size=(n, 12)).astype(float)
    choque = rng.normal(0, 2, size=(n_sku, 12))[sku]          # común a las series del SKU
    pred = {
        "mezcla_prom": y_p + rng.normal(0, 2, size=y_p.shape),
        "motor": y_p + choque + rng.normal(0, 4, size=y_p.shape),
        "naive_m1": y_p + choque + rng.normal(0, 4, size=y_p.shape),
        "ma_12": y_p + 0.4 * nivel[:, None] + rng.normal(0, 4, size=y_p.shape),
    }
    costo = rng.uniform(1, 5, size=n)

    # series_y_predicciones.csv (entrenamiento + prueba)
    filas = []
    for i, s in enumerate(claves):
        for t, mes in enumerate(meses_e):
            filas.append({"periodo": mes, "serie": s, "y_real": y_e[i, t], "bloque": "entrenamiento",
                          **{m: np.nan for m in MODELOS}})
        for t, mes in enumerate(meses_p):
            filas.append({"periodo": mes, "serie": s, "y_real": y_p[i, t], "bloque": "prueba",
                          **{m: pred[m][i, t] for m in MODELOS}})
    pd.DataFrame(filas).to_csv(directorio / "series_y_predicciones.csv", index=False)

    pd.DataFrame({"serie": claves, "valor_entrenamiento": costo * y_e.sum(axis=1)}).to_csv(
        directorio / "fuera_de_muestra.csv", index=False)

    matriz, errores, tabla_val, por_mes = {"serie": claves}, [], [], []
    for m in MODELOS:
        e = pred[m] - y_p
        matriz[m] = np.abs(e).mean(axis=1) + np.abs(e.mean(axis=1))
        errores.append(pd.DataFrame({
            "serie": claves, "modelo": m, "mae": np.abs(e).mean(axis=1),
            "media_real": y_p.mean(axis=1), "media_predicha": pred[m].mean(axis=1),
            "n_observaciones": 12}))
        err, real, pr = (costo[:, None] * np.abs(e)), costo[:, None] * y_p, costo[:, None] * pred[m]
        d = 100 * err.sum() / real.sum() + 100 * abs(pr.sum() - real.sum()) / real.sum()
        tabla_val.append({"Modelo": m, "D = WMAPE + |Bias| (%)": d})
        for t, mes in enumerate(meses_p):
            por_mes.append({"periodo": mes, "modelo": m,
                            "wmape_val": 100 * err[:, t].sum() / real[:, t].sum(),
                            "bias_val": 100 * (pr[:, t].sum() - real[:, t].sum()) / real[:, t].sum()})
    matriz = pd.DataFrame(matriz)
    matriz.to_csv(directorio / "matriz_contraste.csv", index=False)
    pd.concat(errores).to_csv(directorio / "errores_por_serie.csv", index=False)
    pd.DataFrame(tabla_val).to_csv(directorio / "tabla_valorizada.csv", index=False)
    pd.DataFrame(por_mes).to_csv(directorio / "error_por_mes.csv", index=False)

    m = matriz.set_index("serie")
    lineas = ["CONTRASTE", "", "1. WILCOXON DE RANGOS CON SIGNO", ""]
    for prop in PROPUESTOS:
        for ref in REFERENCIAS:
            lineas += _bloque_wilcoxon(wilcoxon_pareado(m[prop].rename(prop), m[ref].rename(ref)))
    lineas += ["2. FRIEDMAN OMNIBUS + POST HOC DE NEMENYI", "  mezcla_prom   7.2"]
    (directorio / "pruebas_estadisticas.txt").write_text("\n".join(lineas), encoding="utf-8")
    _sellar(directorio)
    return directorio


@pytest.fixture()
def gestiones(tmp_path):
    return {"normal": _carpeta_sintetica(tmp_path / "normal", 1),
            "quiebre": _carpeta_sintetica(tmp_path / "quiebre", 2)}


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------

def test_leer_publicados_toma_r_rango_y_p_de_cada_bloque():
    texto = "\n".join([
        "1. WILCOXON",
        "  mezcla_h  vs  ma_12",
        "    Tamaño del efecto  r = -0.3429  (mediano)",
        "    Rango-biserial       = -0.3959  (Kerby: proporción neta de series a favor del propuesto)",
        "    p                    = < 1e-16   ->  SIGNIFICATIVO al 0.05",
        "    Shapiro-Wilk de las diferencias: W = 0.1967, p = < 1e-16  (n = 10)",
        "  motor  vs  naive_m1",
        "    Tamaño del efecto  r = +0.0085  (insignificante)",
        "    Rango-biserial       = +0.0103  (Kerby)",
        "    p                    = 0.3643   ->  no significativo al 0.05",
        "2. FRIEDMAN OMNIBUS",
        "  motor  vs  otra_cosa",
    ])
    pub = ie.leer_publicados(texto)
    assert set(pub) == {("mezcla_h", "ma_12"), ("motor", "naive_m1")}
    assert pub[("mezcla_h", "ma_12")] == {"r": -0.3429, "rango_biserial": -0.3959, "p_texto": "< 1e-16"}
    assert pub[("motor", "naive_m1")]["p_texto"] == "0.3643"
    assert pub[("motor", "naive_m1")]["r"] == pytest.approx(0.0085)


def test_comparaciones_por_omision_son_las_doce_de_la_extension():
    comp = ie.comparaciones_de(ie.PROPUESTOS, ie.REFERENCIAS)
    assert len(comp) == 12
    assert {p for p, _ in comp} == {"mezcla_prom", "mezcla_pond", "mezcla_h",
                                     "mezcla_conmutada", "motor", "lightgbm"}
    assert {r for _, r in comp} == {"ma_12", "naive_m1"}


def test_extremo_a_extremo_sobre_dos_gestiones_sinteticas(gestiones, tmp_path):
    comp = ie.comparaciones_de(PROPUESTOS, REFERENCIAS)
    tabla, resumenes = ie.construir(gestiones, comp, 300, 20260408)
    # 2 gestiones × 4 comparaciones × 3 magnitudes × 2 remuestreos
    assert len(tabla) == 48
    assert set(resumenes) == {"normal", "quiebre"}
    assert all(r["comparaciones_reproducidas"] == 4 for r in resumenes.values())
    # El r de la tabla es el publicado en el txt de cada gestión.
    for clave, carpeta in gestiones.items():
        pub = ie.leer_publicados((carpeta / "pruebas_estadisticas.txt").read_text(encoding="utf-8"))
        sub = tabla[tabla["gestion"] == clave]
        for (prop, ref), v in pub.items():
            fila = sub[(sub["propuesto"] == prop) & (sub["referencia"] == ref)].iloc[0]
            assert round(fila["r_publicado"], 4) == v["r"]
            assert fila["p_publicado_texto"] == v["p_texto"]
    # ma_12 sobrepronostica en proporción al nivel: la mezcla lo supera con
    # intervalos agrupados que excluyen el cero; el motor y el ingenuo
    # comparten el choque del SKU y difieren sólo en ruido.
    sku = tabla[tabla["remuestreo"] == "sku"]
    mezcla_ma = sku[(sku["comparacion"] == "mezcla_prom - ma_12")]
    assert (mezcla_ma["excluye_cero"] == "sí").all()
    assert (mezcla_ma["ic95_superior"] < 0).all()
    motor_naive = sku[(sku["comparacion"] == "motor - naive_m1") & (sku["magnitud"] == "media")]
    assert (motor_naive["excluye_cero"] == "no").all()
    # La salida escrita por main().
    destino = tmp_path / "tablas"
    ie.main(["--normal", str(gestiones["normal"]), "--quiebre", str(gestiones["quiebre"]),
             "--propuestos", *PROPUESTOS, "--referencias", *REFERENCIAS,
             "--destino", str(destino), "--replicas", "300", "--sin-cruce"])
    csv = pd.read_csv(destino / "tabla_H4_incertidumbre_extension.csv")
    assert len(csv) == 48 and {"gestion", "excluye_cero", "r_publicado"} <= set(csv.columns)
    md = (destino / "tabla_H4_incertidumbre_extension.md").read_text(encoding="utf-8")
    assert md.startswith("Tabla H4.") and "P000" not in md and "P000" not in csv.to_string()


def test_misma_semilla_mismo_resultado(gestiones):
    comp = ie.comparaciones_de(PROPUESTOS, REFERENCIAS)
    a, _ = ie.construir(gestiones, comp, 200, 7)
    b, _ = ie.construir(gestiones, comp, 200, 7)
    pd.testing.assert_frame_equal(a, b)


def _alterar_txt(carpeta: Path, viejo: str, nuevo: str) -> None:
    ruta = carpeta / "pruebas_estadisticas.txt"
    texto = ruta.read_text(encoding="utf-8")
    assert viejo in texto
    ruta.write_text(texto.replace(viejo, nuevo, 1), encoding="utf-8")
    _sellar(carpeta)          # se vuelve a sellar: el control que falla es el del Wilcoxon


def test_se_detiene_si_el_r_publicado_no_se_reproduce(gestiones):
    carpeta = gestiones["normal"]
    linea = next(l for l in (carpeta / "pruebas_estadisticas.txt").read_text(encoding="utf-8").splitlines()
                 if "r =" in l)
    valor = linea.split("r = ")[1].split()[0]
    _alterar_txt(carpeta, f"r = {valor}", f"r = {float(valor) + 0.01:+.4f}")
    with pytest.raises(SystemExit, match="Wilcoxon publicado"):
        ie.construir_gestion(carpeta, ie.comparaciones_de(PROPUESTOS, REFERENCIAS), 50, 1)


def test_se_detiene_si_el_p_publicado_no_se_reproduce(gestiones):
    carpeta = gestiones["normal"]
    linea = next(l for l in (carpeta / "pruebas_estadisticas.txt").read_text(encoding="utf-8").splitlines()
                 if l.strip().startswith("p "))
    impreso = linea.split("= ")[1].split("   ->")[0]
    _alterar_txt(carpeta, f"= {impreso}   ->", "= 0.1234   ->")
    with pytest.raises(SystemExit, match="p publicado"):
        ie.construir_gestion(carpeta, ie.comparaciones_de(PROPUESTOS, REFERENCIAS), 50, 1)


def test_se_detiene_si_falta_una_comparacion(gestiones):
    with pytest.raises(SystemExit, match="no publica"):
        ie.construir_gestion(gestiones["normal"], [("motor", "ma_12_gf")], 50, 1)


def test_se_detiene_si_un_insumo_no_coincide_con_el_manifiesto(gestiones):
    ruta = gestiones["quiebre"] / "matriz_contraste.csv"
    ruta.write_text(ruta.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="hash del manifiesto"):
        ie.construir_gestion(gestiones["quiebre"], ie.comparaciones_de(PROPUESTOS, REFERENCIAS), 50, 1)


def test_r_conserva_signo_y_no_queda_en_cero():
    assert ie._r(-0.3691) == "−0,37"
    assert ie._r(0.0867) == "0,09"
    assert ie._r(-0.0028) == "−0,003"


def test_el_csv_conserva_la_precision_de_la_tabla(gestiones, tmp_path):
    comp = ie.comparaciones_de(PROPUESTOS, REFERENCIAS)
    tabla, _ = ie.construir(gestiones, comp, 100, 3)
    destino = tmp_path / "tablas"
    ie.main(["--normal", str(gestiones["normal"]), "--quiebre", str(gestiones["quiebre"]),
             "--propuestos", *PROPUESTOS, "--referencias", *REFERENCIAS,
             "--destino", str(destino), "--replicas", "100", "--semilla", "3", "--sin-cruce"])
    csv = pd.read_csv(destino / "tabla_H4_incertidumbre_extension.csv")
    for col in ("estimacion", "ic95_inferior", "ic95_superior"):
        np.testing.assert_allclose(csv[col].to_numpy(), tabla[col].to_numpy(), rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Combinación frente al motor validado (dos corridas)
# ---------------------------------------------------------------------------

def test_cruce_con_el_motor_validado(gestiones, tmp_path):
    # Misma semilla que la gestión normal: mismas series, venta, costo y modelos.
    base = _carpeta_sintetica(tmp_path / "base", 1)
    tabla, resumen = ie.construir_cruce(base, gestiones["normal"], ["mezcla_prom"], 200, 5)
    assert len(tabla) == 6                                 # 3 magnitudes × 2 remuestreos
    assert set(tabla["referencia"]) == {ie.VALIDADO}
    assert tabla["p_wilcoxon_publicado"].isna().all()
    assert resumen["modelos_comunes_identicos"] == ["ma_12", "mezcla_prom", "naive_m1"]
    # La estimación de D es la resta de las D publicadas de cada corrida.
    tv_e = pd.read_csv(gestiones["normal"] / "tabla_valorizada.csv").set_index("Modelo")
    tv_b = pd.read_csv(base / "tabla_valorizada.csv").set_index("Modelo")
    d = tabla[(tabla["magnitud"] == "d_portafolio") & (tabla["remuestreo"] == "sku")]["estimacion"].iloc[0]
    esperado = tv_e.loc["mezcla_prom", "D = WMAPE + |Bias| (%)"] - tv_b.loc["motor", "D = WMAPE + |Bias| (%)"]
    assert d == pytest.approx(esperado, abs=1e-9)
    # Con la base idéntica a la gestión, la diferencia por serie es la de la
    # mezcla frente al motor de la propia gestión.
    m = pd.read_csv(gestiones["normal"] / "matriz_contraste.csv").set_index("serie")
    media = tabla[(tabla["magnitud"] == "media") & (tabla["remuestreo"] == "sku")]["estimacion"].iloc[0]
    assert media == pytest.approx(float((m["mezcla_prom"] - m["motor"]).mean()))


def test_cruce_de_punta_a_punta(gestiones, tmp_path):
    base = _carpeta_sintetica(tmp_path / "base", 1)
    destino = tmp_path / "tablas"
    ie.main(["--normal", str(gestiones["normal"]), "--quiebre", str(gestiones["quiebre"]),
             "--propuestos", *PROPUESTOS, "--referencias", *REFERENCIAS,
             "--base", str(base), "--propuestos-cruce", "mezcla_prom",
             "--destino", str(destino), "--replicas", "100"])
    csv = pd.read_csv(destino / "tabla_H4_incertidumbre_extension.csv")
    assert len(csv) == 48 + 6
    assert (csv[csv["referencia"] == ie.VALIDADO]["gestion"] == "normal").all()
    md = (destino / "tabla_H4_incertidumbre_extension.md").read_text(encoding="utf-8")
    assert "frente al motor validado" in md and "P000" not in md


def test_cruce_se_detiene_si_las_corridas_no_comparten_series(gestiones, tmp_path):
    otra = _carpeta_sintetica(tmp_path / "otra", 2)        # otra venta y otros modelos
    with pytest.raises(SystemExit, match="no coinciden"):
        ie.construir_cruce(otra, gestiones["normal"], ["mezcla_prom"], 50, 1)
