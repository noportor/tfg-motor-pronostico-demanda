"""Los lectores del tablero: puros, sin streamlit, contra directorios reales.

El tablero completo no se prueba aquí (streamlit no está en la imagen del
pipeline, a propósito); lo que SÍ se garantiza es que la capa de acceso a datos
—la única con lógica— funciona y que no arrastra la dependencia.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tablero import lectores


def _corrida_falsa(base: Path, nombre: str, anulaciones: list[str] | None = None):
    """Directorio salidas*/ mínimo pero con la forma real."""
    directorio = base / nombre
    directorio.mkdir()
    (directorio / "manifiesto.json").write_text(json.dumps({
        "generado_en": "2026-08-07T12:00:00+00:00",
        "datos": {"sha256": "d" * 64},
        "configuracion": {
            "sha256": "c" * 64,
            "contenido": {
                "_anulaciones": anulaciones or [],
                "modelos": {"benchmark_promedio_movil": "ma_12",
                            "benchmark_naive": "naive_m1"},
            },
        },
        "codigo": {"commit": "a" * 40, "arbol_limpio": True},
        "salidas": [],
    }), encoding="utf-8")
    (directorio / "flujo.json").write_text(json.dumps({
        "version": 1,
        "corrida": {"config_sha256": "c" * 64},
        "etapas": [
            {"id": "carga", "titulo": "Carga", "rf": "RF-1",
             "entrada": {}, "salida": {}, "decisiones": {}, "conteos": {},
             "artefactos": ["una_tabla.csv"], "notas": []},
        ],
    }), encoding="utf-8")
    pd.DataFrame({"modelo": ["ma_12"], "mae_mediana": [1.0]}).to_csv(
        directorio / "una_tabla.csv", index=False
    )
    (directorio / "figura.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return directorio


def test_descubre_corridas_con_la_principal_primero(tmp_path):
    _corrida_falsa(tmp_path, "salidas_zeta_ablacion")
    _corrida_falsa(tmp_path, "salidas")
    _corrida_falsa(tmp_path, "salidas_alfa")
    # Un directorio sin manifiesto NO es una corrida.
    (tmp_path / "salidas_rota").mkdir()

    corridas = lectores.descubrir_corridas(tmp_path)
    assert [c.nombre for c in corridas] == [
        "salidas", "salidas_alfa", "salidas_zeta_ablacion"
    ]


def test_la_corrida_expone_sus_documentos(tmp_path):
    _corrida_falsa(tmp_path, "salidas")
    corrida = lectores.descubrir_corridas(tmp_path)[0]

    assert corrida.manifiesto()["codigo"]["commit"] == "a" * 40
    assert corrida.flujo()["version"] == 1
    assert corrida.tabla("una_tabla.csv").iloc[0]["modelo"] == "ma_12"
    assert corrida.tabla("no_existe.csv") is None
    assert corrida.csvs() == ["una_tabla.csv"]
    assert [f.name for f in corrida.figuras()] == ["figura.png"]
    assert corrida.benchmarks() == {"promedio_movil": "ma_12", "naive": "naive_m1"}


def test_la_etiqueta_distingue_las_ablaciones(tmp_path):
    _corrida_falsa(tmp_path, "salidas")
    _corrida_falsa(tmp_path, "salidas_ab", anulaciones=["modelos.motor_regla=mae_mas_bias"])
    principal, ablacion = lectores.descubrir_corridas(tmp_path)

    assert principal.etiqueta() == "salidas"
    assert "mae_mas_bias" in ablacion.etiqueta()
    assert ablacion.anulaciones() == ["modelos.motor_regla=mae_mas_bias"]


def test_la_cache_se_invalida_por_mtime(tmp_path):
    import os

    directorio = _corrida_falsa(tmp_path, "salidas")
    corrida = lectores.descubrir_corridas(tmp_path)[0]
    assert len(corrida.tabla("una_tabla.csv")) == 1

    # El pipeline vuelve a correr y reescribe el CSV: el tablero tiene que ver
    # lo nuevo sin reiniciar nada.
    ruta = directorio / "una_tabla.csv"
    pd.DataFrame({"modelo": ["ma_12", "motor"], "mae_mediana": [1.0, 0.8]}).to_csv(
        ruta, index=False
    )
    os.utime(ruta, (ruta.stat().st_atime + 5, ruta.stat().st_mtime + 5))
    assert len(corrida.tabla("una_tabla.csv")) == 2


def test_mutar_lo_leido_no_envenena_la_cache(tmp_path):
    _corrida_falsa(tmp_path, "salidas")
    corrida = lectores.descubrir_corridas(tmp_path)[0]

    primera = corrida.tabla("una_tabla.csv")
    primera["mae_mediana"] = 999.0
    segunda = corrida.tabla("una_tabla.csv")
    assert segunda["mae_mediana"].iloc[0] == 1.0


def test_artefactos_declarados_permite_detectar_los_nuevos(tmp_path):
    directorio = _corrida_falsa(tmp_path, "salidas")
    pd.DataFrame({"x": [1]}).to_csv(directorio / "mejora_nueva.csv", index=False)
    corrida = lectores.descubrir_corridas(tmp_path)[0]

    declarados = lectores.artefactos_declarados(corrida)
    assert "una_tabla.csv" in declarados
    nuevos = set(corrida.csvs()) - declarados
    assert nuevos == {"mejora_nueva.csv"}, (
        "Un CSV nuevo debe poder identificarse como 'sin vista todavía': es el "
        "mecanismo de aditividad del tablero"
    )


def test_los_lectores_no_dependen_de_streamlit():
    """La capa de datos se prueba en la imagen del pipeline, que no trae
    streamlit. Si alguien le agrega el import, esta prueba lo frena antes de
    que la suite entera falle por un ImportError confuso."""
    import ast

    arbol = ast.parse(Path(lectores.__file__).read_text(encoding="utf-8"))
    importados = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            importados.update(alias.name.split(".")[0] for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            importados.add(nodo.module.split(".")[0])
    assert "streamlit" not in importados, (
        "lectores.py debe seguir siendo puro: es lo que permite probarlo aquí"
    )
    assert "altair" not in importados
