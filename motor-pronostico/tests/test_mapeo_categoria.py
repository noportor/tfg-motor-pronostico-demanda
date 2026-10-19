"""Seudonimización de las categorías al cargar (tabla local de correspondencia).

La captura conserva el nombre real de cada categoría; ``src/carga.py`` lo
reemplaza por su seudónimo con ``mapeo_categoria.csv``, junto al archivo de
ventas y fuera del repositorio. Se exige que con la tabla se aplique, que sin
la tabla la columna pase tal cual, que una categoría sin fila detenga la carga y
que la tabla conserve el orden alfabético de los nombres reales (así los códigos
de la columna categórica no cambian). Datos sintéticos con nombres ficticios
(RN-1).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.carga import (
    COLUMNAS_MAPEO_CATEGORIA,
    NOMBRE_MAPEO_CATEGORIA,
    ErrorDeCarga,
    cargar,
    conserva_el_orden,
)
from src.reporte import archivos_auxiliares

PROYECTO = Path(__file__).resolve().parent.parent


def _fila(categoria, mes=1):
    return {"fecha": f"2020-{mes:02d}-01", "sku": "SKU-0001", "canal": "CANAL-1",
            "regional": "REGIONAL-A", "cantidad": "10", "categoria": categoria,
            "precio": "2.5", "perdidas": "0"}


def _captura(directorio: Path, categorias, mapeo: dict[str, str] | None) -> Path:
    ruta = directorio / "ventas.csv"
    pd.DataFrame([_fila(c, i % 12 + 1) for i, c in enumerate(categorias)]).to_csv(
        ruta, index=False
    )
    if mapeo is not None:
        pd.DataFrame(
            {"categoria_real": list(mapeo), "categoria_seudonimo": list(mapeo.values())}
        ).to_csv(directorio / NOMBRE_MAPEO_CATEGORIA, index=False)
    return ruta


MAPEO = {"NOMBRE REAL 01": "Categoría A", "NOMBRE REAL 05": "Categoría E",
         "NOMBRE REAL 11": "Categoría K"}


def test_con_la_tabla_cada_categoria_pasa_a_su_seudonimo(cfg, tmp_path):
    ruta = _captura(tmp_path, ["NOMBRE REAL 01", "NOMBRE REAL 05", "NOMBRE REAL 01"], MAPEO)

    df, informe = cargar(cfg, ruta)

    assert df["categoria"].tolist() == ["Categoría A", "Categoría E", "Categoría A"]
    tabla = tmp_path / NOMBRE_MAPEO_CATEGORIA
    assert informe.mapeo_categoria == tabla
    assert informe.mapeo_categoria_sha256 == hashlib.sha256(tabla.read_bytes()).hexdigest()
    # El informe registra la tabla por nombre y SHA-256, nunca su contenido.
    texto = "\n".join(informe.lineas())
    assert NOMBRE_MAPEO_CATEGORIA in texto
    assert "NOMBRE REAL" not in texto


def test_sin_la_tabla_la_categoria_pasa_tal_cual(cfg, tmp_path):
    ruta = _captura(tmp_path, ["Categoría C", "Categoría A"], None)

    df, informe = cargar(cfg, ruta)

    assert df["categoria"].tolist() == ["Categoría C", "Categoría A"]
    assert informe.mapeo_categoria is None
    assert not any("seudonimizadas" in linea for linea in informe.lineas())


def test_una_categoria_sin_fila_detiene_la_carga(cfg, tmp_path):
    ruta = _captura(tmp_path, ["NOMBRE REAL 01", "NOMBRE REAL 07"], MAPEO)

    with pytest.raises(ErrorDeCarga) as excepcion:
        cargar(cfg, ruta)

    assert "NOMBRE REAL 07" in str(excepcion.value)
    assert "sin fila" in str(excepcion.value)


def test_las_categorias_nulas_siguen_nulas(cfg, tmp_path):
    ruta = _captura(tmp_path, ["NOMBRE REAL 01", None], MAPEO)

    df, _ = cargar(cfg, ruta)

    assert df["categoria"].iloc[0] == "Categoría A"
    assert pd.isna(df["categoria"].iloc[1])


@pytest.mark.parametrize("filas", [
    [("NOMBRE REAL 01", "Categoría A"), ("NOMBRE REAL 02", "Categoría A")],
    [("NOMBRE REAL 01", "Categoría A"), ("NOMBRE REAL 01", "Categoría B")],
    [("NOMBRE REAL 01", "")],
    # Desordenada: el código categórico de cada categoría cambiaría.
    [("NOMBRE REAL 01", "Categoría B"), ("NOMBRE REAL 02", "Categoría A")],
])
def test_una_tabla_ambigua_o_incompleta_detiene_la_carga(cfg, tmp_path, filas):
    """Dos categorías con un mismo seudónimo se fundirían en una sola; una
    tabla desordenada cambiaría los códigos de la columna categórica."""
    ruta = _captura(tmp_path, ["NOMBRE REAL 01"], None)
    pd.DataFrame(filas, columns=list(COLUMNAS_MAPEO_CATEGORIA)).to_csv(
        tmp_path / NOMBRE_MAPEO_CATEGORIA, index=False
    )

    with pytest.raises(ErrorDeCarga):
        cargar(cfg, ruta)


def test_una_tabla_desordenada_dice_por_que_detiene_la_carga(cfg, tmp_path):
    ruta = _captura(tmp_path, ["NOMBRE REAL 01"],
                    {"NOMBRE REAL 01": "Categoría B", "NOMBRE REAL 02": "Categoría A"})

    with pytest.raises(ErrorDeCarga) as excepcion:
        cargar(cfg, ruta)

    assert "orden alfabético" in str(excepcion.value)

def test_el_manifiesto_sella_la_tabla_como_leida(cfg, tmp_path):
    ruta = _captura(tmp_path, ["NOMBRE REAL 01"], MAPEO)
    (tmp_path / "costos.csv").write_text("sku,costo\nX,1\n", encoding="utf-8")

    sellos = {a["archivo"]: a for a in archivos_auxiliares(cfg, ruta)}

    tabla = tmp_path / NOMBRE_MAPEO_CATEGORIA
    assert sellos[NOMBRE_MAPEO_CATEGORIA]["leido_por_la_corrida"] is True
    assert sellos[NOMBRE_MAPEO_CATEGORIA]["sha256"] == (
        hashlib.sha256(tabla.read_bytes()).hexdigest()
    )


def test_la_plantilla_versionada_cubre_los_doce_seudonimos_sin_nombres_reales():
    plantilla = pd.read_csv(
        PROYECTO / "config" / "mapeo_categoria.ejemplo.csv", dtype=str
    )
    assert list(plantilla.columns) == list(COLUMNAS_MAPEO_CATEGORIA)
    assert plantilla["categoria_seudonimo"].tolist() == [
        f"Categoría {letra}" for letra in "ABCDEFGHIJKL"
    ]
    assert plantilla["categoria_real"].str.fullmatch(r"NOMBRE REAL \d+").all()


def _plantilla() -> dict[str, str]:
    tabla = pd.read_csv(PROYECTO / "config" / "mapeo_categoria.ejemplo.csv", dtype=str)
    return dict(zip(tabla["categoria_real"], tabla["categoria_seudonimo"]))


def test_la_plantilla_conserva_el_orden_alfabetico_de_los_nombres_reales():
    """Ordenar por nombre real y por seudónimo da la misma permutación."""
    mapeo = _plantilla()
    assert sorted(mapeo.values()) == list(mapeo.values())
    assert sorted(mapeo, key=mapeo.get) == sorted(mapeo)
    assert conserva_el_orden(mapeo)


@pytest.mark.parametrize("cuantas", [12, 5])
def test_los_codigos_categoricos_no_cambian_con_el_seudonimo(cuantas):
    """Mismos códigos ``category`` con el nombre real y con el seudónimo, en
    la población completa (12) y en un subconjunto como la muestra (5)."""
    mapeo = _plantilla()
    reales = pd.Series(sorted(mapeo)[::-1][:cuantas] * 3)
    seudonimos = reales.map(mapeo)

    assert reales.astype("category").cat.codes.tolist() == (
        seudonimos.astype("category").cat.codes.tolist()
    )
