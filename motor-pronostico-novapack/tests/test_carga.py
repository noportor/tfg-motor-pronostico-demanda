"""RF-1 — Criterio de aceptación: con un archivo cuyas columnas no coinciden,
el error indica qué falta y qué hay."""

from __future__ import annotations

import pandas as pd
import pytest

from src.carga import ErrorDeCarga, cargar


def _escribir(tmp_path, filas, nombre="ventas.csv"):
    ruta = tmp_path / nombre
    pd.DataFrame(filas).to_csv(ruta, index=False)
    return ruta


def test_error_enumera_lo_que_falta_y_lo_que_hay(cfg, tmp_path):
    ruta = _escribir(tmp_path, [
        {"fecha": "2020-01-01", "codigo": "A", "canal": "CANAL-1",
         "regional": "REGIONAL-A", "cantidad": 10, "categoria": "PLASTICOS", "precio": "2.5", "perdidas": "0"},
    ])

    with pytest.raises(ErrorDeCarga) as excepcion:
        cargar(cfg, ruta)

    mensaje = str(excepcion.value)
    assert "sku" in mensaje, "El error debe decir qué columna canónica falta"
    assert "codigo" in mensaje, "El error debe enumerar las columnas encontradas"
    assert "canal" in mensaje


def test_lee_csv_y_convierte_tipos(cfg, tmp_path):
    ruta = _escribir(tmp_path, [
        {"fecha": "2020-01-01", "sku": "A", "canal": "CANAL-1",
         "regional": "REGIONAL-A", "cantidad": "10.5", "categoria": "PLASTICOS", "precio": "2.5", "perdidas": "0"},
        {"fecha": "2020-02-01", "sku": "A", "canal": "CANAL-1",
         "regional": "REGIONAL-A", "cantidad": "20", "categoria": "PLASTICOS", "precio": "2.5", "perdidas": "0"},
    ])
    df, informe = cargar(cfg, ruta)

    assert len(df) == 2
    assert informe.filas_validas == 2
    assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
    assert df["cantidad"].tolist() == [10.5, 20.0]


def test_cuenta_los_valores_que_no_pudo_convertir(cfg, tmp_path):
    """RN-1: lo inconvertible se descarta y se CUENTA; nunca se rellena."""
    ruta = _escribir(tmp_path, [
        {"fecha": "2020-01-01", "sku": "A", "canal": "CANAL-1",
         "regional": "REGIONAL-A", "cantidad": "10", "categoria": "PLASTICOS", "precio": "2.5", "perdidas": "0"},
        {"fecha": "no-es-fecha", "sku": "A", "canal": "CANAL-1",
         "regional": "REGIONAL-A", "cantidad": "20", "categoria": "PLASTICOS", "precio": "2.5", "perdidas": "0"},
        {"fecha": "2020-03-01", "sku": "A", "canal": "CANAL-1",
         "regional": "REGIONAL-A", "cantidad": "diez", "categoria": "PLASTICOS", "precio": "2.5", "perdidas": "0"},
    ])
    df, informe = cargar(cfg, ruta)

    assert informe.filas_leidas == 3
    assert informe.filas_validas == 1
    assert informe.fechas_no_convertidas == 1
    assert informe.cantidades_no_convertidas == 1
    assert informe.filas_descartadas == 2


def test_falla_si_el_archivo_no_existe(cfg, tmp_path):
    with pytest.raises(ErrorDeCarga) as excepcion:
        cargar(cfg, tmp_path / "no_existe.csv")
    assert "extraer_snapshot" in str(excepcion.value)


def test_extension_desconocida(cfg, tmp_path):
    ruta = tmp_path / "datos.parquet"
    ruta.write_bytes(b"x")
    with pytest.raises(ErrorDeCarga) as excepcion:
        cargar(cfg, ruta)
    assert "parquet" in str(excepcion.value)
