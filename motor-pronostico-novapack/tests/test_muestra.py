"""Población objetivo, estados y sensibilidad: la muestra se decide a la vista.

Tres piezas de la decisión de muestra, cada una verificada a mano:
- el filtro de población excluye Y cuenta (jamás en silencio);
- la clasificación por estados pone nombre a cada serie fuera de la muestra;
- la tabla de sensibilidad reproduce EXACTO el N de la cascada en los umbrales
  base (si divergen, mide otra cosa que los criterios reales).
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import panel_de
from src.carga import ErrorDeCarga, aplicar_poblacion
from src.config import ErrorDeConfiguracion, cargar_config
from src.particion import (
    aplicar_criterios_inclusion, clasificar_estados, particionar,
    sensibilidad_umbrales, tabla_fuera_de_muestra,
)


# ---------------------------------------------------------------------------
# Población objetivo
# ---------------------------------------------------------------------------

def _crudo(categorias: list[str]) -> pd.DataFrame:
    filas = []
    for i, categoria in enumerate(categorias):
        filas.append({
            "fecha": pd.Timestamp("2020-01-01"),
            "sku": f"S{i}", "canal": "C1", "regional": "R1",
            "cantidad": 10.0, "categoria": categoria,
        })
    return pd.DataFrame(filas)


def test_poblacion_excluye_y_cuenta(cfg):
    cfg_pop = cargar_config(
        cfg.ruta_config,
        anulaciones=["poblacion.categorias=[PLASTICOS, CUADERNOS]"],
    )
    df = _crudo(["PLASTICOS", "CUADERNOS", "LIMPIEZA", "LIMPIEZA"])

    filtrado, informe = aplicar_poblacion(df, cfg_pop)

    assert len(filtrado) == 2
    assert set(filtrado["categoria"]) == {"PLASTICOS", "CUADERNOS"}
    assert informe.filas_fuera == 2
    assert informe.series_fuera == 2          # S2 y S3: series distintas
    assert informe.skus_fuera == 2
    assert informe.filas_por_categoria_excluida == {"LIMPIEZA": 2}


def test_poblacion_con_lista_vacia_es_passthrough_declarado(cfg):
    cfg_todas = cargar_config(
        cfg.ruta_config, anulaciones=["poblacion.categorias=[]"]
    )
    df = _crudo(["PLASTICOS", "LIMPIEZA"])
    filtrado, informe = aplicar_poblacion(df, cfg_todas)
    assert len(filtrado) == 2
    assert informe.filas_fuera == 0
    assert not informe.categorias_incluidas
    assert "PORTAFOLIO COMPLETO" in informe.lineas()[0]


def test_poblacion_con_categoria_inexistente_falla_con_la_lista(cfg):
    """Un error tipográfico no puede descartar media población en silencio."""
    cfg_typo = cargar_config(
        cfg.ruta_config, anulaciones=["poblacion.categorias=[PLASTIC0S]"]
    )
    df = _crudo(["PLASTICOS", "CUADERNOS"])
    with pytest.raises(ErrorDeCarga) as excepcion:
        aplicar_poblacion(df, cfg_typo)
    assert "PLASTIC0S" in str(excepcion.value)
    assert "PLASTICOS" in str(excepcion.value), (
        "El error debe enumerar las categorías realmente disponibles"
    )


def test_poblacion_sin_columna_categoria_falla_explicito(cfg):
    df = _crudo(["PLASTICOS"]).drop(columns="categoria")
    with pytest.raises(ErrorDeCarga) as excepcion:
        aplicar_poblacion(df, cfg)
    assert "columna_categoria" in str(excepcion.value)


def test_config_exige_mapeo_de_categoria_si_hay_poblacion(cfg, tmp_path):
    """Población declarada sin columna de categoría mapeada = configuración
    incoherente: el error tiene que saltar al cargar, no al filtrar."""
    import yaml

    alterado = yaml.safe_load(yaml.safe_dump(cfg.crudo, allow_unicode=True))
    del alterado["datos"]["columnas"]["categoria"]
    assert alterado["poblacion"]["categorias"], "La base debe declarar población"
    ruta = tmp_path / "config_sin_categoria.yaml"
    ruta.write_text(yaml.safe_dump(alterado, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ErrorDeConfiguracion) as excepcion:
        cargar_config(ruta)
    assert "categoria" in str(excepcion.value)


# ---------------------------------------------------------------------------
# Estados
# ---------------------------------------------------------------------------

def test_estados_a_mano(cfg):
    """Fronteras exactas: activa (<6m sin venta), dormida (6–11), descontinuada
    (>=12), nueva (<6m de vida), reciente (<12m)."""
    fin = pd.Period("2024-03", "M")
    n = 84  # 2017-04 .. 2024-03

    def con_ultima_venta(meses_atras: int) -> list[float]:
        valores = [5.0] * n
        for k in range(meses_atras):
            valores[n - 1 - k] = 0.0
        return valores

    largo = pd.concat([
        panel_de({"ACTIVA": con_ultima_venta(0)}),
        panel_de({"CASI": con_ultima_venta(5)}),        # 5m sin venta -> activa
        panel_de({"DORMIDA": con_ultima_venta(6)}),
        panel_de({"DESCONTINUADA": con_ultima_venta(12)}),
        panel_de({"NUEVA": [3.0] * 4}, inicio="2023-12"),      # vida 4m
        panel_de({"RECIENTE": [3.0] * 9}, inicio="2023-07"),   # vida 9m
    ], ignore_index=True)

    estados = clasificar_estados(largo, fin)
    por_nombre = {s.split("|")[0]: e for s, e in estados.items()}
    assert por_nombre == {
        "ACTIVA": "activa",
        "CASI": "activa",
        "DORMIDA": "dormida",
        "DESCONTINUADA": "descontinuada",
        "NUEVA": "nueva",
        "RECIENTE": "reciente",
    }


# ---------------------------------------------------------------------------
# Fuera de muestra y sensibilidad
# ---------------------------------------------------------------------------

def _panel_variado() -> pd.DataFrame:
    """Series con destinos distintos frente a los criterios base."""
    return pd.concat([
        # 84 meses sanos: incluida.
        panel_de({"SANA": [10.0 + (t % 4) for t in range(108)]}),
        # Nace tarde: historial corto en train.
        panel_de({"CORTA": [8.0] * 30}, inicio="2023-10"),
        # Espóradica crónica: demasiados ceros.
        panel_de({"RALA": [0.0, 0.0, 0.0, 6.0] * 27}),
        # Muere antes de validación: ausente en los bloques de evaluación.
        panel_de({"MUERTA": [9.0] * 60}),
    ], ignore_index=True)


def test_fuera_de_muestra_atribuye_motivos(cfg):
    corte = particionar(cfg)
    panel = _panel_variado()
    cohorte, _ = aplicar_criterios_inclusion(panel, cfg, corte)
    series_cohorte = set(cohorte["serie"].unique())

    tabla = tabla_fuera_de_muestra(panel, panel, series_cohorte, cfg, corte)
    por_nombre = tabla.assign(nombre=tabla["serie"].str.split("|").str[0]) \
                      .set_index("nombre")

    assert bool(por_nombre.loc["SANA", "incluida"])
    assert por_nombre.loc["SANA", "motivo_exclusion"] == "incluida"
    assert por_nombre.loc["CORTA", "motivo_exclusion"].startswith("historial<")
    assert por_nombre.loc["RALA", "motivo_exclusion"].startswith("ceros>")
    assert por_nombre.loc["MUERTA", "motivo_exclusion"] == "ausente_en_validacion"
    assert por_nombre.loc["MUERTA", "estado"] == "descontinuada"


def test_sensibilidad_base_reproduce_la_cascada(cfg):
    """La garantía de consistencia: la fila base de la grilla == N de la
    cascada real, sobre el mismo panel."""
    corte = particionar(cfg)
    panel = _panel_variado()
    _, informe = aplicar_criterios_inclusion(panel, cfg, corte)

    grilla = sensibilidad_umbrales(panel, corte)
    base = grilla.loc[
        (grilla["historial_minimo_meses"] == cfg.inclusion.historial_minimo_meses)
        & (grilla["proporcion_maxima_ceros"]
           == cfg.inclusion.proporcion_maxima_ceros)
        & (grilla["volumen_minimo_unidades"]
           == cfg.inclusion.volumen_minimo_unidades)
    ]
    assert len(base) == 1
    assert int(base["n_series"].iloc[0]) == informe.series_finales
