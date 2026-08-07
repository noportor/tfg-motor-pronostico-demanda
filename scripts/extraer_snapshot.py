"""Extracción y congelado del histórico de ventas de NOVAPACK S.A.

Este script se ejecuta UNA sola vez, fuera del pipeline. Su producto es un
archivo CSV inmutable en ``datos/crudo/`` acompañado de su SHA-256. A partir de
ahí el experimento no vuelve a hablar con la base de datos: corre siempre contra
el mismo snapshot congelado.

Por qué congelar y no leer en vivo
----------------------------------
La base de origen se recarga a diario. Si el pipeline leyera en vivo, dos
ejecuciones del mismo código en días distintos darían números distintos y la
RN-5 (reproducibilidad) sería imposible de sostener ante el tribunal. El
snapshot fecha el corte y lo vuelve verificable: cualquiera que reciba el
archivo puede recalcular el hash y comprobar que es el mismo que produjo los
números del documento.

Qué se extrae
-------------
Registros de venta efectiva mensual por SKU, canal y regional, para las nueve
gestiones fiscales completas 2018–2026 (abril-2017 a marzo-2026).

Se extraen SOLO los meses con venta distinta de cero. Los meses en cero
intermedios los reconstruye ``src/series.py`` a partir de la primera y la última
venta de cada serie, que es exactamente lo que exige la RF-3: un mes anterior al
lanzamiento del producto no es demanda cero, es ausencia de producto, y por
tanto no puede venir precargado desde el origen.

Seudonimización
---------------
``errores_por_serie.csv`` se adjunta como anexo del documento académico, de modo
que los códigos de producto reales quedarían publicados. Por defecto los SKU se
reemplazan por identificadores estables (``SKU-0001``, ``SKU-0002``, …) y la
tabla de correspondencia se guarda aparte, en ``datos/crudo/`` — que está en
``.gitignore``. La seudonimización es determinista: el mismo código real recibe
siempre el mismo identificador, así que el snapshot sigue siendo reproducible.

Alcance de la seudonimización, para no prometer de más: los identificadores se
asignan siguiendo el orden alfabético de los códigos reales, de modo que evitan
publicar el maestro de productos pero **no son anonimato criptográfico**. Quien
ya tenga el catálogo de la empresa podría reconstruir la correspondencia. Para el
uso previsto —un anexo académico— es suficiente; si hiciera falta más, habría que
permutar con una semilla secreta.

Uso
---
    export TFG_DB_HOST=... TFG_DB_PORT=... TFG_DB_NAME=...
    export TFG_DB_USER=... TFG_DB_PASSWORD=...
    python scripts/extraer_snapshot.py [--sin-seudonimizar]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import cargar_config, gestion_de, mes_a_periodo  # noqa: E402

TABLA_ORIGEN = "hub_thales.thales_demanda_consolidada"

# La columna objetivo es `ventas_efectivas` y no `demanda_total`.
#
# Justificación (se declara en la memoria): `demanda_total` = ventas efectivas +
# venta perdida registrada. La venta perdida NO se registró antes de 2020 — es
# cero en todos los meses de 2017, 2018 y 2019 — de modo que `demanda_total`
# cambia de definición a mitad de la serie. Entrenar y evaluar sobre una variable
# con un quiebre estructural en el medio contamina la comparación entre modelos.
# `ventas_efectivas` es homogénea en las nueve gestiones y es además lo que el
# documento declara como fuente: «datos históricos de ventas».
COLUMNA_OBJETIVO = "ventas_efectivas"

CONSULTA = f"""
SELECT
    (t.year * 100 + t.month)      AS periodo,
    t.sku                         AS sku,
    t.fv                          AS canal,
    t.dc                          AS regional,
    t.{COLUMNA_OBJETIVO}          AS cantidad
FROM {TABLA_ORIGEN} AS t
WHERE (t.year * 100 + t.month) BETWEEN %(desde)s AND %(hasta)s
  AND t.{COLUMNA_OBJETIVO} IS NOT NULL
  AND t.{COLUMNA_OBJETIVO} <> 0
ORDER BY t.sku, t.dc, t.fv, t.year, t.month
"""

CONSULTA_CONTROL = f"""
SELECT
    COUNT(*)                                                        AS filas_rejilla,
    COUNT(*) FILTER (WHERE t.{COLUMNA_OBJETIVO} <> 0)               AS filas_con_venta,
    COUNT(*) FILTER (WHERE t.{COLUMNA_OBJETIVO} <  0)               AS filas_negativas,
    COUNT(*) FILTER (WHERE t.{COLUMNA_OBJETIVO} IS NULL)            AS filas_nulas,
    COUNT(DISTINCT t.sku)                                           AS skus,
    COUNT(DISTINCT t.dc)                                            AS regionales,
    COUNT(DISTINCT t.fv)                                            AS canales,
    COUNT(DISTINCT (t.sku || '|' || t.dc || '|' || t.fv))           AS combinaciones,
    SUM(t.ventas_perdidas) FILTER (WHERE t.year < 2020)             AS perdidas_antes_2020,
    SUM(t.ventas_perdidas) FILTER (WHERE t.year >= 2020)            AS perdidas_desde_2020
FROM {TABLA_ORIGEN} AS t
WHERE (t.year * 100 + t.month) BETWEEN %(desde)s AND %(hasta)s
"""


def _periodo_entero(texto: str) -> int:
    """'2017-04' -> 201704."""
    p = mes_a_periodo(texto)
    return p.year * 100 + p.month


def _fecha_iso(periodo_entero: int) -> str:
    """201704 -> '2017-04-01'."""
    return f"{periodo_entero // 100:04d}-{periodo_entero % 100:02d}-01"


def _conexion():
    """Abre la conexión leyendo credenciales del entorno. Nunca del código."""
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - entorno sin la dependencia
        raise SystemExit(
            "Falta psycopg2. Instalá las dependencias de extracción:\n"
            "    pip install -r requirements-extraccion.txt"
        ) from exc

    faltantes = [v for v in ("TFG_DB_HOST", "TFG_DB_NAME", "TFG_DB_USER", "TFG_DB_PASSWORD")
                 if not os.environ.get(v)]
    if faltantes:
        raise SystemExit(
            "Faltan variables de entorno con las credenciales: "
            + ", ".join(faltantes)
            + "\nNo se hardcodean en el repositorio a propósito."
        )

    return psycopg2.connect(
        host=os.environ["TFG_DB_HOST"],
        port=int(os.environ.get("TFG_DB_PORT", "5432")),
        dbname=os.environ["TFG_DB_NAME"],
        user=os.environ["TFG_DB_USER"],
        password=os.environ["TFG_DB_PASSWORD"],
        connect_timeout=30,
    )


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sin-seudonimizar", action="store_true",
        help="Escribe los códigos de SKU reales. Solo para uso interno; el "
             "archivo resultante no puede adjuntarse al documento académico.",
    )
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    desde = _periodo_entero(str(cfg.crudo["periodo"]["inicio"]))
    hasta = _periodo_entero(str(cfg.crudo["periodo"]["fin"]))

    destino = cfg.ruta_datos
    destino.parent.mkdir(parents=True, exist_ok=True)

    print(f"Origen  : {TABLA_ORIGEN}  (columna objetivo: {COLUMNA_OBJETIVO})")
    print(f"Período : {desde} .. {hasta}")
    print(f"Destino : {destino}")

    conexion = _conexion()
    try:
        with conexion.cursor() as cur:
            cur.execute(CONSULTA_CONTROL, {"desde": desde, "hasta": hasta})
            columnas_control = [c[0] for c in cur.description]
            control = dict(zip(columnas_control, cur.fetchone()))

        print("\nControles sobre el origen (antes de filtrar):")
        for clave, valor in control.items():
            print(f"  {clave:<24} {valor}")

        if control["filas_negativas"]:
            print(
                f"\n  AVISO: {control['filas_negativas']} filas con "
                f"{COLUMNA_OBJETIVO} negativa (devoluciones). Se extraen tal cual; "
                f"el tratamiento lo decide series.devoluciones en config.yaml."
            )

        # Cursor del lado del servidor: la consulta devuelve cientos de miles de
        # filas y no tiene sentido materializarlas todas en memoria.
        seudonimos: dict[str, str] = {}
        filas_escritas = 0
        negativas = 0
        suma = 0.0
        periodo_min: int | None = None
        periodo_max: int | None = None

        with conexion.cursor(name="tfg_snapshot") as cur:
            cur.itersize = 50_000
            cur.execute(CONSULTA, {"desde": desde, "hasta": hasta})

            with destino.open("w", encoding="utf-8", newline="") as fh:
                escritor = csv.writer(fh, delimiter=cfg.datos.separador,
                                      lineterminator="\n")
                escritor.writerow(["fecha", "sku", "canal", "regional", "cantidad"])

                for periodo, sku, canal, regional, cantidad in cur:
                    if not args.sin_seudonimizar:
                        if sku not in seudonimos:
                            seudonimos[sku] = f"SKU-{len(seudonimos) + 1:04d}"
                        sku_salida = seudonimos[sku]
                    else:
                        sku_salida = sku

                    escritor.writerow([
                        _fecha_iso(int(periodo)),
                        sku_salida,
                        canal,
                        regional,
                        # repr corto y estable: evita notación científica y
                        # diferencias de redondeo entre corridas.
                        f"{float(cantidad):.6f}".rstrip("0").rstrip("."),
                    ])
                    filas_escritas += 1
                    suma += float(cantidad)
                    if float(cantidad) < 0:
                        negativas += 1
                    periodo_min = periodo if periodo_min is None else min(periodo_min, periodo)
                    periodo_max = periodo if periodo_max is None else max(periodo_max, periodo)
    finally:
        conexion.close()

    if filas_escritas == 0:
        raise SystemExit(
            "La consulta no devolvió ninguna fila. No se escribe un archivo vacío: "
            "revisá el período y la tabla de origen."
        )

    # Tabla de correspondencia: se guarda aparte y nunca se versiona.
    if seudonimos:
        ruta_mapeo = destino.parent / "mapeo_sku.csv"
        with ruta_mapeo.open("w", encoding="utf-8", newline="") as fh:
            escritor = csv.writer(fh, lineterminator="\n")
            escritor.writerow(["sku_real", "sku_seudonimo"])
            for real, seudo in seudonimos.items():
                escritor.writerow([real, seudo])
        print(f"\nCorrespondencia de SKU -> {ruta_mapeo} ({len(seudonimos)} códigos)")

    digest = _sha256(destino)
    manifiesto = {
        "extraido_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origen": TABLA_ORIGEN,
        "columna_objetivo": COLUMNA_OBJETIVO,
        "base_de_datos": os.environ.get("TFG_DB_NAME"),
        "periodo_solicitado": {"desde": desde, "hasta": hasta},
        "periodo_observado": {"desde": periodo_min, "hasta": periodo_max},
        "gestiones": {
            "desde": gestion_de(mes_a_periodo(str(cfg.crudo["periodo"]["inicio"])),
                                cfg.periodo.mes_inicio_gestion),
            "hasta": gestion_de(mes_a_periodo(str(cfg.crudo["periodo"]["fin"])),
                                cfg.periodo.mes_inicio_gestion),
        },
        "filas_escritas": filas_escritas,
        "filas_negativas": negativas,
        "suma_cantidad": round(suma, 6),
        "skus_distintos": len(seudonimos) if seudonimos else None,
        "seudonimizado": not args.sin_seudonimizar,
        "controles_origen": {k: (float(v) if isinstance(v, (int, float)) and v is not None else v)
                             for k, v in control.items()},
        "archivo": str(destino.relative_to(RAIZ)),
        "sha256": digest,
    }
    ruta_manifiesto = destino.parent / "manifiesto_extraccion.json"
    ruta_manifiesto.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nFilas escritas : {filas_escritas:,}")
    print(f"SHA-256        : {digest}")
    print(f"Manifiesto     : {ruta_manifiesto}")
    print("\nSnapshot congelado. El pipeline no vuelve a consultar la base de datos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
