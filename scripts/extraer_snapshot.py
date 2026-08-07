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

Dónde vive el esquema de origen
-------------------------------
En ``config/extraccion.local.yaml``, que NO se versiona. Este archivo no
contiene ningún nombre de esquema, tabla ni columna de la empresa: el código de
la tesis se publica y el esquema interno está cubierto por el acuerdo de
confidencialidad igual que los datos. Ver ``config/extraccion.ejemplo.yaml``.

Qué se extrae
-------------
Registros de venta mensual por SKU, canal y regional, para el período declarado
en ``config/config.yaml``.

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
    cp config/extraccion.ejemplo.yaml config/extraccion.local.yaml
    # completar config/extraccion.local.yaml con el esquema real
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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import cargar_config, gestion_de, mes_a_periodo  # noqa: E402

RUTA_ORIGEN = RAIZ / "config" / "extraccion.local.yaml"
RUTA_PLANTILLA = RAIZ / "config" / "extraccion.ejemplo.yaml"

ROLES = ("anio", "mes", "sku", "canal", "regional", "cantidad")

# Identificadores SQL admitidos. La configuración la escribe el propio autor, no
# un tercero, pero un nombre de tabla se interpola en la consulta y no se puede
# parametrizar como un valor: validarlo cuesta una línea y cierra la puerta.
PATRON_IDENTIFICADOR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


class ErrorDeExtraccion(Exception):
    """La configuración de la extracción es inválida o falta."""


def cargar_origen(ruta: Path = RUTA_ORIGEN) -> dict:
    """Lee y valida ``config/extraccion.local.yaml``."""
    if not ruta.exists():
        raise ErrorDeExtraccion(
            f"Falta {ruta.relative_to(RAIZ)}.\n"
            f"Copiá la plantilla y completala con el esquema real:\n"
            f"    cp {RUTA_PLANTILLA.relative_to(RAIZ)} {ruta.relative_to(RAIZ)}\n"
            f"Ese archivo no se versiona: el esquema interno de la empresa está "
            f"cubierto por el acuerdo de confidencialidad igual que los datos."
        )

    contenido = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    origen = contenido.get("origen")
    if not isinstance(origen, dict):
        raise ErrorDeExtraccion(f"{ruta.name} no tiene una sección 'origen'.")

    tabla = str(origen.get("tabla", "")).strip()
    if not PATRON_IDENTIFICADOR.match(tabla):
        raise ErrorDeExtraccion(
            f"origen.tabla = '{tabla}' no es un identificador SQL válido "
            f"(esquema.tabla, solo letras, dígitos y guion bajo)."
        )

    columnas = origen.get("columnas") or {}
    faltantes = [rol for rol in ROLES if rol not in columnas]
    if faltantes:
        raise ErrorDeExtraccion(
            f"Faltan columnas en origen.columnas: {faltantes}. "
            f"Declaradas: {sorted(columnas)}"
        )
    for rol, nombre in columnas.items():
        if not PATRON_IDENTIFICADOR.match(str(nombre)):
            raise ErrorDeExtraccion(
                f"origen.columnas.{rol} = '{nombre}' no es un identificador válido."
            )

    control = origen.get("columna_de_control")
    if control is not None and not PATRON_IDENTIFICADOR.match(str(control)):
        raise ErrorDeExtraccion(
            f"origen.columna_de_control = '{control}' no es un identificador válido."
        )

    return {
        "tabla": tabla,
        "columnas": {rol: str(columnas[rol]) for rol in ROLES},
        "columna_de_control": str(control) if control else None,
        "anio_de_corte_del_control": int(origen.get("anio_de_corte_del_control", 2020)),
    }


def construir_consultas(origen: dict) -> tuple[str, str]:
    """Arma la consulta de extracción y la de controles previos."""
    c = origen["columnas"]
    tabla = origen["tabla"]
    control = origen["columna_de_control"]
    corte = origen["anio_de_corte_del_control"]

    consulta = f"""
SELECT
    (t.{c['anio']} * 100 + t.{c['mes']}) AS periodo,
    t.{c['sku']}                          AS sku,
    t.{c['canal']}                        AS canal,
    t.{c['regional']}                     AS regional,
    t.{c['cantidad']}                     AS cantidad
FROM {tabla} AS t
WHERE (t.{c['anio']} * 100 + t.{c['mes']}) BETWEEN %(desde)s AND %(hasta)s
  AND t.{c['cantidad']} IS NOT NULL
  AND t.{c['cantidad']} <> 0
ORDER BY t.{c['sku']}, t.{c['regional']}, t.{c['canal']}, t.{c['anio']}, t.{c['mes']}
"""

    lineas_control = [
        "COUNT(*)                                                   AS filas_totales",
        f"COUNT(*) FILTER (WHERE t.{c['cantidad']} <> 0)             AS filas_con_venta",
        f"COUNT(*) FILTER (WHERE t.{c['cantidad']} <  0)             AS filas_negativas",
        f"COUNT(*) FILTER (WHERE t.{c['cantidad']} IS NULL)          AS filas_nulas",
        f"COUNT(DISTINCT t.{c['sku']})                               AS skus",
        f"COUNT(DISTINCT t.{c['regional']})                          AS regionales",
        f"COUNT(DISTINCT t.{c['canal']})                             AS canales",
        f"COUNT(DISTINCT (t.{c['sku']} || '|' || t.{c['regional']} || '|' || t.{c['canal']}))"
        "                                                            AS combinaciones",
    ]
    if control:
        lineas_control += [
            f"SUM(t.{control}) FILTER (WHERE t.{c['anio']} <  {corte})"
            f"                                                        AS control_antes",
            f"SUM(t.{control}) FILTER (WHERE t.{c['anio']} >= {corte})"
            f"                                                        AS control_desde",
        ]

    consulta_control = (
        "SELECT\n    " + ",\n    ".join(lineas_control) + f"\nFROM {tabla} AS t\n"
        f"WHERE (t.{c['anio']} * 100 + t.{c['mes']}) BETWEEN %(desde)s AND %(hasta)s"
    )
    return consulta, consulta_control


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
    parser.add_argument("--origen", default=None, help="Ruta a extraccion.local.yaml")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    origen = cargar_origen(Path(args.origen) if args.origen else RUTA_ORIGEN)
    consulta, consulta_control = construir_consultas(origen)

    desde = _periodo_entero(str(cfg.crudo["periodo"]["inicio"]))
    hasta = _periodo_entero(str(cfg.crudo["periodo"]["fin"]))

    destino = cfg.ruta_datos
    destino.parent.mkdir(parents=True, exist_ok=True)

    print(f"Origen  : declarado en {RUTA_ORIGEN.name} (fuera del repositorio)")
    print(f"Período : {desde} .. {hasta}")
    print(f"Destino : {destino}")

    conexion = _conexion()
    try:
        with conexion.cursor() as cur:
            cur.execute(consulta_control, {"desde": desde, "hasta": hasta})
            columnas_control = [c[0] for c in cur.description]
            control = dict(zip(columnas_control, cur.fetchone()))

        print("\nControles sobre el origen (antes de filtrar):")
        for clave, valor in control.items():
            print(f"  {clave:<20} {valor}")

        if control.get("filas_negativas"):
            print(
                f"\n  AVISO: {control['filas_negativas']} filas con cantidad negativa "
                f"(devoluciones). Se extraen tal cual; el tratamiento lo decide "
                f"series.devoluciones en config.yaml."
            )

        antes, desde_corte = control.get("control_antes"), control.get("control_desde")
        if antes is not None and desde_corte is not None:
            corte = origen["anio_de_corte_del_control"]
            if not antes and desde_corte:
                print(
                    f"\n  CONTROL DE INTEGRIDAD: la columna de control es 0 antes de "
                    f"{corte} y {desde_corte:,.0f} desde {corte}. Confirma que esa "
                    f"medida CAMBIA de definición a mitad del período y que no puede "
                    f"usarse como variable objetivo."
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
            cur.execute(consulta, {"desde": desde, "hasta": hasta})

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
        # El nombre de la tabla y el de la base NO se registran aquí: este
        # archivo vive junto al snapshot, que puede compartirse. Lo que hace
        # verificable la extracción es el hash, no el origen.
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
        "controles_origen": {
            k: (float(v) if isinstance(v, (int, float)) and v is not None else v)
            for k, v in control.items()
        },
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
