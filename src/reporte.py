"""RF-10 — Consolidación de salidas y manifiesto de trazabilidad.

Es la única capa del programa que escribe archivos (§9: sin efectos secundarios
ocultos; ninguna otra función toca el disco).

Cada archivo de esta carpeta alimenta un punto concreto del documento:

=============================  ================================================
Archivo                        Destino en la tesis
=============================  ================================================
``inspeccion_datos.txt``       Criterios de inclusión y exclusión; N de la muestra
``errores_por_serie.csv``      Insumo de las pruebas; se adjunta como anexo
``tabla8_resultados.csv``      Tabla 8 — MAE, MAPE, RMSE, Bias por modelo
``pruebas_estadisticas.txt``   Apartado «Contraste estadístico de la hipótesis»
``seleccion_motor.csv``        Evidencia de la selección por serie
``figura2_error.png``          Figura 2
``figura3_dispersion.png``     Figura 3
``manifiesto.json``            Trazabilidad y reproducibilidad
=============================  ================================================

El manifiesto (RN-6) registra el hash del archivo de datos, la versión del
código, la configuración usada y el hash de cada salida producida. Con él,
cualquier número del documento se puede rastrear hasta la corrida que lo generó.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _version_del_codigo(raiz: Path) -> dict:
    """Commit que generó los números. Si no hay repositorio, se dice."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(raiz), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        sucio = subprocess.run(
            ["git", "-C", str(raiz), "status", "--porcelain"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        return {
            "commit": commit,
            "arbol_limpio": not bool(sucio),
            "archivos_modificados": [l[3:] for l in sucio.splitlines()] if sucio else [],
        }
    except Exception:
        return {
            "commit": None,
            "arbol_limpio": None,
            "advertencia": (
                "No se pudo determinar el commit: el directorio no es un repositorio "
                "git o git no está disponible. La RN-6 exige poder rastrear cada "
                "número hasta el commit que lo generó."
            ),
        }


def _versiones_de_dependencias() -> dict:
    """Versiones REALMENTE cargadas, no las declaradas en requirements.txt."""
    versiones = {"python": sys.version.split()[0], "plataforma": platform.platform()}
    for modulo in ("pandas", "numpy", "scipy", "sklearn", "matplotlib",
                   "lightgbm", "scikit_posthocs", "yaml"):
        try:
            importado = __import__(modulo)
            versiones[modulo] = getattr(importado, "__version__", "desconocida")
        except ImportError:
            versiones[modulo] = None
    return versiones


class Reporte:
    """Acumula las salidas y escribe el manifiesto al cerrar."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.directorio = cfg.ruta_salidas
        self.directorio.mkdir(parents=True, exist_ok=True)
        self.inicio = datetime.now(timezone.utc)
        self.archivos: list[Path] = []
        self.notas: dict = {}

    # -- escritura ----------------------------------------------------------

    def texto(self, nombre: str, contenido: str | list[str]) -> Path:
        if isinstance(contenido, list):
            contenido = "\n".join(contenido)
        ruta = self.directorio / nombre
        ruta.write_text(contenido.rstrip() + "\n", encoding="utf-8")
        self.archivos.append(ruta)
        return ruta

    def tabla(self, nombre: str, df: pd.DataFrame, indice: bool = False) -> Path:
        ruta = self.directorio / nombre
        # Separador decimal punto y sin índice: es lo que consume LibreOffice y
        # Excel sin reinterpretar los números según la configuración regional.
        df.to_csv(ruta, index=indice, encoding="utf-8", lineterminator="\n")
        self.archivos.append(ruta)
        return ruta

    def figura(self, ruta: Path) -> Path:
        self.archivos.append(Path(ruta))
        return Path(ruta)

    def nota(self, clave: str, valor) -> None:
        self.notas[clave] = valor

    # -- cierre -------------------------------------------------------------

    def escribir_manifiesto(self, ruta_datos: Path) -> Path:
        fin = datetime.now(timezone.utc)
        manifiesto = {
            "generado_en": fin.isoformat(timespec="seconds"),
            "duracion_segundos": round((fin - self.inicio).total_seconds(), 1),
            "datos": {
                "archivo": str(ruta_datos),
                "sha256": _sha256(ruta_datos) if Path(ruta_datos).exists() else None,
                "bytes": Path(ruta_datos).stat().st_size if Path(ruta_datos).exists() else None,
            },
            "codigo": _version_del_codigo(self.cfg.raiz),
            "dependencias": _versiones_de_dependencias(),
            "configuracion": {
                "archivo": str(self.cfg.ruta_config),
                "sha256": self.cfg.hash_configuracion(),
                "contenido": self.cfg.crudo,
            },
            "resultados": _serializable(self.notas),
            "salidas": [
                {
                    "archivo": ruta.name,
                    "bytes": ruta.stat().st_size,
                    "sha256": _sha256(ruta),
                }
                for ruta in sorted(set(self.archivos), key=lambda r: r.name)
                if ruta.exists()
            ],
        }
        ruta = self.directorio / "manifiesto.json"
        ruta.write_text(
            json.dumps(manifiesto, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return ruta


def _serializable(objeto):
    """Convierte dataclasses, Series y DataFrames a estructuras JSON."""
    if is_dataclass(objeto) and not isinstance(objeto, type):
        return _serializable(asdict(objeto))
    if isinstance(objeto, dict):
        return {str(k): _serializable(v) for k, v in objeto.items()}
    if isinstance(objeto, (list, tuple)):
        return [_serializable(v) for v in objeto]
    if isinstance(objeto, pd.DataFrame):
        return objeto.to_dict(orient="records")
    if isinstance(objeto, pd.Series):
        return {str(k): _serializable(v) for k, v in objeto.items()}
    if isinstance(objeto, pd.Period):
        return str(objeto)
    return objeto


# ---------------------------------------------------------------------------
# Informe de las pruebas estadísticas
# ---------------------------------------------------------------------------

def informe_pruebas(
    resultados: dict,
    cfg: Config,
    metrica: str,
    acierto_motor: dict | None = None,
) -> list[str]:
    """Redacta ``pruebas_estadisticas.txt``.

    El informe se encabeza con el tamaño del efecto y no con el p-valor: con
    miles de series cualquier diferencia sale significativa, y leer solo el p
    llevaría a presentar como hallazgo una diferencia irrelevante en la práctica.
    """
    L: list[str] = []
    L += ["=" * 78,
          "CONTRASTE ESTADÍSTICO DE LA HIPÓTESIS",
          "=" * 78,
          "",
          f"Métrica de contraste     : {metrica.upper()} por serie, bloque de prueba",
          f"Unidad de análisis       : la serie SKU–canal–regional",
          f"Bloques completos        : {resultados['n_bloques']:,}",
          f"Nivel de significación   : {cfg.pruebas.alfa}",
          f"Hipótesis alternativa    : el modelo propuesto tiene MENOS error "
          f"(unilateral, '{cfg.pruebas.alternativa}')",
          ""]

    # -- 1. Wilcoxon ---------------------------------------------------------
    L += ["", "-" * 78,
          "1. WILCOXON DE RANGOS CON SIGNO (pareado, unilateral)",
          "-" * 78,
          "",
          "Referencia metodológica: Demšar (2006) recomienda Wilcoxon para comparar",
          "dos modelos sobre múltiples conjuntos, por no asumir normalidad ni",
          "homogeneidad de varianzas.",
          ""]

    for resultado in resultados["wilcoxon"]:
        magnitud = _magnitud_efecto(abs(resultado.r))
        L += [
            f"  {resultado.propuesto}  vs  {resultado.referencia}",
            f"    Tamaño del efecto  r = {resultado.r:+.4f}  ({magnitud})",
            f"    Rango-biserial       = {resultado.rango_biserial:+.4f}  "
            f"(Kerby: proporción neta de series a favor del propuesto)",
            f"    Series a favor       : {resultado.gana_propuesto:,} de "
            f"{resultado.n_pares:,} "
            f"({100 * resultado.gana_propuesto / max(resultado.n_pares, 1):.1f} %)",
            f"    Series en contra     : {resultado.gana_referencia:,}"
            f"   ·  empates exactos: {resultado.empates:,}",
            f"    Mediana de la diferencia de {metrica.upper()}: "
            f"{resultado.mediana_diferencia:+,.4f}",
            f"    Estadístico W        = {resultado.estadistico:,.1f}"
            f"   ·  Z = {resultado.z:+.4f}",
            f"    p                    = {_formato_p(resultado.p)}"
            f"   ->  {'SIGNIFICATIVO' if resultado.significativo else 'no significativo'}"
            f" al {cfg.pruebas.alfa}",
            f"    Shapiro-Wilk de las diferencias: W = "
            f"{resultado.normalidad.get('estadistico', float('nan')):.4f}, "
            f"p = {_formato_p(resultado.normalidad.get('p', float('nan')))}"
            f"  (n = {resultado.normalidad.get('n_usado', 0):,}"
            f"{'; submuestreado' if resultado.normalidad.get('submuestreado') else ''})",
            "",
        ]

    L += [
        "  Lectura del Shapiro-Wilk: se aplica a las DIFERENCIAS PAREADAS entre los",
        "  errores de dos modelos, no a las ventas, y aquí es un diagnóstico, no una",
        "  puerta de decisión. Con n de miles la prueba rechaza la normalidad casi",
        "  siempre; el uso de pruebas no paramétricas se justifica a priori por la",
        "  asimetría y los valores extremos propios de los errores de pronóstico.",
        "",
    ]

    # -- 2. Friedman + Nemenyi ----------------------------------------------
    friedman = resultados["friedman"]
    L += ["", "-" * 78,
          "2. FRIEDMAN OMNIBUS + POST HOC DE NEMENYI",
          "-" * 78,
          "",
          f"  Modelos comparados (k)  : {friedman.k}",
          f"  Bloques completos (N)   : {friedman.n_bloques:,}",
          f"  Chi-cuadrado de Friedman: {friedman.chi2:,.4f}",
          f"  p                       : {_formato_p(friedman.p)}"
          f"   ->  {'SE RECHAZA' if friedman.significativo else 'NO se rechaza'} "
          f"la igualdad de los modelos",
          f"  Kendall W               : {friedman.kendall_w:.4f}  "
          f"({_magnitud_kendall(friedman.kendall_w)})",
          "  Nota: Kendall W no es comparable entre análisis con distinto k.",
          "",
          "  Rangos medios (1 = mejor):",
          ]
    for modelo, rango in friedman.rangos_medios.items():
        L.append(f"    {modelo:<20} {rango:6.3f}")

    L += ["", f"  Diferencia crítica de Nemenyi (α = {cfg.pruebas.alfa}): "
              f"{friedman.diferencia_critica:.4f}",
          "  Dos modelos cuyos rangos medios difieren menos que la CD no son",
          "  distinguibles entre sí.", ""]

    if friedman.nemenyi is not None:
        L += ["  Matriz de p-valores de Nemenyi:", ""]
        L += ("    " + friedman.nemenyi.round(4).to_string()).split("\n")
        L += ["",
              f"  Contraste cruzado con scikit-posthocs: "
              f"{'coincide' if friedman.contraste_scikit_posthocs else 'no verificado'}",
              ""]
    else:
        L += ["  El omnibus no rechaza; el post hoc no se ejecuta (post hoc protegido).",
              ""]

    L += ["  No se aplica corrección de Holm sobre Nemenyi: el rango studentizado ya",
          "  controla la tasa de error por familia de las k(k−1)/2 comparaciones, y",
          "  corregir dos veces sería innecesariamente conservador.", ""]

    # -- 3. Reparto de victorias --------------------------------------------
    L += ["", "-" * 78,
          "3. PORCENTAJE DE SERIES EN QUE GANA CADA MODELO",
          "-" * 78,
          ""]
    victorias = resultados["victorias"].copy()
    victorias["victorias"] = victorias["victorias"].round(1)
    victorias["porcentaje"] = victorias["porcentaje"].round(2)
    L += ("  " + victorias.to_string(index=False)).split("\n")
    L += ["",
          "  Los empates se reparten proporcionalmente entre los modelos empatados;",
          "  adjudicárselos al primero por orden de columna sería un artefacto.", ""]

    # -- 4. Tasa de acierto del motor ---------------------------------------
    if acierto_motor:
        L += ["", "-" * 78,
              "4. TASA DE ACIERTO DEL MOTOR",
              "-" * 78,
              "",
              "  ¿Cuántas veces el modelo elegido mirando VALIDACIÓN resultó ser",
              "  efectivamente el mejor en PRUEBA?",
              "",
              f"  Series comparadas        : {acierto_motor['series_comparadas']:,}",
              f"  Aciertos                 : {acierto_motor['aciertos']:,}",
              f"  Tasa de acierto          : {100 * acierto_motor['tasa_acierto']:.2f} %",
              f"  Acierto esperado al azar : {100 * acierto_motor['azar_esperado']:.2f} %",
              f"  Exceso de MAE por no elegir el óptimo de prueba:",
              f"     media   {acierto_motor['exceso_mae_medio']:,.4f}",
              f"     mediana {acierto_motor['exceso_mae_mediano']:,.4f}",
              "",
              "  Interpretación: elegir el mínimo entre varios candidatos sobre una",
              "  ventana corta sobreestima siempre lo bien que le irá al ganador fuera",
              "  de la muestra (winner's curse). La brecha entre la tasa de acierto y",
              "  el 100 % es la magnitud de ese efecto, y es un resultado por sí sola.",
              "",
              "  Asimetría declarada: LightGBM fija su número de árboles por parada",
              "  temprana contra validación, que es la misma ventana en que el motor",
              "  decide; los demás candidatos llegan a validación completamente a",
              "  ciegas. El motor lo elegirá algo más a menudo de lo que merecería con",
              "  un protocolo simétrico. La ventaja desaparece en prueba, que es donde",
              "  se mide la tasa de acierto de arriba.",
              ""]

    # -- 5. Limitaciones declaradas -----------------------------------------
    L += ["", "-" * 78,
          "5. LIMITACIONES DEL CONTRASTE (declaradas, no descubiertas después)",
          "-" * 78,
          "",
          "  a) Un solo corte, no origen móvil. El desempeño reportado corresponde a",
          "     UNA gestión de prueba. Un año atípico movería los números; con orígenes",
          "     móviles el resultado sería más estable, a costa de un diseño que ya no",
          "     sería el declarado en la metodología del documento.",
          "",
          "  b) El contraste es unilateral en la dirección de la hipótesis. Está fijado",
          "     antes de ver resultados, y la contrapartida es que NO puede detectar",
          "     que un modelo propuesto sea peor: en ese caso el p-valor simplemente no",
          "     resulta significativo. El porcentaje de series a favor y en contra, que",
          "     aparece arriba, sí muestra la dirección real.",
          "",
          "  c) Las series del mismo SKU en distintas regionales y canales NO son",
          "     independientes entre sí. Las pruebas las tratan como bloques separados,",
          "     lo que hace los p-valores algo optimistas. Con n de miles el efecto",
          "     sobre la conclusión es menor que el del tamaño del efecto, que es lo",
          "     que se reporta primero.",
          "",
          "  d) Evaluación a un paso. Los resultados no se extrapolan a horizontes",
          "     mayores; ahí el promedio móvil recursivo y LightGBM se comportan de",
          "     forma distinta y haría falta un experimento aparte.",
          "",
          "  e) Con n de miles, cualquier diferencia sale significativa. La lectura",
          "     válida es el tamaño del efecto y la diferencia mediana en unidades, no",
          "     el p-valor.",
          ""]

    L += ["", "=" * 78,
          "Fin del contraste.",
          "=" * 78]
    return L


def _formato_p(p: float) -> str:
    if p != p:  # NaN
        return "n/d"
    if p < 1e-16:
        return "< 1e-16"
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.4f}"


def _magnitud_efecto(r: float) -> str:
    """Convenciones de Cohen para r, declaradas explícitamente."""
    if r != r:
        return "n/d"
    if r < 0.1:
        return "insignificante"
    if r < 0.3:
        return "pequeño"
    if r < 0.5:
        return "mediano"
    return "grande"


def _magnitud_kendall(w: float) -> str:
    if w != w:
        return "n/d"
    if w < 0.1:
        return "concordancia muy débil"
    if w < 0.3:
        return "concordancia débil"
    if w < 0.5:
        return "concordancia moderada"
    return "concordancia fuerte"
