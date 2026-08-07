"""Carga y validación de la configuración del experimento.

Todas las decisiones metodológicas viven en ``config/config.yaml`` y se leen a
través de este módulo. Ninguna otra parte del programa lee el YAML directamente:
así hay un único punto donde validar y un único punto que auditar.

La configuración se valida al cargarse. Si falta una clave o un valor es
incoherente (por ejemplo, un corte de validación anterior al de entrenamiento)
el programa falla aquí, antes de tocar los datos.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Raíz del repositorio: este archivo vive en <raiz>/src/config.py
RAIZ = Path(__file__).resolve().parent.parent

RUTA_CONFIG_POR_DEFECTO = RAIZ / "config" / "config.yaml"


class ErrorDeConfiguracion(Exception):
    """La configuración es inválida. Se falla explícitamente, no se adivina."""


# ---------------------------------------------------------------------------
# Utilidades de período
# ---------------------------------------------------------------------------

def mes_a_periodo(texto: str) -> pd.Period:
    """'2024-03' -> Period('2024-03', 'M'). Falla explícitamente si no parsea."""
    try:
        return pd.Period(str(texto), freq="M")
    except Exception as exc:  # noqa: BLE001 - se re-lanza con contexto útil
        raise ErrorDeConfiguracion(
            f"No se pudo interpretar '{texto}' como un mes con formato AAAA-MM."
        ) from exc


def gestion_de(periodo: pd.Period, mes_inicio: int) -> int:
    """Gestión fiscal a la que pertenece un mes.

    NOVAPACK cierra en marzo y una gestión se nombra por el año en que CIERRA:
    la gestión 2018 va de abril-2017 a marzo-2018. Con ``mes_inicio = 4``:

        2017-04 .. 2017-12  -> 2018
        2018-01 .. 2018-03  -> 2018
        2018-04            -> 2019

    Se usa para cortar los bloques en frontera de ciclo comercial completo, de
    modo que ninguna campaña escolar quede partida entre dos bloques.
    """
    return periodo.year + 1 if periodo.month >= mes_inicio else periodo.year


# ---------------------------------------------------------------------------
# Estructura tipada
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigDatos:
    archivo: Path
    columnas: dict[str, str]
    separador: str
    codificacion: str


@dataclass(frozen=True)
class ConfigPeriodo:
    mes_inicio_gestion: int
    inicio: pd.Period
    fin: pd.Period


@dataclass(frozen=True)
class ConfigParticion:
    fin_entrenamiento: pd.Period
    fin_validacion: pd.Period


@dataclass(frozen=True)
class ConfigSeries:
    devoluciones: str
    rellenar_ceros_intermedios: bool


@dataclass(frozen=True)
class ConfigInclusion:
    historial_minimo_meses: int
    proporcion_maxima_ceros: float
    volumen_minimo_unidades: float


@dataclass(frozen=True)
class ConfigFeatures:
    rezagos: list[int]
    ventanas_moviles: list[int]
    incluir_mes: bool
    incluir_anio: bool
    incluir_tendencia: bool
    categoricas: list[str]


@dataclass(frozen=True)
class ConfigPruebas:
    alfa: float
    alternativa: str
    metrica_contraste: str
    shapiro_n_maximo: int
    semilla: int


@dataclass(frozen=True)
class Config:
    """Configuración completa, validada y ya tipada."""

    datos: ConfigDatos
    periodo: ConfigPeriodo
    particion: ConfigParticion
    series: ConfigSeries
    inclusion: ConfigInclusion
    modelos: dict[str, Any]
    features: ConfigFeatures
    pruebas: ConfigPruebas
    figuras: dict[str, Any]
    salidas: dict[str, Any]

    ruta_config: Path = field(default=RUTA_CONFIG_POR_DEFECTO)
    crudo: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- rutas derivadas ----------------------------------------------------

    @property
    def raiz(self) -> Path:
        return RAIZ

    @property
    def ruta_datos(self) -> Path:
        return RAIZ / self.datos.archivo

    @property
    def ruta_salidas(self) -> Path:
        return RAIZ / str(self.salidas["directorio"])

    # -- bloques ------------------------------------------------------------

    @property
    def inicio_validacion(self) -> pd.Period:
        return self.particion.fin_entrenamiento + 1

    @property
    def inicio_prueba(self) -> pd.Period:
        return self.particion.fin_validacion + 1

    def bloque_de(self, periodo: pd.Period) -> str:
        """Bloque ('entrenamiento' | 'validacion' | 'prueba') de un mes dado."""
        if periodo <= self.particion.fin_entrenamiento:
            return "entrenamiento"
        if periodo <= self.particion.fin_validacion:
            return "validacion"
        return "prueba"

    @property
    def modelos_activos(self) -> list[str]:
        return list(self.modelos["activos"])

    @property
    def modelos_candidatos(self) -> list[str]:
        """Brazos entre los que el motor elige: todos los activos menos él mismo."""
        return [m for m in self.modelos_activos if m != "motor"]

    def hash_configuracion(self) -> str:
        """SHA-256 del contenido efectivo de la configuración (RN-6)."""
        volcado = json.dumps(self.crudo, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(volcado.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def _exigir(dic: dict, clave: str, contexto: str) -> Any:
    if clave not in dic:
        raise ErrorDeConfiguracion(
            f"Falta la clave '{clave}' en la sección '{contexto}' de la configuración. "
            f"Claves presentes: {sorted(dic)}"
        )
    return dic[clave]


def aplicar_anulaciones(crudo: dict, anulaciones: list[str]) -> dict:
    """Aplica anulaciones puntuales del tipo ``modelos.motor_regla=mae_mas_bias``.

    Sirve para correr ablaciones —cambiar UNA decisión y volver a medir— sin
    duplicar el archivo de configuración. Duplicarlo sería peor: las dos copias
    se separan con el tiempo y deja de saberse en qué se diferenciaban las
    corridas.

    Las anulaciones quedan registradas en el manifiesto junto con el resto de la
    configuración efectiva, de modo que la trazabilidad de la RN-6 se mantiene:
    lo que se guarda es la configuración REALMENTE usada, no el archivo base.
    """
    resultado = yaml.safe_load(yaml.safe_dump(crudo))  # copia profunda
    for anulacion in anulaciones:
        if "=" not in anulacion:
            raise ErrorDeConfiguracion(
                f"Anulación mal formada: '{anulacion}'. Se espera clave.subclave=valor."
            )
        ruta_clave, valor_texto = anulacion.split("=", 1)
        claves = ruta_clave.strip().split(".")
        nodo = resultado
        for clave in claves[:-1]:
            if clave not in nodo or not isinstance(nodo[clave], dict):
                raise ErrorDeConfiguracion(
                    f"La anulación '{anulacion}' apunta a '{clave}', que no existe "
                    f"en la configuración. Claves disponibles: {sorted(nodo)}"
                )
            nodo = nodo[clave]
        if claves[-1] not in nodo:
            raise ErrorDeConfiguracion(
                f"La anulación '{anulacion}' crearía la clave nueva '{claves[-1]}'. "
                f"Solo se permite anular claves existentes, para que una errata no "
                f"pase inadvertida. Claves disponibles: {sorted(nodo)}"
            )
        nodo[claves[-1]] = yaml.safe_load(valor_texto)
    resultado["_anulaciones"] = list(anulaciones)
    return resultado


def cargar_config(
    ruta: str | Path | None = None, anulaciones: list[str] | None = None
) -> Config:
    """Lee, valida y devuelve la configuración del experimento."""
    ruta = Path(ruta) if ruta is not None else RUTA_CONFIG_POR_DEFECTO
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    if not ruta.exists():
        raise ErrorDeConfiguracion(f"No existe el archivo de configuración: {ruta}")

    with ruta.open("r", encoding="utf-8") as fh:
        crudo = yaml.safe_load(fh)

    if not isinstance(crudo, dict):
        raise ErrorDeConfiguracion(f"El archivo {ruta} no contiene un mapeo YAML válido.")

    if anulaciones:
        crudo = aplicar_anulaciones(crudo, anulaciones)

    for seccion in (
        "datos", "periodo", "particion", "series", "inclusion",
        "modelos", "features", "pruebas", "figuras", "salidas",
    ):
        _exigir(crudo, seccion, "raíz")

    d = crudo["datos"]
    datos = ConfigDatos(
        archivo=Path(_exigir(d, "archivo", "datos")),
        columnas=dict(_exigir(d, "columnas", "datos")),
        separador=d.get("separador", ","),
        codificacion=d.get("codificacion", "utf-8"),
    )
    for canonica in ("fecha", "sku", "canal", "regional", "cantidad"):
        if canonica not in datos.columnas:
            raise ErrorDeConfiguracion(
                f"datos.columnas debe mapear la columna canónica '{canonica}'. "
                f"Mapeadas: {sorted(datos.columnas)}"
            )

    p = crudo["periodo"]
    periodo = ConfigPeriodo(
        mes_inicio_gestion=int(_exigir(p, "mes_inicio_gestion", "periodo")),
        inicio=mes_a_periodo(_exigir(p, "inicio", "periodo")),
        fin=mes_a_periodo(_exigir(p, "fin", "periodo")),
    )
    if not 1 <= periodo.mes_inicio_gestion <= 12:
        raise ErrorDeConfiguracion(
            f"periodo.mes_inicio_gestion debe estar entre 1 y 12; "
            f"llegó {periodo.mes_inicio_gestion}."
        )
    if periodo.inicio >= periodo.fin:
        raise ErrorDeConfiguracion(
            f"periodo.inicio ({periodo.inicio}) debe ser anterior a "
            f"periodo.fin ({periodo.fin})."
        )

    pa = crudo["particion"]
    particion = ConfigParticion(
        fin_entrenamiento=mes_a_periodo(_exigir(pa, "fin_entrenamiento", "particion")),
        fin_validacion=mes_a_periodo(_exigir(pa, "fin_validacion", "particion")),
    )

    # RN-2: el orden de los bloques es la restricción más grave del experimento.
    # Se verifica aquí y además hay una prueba dedicada en tests/test_particion.py.
    if not (periodo.inicio
            <= particion.fin_entrenamiento
            < particion.fin_validacion
            < periodo.fin):
        raise ErrorDeConfiguracion(
            "Los cortes deben cumplir "
            "periodo.inicio <= fin_entrenamiento < fin_validacion < periodo.fin. "
            f"Llegó inicio={periodo.inicio}, fin_entrenamiento={particion.fin_entrenamiento}, "
            f"fin_validacion={particion.fin_validacion}, fin={periodo.fin}."
        )

    s = crudo["series"]
    devoluciones = str(_exigir(s, "devoluciones", "series")).lower()
    if devoluciones not in ("descartar", "netear"):
        raise ErrorDeConfiguracion(
            f"series.devoluciones debe ser 'descartar' o 'netear'; llegó '{devoluciones}'."
        )
    series = ConfigSeries(
        devoluciones=devoluciones,
        rellenar_ceros_intermedios=bool(s.get("rellenar_ceros_intermedios", True)),
    )

    i = crudo["inclusion"]
    inclusion = ConfigInclusion(
        historial_minimo_meses=int(_exigir(i, "historial_minimo_meses", "inclusion")),
        proporcion_maxima_ceros=float(_exigir(i, "proporcion_maxima_ceros", "inclusion")),
        volumen_minimo_unidades=float(_exigir(i, "volumen_minimo_unidades", "inclusion")),
    )
    if not 0.0 <= inclusion.proporcion_maxima_ceros <= 1.0:
        raise ErrorDeConfiguracion(
            "inclusion.proporcion_maxima_ceros es una proporción entre 0 y 1; "
            f"llegó {inclusion.proporcion_maxima_ceros}."
        )

    f = crudo["features"]
    features = ConfigFeatures(
        rezagos=[int(x) for x in _exigir(f, "rezagos", "features")],
        ventanas_moviles=[int(x) for x in _exigir(f, "ventanas_moviles", "features")],
        incluir_mes=bool(f.get("incluir_mes", True)),
        incluir_anio=bool(f.get("incluir_anio", True)),
        incluir_tendencia=bool(f.get("incluir_tendencia", True)),
        categoricas=[str(x) for x in f.get("categoricas", [])],
    )
    if min(features.rezagos) < 1:
        raise ErrorDeConfiguracion(
            "Todos los rezagos deben ser >= 1: un rezago 0 sería el propio valor "
            "que se quiere predecir (fuga temporal, RN-3)."
        )

    pr = crudo["pruebas"]
    pruebas = ConfigPruebas(
        alfa=float(pr.get("alfa", 0.05)),
        alternativa=str(pr.get("alternativa", "less")),
        metrica_contraste=str(pr.get("metrica_contraste", "mae")),
        shapiro_n_maximo=int(pr.get("shapiro_n_maximo", 5000)),
        semilla=int(pr.get("semilla", 0)),
    )
    if pruebas.alternativa not in ("less", "greater", "two-sided"):
        raise ErrorDeConfiguracion(
            f"pruebas.alternativa debe ser 'less', 'greater' o 'two-sided'; "
            f"llegó '{pruebas.alternativa}'."
        )

    modelos = dict(crudo["modelos"])
    activos = list(_exigir(modelos, "activos", "modelos"))
    if "motor" not in activos:
        raise ErrorDeConfiguracion("La lista modelos.activos debe incluir 'motor'.")
    for clave in ("benchmark_promedio_movil", "benchmark_naive"):
        referencia = _exigir(modelos, clave, "modelos")
        if referencia not in activos:
            raise ErrorDeConfiguracion(
                f"modelos.{clave} = '{referencia}' no está en modelos.activos: {activos}"
            )
    regla = modelos.get("motor_regla", "mae")
    if regla not in ("mae", "mae_mas_bias"):
        raise ErrorDeConfiguracion(
            f"modelos.motor_regla debe ser 'mae' o 'mae_mas_bias'; llegó '{regla}'."
        )

    return Config(
        datos=datos,
        periodo=periodo,
        particion=particion,
        series=series,
        inclusion=inclusion,
        modelos=modelos,
        features=features,
        pruebas=pruebas,
        figuras=dict(crudo["figuras"]),
        salidas=dict(crudo["salidas"]),
        ruta_config=ruta,
        crudo=crudo,
    )
