"""Acceso de solo lectura a las corridas del experimento.

Este módulo NO importa streamlit a propósito: son funciones puras sobre el
sistema de archivos, y así se prueban con pytest dentro de la imagen del
pipeline (que no tiene streamlit instalado).

Convenciones de descubrimiento — el mecanismo de la aditividad:

- Todo directorio ``salidas*/`` que contenga un ``manifiesto.json`` es una
  corrida (la principal y las ablaciones por igual).
- Todo ``.csv`` de una corrida es una tabla visible; todo ``.png``, una figura.

La caché se invalida sola por *mtime*: si el pipeline vuelve a correr, el
tablero muestra lo nuevo sin reiniciar nada.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Caché por (ruta, mtime)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _json_cacheado(ruta: str, mtime: float) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def _csv_cacheado(ruta: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(ruta)


def cargar_json(ruta: Path) -> dict | None:
    if not ruta.exists():
        return None
    return _json_cacheado(str(ruta), ruta.stat().st_mtime)


def cargar_csv(ruta: Path) -> pd.DataFrame | None:
    if not ruta.exists():
        return None
    # .copy(): la caché guarda el original; quien lo reciba puede mutarlo sin
    # envenenar las lecturas siguientes.
    return _csv_cacheado(str(ruta), ruta.stat().st_mtime).copy()


# ---------------------------------------------------------------------------
# Corridas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Corrida:
    """Una ejecución del pipeline: un directorio ``salidas*/`` con manifiesto."""

    nombre: str
    ruta: Path

    # -- documentos ---------------------------------------------------------

    def manifiesto(self) -> dict | None:
        return cargar_json(self.ruta / "manifiesto.json")

    def flujo(self) -> dict | None:
        return cargar_json(self.ruta / "flujo.json")

    def inspeccion(self) -> dict | None:
        return cargar_json(self.ruta / "inspeccion.json")

    def tabla(self, nombre: str) -> pd.DataFrame | None:
        return cargar_csv(self.ruta / nombre)

    # -- inventario ---------------------------------------------------------

    def csvs(self) -> list[str]:
        return sorted(p.name for p in self.ruta.glob("*.csv"))

    def figuras(self) -> list[Path]:
        return sorted(self.ruta.glob("*.png"))

    def textos(self) -> list[str]:
        return sorted(p.name for p in self.ruta.glob("*.txt"))

    # -- metadatos ----------------------------------------------------------

    def anulaciones(self) -> list[str]:
        m = self.manifiesto() or {}
        return list(
            m.get("configuracion", {}).get("contenido", {}).get("_anulaciones", [])
        )

    def configuracion(self) -> dict:
        m = self.manifiesto() or {}
        return m.get("configuracion", {}).get("contenido", {}) or {}

    def benchmarks(self) -> dict[str, str]:
        """Modelos de referencia declarados en la configuración de la corrida."""
        modelos = self.configuracion().get("modelos", {})
        return {
            "promedio_movil": modelos.get("benchmark_promedio_movil", "ma_12"),
            "naive": modelos.get("benchmark_naive", "naive_m1"),
        }

    def etiqueta(self) -> str:
        """Nombre humano para selectores: directorio + anulaciones si las hay."""
        anulaciones = self.anulaciones()
        if anulaciones:
            return f"{self.nombre}  ({'; '.join(anulaciones)})"
        return self.nombre


def descubrir_corridas(raiz: Path | None = None) -> list[Corrida]:
    """Las corridas disponibles, con ``salidas/`` (la principal) primero."""
    raiz = raiz or RAIZ_PROYECTO
    corridas = [
        Corrida(nombre=directorio.name, ruta=directorio)
        for directorio in sorted(raiz.glob("salidas*"))
        if directorio.is_dir() and (directorio / "manifiesto.json").exists()
    ]
    return sorted(corridas, key=lambda c: (c.nombre != "salidas", c.nombre))


# ---------------------------------------------------------------------------
# Ayudas de presentación (siguen siendo puras)
# ---------------------------------------------------------------------------

def etapas_de(corrida: Corrida) -> list[dict]:
    flujo = corrida.flujo() or {}
    return list(flujo.get("etapas", []))


def artefactos_declarados(corrida: Corrida) -> set[str]:
    """Artefactos que alguna etapa ya reclama como suyos.

    Lo que NO esté aquí es lo que la página genérica de artefactos muestra sin
    que nadie haya escrito una vista: el mecanismo de 'aparece solo'.
    """
    declarados: set[str] = set()
    for etapa in etapas_de(corrida):
        declarados.update(etapa.get("artefactos", []))
    return declarados
