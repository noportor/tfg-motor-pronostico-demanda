"""Figura del microservicio propuesto para la operación mensual del motor.

Dibuja lo que el apartado 4.2.5 del documento describe en prosa, en dos
franjas. La franja superior es el contexto de la operación: el sistema de
gestión de la empresa, la captura mensual sellada con huella, el microservicio
con su base de pronósticos y métricas, la plataforma de planificación
colaborativa con sus etapas (base del motor, ajuste de Comercial y plan
consensuado) y la reunión mensual, más la flecha punteada de vuelta que lleva
el plan consensuado y la venta real a la corrida siguiente, donde se miden el
error por etapa y su valor agregado (el mismo trazo del lazo de la Figura 5).
La franja inferior abre el microservicio en un marco de puntos con sus
componentes (API y autenticación, permisos por rol, orquestador, núcleo de
los cinco módulos de la Figura 7, insumos declarados, almacén de corridas,
base de pronósticos, selección y métricas, monitor) y dos bandas laterales:
la de observabilidad, donde están el recolector de métricas y tablero y las
alertas, y la de integración continua y registro de modelos.

Convenciones de trazo: la flecha de vuelta usa el trazo discontinuo (4, 2)
del lazo de la Figura 5; el marco del microservicio y las dos líneas de
ampliación usan un trazo de puntos finos, para que la nota de la figura pueda
nombrar el lazo sin ambigüedad. Rellenos en escala de grises: blanco para lo
propuesto, gris medio para el núcleo construido en el estudio y gris claro
para lo que queda fuera del microservicio.

Tipografía: la figura mide unas 7 × 4,5 pulgadas y se inserta a 6,10
pulgadas de ancho, como la Figura 7 (escala 0,86). Los cuerpos de letra van de
6,4 a 7,0 puntos, que impresos quedan entre 5,5 y 6,0 puntos.

Es un diagrama de diseño: no lee datos de ninguna corrida ni nombra
regionales, productos ni sistemas de la empresa. Uso dentro del repositorio:

    docker compose run --rm tfg python scripts/figura_microservicio.py

o fuera de él, indicando el destino:

    python figura_microservicio.py ruta/figura08_microservicio.png
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

if (RAIZ / "src" / "figuras.py").exists():
    sys.path.insert(0, str(RAIZ))
    from src.figuras import _estilo, _guardar  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402
else:  # Copia fiel de los dos auxiliares de src/figuras.py, para correr suelto.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _estilo(dpi: int) -> None:
        plt.rcParams.update({
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "0.85",
            "grid.linewidth": 0.5,
            "axes.edgecolor": "0.2",
            "axes.labelcolor": "0.1",
            "text.color": "0.1",
            "xtick.color": "0.2",
            "ytick.color": "0.2",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        })

    def _guardar(figura, destino: Path) -> Path:
        destino.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(destino)
        plt.close(figura)
        return destino

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

# Rellenos en escala de grises. Separados en al menos 0,10 de luminancia para
# que se distingan impresos en blanco y negro.
EXTERNO = "0.90"      # sistema, plataforma o instancia fuera del microservicio
NUCLEO = "0.76"       # núcleo ya construido en el estudio (módulos de la Figura 7)
PROPUESTO = "white"   # componente propuesto del microservicio
GRIS = "0.40"
DISCONTINUO = (0, (4, 2))     # lazo de retroalimentación (trazo de la Figura 5)
PUNTEADO = (0, (1.2, 1.4))    # límite del microservicio y su ampliación

# Cuerpos de letra (puntos, a 7 pulgadas de ancho).
T_CAJA = 6.8
T_MICRO = 7.0
T_PLAT = 6.6
T_ETAPA = 6.4
T_RETRO = 6.6
T_MARCO = 6.6
T_BANDA = 6.4
T_BANDA_TIT = 6.6
T_LEYENDA = 6.8

# Franja superior: contexto de la operación mensual.
SISTEMA = "Sistema de\ngestión de\nla empresa"
CAPTURA = "Captura\nmensual sellada\n(huella SHA-256)"
MICRO = "Microservicio\nde pronóstico"
BASE_CTX = "Base de\npronósticos\ny métricas"
PLATAFORMA = "Plataforma de\nplanificación colaborativa"
ETAPAS = ["Base del motor", "Ajuste de Comercial", "Plan consensuado"]
REUNION = "Reunión\nmensual de\nplanificación"
RETRO = ("Plan consensuado y venta real: error por etapa y valor agregado "
         "(FVA), medidos en la corrida siguiente")

# Franja inferior: interior del microservicio (nombres de la Tabla O1).
API = "API y\nautenticación"
PERMISOS = "Permisos\npor rol"
ORQUESTADOR = "Orquestador\nde la corrida\nmensual"
NUCLEO_TXT = "Núcleo:\nmódulos 1 a 5\n(Figura 7)"
INSUMOS = ("Insumos declarados: registro de eventos,\n"
           "maestro de equivalencias y política por estado")
ALMACEN = "Almacén\ninmutable de\ncorridas"
BASE = ("Base de pronósticos, selección\ny métricas (esquema propio\n"
        "en la base de la empresa)")
MONITOR = "Monitor de\ndesempeño y\nseñal de régimen"
MARCO = "Interior del microservicio propuesto, en un servidor interno"

OBSERVABILIDAD = "Observabilidad"
# (texto, relleno): el tablero de la plataforma es parte de la plataforma.
OBS_ITEMS = [
    ("Registros\nestructurados\npor corrida", PROPUESTO),
    ("Recolector de\nmétricas y tablero,\np. ej. Prometheus\ny Grafana",
     PROPUESTO),
    ("Alertas con\nacción", PROPUESTO),
    ("Visor analítico\ndel planificador", PROPUESTO),
    ("Tablero de la\nplataforma", EXTERNO),
]
INTEGRACION = "Integración\ncontinua y registro\nde modelos"
INT_ITEMS = [
    ("Pruebas\nautomatizadas\nen cada cambio", PROPUESTO),
    ("Imagen etiquetada\ncon la versión del\ncódigo y base\n"
     "fijada por su huella", PROPUESTO),
    ("Registro de\nmodelos entrenados\ny parámetros,\ncon huella", PROPUESTO),
    ("Configuración\nversionada", PROPUESTO),
]


def _caja(eje, x, y, ancho, alto, texto, relleno=PROPUESTO, tam=T_CAJA,
          borde="black", grosor=1.0):
    eje.add_patch(Rectangle((x, y), ancho, alto, facecolor=relleno,
                            edgecolor=borde, linewidth=grosor))
    if texto:
        eje.text(x + ancho / 2, y + alto / 2, texto, ha="center",
                 va="center", fontsize=tam, linespacing=1.15)


def _flecha(eje, desde, hasta, estilo="-|>", color="black", grosor=0.9,
            escala=8, linea="solid"):
    eje.add_patch(FancyArrowPatch(desde, hasta, arrowstyle=estilo,
                                  mutation_scale=escala, color=color,
                                  linewidth=grosor, linestyle=linea,
                                  shrinkA=0, shrinkB=0))


def _banda(eje, x, y, ancho, alto, titulo, items, tam=T_BANDA):
    """Banda lateral: marco gris con título y una pila de cajas."""
    eje.add_patch(Rectangle((x, y), ancho, alto, facecolor="none",
                            edgecolor=GRIS, linewidth=0.9))
    lineas_titulo = titulo.count("\n") + 1
    alto_titulo = 1.3 + 1.75 * lineas_titulo
    eje.text(x + ancho / 2, y + alto - 0.7, titulo, ha="center", va="top",
             fontsize=T_BANDA_TIT, fontweight="bold", linespacing=1.1)
    margen, sep = 0.8, 0.8
    disponible = alto - alto_titulo - margen - sep * (len(items) - 1)
    pesos = [texto.count("\n") + 1.5 for texto, _ in items]
    unidad = disponible / sum(pesos)
    tope = y + alto - alto_titulo
    for (texto, relleno), peso in zip(items, pesos):
        alto_i = peso * unidad
        _caja(eje, x + margen, tope - alto_i, ancho - 2 * margen, alto_i,
              texto, relleno=relleno, tam=tam, grosor=0.8)
        tope -= alto_i + sep


def figura_microservicio(destino: Path, dpi: int = 300) -> Path:
    _estilo(dpi)
    figura, eje = plt.subplots(figsize=(7.0, 4.45))
    eje.set_xlim(0, 100)
    eje.set_ylim(0, 68)
    eje.axis("off")

    # ---------------- Franja superior: contexto ----------------
    y0, alto = 45.5, 15.5
    ym = y0 + alto / 2
    cajas = {
        "sistema": (0.5, 12.0),
        "captura": (15.5, 13.5),
        "micro": (32.0, 14.5),
        "base": (49.5, 12.5),
        "plataforma": (65.0, 21.0),
        "reunion": (89.0, 10.5),
    }
    x, a = cajas["sistema"]
    _caja(eje, x, y0, a, alto, SISTEMA, relleno=EXTERNO)
    x, a = cajas["captura"]
    _caja(eje, x, y0, a, alto, CAPTURA)
    x, a = cajas["micro"]
    _caja(eje, x, y0, a, alto, MICRO, tam=T_MICRO, grosor=1.4)
    x, a = cajas["base"]
    _caja(eje, x, y0, a, alto, BASE_CTX)

    # Plataforma con sus etapas apiladas.
    px, pa = cajas["plataforma"]
    eje.add_patch(Rectangle((px, y0), pa, alto, facecolor=EXTERNO,
                            edgecolor="black", linewidth=1.0))
    eje.text(px + pa / 2, y0 + alto - 0.6, PLATAFORMA, ha="center", va="top",
             fontsize=T_PLAT, linespacing=1.1)
    ealto, esep, ebase = 2.75, 1.15, 0.8
    ey = y0 + ebase
    for texto in reversed(ETAPAS):
        _caja(eje, px + 1.5, ey, pa - 3.0, ealto, texto, tam=T_ETAPA,
              grosor=0.8)
        ey += ealto + esep
    # Flechas entre etapas (de arriba hacia abajo: base, Comercial, consenso).
    xc = px + pa / 2
    tope_etapas = y0 + ebase + 3 * ealto + 2 * esep
    for i in range(2):
        yt = tope_etapas - ealto - i * (ealto + esep)
        _flecha(eje, (xc, yt), (xc, yt - esep), escala=5, grosor=0.7)

    x, a = cajas["reunion"]
    _caja(eje, x, y0, a, alto, REUNION, relleno=EXTERNO)

    # Flujo de ida, de izquierda a derecha.
    orden = ["sistema", "captura", "micro", "base", "plataforma", "reunion"]
    for izq, der in zip(orden[:-1], orden[1:]):
        xi, ai = cajas[izq]
        xd, _ = cajas[der]
        estilo = "<|-|>" if izq == "plataforma" else "-|>"
        _flecha(eje, (xi + ai, ym), (xd, ym), estilo=estilo, escala=9,
                grosor=1.0)

    # Lazo de retroalimentación: del plan consensuado al microservicio.
    xr = px + pa / 2
    xm = cajas["micro"][0] + cajas["micro"][1] / 2
    yl = 63.6
    eje.plot([xr, xr, xm], [y0 + alto, yl, yl], color=GRIS, linewidth=1.1,
             linestyle=DISCONTINUO)
    _flecha(eje, (xm, yl), (xm, y0 + alto), color=GRIS, grosor=1.1,
            escala=10, linea=DISCONTINUO)
    eje.text((xr + xm) / 2, yl + 0.7, RETRO, ha="center", va="bottom",
             fontsize=T_RETRO, color="0.25")

    # ---------------- Franja inferior: interior ----------------
    mx0, my0, mancho, malto = 19.0, 5.4, 62.0, 34.4
    eje.add_patch(Rectangle((mx0, my0), mancho, malto, facecolor="none",
                            edgecolor=GRIS, linewidth=1.1, linestyle=PUNTEADO))
    eje.text(mx0 + mancho / 2, my0 - 0.8, MARCO, ha="center", va="top",
             fontsize=T_MARCO, color="0.25")

    # Ampliación: el microservicio y su base, abiertos en el marco.
    xa0 = cajas["micro"][0]
    xa1 = cajas["base"][0] + cajas["base"][1]
    for xs, xi in ((xa0, mx0), (xa1, mx0 + mancho)):
        eje.plot([xs, xi], [y0, my0 + malto], color=GRIS, linewidth=0.9,
                 linestyle=PUNTEADO)

    # Filas del interior.
    fila_sup = (31.4, 6.2)
    fila_med = (18.2, 9.2)
    fila_inf = (7.4, 7.2)
    col = {
        "api": (21.0, 11.5),
        "orq": (35.5, 11.5),
        "nucleo": (50.0, 13.0),
        "almacen": (66.0, 13.0),
    }
    # Fila media: API, orquestador, núcleo y almacén de corridas.
    ys, al = fila_med
    _caja(eje, col["api"][0], ys, col["api"][1], al, API)
    _caja(eje, col["orq"][0], ys, col["orq"][1], al, ORQUESTADOR)
    _caja(eje, col["nucleo"][0], ys, col["nucleo"][1], al, NUCLEO_TXT,
          relleno=NUCLEO, grosor=1.3)
    _caja(eje, col["almacen"][0], ys, col["almacen"][1], al, ALMACEN)
    yc = ys + al / 2
    for izq, der in (("api", "orq"), ("orq", "nucleo"), ("nucleo", "almacen")):
        xi, ai = col[izq]
        _flecha(eje, (xi + ai, yc), (col[der][0], yc), escala=8)

    # Fila superior: permisos sobre la API e insumos sobre el núcleo.
    ys_s, al_s = fila_sup
    _caja(eje, col["api"][0], ys_s, col["api"][1], al_s, PERMISOS)
    eje.plot([col["api"][0] + col["api"][1] / 2] * 2, [ys + al, ys_s],
             color="black", linewidth=0.9)
    nx, na = col["nucleo"]
    ix0, ix1 = 38.5, 79.0
    _caja(eje, ix0, ys_s, ix1 - ix0, al_s, INSUMOS)
    _flecha(eje, (nx + na / 2, ys_s), (nx + na / 2, ys + al), escala=8)

    # Fila inferior: base de pronósticos, selección y métricas, y monitor.
    ys_i, al_i = fila_inf
    bx0, bx1 = col["orq"][0], nx + na
    _caja(eje, bx0, ys_i, bx1 - bx0, al_i, BASE)
    _caja(eje, col["almacen"][0], ys_i, col["almacen"][1], al_i, MONITOR,
          tam=6.6)
    _flecha(eje, (nx + na / 2, ys), (nx + na / 2, ys_i + al_i), escala=8)
    _flecha(eje, (bx1, ys_i + al_i / 2), (col["almacen"][0], ys_i + al_i / 2),
            escala=8)
    # La API consulta la base y registra en ella los ajustes (doble sentido).
    xapi = col["api"][0] + col["api"][1] / 2
    ybase = ys_i + al_i / 2
    eje.plot([xapi, xapi, bx0 - 1.2], [ys, ybase, ybase], color="black",
             linewidth=0.9)
    _flecha(eje, (bx0 - 1.3, ybase), (bx0, ybase), escala=8)
    _flecha(eje, (xapi, ys - 1.3), (xapi, ys), escala=8)

    # Bandas laterales.
    _banda(eje, 0.5, my0, 16.0, malto, OBSERVABILIDAD, OBS_ITEMS)
    _banda(eje, 83.5, my0, 16.0, malto, INTEGRACION, INT_ITEMS)
    yb = my0 + malto / 2
    _flecha(eje, (16.5, yb), (mx0, yb), estilo="<|-|>", color=GRIS,
            escala=7, grosor=0.9)
    _flecha(eje, (mx0 + mancho, yb), (83.5, yb), estilo="<|-|>", color=GRIS,
            escala=7, grosor=0.9)

    # Leyenda fuera del área del diagrama.
    marcas = [
        Patch(facecolor=PROPUESTO, edgecolor="black", linewidth=0.8,
              label="Componente propuesto"),
        Patch(facecolor=NUCLEO, edgecolor="black", linewidth=0.8,
              label="Núcleo construido en el estudio"),
        Patch(facecolor=EXTERNO, edgecolor="black", linewidth=0.8,
              label="Fuera del microservicio"),
        Line2D([0], [0], color=GRIS, linewidth=1.1, linestyle=DISCONTINUO,
               label="Lazo de retroalimentación"),
    ]
    eje.legend(handles=marcas, loc="upper center", bbox_to_anchor=(0.5, 0.03),
               ncol=4, frameon=False, fontsize=T_LEYENDA, handlelength=2.2,
               columnspacing=1.4)

    figura.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
    return _guardar(figura, destino)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        salida = Path(sys.argv[1])
    else:
        salida = RAIZ / "salidas" / "figura_microservicio.png"
    ruta = figura_microservicio(salida)
    print(f"Figura escrita en {ruta}")
