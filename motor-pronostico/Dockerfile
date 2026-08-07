# syntax=docker/dockerfile:1.7
#
# Entorno reproducible del experimento (RN-5).
#
# La máquina de desarrollo no tiene Python instalado: todo se ejecuta aquí.
# La imagen fija el intérprete y las versiones de las dependencias, de modo que
# la misma entrada produce exactamente la misma salida en cualquier máquina.

FROM python:3.13-slim

# libgomp1 es requisito de LightGBM (OpenMP).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Determinismo numérico: una sola hebra en las bibliotecas BLAS. Sin esto, la
# reducción en punto flotante depende del número de núcleos de la máquina y los
# resultados difieren en los últimos dígitos entre corridas.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

COPY requirements.txt requirements-extraccion.txt ./

# La instalación va en pasos separados y con caché persistente de pip. Sobre una
# conexión lenta, un `pip install` monolítico con `--no-cache-dir` obliga a
# volver a descargar TODO si algo se corta; así cada paquete ya bajado se
# reaprovecha entre intentos.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy==2.4.4"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "pandas==3.0.2" "scipy==1.17.1"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "scikit-learn==1.8.0" "lightgbm==4.7.0"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "matplotlib==3.10.8" "openpyxl==3.1.5" "PyYAML==6.0.2" "pytest==8.3.4"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "scikit-posthocs==0.14.0"
# Optuna: SOLO lo usa scripts/ajustar_lightgbm.py (la búsqueda corre una vez,
# fuera del pipeline). Vive en la imagen para que la búsqueda sea reproducible
# con las mismas versiones que el resto del experimento.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "optuna==4.1.0"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-extraccion.txt

# Verificación final: las versiones instaladas deben ser EXACTAMENTE las
# declaradas. Si un paso anterior resolvió otra cosa, la imagen no se construye.
RUN pip install --no-deps --dry-run -r requirements.txt > /dev/null \
 && pip check \
 && python -c "import pandas, numpy, scipy, sklearn, matplotlib, lightgbm, scikit_posthocs, yaml, openpyxl; print('dependencias OK')"

# git NO influye en ningún número, pero hacen falta dos cosas suyas:
#   - el manifiesto registra el commit y si el árbol tenía cambios sin commitear
#     al ejecutar (RN-6); sin git solo puede saber lo primero;
#   - el control de confidencialidad necesita saber qué archivos rastrea git,
#     porque son exactamente los que se publicarían.
# Va DESPUÉS de las capas de pip a propósito: ponerlo arriba invalidaría la
# caché de todas ellas ante cualquier cambio.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git config --global --add safe.directory /app

COPY . .

CMD ["python", "main.py", "--help"]
