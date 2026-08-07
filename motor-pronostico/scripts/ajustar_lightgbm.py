"""Búsqueda de hiperparámetros del brazo LightGBM — Optuna TPE, determinista.

Corre UNA vez, fuera del pipeline (como la extracción): sus artefactos quedan
en ``salidas_tuning/`` y los hiperparámetros ganadores se DECLARAN a mano en
``modelos.lightgbm`` de config.yaml, con un comentario que cita esta corrida.
El pipeline nunca re-optimiza: corre siempre con lo declarado.

Reproducibilidad (RN-5)
-----------------------
- Sampler TPE con SEMILLA FIJA y ejecución SECUENCIAL (un worker): la
  secuencia de ensayos es determinista de punta a punta.
- El espacio de búsqueda vive en config.yaml (``lightgbm_tuning``): el mismo
  principio que la rejilla de alfa del SES, a otra escala.
- La tabla completa de ensayos es un artefacto: el tribunal ve el CAMINO de la
  búsqueda, no solo al ganador.

Disciplina temporal
-------------------
El fold interno es la ÚLTIMA gestión del bloque de entrenamiento: se entrena
con lo anterior y se evalúa ahí con la D interna valorizada (WMAPE + |Bias|,
en dinero). La validación real queda intacta para el motor (RN-2) y el bloque
de prueba jamás se toca. El guardia de abajo lo verifica activamente.

Uso::

    docker compose run --rm tfg python scripts/ajustar_lightgbm.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import carga, eventos, particion, series  # noqa: E402
from src.config import cargar_config, mes_a_periodo  # noqa: E402
from src.costos import cargar_costos  # noqa: E402
from src.exogenas import cargar_exogenas  # noqa: E402
from src.features import (  # noqa: E402
    categoricas_de, construir_features, nombres_de_features,
)


def _d_interna(y: np.ndarray, p: np.ndarray, costo: np.ndarray) -> dict:
    """D interna valorizada sobre el fold: la misma definición del estudio."""
    demanda = float((costo * y).sum())
    error = float((costo * np.abs(p - y)).sum())
    predicho = float((costo * p).sum())
    if demanda <= 0:
        return {"wmape": float("nan"), "bias": float("nan"), "D": float("inf")}
    wmape = 100.0 * error / demanda
    bias = 100.0 * (predicho - demanda) / demanda
    return {"wmape": wmape, "bias": bias, "D": wmape + abs(bias)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    import lightgbm as lgb
    import optuna

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Anula lightgbm_tuning.n_trials (para pruebas de humo).",
    )
    parser.add_argument(
        "--destino", default=None,
        help="Directorio de artefactos (por defecto, salidas_tuning/).",
    )
    args = parser.parse_args(argv)

    cfg = cargar_config(args.config)
    ajuste = dict(cfg.crudo.get("lightgbm_tuning") or {})
    if args.n_trials is not None:
        ajuste["n_trials"] = args.n_trials
    if not ajuste:
        raise SystemExit("config.yaml no tiene la sección lightgbm_tuning.")

    fold_desde = mes_a_periodo(str(ajuste["fold_desde"]))
    fold_hasta = mes_a_periodo(str(ajuste["fold_hasta"]))
    # Guardia activo: el fold interno cae ÍNTEGRO dentro de entrenamiento.
    if fold_hasta > cfg.particion.fin_entrenamiento:
        raise SystemExit(
            f"El fold interno termina en {fold_hasta}, después del cierre de "
            f"entrenamiento ({cfg.particion.fin_entrenamiento}): estaría "
            "optimizando sobre validación o prueba (RN-2)."
        )

    print("=" * 78)
    print("AJUSTE DE HIPERPARÁMETROS — LightGBM · Optuna TPE (semilla "
          f"{ajuste['semilla']}, {ajuste['n_trials']} ensayos)")
    print("=" * 78)
    print(f"Fold interno : {fold_desde} .. {fold_hasta} (última gestión de "
          "entrenamiento)")

    # --- Los mismos pasos de datos del pipeline (1 .. 6) --------------------
    df, informe_carga = carga.cargar(cfg)
    sha_datos = informe_carga.sha256   # certificado ANTES de horas de búsqueda
    df, informe_poblacion = carga.aplicar_poblacion(df, cfg)
    print(f"Población    : {informe_poblacion.filas_en_poblacion:,} filas")
    panel_largo, _ = series.construir_series(df, cfg)
    panel_ajuste_largo, _ = eventos.aplicar_tratamiento(panel_largo, cfg)
    corte = particion.particionar(cfg)
    cohorte, informe_cohorte = particion.aplicar_criterios_inclusion(
        panel_ajuste_largo, cfg, corte
    )
    print(f"Cohorte      : {informe_cohorte.series_finales:,} series")
    exogenas = cargar_exogenas(cfg, df)
    tabla = construir_features(cohorte, cfg, exogenas)

    columnas = nombres_de_features(cfg)
    categoricas = [c for c in categoricas_de(cfg) if c in columnas]

    entrena = tabla.loc[tabla["periodo"] < fold_desde]
    fold = tabla.loc[
        (tabla["periodo"] >= fold_desde) & (tabla["periodo"] <= fold_hasta)
    ]
    print(f"Filas        : {len(entrena):,} de ajuste · {len(fold):,} de fold")

    maestro = cargar_costos(cfg)
    if not maestro.disponible:
        raise SystemExit(
            "Sin maestro de costos no hay D interna valorizada: el criterio "
            "de la búsqueda debe ser el del estudio."
        )
    costo_fold = fold["serie"].map(maestro.costo_por_serie)
    con_costo = costo_fold.notna().to_numpy()
    excluidas = int((~con_costo).sum())
    print(f"Fold valorizable: {int(con_costo.sum()):,} celdas "
          f"({excluidas:,} sin costo, excluidas y declaradas)")

    y_fold = fold["y"].to_numpy(dtype=float)[con_costo]
    costo = costo_fold.to_numpy(dtype=float)[con_costo]
    x_fold = fold[columnas]

    dtrain = lgb.Dataset(
        entrena[columnas], label=entrena["y"],
        categorical_feature=categoricas, free_raw_data=False,
    )
    dfold = lgb.Dataset(
        fold[columnas], label=fold["y"],
        categorical_feature=categoricas, reference=dtrain, free_raw_data=False,
    )

    espacio = ajuste["espacio"]
    fijos = {
        "seed": int(cfg.modelos["lightgbm"].get("seed", 0)),
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        "bagging_freq": 1,
    }

    filas_ensayos: list[dict] = []

    def objetivo(trial: "optuna.Trial") -> float:
        params = dict(fijos)
        params["objective"] = trial.suggest_categorical(
            "objective", list(espacio["objective"])
        )
        if params["objective"] == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float(
                "tweedie_variance_power", *espacio["tweedie_variance_power"]
            )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", *espacio["learning_rate"], log=True
        )
        params["num_leaves"] = trial.suggest_int(
            "num_leaves", *espacio["num_leaves"], log=True
        )
        params["min_data_in_leaf"] = trial.suggest_int(
            "min_data_in_leaf", *espacio["min_data_in_leaf"], log=True
        )
        params["feature_fraction"] = trial.suggest_float(
            "feature_fraction", *espacio["feature_fraction"]
        )
        params["bagging_fraction"] = trial.suggest_float(
            "bagging_fraction", *espacio["bagging_fraction"]
        )
        params["lambda_l2"] = trial.suggest_float(
            "lambda_l2", *espacio["lambda_l2"]
        )

        marca = time.perf_counter()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=int(ajuste["num_boost_round"]),
            valid_sets=[dfold], valid_names=["fold"],
            callbacks=[
                lgb.early_stopping(int(ajuste["early_stopping_rounds"]),
                                   verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        predicho = np.maximum(
            booster.predict(x_fold, num_iteration=booster.best_iteration), 0.0
        )[con_costo]
        resultado = _d_interna(y_fold, predicho, costo)

        filas_ensayos.append({
            "ensayo": trial.number,
            **{k: v for k, v in params.items() if k not in fijos},
            "mejor_iteracion": int(booster.best_iteration),
            "wmape_interna": round(resultado["wmape"], 3),
            "bias_interna": round(resultado["bias"], 3),
            "D_interna": round(resultado["D"], 3),
            "segundos": round(time.perf_counter() - marca, 1),
        })
        print(f"  ensayo {trial.number:>3}: D={resultado['D']:7.2f} "
              f"(wmape {resultado['wmape']:6.2f}, bias {resultado['bias']:+7.2f}) "
              f"obj={params['objective']:<14} "
              f"iter={booster.best_iteration:>4} "
              f"({time.perf_counter() - marca:5.1f} s)", flush=True)
        return resultado["D"]

    sampler = optuna.samplers.TPESampler(seed=int(ajuste["semilla"]))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="minimize", sampler=sampler)
    estudio.optimize(objetivo, n_trials=int(ajuste["n_trials"]), n_jobs=1)

    # --- Artefactos ---------------------------------------------------------
    destino = Path(args.destino) if args.destino else RAIZ / "salidas_tuning"
    destino.mkdir(exist_ok=True)
    tabla_ensayos = pd.DataFrame(filas_ensayos).sort_values("D_interna")
    tabla_ensayos.to_csv(destino / "lightgbm_tuning_ensayos.csv", index=False)

    mejor = estudio.best_trial
    resumen = {
        "semilla": int(ajuste["semilla"]),
        "n_trials": int(ajuste["n_trials"]),
        "fold_interno": {"desde": str(fold_desde), "hasta": str(fold_hasta)},
        "criterio": "D interna valorizada (WMAPE + |Bias|)",
        "mejor_D_interna": float(mejor.value),
        "mejores_parametros": mejor.params,
        "mejor_iteracion": int(
            tabla_ensayos.loc[tabla_ensayos["ensayo"] == mejor.number,
                              "mejor_iteracion"].iloc[0]
        ),
        "sha256_datos": sha_datos,
        "sha256_configuracion": cfg.hash_configuracion(),
        "optuna_version": optuna.__version__,
    }
    (destino / "lightgbm_tuning_mejor.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("MEJOR ENSAYO")
    print(json.dumps(resumen["mejores_parametros"], indent=2))
    print(f"D interna: {mejor.value:.2f}")
    print(f"\nArtefactos en {destino}/")
    print("Siguiente paso: declarar los ganadores en modelos.lightgbm de "
          "config.yaml citando esta corrida, y re-correr el pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
