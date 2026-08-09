"""Búsqueda de hiperparámetros de los brazos neuronales (ML2-V3) — Optuna TPE.

La V2 midió que la potencia Tweedie del campeón NO es transferible a ciegas:
cada red necesita su propia búsqueda, exactamente como la tuvo LightGBM. Este
script replica la disciplina de ``ajustar_lightgbm.py``:

- Sampler TPE con semilla fija, ejecución secuencial: la SECUENCIA de ensayos
  es determinista (el entrenamiento en GPU no es bit a bit reproducible entre
  hardware distinto y se declara — RN-5).
- Fold interno = la última gestión del bloque de entrenamiento: se entrena con
  lo anterior, la parada temprana y la evaluación miran SOLO el fold. La
  validación real queda intacta para el motor (RN-2) y la prueba jamás se toca.
- Criterio: D interna valorizada (WMAPE + |Bias|, en dinero) sobre el
  protocolo a un paso — la misma vara del estudio.
- La tabla completa de ensayos es un artefacto: el camino, no solo el ganador.
- Los ganadores se DECLARAN a mano en ``modelos.neuronales`` de la
  configuración ML2-V3, citando esta corrida. El pipeline nunca re-optimiza.

Uso::

    python scripts/ajustar_neuronal.py --brazo tft   --config config/config_ml2_v3.yaml
    python scripts/ajustar_neuronal.py --brazo nhits --config config/config_ml2_v3.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import carga, eventos, particion, series  # noqa: E402
from src.config import cargar_config, mes_a_periodo  # noqa: E402
from src.costos import cargar_costos  # noqa: E402
from src.exogenas import cargar_exogenas  # noqa: E402
from src.features import construir_features  # noqa: E402
from src.modelos.base import truncar_en_cero  # noqa: E402


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


class _CfgFold:
    """Lo mínimo que ModeloNeural consulta, con el fold como «validación».

    El proxy corre el MISMO envoltorio del pipeline con otra frontera
    temporal: fin_entrenamiento = el mes anterior al fold, de modo que
    ``predecir`` evalúe exactamente los meses del fold interno.
    """

    def __init__(self, neuronales: dict, fin_entrenamiento: pd.Period):
        self.modelos = {"neuronales": neuronales}
        self.particion = SimpleNamespace(fin_entrenamiento=fin_entrenamiento)


def main(argv: list[str] | None = None) -> int:
    import argparse

    import optuna

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brazo", required=True, choices=["tft", "nhits"])
    parser.add_argument("--config", default=None, help="Ruta a la configuración")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--destino", default=None,
                        help="Directorio de artefactos (salidas_tuning_ml2/).")
    args = parser.parse_args(argv)

    cfg = cargar_config(args.config)
    ajuste = dict(cfg.crudo.get("neuronal_tuning") or {})
    if not ajuste:
        raise SystemExit("La configuración no tiene la sección neuronal_tuning.")
    if args.n_trials is not None:
        ajuste["n_trials"] = args.n_trials
    espacio = dict(ajuste["espacio"][args.brazo])

    fold_desde = mes_a_periodo(str(ajuste["fold_desde"]))
    fold_hasta = mes_a_periodo(str(ajuste["fold_hasta"]))
    if fold_hasta > cfg.particion.fin_entrenamiento:
        raise SystemExit(
            f"El fold interno termina en {fold_hasta}, después del cierre de "
            f"entrenamiento ({cfg.particion.fin_entrenamiento}): estaría "
            "optimizando sobre validación o prueba (RN-2)."
        )

    print("=" * 78)
    print(f"AJUSTE DE HIPERPARÁMETROS — {args.brazo} · Optuna TPE (semilla "
          f"{ajuste['semilla']}, {ajuste['n_trials']} ensayos)")
    print("=" * 78)
    print(f"Fold interno : {fold_desde} .. {fold_hasta}")

    # --- Los mismos pasos de datos del pipeline (1 .. 6) --------------------
    df, informe_carga = carga.cargar(cfg)
    sha_datos = informe_carga.sha256
    df, _ = carga.aplicar_poblacion(df, cfg)
    panel_largo, _ = series.construir_series(df, cfg)
    panel_ajuste_largo, _ = eventos.aplicar_tratamiento(panel_largo, cfg)
    corte = particion.particionar(cfg)
    cohorte, informe_cohorte = particion.aplicar_criterios_inclusion(
        panel_ajuste_largo, cfg, corte
    )
    print(f"Cohorte      : {informe_cohorte.series_finales:,} series")
    exogenas = cargar_exogenas(cfg, df)
    tabla = construir_features(cohorte, cfg, exogenas)

    panel = series.a_panel_ancho(cohorte)
    panel_hasta_fold = panel.loc[panel.index <= fold_hasta]
    panel_entrena = panel.loc[panel.index < fold_desde]
    panel_fold = panel.loc[
        (panel.index >= fold_desde) & (panel.index <= fold_hasta)
    ]

    maestro = cargar_costos(cfg)
    if not maestro.disponible:
        raise SystemExit("Sin maestro de costos no hay D interna valorizada.")

    real_fold = panel_fold.stack().rename("y").reset_index()
    real_fold.columns = ["periodo", "serie", "y"]
    real_fold["costo"] = real_fold["serie"].map(maestro.costo_por_serie)
    real_fold = real_fold.dropna(subset=["costo"])
    print(f"Fold valorizable: {len(real_fold):,} celdas")

    comun = dict(cfg.modelos.get("neuronales", {}))
    exog_brazo = dict(comun.get(args.brazo, {}).get("exogenas", {}) or {})
    fin_entrena_fold = fold_desde - 1

    filas_ensayos: list[dict] = []

    def objetivo(trial: "optuna.Trial") -> float:
        perdida = trial.suggest_categorical("perdida", list(espacio["perdida"]))
        params_brazo: dict = {"perdida": perdida}
        if exog_brazo:
            params_brazo["exogenas"] = exog_brazo
        neuronales = {
            "semilla": int(ajuste["semilla"]),
            "horizonte": int(comun.get("horizonte", 12)),
            "input_size": trial.suggest_categorical(
                "input_size", list(espacio["input_size"])),
            "max_steps": int(ajuste.get("max_steps", 1000)),
            args.brazo: params_brazo,
        }
        if perdida == "tweedie":
            neuronales["tweedie_rho"] = trial.suggest_float(
                "tweedie_rho", *espacio["tweedie_rho"])
        params_brazo["learning_rate"] = trial.suggest_float(
            "learning_rate", *espacio["learning_rate"], log=True)
        params_brazo["scaler_type"] = trial.suggest_categorical(
            "scaler_type", list(espacio["scaler_type"]))
        if "hidden_size" in espacio:
            params_brazo["hidden_size"] = trial.suggest_categorical(
                "hidden_size", list(espacio["hidden_size"]))

        from src.modelos.neuronales import ModeloNeural

        marca = time.perf_counter()
        try:
            modelo = ModeloNeural(
                _CfgFold(neuronales, fin_entrena_fold), args.brazo, tabla
            )
            modelo.ajustar(panel_entrena, panel_fold)
            pred = truncar_en_cero(modelo.predecir(panel_hasta_fold))
        except Exception as error:  # un ensayo que diverge no mata la búsqueda
            print(f"  ensayo {trial.number:>3}: FALLÓ ({type(error).__name__}: "
                  f"{error})", flush=True)
            filas_ensayos.append({
                "ensayo": trial.number, **trial.params,
                "D_interna": float("inf"), "fallo": str(error)[:120],
                "segundos": round(time.perf_counter() - marca, 1),
            })
            return float("inf")

        largo_pred = pred.loc[panel_fold.index].stack().rename("p").reset_index()
        largo_pred.columns = ["periodo", "serie", "p"]
        junto = real_fold.merge(largo_pred, on=["periodo", "serie"], how="inner")
        # Predicciones no finitas = divergencia numérica: se declara y descarta.
        finito = np.isfinite(junto["p"].to_numpy(dtype=float))
        if not finito.all():
            junto = junto.loc[finito]
        resultado = _d_interna(
            junto["y"].to_numpy(dtype=float),
            junto["p"].to_numpy(dtype=float),
            junto["costo"].to_numpy(dtype=float),
        )
        filas_ensayos.append({
            "ensayo": trial.number, **trial.params,
            "wmape_interna": round(resultado["wmape"], 3),
            "bias_interna": round(resultado["bias"], 3),
            "D_interna": round(resultado["D"], 3),
            "celdas_no_finitas": int((~finito).sum()),
            "segundos": round(time.perf_counter() - marca, 1),
        })
        print(f"  ensayo {trial.number:>3}: D={resultado['D']:7.2f} "
              f"(wmape {resultado['wmape']:6.2f}, bias {resultado['bias']:+7.2f}) "
              f"perdida={perdida:<8} "
              f"({time.perf_counter() - marca:5.1f} s)", flush=True)
        return resultado["D"]

    sampler = optuna.samplers.TPESampler(seed=int(ajuste["semilla"]))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="minimize", sampler=sampler)
    estudio.optimize(objetivo, n_trials=int(ajuste["n_trials"]), n_jobs=1)

    destino = Path(args.destino) if args.destino else RAIZ / "salidas_tuning_ml2"
    destino.mkdir(exist_ok=True)
    tabla_ensayos = pd.DataFrame(filas_ensayos).sort_values("D_interna")
    tabla_ensayos.to_csv(destino / f"{args.brazo}_tuning_ensayos.csv", index=False)

    mejor = estudio.best_trial
    resumen = {
        "brazo": args.brazo,
        "semilla": int(ajuste["semilla"]),
        "n_trials": int(ajuste["n_trials"]),
        "fold_interno": {"desde": str(fold_desde), "hasta": str(fold_hasta)},
        "criterio": "D interna valorizada (WMAPE + |Bias|), un paso",
        "exogenas": exog_brazo,
        "mejor_D_interna": float(mejor.value),
        "mejores_parametros": mejor.params,
        "sha256_datos": sha_datos,
        "sha256_configuracion": cfg.hash_configuracion(),
    }
    (destino / f"{args.brazo}_tuning_mejor.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("MEJOR ENSAYO")
    print(json.dumps(resumen["mejores_parametros"], indent=2))
    print(f"D interna: {mejor.value:.2f}")
    print(f"\nArtefactos en {destino}/")
    print("Siguiente paso: declarar los ganadores en modelos.neuronales de la "
          "configuración ML2-V3 citando esta corrida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
