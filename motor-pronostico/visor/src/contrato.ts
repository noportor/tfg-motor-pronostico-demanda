/**
 * Tipos del CONTRATO entre el pipeline y el visor.
 *
 * El pipeline (Python) escribe estos documentos en cada corrida; el visor solo
 * los lee. Si algo cambia aquí, cambia primero en `src/reporte.py` y en la
 * prueba de schema (`tests/test_pipeline.py`), que es la dueña del contrato.
 */

// --------------------------------------------------------------------------
// flujo.json (versión 1)
// --------------------------------------------------------------------------

export interface Etapa {
  id: string;
  titulo: string;
  rf: string | null;
  entrada: Record<string, unknown>;
  salida: Record<string, unknown>;
  decisiones: Record<string, unknown>;
  conteos: Record<string, unknown>;
  artefactos: string[];
  notas: string[];
  duracion_s: number | null;
}

export interface Flujo {
  version: number;
  corrida: {
    directorio: string;
    generado_en: string;
    config_sha256: string;
    datos_sha256: string | null;
    commit: string | null;
    anulaciones: string[];
  };
  etapas: Etapa[];
}

// --------------------------------------------------------------------------
// manifiesto.json
// --------------------------------------------------------------------------

export interface ResultadoWilcoxon {
  propuesto: string;
  referencia: string;
  r: number;
  z: number;
  p: number;
  significativo: boolean;
  gana_propuesto: number;
  n_pares: number;
}

export interface ResultadoFriedman {
  chi2: number;
  p: number;
  kendall_w: number;
  n_bloques: number;
  k: number;
  diferencia_critica: number;
  rangos_medios: Record<string, number>;
}

export interface AciertoMotor {
  series_comparadas: number;
  aciertos: number;
  tasa_acierto: number;
  azar_esperado: number;
  exceso_mae_medio: number;
  exceso_mae_mediano: number;
}

export interface NotasMotor {
  regla: string;
  empates: number;
  respaldos: number;
  reparto: Record<string, number>;
  acierto: AciertoMotor;
}

export interface Manifiesto {
  generado_en: string;
  duracion_segundos: number;
  datos: { archivo: string; sha256: string | null; bytes: number | null };
  codigo: {
    commit: string | null;
    arbol_limpio: boolean | null;
    origen?: string | null;
    advertencia?: string;
    archivos_modificados?: string[];
  };
  dependencias: Record<string, string | null>;
  configuracion: {
    archivo: string;
    sha256: string;
    contenido: Record<string, unknown> & {
      _anulaciones?: string[];
      modelos?: {
        benchmark_promedio_movil?: string;
        benchmark_naive?: string;
      } & Record<string, unknown>;
    };
  };
  resultados: {
    motor?: NotasMotor;
    friedman?: ResultadoFriedman;
    wilcoxon?: ResultadoWilcoxon[];
    lightgbm_mejor_iteracion?: number;
    respaldos_por_modelo?: Record<string, number>;
  } & Record<string, unknown>;
  salidas: { archivo: string; bytes: number; sha256: string }[];
}

// --------------------------------------------------------------------------
// inspeccion.json — el dict `datos` del informe de inspección
// --------------------------------------------------------------------------

export interface Inspeccion {
  rango_fechas: [string, string];
  meses_con_registro: number;
  meses_esperados: number;
  n_sku: number;
  n_canales: number;
  n_regionales: number;
  n_combinaciones: number;
  registros_duplicados: number;
  nulos_por_columna: Record<string, number>;
  registros_negativos: number;
  unidades_negativas: number;
  registros_en_cero: number;
  volumen_por_gestion: {
    gestion: number;
    registros: number;
    unidades: number;
    skus: number;
    combinaciones: number;
  }[];
  volumen_por_anio: { anio: number; registros: number; unidades: number }[];
  estacionalidad_mensual: { mes: number; cantidad: number; porcentaje: number }[];
  reparto_canal_regional: {
    regional: string;
    canal: string;
    registros: number;
    unidades: number;
    skus: number;
    combinaciones: number;
  }[];
  registros_por_combinacion: Record<string, number>;
  supervivencia_historial: Record<string, number>[];
  volumen_por_combinacion: Record<string, number>;
  panel_series?: number;
  panel_filas?: number;
  proporcion_ceros?: Record<string, number>;
  series_intermitentes_adi_1_32?: number;
}

// --------------------------------------------------------------------------
// Filas de los CSVs (papaparse con dynamicTyping)
// --------------------------------------------------------------------------

/** resumen_metricas.csv — una fila por modelo, con <metrica>_{media,mediana}. */
export interface FilaResumen extends Record<string, unknown> {
  modelo: string;
  series: number;
}

/** errores_por_serie.csv — una fila por (serie, modelo). */
export interface FilaErrorSerie extends Record<string, unknown> {
  serie: string;
  modelo: string;
  mae: number;
  rmse: number;
  mape: number | null;
  bias: number | null;
  mase: number | null;
}

/** seleccion_motor.csv — MAE de cada candidato en validación + el elegido. */
export interface FilaSeleccion extends Record<string, unknown> {
  serie: string;
  modelo_elegido: string;
  criterio_validacion: number | null;
  candidatos_validos: number;
  empate: boolean | string;
}

export interface FilaParametro extends Record<string, unknown> {
  serie: string;
  modelo: string;
  parametro: string;
  valor: number;
}

export interface FilaImportancia extends Record<string, unknown> {
  feature: string;
  ganancia: number;
  divisiones: number;
}

// --------------------------------------------------------------------------
// Descubrimiento (autoindex JSON de nginx)
// --------------------------------------------------------------------------

export interface EntradaAutoindex {
  name: string;
  type: "directory" | "file";
  mtime: string;
  size?: number;
}

/** Una corrida = un directorio salidas* con manifiesto. */
export interface Corrida {
  nombre: string;
  /** Etiqueta humana: nombre + anulaciones si es una ablación. */
  etiqueta: string;
  anulaciones: string[];
  generado_en: string | null;
}

export const METRICAS = [
  { clave: "mae", titulo: "MAE (unidades)", formato: "unidades" },
  { clave: "mape", titulo: "MAPE (%)", formato: "porcentaje" },
  { clave: "rmse", titulo: "RMSE (unidades)", formato: "unidades" },
  { clave: "mase", titulo: "MASE", formato: "razon" },
  { clave: "bias", titulo: "Bias (%)", formato: "porcentaje_signo" },
] as const;

export type ClaveMetrica = (typeof METRICAS)[number]["clave"];
export type Agregado = "media" | "mediana";
