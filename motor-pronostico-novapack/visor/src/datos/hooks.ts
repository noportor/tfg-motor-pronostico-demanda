/**
 * Hooks de datos (TanStack Query) — el único camino de la UI a los archivos.
 *
 * Claves de caché por [corrida, archivo]: cambiar de corrida no invalida lo ya
 * bajado, y volver a ella es instantáneo. `staleTime: Infinity` a propósito:
 * una corrida es INMUTABLE por diseño (RN-5/RN-6) — si el pipeline vuelve a
 * correr, el usuario recarga la página, no hay estado vivo que sincronizar.
 */

import { useQuery } from "@tanstack/react-query";

import type {
  Corrida,
  FilaErrorSerie,
  FilaImportancia,
  FilaParametro,
  FilaResumen,
  FilaSeleccion,
  Flujo,
  Inspeccion,
  Manifiesto,
} from "../contrato";
import { leerCsv } from "./csv";
import { descubrirCorridas, listarArtefactos, urlDeArtefacto } from "./indice";

const INMUTABLE = { staleTime: Infinity, retry: 1 } as const;

export function useCorridas() {
  return useQuery<Corrida[]>({
    queryKey: ["corridas"],
    queryFn: descubrirCorridas,
    ...INMUTABLE,
  });
}

function useJson<T>(corrida: string, archivo: string, obligatorio = true) {
  return useQuery<T | null>({
    queryKey: [corrida, archivo],
    queryFn: async () => {
      const respuesta = await fetch(urlDeArtefacto(corrida, archivo));
      if (!respuesta.ok) {
        if (obligatorio) throw new Error(`${archivo}: HTTP ${respuesta.status}`);
        return null;
      }
      return (await respuesta.json()) as T;
    },
    enabled: corrida !== "",
    ...INMUTABLE,
  });
}

function useCsv<T extends Record<string, unknown>>(corrida: string, archivo: string) {
  return useQuery<T[]>({
    queryKey: [corrida, archivo],
    queryFn: () => leerCsv<T>(urlDeArtefacto(corrida, archivo)),
    enabled: corrida !== "",
    ...INMUTABLE,
  });
}

// --- Documentos del contrato -------------------------------------------------

export const useManifiesto = (corrida: string) =>
  useJson<Manifiesto>(corrida, "manifiesto.json");
export const useFlujo = (corrida: string) => useJson<Flujo>(corrida, "flujo.json");
export const useInspeccion = (corrida: string) =>
  useJson<Inspeccion>(corrida, "inspeccion.json");

// --- Tablas ------------------------------------------------------------------

export const useResumenMetricas = (corrida: string) =>
  useCsv<FilaResumen>(corrida, "resumen_metricas.csv");
export const useTabla8 = (corrida: string) =>
  useCsv<Record<string, unknown>>(corrida, "tabla8_resultados.csv");
export const useErroresPorSerie = (corrida: string) =>
  useCsv<FilaErrorSerie>(corrida, "errores_por_serie.csv");
export const useSeleccionMotor = (corrida: string) =>
  useCsv<FilaSeleccion>(corrida, "seleccion_motor.csv");
export const useParametros = (corrida: string) =>
  useCsv<FilaParametro>(corrida, "parametros_por_serie.csv");
export const useImportancias = (corrida: string) =>
  useCsv<FilaImportancia>(corrida, "lightgbm_importancias.csv");
export const useNemenyi = (corrida: string) =>
  useCsv<Record<string, unknown>>(corrida, "nemenyi.csv");
export const useCsvGenerico = (corrida: string, archivo: string) =>
  useCsv<Record<string, unknown>>(corrida, archivo);

// --- Inventario --------------------------------------------------------------

export function useArtefactos(corrida: string) {
  return useQuery({
    queryKey: [corrida, "__listado__"],
    queryFn: () => listarArtefactos(corrida),
    enabled: corrida !== "",
    ...INMUTABLE,
  });
}

// --- Texto plano (informes .txt) ---------------------------------------------

export function useTexto(corrida: string, archivo: string) {
  return useQuery<string>({
    queryKey: [corrida, archivo],
    queryFn: async () => {
      const respuesta = await fetch(urlDeArtefacto(corrida, archivo));
      if (!respuesta.ok) throw new Error(`${archivo}: HTTP ${respuesta.status}`);
      return respuesta.text();
    },
    enabled: corrida !== "",
    ...INMUTABLE,
  });
}

// --- Derivados compartidos ----------------------------------------------------

/** Benchmarks declarados en la configuración de la corrida. */
export function benchmarksDe(manifiesto: Manifiesto | null | undefined): {
  promedioMovil: string;
  naive: string;
} {
  const modelos = manifiesto?.configuracion?.contenido?.modelos;
  return {
    promedioMovil: modelos?.benchmark_promedio_movil ?? "ma_12",
    naive: modelos?.benchmark_naive ?? "naive_m1",
  };
}
