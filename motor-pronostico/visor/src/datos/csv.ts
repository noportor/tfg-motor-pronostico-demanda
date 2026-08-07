/**
 * Lectura de los CSVs del contrato con PapaParse.
 *
 * Sin streaming ni workers: el CSV más grande (`errores_por_serie.csv`, ~6 MB)
 * se parsea una vez por sesión y React Query lo cachea; a partir de ahí todos
 * los filtros son operaciones en memoria.
 */

import Papa from "papaparse";

export async function leerCsv<T extends Record<string, unknown>>(
  url: string,
): Promise<T[]> {
  const respuesta = await fetch(url);
  if (!respuesta.ok) {
    throw new Error(`${url}: HTTP ${respuesta.status}`);
  }
  const texto = await respuesta.text();
  const resultado = Papa.parse<T>(texto, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  if (resultado.errors.length > 0) {
    // Un CSV del contrato no puede venir roto: mejor un error visible que
    // filas silenciosamente descartadas.
    const primero = resultado.errors[0];
    throw new Error(
      `${url}: error de parseo en la fila ${primero.row ?? "?"}: ${primero.message}`,
    );
  }
  return resultado.data;
}

/** Percentil por interpolación lineal (el mismo método que usa pandas). */
export function percentil(valores: number[], q: number): number {
  if (valores.length === 0) return NaN;
  const ordenados = [...valores].sort((a, b) => a - b);
  const posicion = (ordenados.length - 1) * q;
  const base = Math.floor(posicion);
  const resto = posicion - base;
  const siguiente = ordenados[base + 1];
  return siguiente !== undefined
    ? ordenados[base] + resto * (siguiente - ordenados[base])
    : ordenados[base];
}

/** Agrupa filas por una clave y devuelve Map — base de todos los groupby del visor. */
export function agruparPor<T>(filas: T[], clave: (fila: T) => string): Map<string, T[]> {
  const grupos = new Map<string, T[]>();
  for (const fila of filas) {
    const k = clave(fila);
    const grupo = grupos.get(k);
    if (grupo) grupo.push(fila);
    else grupos.set(k, [fila]);
  }
  return grupos;
}
