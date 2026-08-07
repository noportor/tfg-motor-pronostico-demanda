/**
 * Descubrimiento de corridas — la aditividad del visor.
 *
 * nginx expone el directorio del proyecto como listado JSON (`autoindex`) bajo
 * /datos/. Toda entrada `salidas*` que tenga `manifiesto.json` es una corrida:
 * una ablación nueva aparece en el selector sin tocar el visor.
 */

import type { Corrida, EntradaAutoindex, Manifiesto } from "../contrato";

export const BASE_DATOS = "/datos";

export function urlDeArtefacto(corrida: string, archivo: string): string {
  return `${BASE_DATOS}/${corrida}/${archivo}`;
}

async function jsonDe<T>(url: string): Promise<T | null> {
  const respuesta = await fetch(url);
  if (!respuesta.ok) return null;
  return (await respuesta.json()) as T;
}

export async function descubrirCorridas(): Promise<Corrida[]> {
  const listado = await jsonDe<EntradaAutoindex[]>(`${BASE_DATOS}/`);
  if (!listado) {
    throw new Error(
      "No se pudo listar /datos/. ¿El visor está corriendo con el repositorio montado?",
    );
  }

  const candidatas = listado
    .filter((e) => e.type === "directory" && e.name.startsWith("salidas"))
    .map((e) => e.name);

  const corridas: Corrida[] = [];
  await Promise.all(
    candidatas.map(async (nombre) => {
      const manifiesto = await jsonDe<Manifiesto>(
        urlDeArtefacto(nombre, "manifiesto.json"),
      );
      if (!manifiesto) return; // sin manifiesto no es una corrida
      const anulaciones = manifiesto.configuracion?.contenido?._anulaciones ?? [];
      corridas.push({
        nombre,
        etiqueta: anulaciones.length
          ? `${nombre} (${anulaciones.join("; ")})`
          : nombre,
        anulaciones,
        generado_en: manifiesto.generado_en ?? null,
      });
    }),
  );

  // `salidas` (la principal) primero; el resto alfabético — orden estable.
  return corridas.sort((a, b) =>
    a.nombre === "salidas" ? -1 : b.nombre === "salidas" ? 1 : a.nombre.localeCompare(b.nombre),
  );
}

/** Archivos presentes en una corrida (para la vista Artefactos). */
export async function listarArtefactos(
  corrida: string,
): Promise<EntradaAutoindex[]> {
  const listado = await jsonDe<EntradaAutoindex[]>(
    `${BASE_DATOS}/${corrida}/`,
  );
  return listado ?? [];
}
