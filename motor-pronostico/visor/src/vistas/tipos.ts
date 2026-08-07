/**
 * Tipos compartidos por las vistas. Vive separado de registro.ts a propósito:
 * el registro glob-importa las vistas, así que si las vistas importaran del
 * registro habría un ciclo. De acá pueden importar todos.
 */

export type { Etapa, Flujo, Inspeccion, Manifiesto } from "../contrato";

export interface MetaVista {
  ruta: string;
  titulo: string;
  orden: number;
  descripcion?: string;
}
