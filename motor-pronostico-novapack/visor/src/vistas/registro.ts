/**
 * Registro ADITIVO de vistas: `import.meta.glob` recoge en build todo archivo
 * `*.tsx` de esta carpeta que exporte `vista` (metadatos) + componente default.
 * Agregar una vista = soltar un archivo acá. No hay lista central que editar.
 */

import type { ComponentType } from "react";

import type { MetaVista } from "./tipos";

export type { MetaVista };

interface ModuloVista {
  vista?: MetaVista;
  default?: ComponentType;
}

const modulos = import.meta.glob<ModuloVista>("./*.tsx", { eager: true });

export const VISTAS: (MetaVista & { Componente: ComponentType })[] = Object.values(
  modulos,
)
  .filter(
    (modulo): modulo is Required<ModuloVista> =>
      modulo.vista !== undefined && modulo.default !== undefined,
  )
  .map((modulo) => ({ ...modulo.vista, Componente: modulo.default }))
  .sort((a, b) => a.orden - b.orden);
