/**
 * Paleta del visor — la misma regla y los mismos hex validados del tablero.
 *
 * El color codifica el ROL del modelo en el experimento, nunca su identidad:
 * la identidad la lleva siempre la etiqueta. Los dos acentos (azul/naranja)
 * pasaron el verificador de la guía de visualización en todos los pares sobre
 * superficie clara (CVD ΔE 24,7 · visión normal 33,6 · ambos ≥3:1); los grises
 * son contexto deliberadamente neutro, con la regla de alivio cubierta porque
 * cada gráfico lleva etiquetas visibles y su tabla al lado.
 */

export const ROLES = [
  "Motor (propuesta)",
  "LightGBM",
  "Benchmark declarado",
  "Otros métodos",
] as const;
export type Rol = (typeof ROLES)[number];

export const COLOR_ROL: Record<Rol, string> = {
  "Motor (propuesta)": "#2a78d6",
  LightGBM: "#eb6834",
  "Benchmark declarado": "#52514e",
  "Otros métodos": "#898781",
};

// Tinta y cromo (instancia de referencia de la guía, superficie clara).
export const TINTA = "#0b0b0b";
export const TINTA_SECUNDARIA = "#52514e";
export const TINTA_MUTED = "#898781";
export const GRILLA = "#e1e0d9";
export const EJE = "#c3c2b7";
export const ACENTO = "#2a78d6";
export const NEUTRO_CLARO = "#eeedea";
export const SUPERFICIE = "#ffffff";
export const PLANO_PAGINA = "#f9f9f7";

export function rolDe(
  modelo: string,
  benchmarks: { promedioMovil: string; naive: string },
): Rol {
  if (modelo === "motor") return "Motor (propuesta)";
  if (modelo === "lightgbm") return "LightGBM";
  if (modelo === benchmarks.promedioMovil || modelo === benchmarks.naive) {
    return "Benchmark declarado";
  }
  return "Otros métodos";
}

export function colorDeModelo(
  modelo: string,
  benchmarks: { promedioMovil: string; naive: string },
): string {
  return COLOR_ROL[rolDe(modelo, benchmarks)];
}

// --- Formato de números (es-BO: coma decimal, punto de miles) ---------------

const formateadores = new Map<string, Intl.NumberFormat>();

export function numero(
  valor: number | null | undefined,
  decimales = 2,
  signo = false,
): string {
  if (valor === null || valor === undefined || Number.isNaN(valor)) return "n/d";
  const clave = `${decimales}|${signo}`;
  let formateador = formateadores.get(clave);
  if (!formateador) {
    formateador = new Intl.NumberFormat("es-BO", {
      minimumFractionDigits: decimales,
      maximumFractionDigits: decimales,
      signDisplay: signo ? "exceptZero" : "auto",
    });
    formateadores.set(clave, formateador);
  }
  return formateador.format(valor);
}

export function entero(valor: number | null | undefined): string {
  return numero(valor, 0);
}
