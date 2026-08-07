/**
 * Capa de datos del visor. Fixtures SINTÉTICAS (RN-1: nada real en pruebas).
 */

import { describe, expect, it, vi } from "vitest";

import { agruparPor, leerCsv, percentil } from "./csv";

describe("percentil (interpolación lineal, como pandas)", () => {
  const valores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

  it("mediana de una lista par interpola", () => {
    expect(percentil(valores, 0.5)).toBe(5.5);
  });

  it("extremos devuelven mínimo y máximo", () => {
    expect(percentil(valores, 0)).toBe(1);
    expect(percentil(valores, 1)).toBe(10);
  });

  it("coincide con pandas en un caso no trivial", () => {
    // pandas: Series([2, 4, 7, 11]).quantile(0.25) == 3.5
    expect(percentil([2, 4, 7, 11], 0.25)).toBe(3.5);
  });

  it("no muta el arreglo de entrada", () => {
    const desordenado = [3, 1, 2];
    percentil(desordenado, 0.5);
    expect(desordenado).toEqual([3, 1, 2]);
  });

  it("lista vacía devuelve NaN, no revienta", () => {
    expect(Number.isNaN(percentil([], 0.5))).toBe(true);
  });
});

describe("agruparPor", () => {
  it("agrupa preservando el orden de aparición", () => {
    const filas = [
      { modelo: "a", v: 1 },
      { modelo: "b", v: 2 },
      { modelo: "a", v: 3 },
    ];
    const grupos = agruparPor(filas, (f) => f.modelo);
    expect([...grupos.keys()]).toEqual(["a", "b"]);
    expect(grupos.get("a")).toHaveLength(2);
  });
});

describe("leerCsv", () => {
  it("parsea con tipos y sin filas vacías", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        text: async () => "modelo,mae\nnaive_m1,10.5\nmotor,8.25\n",
      })),
    );
    const filas = await leerCsv<{ modelo: string; mae: number }>("/x.csv");
    expect(filas).toEqual([
      { modelo: "naive_m1", mae: 10.5 },
      { modelo: "motor", mae: 8.25 },
    ]);
    vi.unstubAllGlobals();
  });

  it("un HTTP no-ok es un error visible, no datos vacíos", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 403, text: async () => "" })),
    );
    await expect(leerCsv("/prohibido.csv")).rejects.toThrow("403");
    vi.unstubAllGlobals();
  });
});
