/** Descubrimiento de corridas contra un autoindex simulado (sintético, RN-1). */

import { afterEach, describe, expect, it, vi } from "vitest";

import { descubrirCorridas } from "./indice";

function respuestaJson(cuerpo: unknown): { ok: boolean; status?: number; json: () => Promise<unknown> } {
  return { ok: true, json: async () => cuerpo };
}

afterEach(() => vi.unstubAllGlobals());

describe("descubrirCorridas", () => {
  it("toma solo salidas* con manifiesto, la principal primero", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url === "/datos/") {
          return respuestaJson([
            { name: "salidas_zeta", type: "directory", mtime: "" },
            { name: "src", type: "directory", mtime: "" },
            { name: "salidas", type: "directory", mtime: "" },
            { name: "salidas_rota", type: "directory", mtime: "" },
            { name: "main.py", type: "file", mtime: "" },
          ]);
        }
        if (url.includes("salidas_rota")) {
          return { ok: false, status: 404, json: async () => null };
        }
        const anulaciones = url.includes("salidas_zeta")
          ? ["modelos.motor_regla=mae_mas_bias"]
          : [];
        return respuestaJson({
          generado_en: "2026-08-07T00:00:00+00:00",
          configuracion: { contenido: { _anulaciones: anulaciones } },
        });
      }),
    );

    const corridas = await descubrirCorridas();
    expect(corridas.map((c) => c.nombre)).toEqual(["salidas", "salidas_zeta"]);
    expect(corridas[1].etiqueta).toContain("mae_mas_bias");
  });

  it("si /datos/ no responde, el error es explícito", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 500, json: async () => null })),
    );
    await expect(descubrirCorridas()).rejects.toThrow("/datos/");
  });
});
