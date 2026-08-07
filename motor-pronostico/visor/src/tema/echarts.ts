/**
 * Base de opciones ECharts del visor: cromo recesivo (grilla y ejes se ven
 * menos que los datos), tipografía del sistema, tooltips consistentes.
 * Cada gráfico parte de aquí y agrega solo lo suyo.
 */

import type { EChartsOption } from "echarts";

import { EJE, GRILLA, TINTA, TINTA_SECUNDARIA } from "./paleta";

export const FUENTE =
  'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif';

export const ejeBase = {
  axisLine: { lineStyle: { color: EJE } },
  axisTick: { lineStyle: { color: EJE } },
  axisLabel: { color: TINTA_SECUNDARIA, fontFamily: FUENTE },
  nameTextStyle: { color: TINTA_SECUNDARIA, fontFamily: FUENTE },
  splitLine: { lineStyle: { color: GRILLA } },
} as const;

export function opcionesBase(extra: EChartsOption): EChartsOption {
  return {
    textStyle: { fontFamily: FUENTE, color: TINTA },
    tooltip: {
      trigger: "item",
      textStyle: { fontFamily: FUENTE, fontSize: 12 },
      borderColor: EJE,
    },
    grid: { left: 8, right: 40, top: 28, bottom: 8, containLabel: true },
    ...extra,
  };
}
