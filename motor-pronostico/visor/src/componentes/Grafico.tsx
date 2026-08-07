/** Envoltorio único de ECharts: todos los gráficos del visor pasan por acá. */

import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";

import { opcionesBase } from "../tema/echarts";

interface Props {
  opciones: EChartsOption;
  alto?: number;
}

export default function Grafico({ opciones, alto = 300 }: Props) {
  return (
    <ReactECharts
      option={opcionesBase(opciones)}
      style={{ height: alto, width: "100%" }}
      notMerge
      lazyUpdate
    />
  );
}
