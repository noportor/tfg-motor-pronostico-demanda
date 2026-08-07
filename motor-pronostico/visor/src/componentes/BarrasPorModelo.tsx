/**
 * LA vista del visor: una magnitud comparada entre los once modelos.
 *
 * Barras horizontales ordenadas por valor, color por ROL (la identidad la da
 * la etiqueta del eje, nunca el color) y valor directo al final de cada barra.
 */

import { useMemo } from "react";
import { Box, Chip, Stack } from "@mui/material";

import Grafico from "./Grafico";
import { ejeBase } from "../tema/echarts";
import {
  COLOR_ROL,
  ROLES,
  TINTA_SECUNDARIA,
  colorDeModelo,
  numero,
} from "../tema/paleta";

interface Fila {
  modelo: string;
  valor: number;
}

interface Props {
  filas: Fila[];
  tituloValor: string;
  benchmarks: { promedioMovil: string; naive: string };
  decimales?: number;
  signo?: boolean;
  ascendente?: boolean;
}

export default function BarrasPorModelo({
  filas,
  tituloValor,
  benchmarks,
  decimales = 1,
  signo = false,
  ascendente = true,
}: Props) {
  const ordenadas = useMemo(
    () =>
      [...filas].sort((a, b) =>
        ascendente ? b.valor - a.valor : a.valor - b.valor,
      ),
    [filas, ascendente],
  );

  const opciones = {
    grid: { left: 8, right: 70, top: 8, bottom: 8, containLabel: true },
    xAxis: {
      type: "value" as const,
      name: tituloValor,
      nameLocation: "middle" as const,
      nameGap: 28,
      ...ejeBase,
    },
    yAxis: {
      type: "category" as const,
      data: ordenadas.map((f) => f.modelo),
      ...ejeBase,
      splitLine: { show: false },
    },
    series: [
      {
        type: "bar" as const,
        data: ordenadas.map((f) => ({
          value: f.valor,
          itemStyle: {
            color: colorDeModelo(f.modelo, benchmarks),
            borderRadius: [0, 3, 3, 0],
          },
        })),
        barWidth: "62%",
        label: {
          show: true,
          position: "right" as const,
          color: TINTA_SECUNDARIA,
          formatter: (p: any) => numero(p.value, decimales, signo),
        },
      },
    ],
    tooltip: {
      formatter: (p: any) =>
        `${p.name}: <b>${numero(p.value, decimales, signo)}</b>`,
    },
  };

  return (
    <Box>
      {/* Leyenda de ROLES (los colores codifican rol, no identidad). */}
      <Stack direction="row" spacing={1} sx={{ mb: 1, flexWrap: "wrap" }}>
        {ROLES.map((rol) => (
          <Chip
            key={rol}
            size="small"
            label={rol}
            sx={{
              bgcolor: "transparent",
              border: "1px solid",
              borderColor: "divider",
              "& .MuiChip-label": { pl: 0.5 },
            }}
            icon={
              <Box
                sx={{
                  width: 10,
                  height: 10,
                  borderRadius: "2px",
                  bgcolor: COLOR_ROL[rol],
                  ml: 1,
                }}
              />
            }
          />
        ))}
      </Stack>
      <Grafico opciones={opciones} alto={Math.max(200, 30 * ordenadas.length + 60)} />
    </Box>
  );
}
