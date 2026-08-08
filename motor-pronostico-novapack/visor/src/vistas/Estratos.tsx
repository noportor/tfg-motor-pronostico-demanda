/**
 * Estratos — ¿de dónde viene la ventaja de cada brazo? La misma D valorizada,
 * partida por régimen ADI-CV² y por tipo de categoría (estacional/recurrente),
 * en los dos protocolos (un paso y multihorizonte).
 */

import { useMemo, useState } from "react";
import {
  Grid2 as Grid,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";

import BarrasPorModelo from "../componentes/BarrasPorModelo";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import { benchmarksDe, useCsvGenerico, useManifiesto } from "../datos/hooks";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "estratos",
  titulo: "Estratos",
  orden: 3.7,
  descripcion: "D por régimen ADI-CV² y por tipo de categoría",
};

interface FilaEstrato {
  dimension: string;
  estrato: string;
  modelo: string;
  wmape_val: number;
  bias_val: number;
  D: number;
  [otra: string]: unknown;
}

const DIMENSIONES: Record<string, string> = {
  regimen_adi_cv2: "Régimen ADI-CV² (Syntetos-Boylan)",
  tipo_de_categoria: "Tipo de categoría (estacional / recurrente)",
};

const redondear = (valor: number, decimales = 1): number =>
  Number.isFinite(valor) ? Number(valor.toFixed(decimales)) : NaN;

export default function Estratos() {
  const corrida = useCorridaActiva();
  const manifiesto = useManifiesto(corrida);
  const unPaso = useCsvGenerico(corrida, "estratos_valorizados.csv");
  const horizonte = useCsvGenerico(corrida, "multihorizonte_estratos.csv");

  const [dimension, setDimension] = useState("tipo_de_categoria");
  const [protocolo, setProtocolo] = useState<"un_paso" | "multihorizonte">(
    "un_paso",
  );

  const benchmarks = benchmarksDe(manifiesto.data);
  const activas = protocolo === "un_paso" ? unPaso : horizonte;
  const filas = useMemo(
    () =>
      ((activas.data ?? []) as unknown as FilaEstrato[]).filter(
        (fila) => fila.dimension === dimension,
      ),
    [activas.data, dimension],
  );

  const estratosVisibles = useMemo(
    () => [...new Set(filas.map((fila) => fila.estrato))].sort(),
    [filas],
  );

  if (unPaso.isPending || manifiesto.isPending) return <Cargando />;
  if (unPaso.error) return <ErrorCarga error={unPaso.error} />;
  if (!((unPaso.data ?? []).length)) {
    return (
      <Aviso>
        Esta corrida no tiene estratos_valorizados.csv (corridas anteriores a
        la lectura por estrato).
      </Aviso>
    );
  }

  return (
    <Seccion
      titulo="¿De dónde viene la ventaja de cada brazo?"
      detalle="La MISMA métrica de decisión D, recalculada dentro de cada estrato: mismo maestro de costos, misma definición, ningún número nuevo — solo la vara, partida donde la lectura importa."
    >
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid size={{ xs: 12, md: 7 }}>
          <ToggleButtonGroup
            exclusive
            size="small"
            value={dimension}
            onChange={(_, valor: string | null) => valor && setDimension(valor)}
          >
            {Object.entries(DIMENSIONES).map(([clave, etiqueta]) => (
              <ToggleButton key={clave} value={clave}>
                {etiqueta}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Grid>
        <Grid size={{ xs: 12, md: 5 }}>
          <ToggleButtonGroup
            exclusive
            size="small"
            value={protocolo}
            onChange={(_, valor: typeof protocolo | null) =>
              valor && setProtocolo(valor)
            }
          >
            <ToggleButton value="un_paso">Un paso</ToggleButton>
            <ToggleButton value="multihorizonte">
              Multihorizonte D(h)
            </ToggleButton>
          </ToggleButtonGroup>
        </Grid>
      </Grid>

      {protocolo === "multihorizonte" && horizonte.error && (
        <Aviso>Esta corrida no tiene multihorizonte_estratos.csv.</Aviso>
      )}

      <Grid container spacing={2}>
        {estratosVisibles.map((estrato) => (
          <Grid key={estrato} size={{ xs: 12, md: 6 }}>
            <Typography variant="subtitle2" gutterBottom>
              {estrato}{" "}
              <Typography component="span" variant="caption" color="text.secondary">
                (
                {(filas.find((fila) => fila.estrato === estrato)
                  ?.series_con_costo as number) ??
                  (filas.find((fila) => fila.estrato === estrato)
                    ?.n_series as number) ?? "?"}{" "}
                series)
              </Typography>
            </Typography>
            <BarrasPorModelo
              filas={filas
                .filter((fila) => fila.estrato === estrato)
                .map((fila) => ({
                  modelo: fila.modelo,
                  valor: redondear(fila.D),
                }))}
              tituloValor="D (%)"
              benchmarks={benchmarks}
            />
          </Grid>
        ))}
      </Grid>

      <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
        Tabla completa
      </Typography>
      <TablaDatos
        filas={filas.map((fila) => ({
          estrato: fila.estrato,
          modelo: fila.modelo,
          "WMAPE val (%)": redondear(fila.wmape_val),
          "Bias val (%)": redondear(fila.bias_val),
          "D (%)": redondear(fila.D),
        }))}
        alto={64 + 36 * Math.min(filas.length, 14)}
        decimales={1}
      />
    </Seccion>
  );
}
