/**
 * Series — la evidencia serie por serie: la realidad con los pronósticos
 * encima. Es la vista de la defensa: cualquier número agregado del estudio
 * puede bajarse acá a una serie concreta y VERSE. Los pronósticos se dibujan
 * desde validación (dentro de entrenamiento serían el ajuste in-sample).
 */

import { useMemo, useState } from "react";
import {
  Autocomplete,
  Chip,
  Grid2 as Grid,
  Stack,
  TextField,
  Typography,
} from "@mui/material";

import Grafico from "../componentes/Grafico";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import {
  benchmarksDe,
  useCsvGenerico,
  useErroresPorSerie,
  useManifiesto,
  useSeleccionMotor,
} from "../datos/hooks";
import { ejeBase } from "../tema/echarts";
import { TINTA_MUTED, colorDeModelo, numero } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "series",
  titulo: "Series",
  orden: 3.4,
  descripcion: "La realidad de cada serie con sus pronósticos encima",
};

interface Fila {
  periodo: string;
  serie: string;
  y_real: number | null;
  bloque: string;
  [modelo: string]: unknown;
}

const MODELOS_DEFECTO = ["motor", "lightgbm"];

/** Redondeo para celdas: número crudo, jamás string con locale. */
const redondear = (valor: number, decimales = 1): number =>
  Number.isFinite(valor) ? Number(valor.toFixed(decimales)) : NaN;

export default function Series() {
  const corrida = useCorridaActiva();
  const manifiesto = useManifiesto(corrida);
  const datos = useCsvGenerico(corrida, "series_y_predicciones.csv");
  const errores = useErroresPorSerie(corrida);
  const seleccion = useSeleccionMotor(corrida);

  const [serie, setSerie] = useState<string | null>(null);
  const [modelos, setModelos] = useState<string[]>(MODELOS_DEFECTO);

  const benchmarks = benchmarksDe(manifiesto.data);
  const filas = useMemo(
    () => (datos.data ?? []) as unknown as Fila[],
    [datos.data],
  );

  const series = useMemo(
    () => [...new Set(filas.map((fila) => fila.serie))].sort(),
    [filas],
  );
  const activa = serie ?? series[0] ?? null;

  const modelosDisponibles = useMemo(() => {
    if (!filas.length) return [];
    const fijas = new Set(["periodo", "serie", "y_real", "bloque"]);
    return Object.keys(filas[0]).filter((columna) => !fijas.has(columna));
  }, [filas]);

  const delaSerie = useMemo(
    () => filas.filter((fila) => fila.serie === activa),
    [filas, activa],
  );

  const elegido = useMemo(() => {
    const fila = (seleccion.data ?? []).find(
      (registro) => String(registro.serie) === activa,
    );
    return fila ? String(fila.modelo_elegido) : null;
  }, [seleccion.data, activa]);

  const metricas = useMemo(
    () =>
      (errores.data ?? [])
        .filter(
          (fila) =>
            String(fila.serie) === activa
            && modelos.includes(String(fila.modelo)),
        )
        .map((fila) => ({
          modelo: String(fila.modelo),
          "MAE (u)": redondear(Number(fila.mae)),
          "Bias (u)": redondear(Number(fila.bias_unidades)),
          "MAE+|Bias| (u)": redondear(Number(fila.error_compuesto_unidades)),
        })),
    [errores.data, activa, modelos],
  );

  if (datos.isPending || manifiesto.isPending) return <Cargando />;
  if (datos.error) return <ErrorCarga error={datos.error} />;
  if (!filas.length) {
    return (
      <Aviso>
        Esta corrida no tiene series_y_predicciones.csv (corridas anteriores a
        la persistencia por serie).
      </Aviso>
    );
  }

  const periodos = delaSerie.map((fila) => fila.periodo);
  const primerValidacion = delaSerie.find(
    (fila) => fila.bloque === "validacion",
  )?.periodo;
  const primerPrueba = delaSerie.find(
    (fila) => fila.bloque === "prueba",
  )?.periodo;

  return (
    <Seccion
      titulo="La evidencia, serie por serie"
      detalle="La línea gris es lo observado. Los pronósticos se dibujan desde validación: el protocolo a un paso, tal como se evaluó. El sombreado es el bloque de prueba."
    >
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid size={{ xs: 12, md: 5 }}>
          <Autocomplete
            options={series}
            value={activa}
            onChange={(_, valor) => setSerie(valor)}
            renderInput={(parametros) => (
              <TextField {...parametros} label="Serie (SKU | canal | regional)"
                         size="small" />
            )}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="El motor eligió"
            valor={elegido ?? "n/d"}
            ayuda="Selección por serie, decidida mirando SOLO validación (RN-2)."
          />
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
            {modelosDisponibles.map((nombre) => (
              <Chip
                key={nombre}
                label={nombre}
                size="small"
                variant={modelos.includes(nombre) ? "filled" : "outlined"}
                onClick={() =>
                  setModelos((antes) =>
                    antes.includes(nombre)
                      ? antes.filter((m) => m !== nombre)
                      : [...antes, nombre],
                  )
                }
                sx={
                  modelos.includes(nombre)
                    ? { bgcolor: colorDeModelo(nombre, benchmarks), color: "#fff" }
                    : undefined
                }
              />
            ))}
          </Stack>
        </Grid>
      </Grid>

      <Grafico
        alto={380}
        opciones={{
          grid: { left: 8, right: 24, top: 34, bottom: 8, containLabel: true },
          legend: { top: 0, type: "scroll", textStyle: { fontSize: 11 } },
          xAxis: {
            type: "category",
            data: periodos,
            ...ejeBase,
            splitLine: { show: false },
            axisLabel: { ...ejeBase.axisLabel, interval: 11 },
            boundaryGap: false,
          },
          yAxis: { type: "value", name: "unidades", ...ejeBase },
          series: [
            {
              type: "line",
              name: "real",
              data: delaSerie.map((fila) => fila.y_real),
              lineStyle: { color: TINTA_MUTED, width: 2 },
              itemStyle: { color: TINTA_MUTED },
              symbol: "none",
              markArea: primerPrueba
                ? {
                    silent: true,
                    itemStyle: { color: "rgba(0,0,0,0.06)" },
                    data: [[{ xAxis: primerPrueba }, { xAxis: periodos.at(-1) }]],
                  }
                : undefined,
              markLine: primerValidacion
                ? {
                    silent: true,
                    symbol: "none",
                    lineStyle: { color: "#888", type: "dashed" },
                    label: { formatter: "validación →", fontSize: 10 },
                    data: [{ xAxis: primerValidacion }],
                  }
                : undefined,
            },
            ...modelos
              .filter((nombre) => modelosDisponibles.includes(nombre))
              .map((nombre) => ({
                type: "line" as const,
                name: nombre,
                data: delaSerie.map((fila) =>
                  fila.bloque === "entrenamiento"
                    ? null
                    : (fila[nombre] as number | null),
                ),
                lineStyle: {
                  color: colorDeModelo(nombre, benchmarks),
                  width: nombre === "motor" ? 2.2 : 1.4,
                },
                itemStyle: { color: colorDeModelo(nombre, benchmarks) },
                symbol: "circle",
                symbolSize: 3,
                connectNulls: false,
              })),
          ],
          tooltip: {
            trigger: "axis",
            valueFormatter: (valor) =>
              typeof valor === "number" ? numero(valor, 1) : "n/d",
          },
        }}
      />

      {metricas.length > 0 && (
        <>
          <Typography variant="subtitle2" sx={{ mt: 2 }} gutterBottom>
            Métricas de esta serie (bloque de prueba)
          </Typography>
          <TablaDatos
            filas={metricas}
            alto={64 + 44 * Math.max(metricas.length, 1)}
            decimales={1}
          />
        </>
      )}
    </Seccion>
  );
}
