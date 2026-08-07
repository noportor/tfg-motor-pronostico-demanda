/**
 * Datos — la inspección (RF-2) interactiva, desde inspeccion.json.
 * De acá salieron los umbrales de inclusión y el N de la muestra.
 */

import { Grid2 as Grid, Typography } from "@mui/material";

import Grafico from "../componentes/Grafico";
import PanelJSON from "../componentes/PanelJSON";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import { useInspeccion } from "../datos/hooks";
import { ejeBase } from "../tema/echarts";
import { ACENTO, TINTA_MUTED, entero, numero } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "datos",
  titulo: "Datos",
  orden: 2,
  descripcion: "Inspección: la evidencia detrás de los criterios de inclusión",
};

export default function Datos() {
  const corrida = useCorridaActiva();
  const { data: datos, isPending, error } = useInspeccion(corrida);

  if (isPending) return <Cargando />;
  if (error) return <ErrorCarga error={error} />;
  if (!datos) return <Aviso>Esta corrida no tiene inspeccion.json.</Aviso>;

  const intermitentes = datos.series_intermitentes_adi_1_32;
  const series = datos.panel_series;

  return (
    <Seccion
      titulo="Inspección de los datos"
      detalle="Comprensión de datos (CRISP-DM). De aquí salen los umbrales de inclusión y el N de la muestra."
    >
      <Grid container spacing={1.5} sx={{ mb: 3 }}>
        <Grid size={{ xs: 6, md: 2.4 }}>
          <TarjetaKPI titulo="SKU" valor={entero(datos.n_sku)} />
        </Grid>
        <Grid size={{ xs: 6, md: 2.4 }}>
          <TarjetaKPI titulo="Canales" valor={entero(datos.n_canales)} />
        </Grid>
        <Grid size={{ xs: 6, md: 2.4 }}>
          <TarjetaKPI titulo="Regionales" valor={entero(datos.n_regionales)} />
        </Grid>
        <Grid size={{ xs: 6, md: 2.4 }}>
          <TarjetaKPI titulo="Combinaciones" valor={entero(datos.n_combinaciones)} />
        </Grid>
        <Grid size={{ xs: 12, md: 2.4 }}>
          <TarjetaKPI
            titulo="Rango"
            valor={`${datos.rango_fechas[0]} → ${datos.rango_fechas[1]}`}
          />
        </Grid>
      </Grid>

      {intermitentes != null && series != null && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          {entero(intermitentes)} de {entero(series)} series (
          {numero((100 * intermitentes) / series, 1)} %) son demanda
          intermitente (ADI ≥ 1,32, Syntetos–Boylan) — la razón de incluir
          Croston entre los modelos.
        </Typography>
      )}

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Typography variant="subtitle2" gutterBottom>
            Estacionalidad agregada — participación de cada mes (%)
          </Typography>
          <Grafico
            alto={280}
            opciones={{
              xAxis: {
                type: "category",
                data: datos.estacionalidad_mensual.map((f) => String(f.mes)),
                name: "Mes",
                ...ejeBase,
                splitLine: { show: false },
              },
              yAxis: { type: "value", name: "%", ...ejeBase },
              series: [
                {
                  type: "bar",
                  data: datos.estacionalidad_mensual.map((f) => f.porcentaje),
                  itemStyle: { color: ACENTO, borderRadius: [3, 3, 0, 0] },
                  barWidth: "60%",
                  // Reparto plano = 100/12: la estacionalidad ES la distancia
                  // a esta línea.
                  markLine: {
                    silent: true,
                    symbol: "none",
                    lineStyle: { color: TINTA_MUTED, type: "dashed" },
                    label: { formatter: "8,33 %", color: TINTA_MUTED },
                    data: [{ yAxis: 100 / 12 }],
                  },
                },
              ],
              tooltip: {
                formatter: (p: any) =>
                  `Mes ${p.name}: <b>${numero(p.value, 2)} %</b>`,
              },
            }}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Typography variant="subtitle2" gutterBottom>
            Volumen por gestión fiscal (unidades)
          </Typography>
          <Grafico
            alto={280}
            opciones={{
              xAxis: {
                type: "category",
                data: datos.volumen_por_gestion.map((f) => String(f.gestion)),
                name: "Gestión",
                ...ejeBase,
                splitLine: { show: false },
              },
              yAxis: { type: "value", ...ejeBase },
              series: [
                {
                  type: "bar",
                  data: datos.volumen_por_gestion.map((f) => f.unidades),
                  itemStyle: { color: ACENTO, borderRadius: [3, 3, 0, 0] },
                  barWidth: "60%",
                },
              ],
              tooltip: {
                formatter: (p: any) =>
                  `Gestión ${p.name}: <b>${entero(p.value)}</b> unidades`,
              },
            }}
          />
        </Grid>
      </Grid>

      <Typography variant="subtitle2" gutterBottom>
        Cuántas combinaciones sobreviven a cada umbral — la evidencia detrás de
        los criterios de inclusión
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, md: 4 }}>
          <Typography variant="caption" color="text.secondary">
            por historial mínimo (meses)
          </Typography>
          <TablaDatos filas={datos.supervivencia_historial} alto={380} />
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Typography variant="caption" color="text.secondary">
            por proporción de meses en cero (percentiles)
          </Typography>
          <TablaDatos
            filas={Object.entries(datos.proporcion_ceros ?? {}).map(
              ([percentil, proporcion]) => ({ percentil, proporcion }),
            )}
            alto={380}
            decimales={3}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Typography variant="caption" color="text.secondary">
            por volumen acumulado (percentiles, unidades)
          </Typography>
          <TablaDatos
            filas={Object.entries(datos.volumen_por_combinacion).map(
              ([percentil, unidades]) => ({ percentil, unidades }),
            )}
            alto={380}
            decimales={1}
          />
        </Grid>
      </Grid>

      <PanelJSON titulo="Reparto por regional y canal" datos={datos.reparto_canal_regional} />
      <PanelJSON titulo="Inspección completa (JSON)" datos={datos} />
    </Seccion>
  );
}
