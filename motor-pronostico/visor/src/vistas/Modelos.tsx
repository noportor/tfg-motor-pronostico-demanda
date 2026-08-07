/**
 * Modelos y optimización — qué parámetros eligió cada modelo y a qué costo.
 * Fuentes: etapa `modelos` del flujo, parametros_por_serie.csv,
 * lightgbm_importancias.csv.
 */

import { useMemo, useState } from "react";
import {
  FormControl,
  Grid2 as Grid,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from "@mui/material";

import BarrasPorModelo from "../componentes/BarrasPorModelo";
import Grafico from "../componentes/Grafico";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import {
  benchmarksDe,
  useFlujo,
  useImportancias,
  useManifiesto,
  useParametros,
} from "../datos/hooks";
import { agruparPor } from "../datos/csv";
import { ejeBase } from "../tema/echarts";
import { ACENTO, COLOR_ROL, entero, numero } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "modelos",
  titulo: "Modelos",
  orden: 4,
  descripcion: "Optimización: parámetros por serie, costo, LightGBM",
};

export default function Modelos() {
  const corrida = useCorridaActiva();
  const flujo = useFlujo(corrida);
  const manifiesto = useManifiesto(corrida);
  const parametros = useParametros(corrida);
  const importancias = useImportancias(corrida);
  const [combinacion, setCombinacion] = useState("");

  const benchmarks = benchmarksDe(manifiesto.data);

  const etapaModelos = flujo.data?.etapas.find((e) => e.id === "modelos");
  const tiempos = (etapaModelos?.conteos?.duracion_por_modelo_s ?? {}) as Record<
    string,
    number
  >;
  const respaldos = (etapaModelos?.conteos?.respaldos_en_validacion_y_prueba ??
    {}) as Record<string, number>;

  const combinaciones = useMemo(() => {
    if (!parametros.data) return [];
    const vistas = new Set<string>();
    for (const fila of parametros.data) {
      vistas.add(`${fila.modelo} · ${fila.parametro}`);
    }
    return [...vistas].sort();
  }, [parametros.data]);

  const activa = combinacion || combinaciones[0] || "";

  const distribucion = useMemo(() => {
    if (!parametros.data || !activa) return [];
    const [modelo, parametro] = activa.split(" · ");
    const filtradas = parametros.data.filter(
      (fila) => fila.modelo === modelo && fila.parametro === parametro,
    );
    const grupos = agruparPor(filtradas, (fila) => String(fila.valor));
    return [...grupos.entries()]
      .map(([valor, filas]) => ({ valor: Number(valor), series: filas.length }))
      .sort((a, b) => a.valor - b.valor);
  }, [parametros.data, activa]);

  if (flujo.isPending || manifiesto.isPending) return <Cargando />;
  if (flujo.error) return <ErrorCarga error={flujo.error} />;

  const mejorIteracion = manifiesto.data?.resultados?.lightgbm_mejor_iteracion;
  const top = (importancias.data ?? [])
    .slice()
    .sort((a, b) => b.ganancia - a.ganancia)
    .slice(0, 15)
    .reverse();

  return (
    <Seccion
      titulo="Modelos y optimización"
      detalle="Los parámetros se estiman SOLO con entrenamiento; validación la usa únicamente la parada temprana de LightGBM (RN-2)."
    >
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Typography variant="subtitle2" gutterBottom>
            Tiempo de ajuste por modelo (s)
          </Typography>
          <BarrasPorModelo
            filas={Object.entries(tiempos).map(([modelo, valor]) => ({
              modelo,
              valor,
            }))}
            tituloValor="segundos"
            benchmarks={benchmarks}
            ascendente={false}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Typography variant="subtitle2" gutterBottom>
            Respaldos en validación + prueba
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
            Meses evaluados donde el modelo no pudo pronosticar y se aplicó la
            cascada declarada. Debería ser ~0: si no, hay que decirlo en el
            documento.
          </Typography>
          <TablaDatos
            filas={Object.entries(respaldos).map(([modelo, cantidad]) => ({
              modelo,
              respaldos: cantidad,
            }))}
            alto={360}
          />
        </Grid>
      </Grid>

      <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
        Qué parámetros ganaron, serie por serie
      </Typography>
      <FormControl size="small" sx={{ minWidth: 280, mb: 1.5 }}>
        <InputLabel>Modelo · parámetro</InputLabel>
        <Select
          value={activa}
          label="Modelo · parámetro"
          onChange={(evento) => setCombinacion(evento.target.value)}
        >
          {combinaciones.map((c) => (
            <MenuItem key={c} value={c}>
              {c}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      {parametros.isPending ? (
        <Cargando />
      ) : (
        <>
          <Grafico
            alto={260}
            opciones={{
              xAxis: {
                type: "category",
                data: distribucion.map((d) => numero(d.valor, 2)),
                name: "valor elegido",
                ...ejeBase,
                splitLine: { show: false },
              },
              yAxis: { type: "value", name: "series", ...ejeBase },
              series: [
                {
                  type: "bar",
                  data: distribucion.map((d) => d.series),
                  itemStyle: { color: ACENTO, borderRadius: [3, 3, 0, 0] },
                  barWidth: "60%",
                },
              ],
              tooltip: {
                formatter: (p: any) =>
                  `valor ${p.name}: <b>${entero(p.value)}</b> series`,
              },
            }}
          />
          <Typography variant="caption" color="text.secondary">
            Un α alto = la serie pide reaccionar rápido; α bajo = suavizar. La
            forma de esta distribución es un retrato del portafolio.
          </Typography>
        </>
      )}

      <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
        LightGBM — el brazo de aprendizaje automático
      </Typography>
      {mejorIteracion != null && (
        <Grid container sx={{ mb: 1.5 }}>
          <Grid size={{ xs: 12, md: 4 }}>
            <TarjetaKPI
              titulo="Árboles (parada temprana contra validación)"
              valor={entero(mejorIteracion)}
            />
          </Grid>
        </Grid>
      )}
      {top.length > 0 && (
        <>
          <Grafico
            alto={30 * top.length + 60}
            opciones={{
              grid: { left: 8, right: 60, top: 8, bottom: 8, containLabel: true },
              xAxis: { type: "value", name: "ganancia", ...ejeBase },
              yAxis: {
                type: "category",
                data: top.map((f) => f.feature),
                ...ejeBase,
                splitLine: { show: false },
              },
              series: [
                {
                  type: "bar",
                  data: top.map((f) => f.ganancia),
                  itemStyle: {
                    color: COLOR_ROL.LightGBM,
                    borderRadius: [0, 3, 3, 0],
                  },
                  barWidth: "62%",
                },
              ],
              tooltip: {
                formatter: (p: any) =>
                  `${p.name}: <b>${entero(p.value)}</b>`,
              },
            }}
          />
          <Typography variant="caption" color="text.secondary">
            Importancia por ganancia (reducción de error acumulada). Que dominen
            los rezagos y medias móviles es lo esperable en demanda mensual.
          </Typography>
        </>
      )}
    </Seccion>
  );
}
