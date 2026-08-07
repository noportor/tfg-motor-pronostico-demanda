/**
 * Contraste — Wilcoxon, Friedman y Nemenyi con su guía de lectura.
 * Con miles de series cualquier diferencia sale significativa: se lee PRIMERO
 * el tamaño del efecto, después el p-valor.
 */

import { useMemo } from "react";
import { Grid2 as Grid, Typography } from "@mui/material";

import Grafico from "../componentes/Grafico";
import PanelTexto from "../componentes/PanelTexto";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import { benchmarksDe, useManifiesto, useNemenyi } from "../datos/hooks";
import { ejeBase } from "../tema/echarts";
import {
  ACENTO,
  NEUTRO_CLARO,
  colorDeModelo,
  numero,
} from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "contraste",
  titulo: "Contraste",
  orden: 6,
  descripcion: "Wilcoxon, Friedman + Nemenyi, tamaños del efecto",
};

const ALFA = 0.05;

function formatearP(p: number): string {
  if (p < 1e-16) return "< 1e-16";
  if (p < 0.001) return p.toExponential(2);
  return numero(p, 4);
}

export default function Contraste() {
  const corrida = useCorridaActiva();
  const manifiesto = useManifiesto(corrida);
  const nemenyi = useNemenyi(corrida);

  const wilcoxon = manifiesto.data?.resultados?.wilcoxon ?? [];
  const friedman = manifiesto.data?.resultados?.friedman;
  const benchmarks = benchmarksDe(manifiesto.data);
  const alfa =
    Number(
      (manifiesto.data?.configuracion?.contenido?.pruebas as
        | Record<string, unknown>
        | undefined)?.alfa,
    ) || ALFA;

  const rangos = useMemo(() => {
    if (!friedman) return [];
    return Object.entries(friedman.rangos_medios)
      .map(([modelo, rango]) => ({ modelo, rango }))
      .sort((a, b) => a.rango - b.rango);
  }, [friedman]);

  const celdas = useMemo(() => {
    if (!nemenyi.data) return { modelos: [] as string[], datos: [] as number[][] };
    const modelos = nemenyi.data.map((fila) => String(fila.modelo));
    const datos: number[][] = [];
    nemenyi.data.forEach((fila, i) => {
      modelos.forEach((otro, j) => {
        const p = Number(fila[otro]);
        if (!Number.isNaN(p)) datos.push([j, i, p]);
      });
    });
    return { modelos, datos };
  }, [nemenyi.data]);

  if (manifiesto.isPending) return <Cargando />;
  if (manifiesto.error) return <ErrorCarga error={manifiesto.error} />;
  if (!friedman) return <Aviso>Esta corrida no tiene notas del contraste.</Aviso>;

  return (
    <Seccion
      titulo="Contraste estadístico"
      detalle="Con miles de series cualquier diferencia sale significativa: la lectura válida es el TAMAÑO DEL EFECTO primero, el p-valor después."
    >
      <Typography variant="subtitle2" gutterBottom>
        Wilcoxon de rangos con signo — pareado, unilateral (Demšar 2006)
      </Typography>
      <TablaDatos
        filas={wilcoxon.map((r) => ({
          propuesto: r.propuesto,
          referencia: r.referencia,
          r: Number(numero(r.r, 3).replace(",", ".")),
          "series a favor (%)": Number(
            ((100 * r.gana_propuesto) / r.n_pares).toFixed(1),
          ),
          p: formatearP(r.p),
          veredicto: r.significativo ? "significativo" : "no significativo",
        }))}
        alto={64 + 44 * Math.max(wilcoxon.length, 1)}
        decimales={3}
      />
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 3 }}>
        r = Z/√N (Cohen: 0,1 pequeño · 0,3 mediano · 0,5 grande). El signo
        negativo indica MENOS error del modelo propuesto.
      </Typography>

      <Typography variant="subtitle2" gutterBottom>
        Friedman + post hoc de Nemenyi — los once modelos a la vez
      </Typography>
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI titulo="Chi² de Friedman" valor={numero(friedman.chi2, 0)} />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI titulo="p" valor={formatearP(friedman.p)} />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="W de Kendall"
            valor={numero(friedman.kendall_w, 3)}
            ayuda="Concordancia del ranking entre series. No comparable entre análisis con distinto número de modelos."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Diferencia crítica (CD)"
            valor={numero(friedman.diferencia_critica, 3)}
            ayuda="Dos modelos con rangos medios a menos de esta distancia no son distinguibles al nivel declarado."
          />
        </Grid>
      </Grid>

      <Grafico
        alto={30 * rangos.length + 70}
        opciones={{
          grid: { left: 8, right: 30, top: 8, bottom: 8, containLabel: true },
          xAxis: {
            type: "value",
            name: "Rango medio de Friedman (1 = mejor)",
            nameLocation: "middle",
            nameGap: 28,
            min: (valor: { min: number }) => Math.floor(valor.min - 0.5),
            ...ejeBase,
          },
          yAxis: {
            type: "category",
            data: rangos.map((r) => r.modelo),
            inverse: true,
            ...ejeBase,
            splitLine: { show: false },
          },
          series: [
            {
              // Puntos, no barras: el rango medio es una POSICIÓN, no una
              // magnitud que se acumule desde cero. Datos como pares planos
              // [x, y] y color por callback a nivel de serie: el objeto
              // {value, itemStyle} por punto no renderizaba sobre eje
              // categórico en esta versión de ECharts.
              type: "scatter",
              symbolSize: 12,
              data: rangos.map((r) => [r.rango, r.modelo]),
              itemStyle: {
                color: (p: any) =>
                  colorDeModelo(rangos[p.dataIndex]?.modelo ?? "", benchmarks),
              },
            },
          ],
          tooltip: {
            formatter: (p: any) =>
              `${p.name}: rango medio <b>${numero(p.value, 2)}</b>`,
          },
        }}
      />

      {celdas.modelos.length > 0 && (
        <>
          <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
            Matriz de Nemenyi — qué pares son distinguibles
          </Typography>
          <Grafico
            alto={34 * celdas.modelos.length + 90}
            opciones={{
              grid: { left: 8, right: 20, top: 8, bottom: 60, containLabel: true },
              xAxis: {
                type: "category",
                data: celdas.modelos,
                axisLabel: { ...ejeBase.axisLabel, rotate: 40 },
                axisLine: ejeBase.axisLine,
                axisTick: ejeBase.axisTick,
                splitLine: { show: false },
              },
              yAxis: {
                type: "category",
                data: celdas.modelos,
                ...ejeBase,
                splitLine: { show: false },
              },
              visualMap: {
                show: false,
                // Dos clases, no un gradiente: la pregunta del post hoc es
                // binaria (¿distinguibles al nivel α?); el p exacto, en tooltip.
                pieces: [
                  { lt: alfa, color: ACENTO },
                  { gte: alfa, color: NEUTRO_CLARO },
                ],
                dimension: 2,
              },
              series: [
                {
                  type: "heatmap",
                  data: celdas.datos,
                  itemStyle: { borderColor: "#ffffff", borderWidth: 2 },
                },
              ],
              tooltip: {
                formatter: (p: any) => {
                  const [x, y, valorP] = p.value;
                  return (
                    `${celdas.modelos[y]} vs ${celdas.modelos[x]}<br/>` +
                    `p = <b>${formatearP(valorP)}</b> — ` +
                    (valorP < alfa ? "distinguibles" : "no distinguibles")
                  );
                },
              },
            }}
          />
          <Typography variant="caption" color="text.secondary">
            Azul: p &lt; {numero(alfa, 2)} (distinguibles). El rango studentizado
            ya controla el error por familia: no se aplica otra corrección
            encima.
          </Typography>
        </>
      )}

      <PanelTexto
        corrida={corrida}
        archivo="pruebas_estadisticas.txt"
        titulo="Informe completo del contraste (como va al documento)"
      />
    </Seccion>
  );
}
