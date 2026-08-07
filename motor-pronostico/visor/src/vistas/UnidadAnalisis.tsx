/**
 * Unidad de análisis — la evidencia de POR QUÉ la serie es SKU-canal-regional.
 * Tres patas: (1) la impone la decisión (unidades no agregables entre SKUs),
 * (2) independencia estadística dentro del SKU, (3) la alternativa ensayada
 * (repartir desde el SKU agregado) no domina. Estructura medida SOLO sobre
 * entrenamiento; contrafactual evaluado SOLO sobre validación.
 */

import { useMemo } from "react";
import { Grid2 as Grid, Typography } from "@mui/material";

import Grafico from "../componentes/Grafico";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import { useCsvGenerico, useManifiesto } from "../datos/hooks";
import { ejeBase } from "../tema/echarts";
import { ACENTO, NEUTRO_CLARO, TINTA_MUTED, entero, numero } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "unidad",
  titulo: "Unidad de análisis",
  orden: 2.5,
  descripcion: "Por qué la serie es SKU-canal-regional: estructura + contrafactual",
};

interface FilaSku {
  sku: string;
  n_combos: number;
  deriva_participacion_pp: number;
  correlacion_mediana: number;
  n_cuadrantes: number;
}

/** Redondeo para celdas de tabla: número crudo, jamás string con locale. */
const redondear = (valor: number, decimales = 1): number =>
  Number.isFinite(valor) ? Number(valor.toFixed(decimales)) : NaN;

export default function UnidadAnalisis() {
  const corrida = useCorridaActiva();
  const manifiesto = useManifiesto(corrida);
  const estructura = useCsvGenerico(corrida, "unidad_estructura_sku.csv");

  const notas = manifiesto.data?.resultados?.unidad_analisis;
  const filas = useMemo(
    () => (estructura.data ?? []) as unknown as FilaSku[],
    [estructura.data],
  );

  const distribucionCombos = useMemo(() => {
    const conteo = new Map<string, number>();
    filas.forEach((fila) => {
      const clave = fila.n_combos >= 10 ? "10+" : String(fila.n_combos);
      conteo.set(clave, (conteo.get(clave) ?? 0) + 1);
    });
    const orden = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"];
    return orden
      .filter((clave) => conteo.has(clave))
      .map((clave) => ({ combos: clave, skus: conteo.get(clave) ?? 0 }));
  }, [filas]);

  const distribucionCorrelacion = useMemo(() => {
    const bordes = [-1, -0.5, 0, 0.25, 0.5, 0.75, 1.0001];
    const etiquetas = ["−1 a −0,5", "−0,5 a 0", "0 a 0,25", "0,25 a 0,5",
                       "0,5 a 0,75", "0,75 a 1"];
    const conteo = new Array(etiquetas.length).fill(0);
    filas.forEach((fila) => {
      // Un SKU sin correlación medida (single-combo o sin pares con solape)
      // llega como celda vacía → null. Number(null) === 0: contarlo sería
      // FABRICAR evidencia de baja correlación. Se descarta explícitamente.
      const bruto = fila.correlacion_mediana as unknown;
      if (bruto === null || bruto === undefined || bruto === "") return;
      const r = Number(bruto);
      if (!Number.isFinite(r)) return;
      for (let i = 0; i < etiquetas.length; i += 1) {
        if (r >= bordes[i] && r < bordes[i + 1]) {
          conteo[i] += 1;
          break;
        }
      }
    });
    return etiquetas.map((etiqueta, i) => ({ rango: etiqueta, skus: conteo[i] }));
  }, [filas]);

  if (manifiesto.isPending || estructura.isPending) return <Cargando />;
  if (manifiesto.error) return <ErrorCarga error={manifiesto.error} />;
  if (estructura.error) return <ErrorCarga error={estructura.error} />;
  if (!notas) {
    return <Aviso>Esta corrida no tiene la etapa de unidad de análisis.</Aviso>;
  }

  const contrafactual = notas.contrafactual ?? [];

  return (
    <Seccion
      titulo="Unidad de análisis: la serie SKU-canal-regional"
      detalle="La decisión que sostiene a las demás. Pata 1 (dominio): la demanda es independiente por combinación y las unidades de medida difieren ENTRE SKUs — no hay agregación física por encima del SKU. Las patas 2 y 3 son las medibles, y se miden acá."
    >
      <Typography variant="subtitle2" gutterBottom>
        Pata 2 — Independencia estadística dentro del SKU (solo entrenamiento)
      </Typography>
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="SKUs en varias combinaciones"
            valor={`${numero(100 * notas.proporcion_multicombo, 1)} %`}
            detalle={`${entero(notas.skus_multicombo)} de ${entero(notas.skus_totales)} · mediana ${entero(notas.mediana_combos_en_multi)} combinaciones`}
            ayuda="Si casi todo SKU viviera en una sola combinación, la discusión sería vacía."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Deriva de participación"
            valor={`${numero(notas.deriva_participacion_pp_media, 1)} pp/mes`}
            ayuda="Cuánto se mueve, mes a mes, la parte de cada combinación dentro de su SKU. Un reparto estable sería prorrateable desde arriba; esta deriva dice que el reparto es una diana móvil."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Correlación intra-SKU"
            valor={numero(notas.correlacion_mediana, 2)}
            detalle={`interanual ${numero(notas.correlacion_yoy_mediana, 2)} · ${numero(100 * notas.proporcion_pares_bajo_05, 0)} % de pares < 0,5`}
            ayuda="Mediana de los pares de combinaciones del mismo SKU con ≥24 meses de solape. La versión interanual descuenta la estacionalidad común (dos combos pueden 'correlacionar' solo porque ambas venden en campaña)."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Regímenes ADI-CV² mixtos"
            valor={`${numero(
              (100 * notas.skus_con_regimenes_mixtos) /
                Math.max(notas.skus_multi_clasificables, 1),
              0,
            )} %`}
            detalle={`${entero(notas.skus_con_regimenes_mixtos)} de ${entero(notas.skus_multi_clasificables)} SKUs clasificables`}
            ayuda="SKUs cuyas combinaciones caen en cuadrantes DISTINTOS de Syntetos-Boylan (suave/errática/intermitente/lumpy): la demanda cambia de naturaleza según el canal y la regional."
          />
        </Grid>
      </Grid>

      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Typography variant="caption" color="text.secondary">
            ¿En cuántas combinaciones vive cada SKU?
          </Typography>
          <Grafico
            alto={260}
            opciones={{
              xAxis: {
                type: "category",
                data: distribucionCombos.map((f) => f.combos),
                name: "combinaciones",
                ...ejeBase,
                splitLine: { show: false },
              },
              yAxis: { type: "value", name: "SKUs", ...ejeBase },
              series: [
                {
                  type: "bar",
                  // Mismo patrón que BarrasPorModelo/Datos (los pares [x, y]
                  // no renderizan en barras sobre eje categórico en esta
                  // versión de ECharts): valores alineados al eje, estilo por
                  // ítem para atenuar la barra de los single-combo.
                  data: distribucionCombos.map((f) => ({
                    value: f.skus,
                    itemStyle: {
                      color: f.combos === "1" ? NEUTRO_CLARO : TINTA_MUTED,
                      borderRadius: [3, 3, 0, 0],
                    },
                  })),
                  barWidth: "60%",
                },
              ],
              tooltip: {
                formatter: (p: any) =>
                  `${p.name} combinaciones: <b>${entero(p.value)}</b> SKUs`,
              },
            }}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Typography variant="caption" color="text.secondary">
            Correlación mediana entre combinaciones del mismo SKU (por SKU)
          </Typography>
          <Grafico
            alto={260}
            opciones={{
              xAxis: {
                type: "category",
                data: distribucionCorrelacion.map((f) => f.rango),
                name: "r (niveles)",
                axisLabel: { ...ejeBase.axisLabel, rotate: 25 },
                axisLine: ejeBase.axisLine,
                axisTick: ejeBase.axisTick,
                nameTextStyle: ejeBase.nameTextStyle,
                splitLine: { show: false },
              },
              yAxis: { type: "value", name: "SKUs", ...ejeBase },
              series: [
                {
                  type: "bar",
                  data: distribucionCorrelacion.map((f) => f.skus),
                  itemStyle: {
                    color: TINTA_MUTED,
                    borderRadius: [3, 3, 0, 0],
                  },
                  barWidth: "60%",
                },
              ],
              tooltip: {
                formatter: (p: any) =>
                  `r en ${p.name}: <b>${entero(p.value)}</b> SKUs`,
              },
            }}
          />
        </Grid>
      </Grid>

      <Grid container spacing={1.5} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, md: 6 }}>
          <TarjetaKPI
            titulo="SKUs con modelos ganadores DISTINTOS por combinación"
            valor={`${numero(100 * notas.proporcion_multiganador, 1)} %`}
            detalle={`${entero(notas.skus_multiganador)} de ${entero(notas.skus_cohorte_multi)} SKUs multi-combinación de la cohorte (selección del motor, validación — RN-2)`}
            ayuda="Si el mejor modelo fuera propiedad del SKU, elegir por combinación sería burocracia. No lo es."
          />
        </Grid>
      </Grid>

      <Typography variant="subtitle2" gutterBottom>
        Pata 3 — El mismo modelo, arriba contra abajo (validación, valorizado)
      </Typography>
      {contrafactual.length === 0 ? (
        <Aviso>
          Sin maestro de costos no hay contrafactual valorizado en esta corrida.
        </Aviso>
      ) : (
        <>
          <TablaDatos
            filas={contrafactual.map((fila) => ({
              modelo: fila.modelo,
              enfoque:
                fila.enfoque === "bottom_up"
                  ? "por combinación (unidad elegida)"
                  : "SKU agregado, repartido",
              // Números CRUDOS redondeados — nunca re-parsear un string
              // formateado con locale (con miles, "1.234,5" → NaN).
              "WMAPE val (%)": redondear(fila.wmape_val),
              "Bias val (%)": redondear(fila.bias_val),
              "D (%)": redondear(fila.D),
              "Δ D top-down": redondear(fila.delta_D_topdown),
              series: fila.series,
              observaciones: fila.observaciones,
            }))}
            alto={64 + 44 * Math.max(contrafactual.length, 1)}
            decimales={1}
          />
          <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
            Mismos modelos (los benchmarks declarados), mismas combinaciones,
            mismos meses de validación, misma valorización — solo cambia el
            nivel al que se pronostica. El reparto usa participaciones de los
            últimos 12 meses <b>cerrados</b> (sin fuga). Δ D positivo = repartir
            desde arriba es peor. Agregar DENTRO del SKU es legítimo (misma
            unidad de medida); por encima del SKU no existe agregación física —
            esa es la pata 1.{" "}
            <span style={{ color: ACENTO }}>
              Conclusión: la alternativa no domina; la unidad elegida no paga
              costo oculto.
            </span>
          </Typography>
        </>
      )}
    </Seccion>
  );
}
