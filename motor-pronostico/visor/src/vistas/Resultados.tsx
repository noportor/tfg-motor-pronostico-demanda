/**
 * Resultados — el dashboard. Todo número mostrado sale de un archivo de la
 * corrida; el visor reagrega para visualizar, nunca calcula resultados nuevos.
 * Filtros 100 % en el cliente: cambiar métrica o modelos es un setState.
 */

import { useMemo, useState } from "react";
import {
  Box,
  Chip,
  Grid2 as Grid,
  Stack,
  Tab,
  Tabs,
  ToggleButton,
  ToggleButtonGroup,
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
  useCsvGenerico,
  useErroresPorSerie,
  useManifiesto,
  useResumenMetricas,
  useTabla8,
} from "../datos/hooks";
import { agruparPor, percentil } from "../datos/csv";
import { urlDeArtefacto } from "../datos/indice";
import { ejeBase } from "../tema/echarts";
import { EJE, colorDeModelo, entero, numero } from "../tema/paleta";
import { METRICAS, type Agregado, type ClaveMetrica } from "../contrato";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "resultados",
  titulo: "Resultados",
  orden: 3,
  descripcion: "Tabla 8, comparación por métrica y distribución por serie",
};

export default function Resultados() {
  const corrida = useCorridaActiva();
  const resumen = useResumenMetricas(corrida);
  const tabla8 = useTabla8(corrida);
  const valorizada = useCsvGenerico(corrida, "tabla_valorizada.csv");
  const manifiesto = useManifiesto(corrida);
  const errores = useErroresPorSerie(corrida);

  // Galería por DESCUBRIMIENTO: toda figura del catálogo que la corrida haya
  // emitido aparece sola (mismo principio aditivo que el resto del visor).
  const FIGURAS = useMemo(
    () =>
      (manifiesto.data?.salidas ?? [])
        .map((salida) => salida.archivo)
        .filter((nombre) => nombre.endsWith(".png"))
        .sort(),
    [manifiesto.data],
  );

  const [metrica, setMetrica] = useState<ClaveMetrica>("mae");
  const [agregado, setAgregado] = useState<Agregado>("mediana");
  const [metricaSerie, setMetricaSerie] = useState<ClaveMetrica>("mae");
  const [figura, setFigura] = useState(0);

  const benchmarks = benchmarksDe(manifiesto.data);

  // --- Percentiles por modelo (client-side, memoizado: filtrar es gratis) ---
  const percentiles = useMemo(() => {
    if (!errores.data) return [];
    const grupos = agruparPor(errores.data, (fila) => fila.modelo);
    return [...grupos.entries()]
      .map(([modelo, filas]) => {
        const valores = filas
          .map((fila) => fila[metricaSerie])
          .filter((v): v is number => typeof v === "number" && !Number.isNaN(v));
        return {
          modelo,
          p10: percentil(valores, 0.1),
          p25: percentil(valores, 0.25),
          p50: percentil(valores, 0.5),
          p75: percentil(valores, 0.75),
          p90: percentil(valores, 0.9),
        };
      })
      .sort((a, b) => a.p50 - b.p50);
  }, [errores.data, metricaSerie]);

  if (resumen.isPending || manifiesto.isPending) return <Cargando />;
  if (resumen.error) return <ErrorCarga error={resumen.error} />;
  if (!resumen.data) return null;

  const infoMetrica = METRICAS.find((m) => m.clave === metrica)!;
  const columna = `${metrica}_${agregado}`;
  const filasBarras = resumen.data.map((fila) => ({
    modelo: fila.modelo,
    valor: Number(fila[columna] ?? NaN),
  }));

  const porModelo = new Map(resumen.data.map((fila) => [fila.modelo, fila]));
  const motor = porModelo.get("motor");
  const benchmark = porModelo.get(benchmarks.promedioMovil);
  const mejora =
    motor && benchmark
      ? (100 * (Number(benchmark.mae_mediana) - Number(motor.mae_mediana))) /
        Number(benchmark.mae_mediana)
      : null;
  const acierto = manifiesto.data?.resultados?.motor?.acierto;
  const mejorMase = [...resumen.data].sort(
    (a, b) => Number(a.mase_mediana) - Number(b.mase_mediana),
  )[0];

  return (
    <Seccion
      titulo="Resultados"
      detalle="Bloque de prueba: la última gestión cerrada. Menor es mejor."
    >
      <Grid container spacing={1.5} sx={{ mb: 3 }}>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Series evaluadas (N)"
            valor={entero(Number(resumen.data[0]?.series ?? 0))}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Mejor MASE mediano"
            valor={String(mejorMase?.modelo ?? "n/d")}
            detalle={numero(Number(mejorMase?.mase_mediana), 3)}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo={`Motor vs ${benchmarks.promedioMovil} (MAE mediano)`}
            valor={mejora != null ? `−${numero(mejora, 1)} %` : "n/d"}
            ayuda="Reducción del MAE mediano del motor frente al benchmark declarado."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Acierto del motor"
            valor={acierto ? `${numero(100 * acierto.tasa_acierto, 1)} %` : "n/d"}
            detalle={
              acierto ? `azar: ${numero(100 * acierto.azar_esperado, 0)} %` : undefined
            }
            ayuda="Cuántas veces el elegido en validación fue el mejor en prueba."
          />
        </Grid>
      </Grid>

      {/* --- Comparación por métrica (instantáneo: estado del cliente) ------ */}
      <Stack direction="row" spacing={2} sx={{ mb: 1.5, flexWrap: "wrap" }}>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={metrica}
          onChange={(_, valor) => valor && setMetrica(valor)}
        >
          {METRICAS.map((m) => (
            <ToggleButton key={m.clave} value={m.clave}>
              {m.clave.toUpperCase()}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={agregado}
          onChange={(_, valor) => valor && setAgregado(valor)}
        >
          <ToggleButton value="mediana">mediana</ToggleButton>
          <ToggleButton value="media">media</ToggleButton>
        </ToggleButtonGroup>
      </Stack>
      <BarrasPorModelo
        filas={filasBarras}
        tituloValor={`${infoMetrica.titulo} — ${agregado}`}
        benchmarks={benchmarks}
        decimales={metrica === "mase" ? 3 : 1}
        signo={metrica === "bias"}
      />
      {metrica === "mape" && (
        <Typography variant="caption" color="text.secondary">
          El MAPE excluye los meses con demanda real cero (~
          {numero(Number(resumen.data[0]?.pct_excluido_del_mape_medio), 0)} % de
          las observaciones); con muchos ceros, leer MASE.
        </Typography>
      )}

      {/* --- Métrica de decisión (valorizada) -------------------------------- */}
      {valorizada.data && valorizada.data.length > 0 && (
        <Box sx={{ my: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Métrica de decisión — D = WMAPE valorizado + |Bias valorizado|
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
            Las unidades difieren entre SKUs: la única agregación con sentido
            físico entre productos es la valorizada (|error| × costo unitario).
            D captura el objetivo del pronóstico — errar poco y sin sesgo
            sistemático — y es la regla con la que el motor selecciona. Menor
            es mejor.
          </Typography>
          <TablaDatos
            filas={valorizada.data}
            alto={64 + 40 * Math.min(valorizada.data.length, 11)}
            decimales={1}
          />
        </Box>
      )}

      {/* --- Tabla 8 -------------------------------------------------------- */}
      <Box sx={{ my: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Tabla 8 — como va al documento (niveles en unidades: referenciales;
          comparar con MASE/MAPE o con la tabla valorizada)
        </Typography>
        {tabla8.data && <TablaDatos filas={tabla8.data} alto={480} />}
      </Box>

      {/* --- Distribución por serie ----------------------------------------- */}
      <Typography variant="subtitle2" gutterBottom>
        Distribución del error por serie — lo que el promedio esconde
      </Typography>
      <Stack direction="row" spacing={1} sx={{ mb: 1 }}>
        {METRICAS.map((m) => (
          <Chip
            key={m.clave}
            size="small"
            label={m.clave.toUpperCase()}
            color={metricaSerie === m.clave ? "primary" : "default"}
            variant={metricaSerie === m.clave ? "filled" : "outlined"}
            onClick={() => setMetricaSerie(m.clave)}
          />
        ))}
      </Stack>
      {errores.isPending ? (
        <Cargando />
      ) : (
        <>
          <Grafico
            alto={Math.max(220, 34 * percentiles.length + 60)}
            opciones={{
              xAxis: { type: "value", name: metricaSerie.toUpperCase(), ...ejeBase },
              yAxis: {
                type: "category",
                data: percentiles.map((p) => p.modelo),
                ...ejeBase,
                splitLine: { show: false },
              },
              series: [
                {
                  // Boxplot con percentiles PRECALCULADOS: [p10,p25,p50,p75,p90]
                  // por modelo — once filas al navegador, no miles.
                  type: "boxplot",
                  data: percentiles.map((p) => ({
                    value: [p.p10, p.p25, p.p50, p.p75, p.p90],
                    itemStyle: {
                      color: `${colorDeModelo(p.modelo, benchmarks)}33`,
                      borderColor: colorDeModelo(p.modelo, benchmarks),
                      borderWidth: 1.5,
                    },
                  })),
                  boxWidth: [12, 22],
                },
              ],
              tooltip: {
                formatter: (parametros: any) => {
                  const [, p10, p25, p50, p75, p90] = parametros.value;
                  return (
                    `<b>${parametros.name}</b><br/>` +
                    `p90: ${numero(p90)}<br/>p75: ${numero(p75)}<br/>` +
                    `<b>mediana: ${numero(p50)}</b><br/>` +
                    `p25: ${numero(p25)}<br/>p10: ${numero(p10)}`
                  );
                },
              },
            }}
          />
          <Typography variant="caption" color="text.secondary">
            Caja: p25–p75 · bigotes: p10–p90 · línea: mediana. Percentiles sobre
            las {entero((errores.data?.length ?? 0) / Math.max(percentiles.length, 1))}{" "}
            series de cada modelo; el detalle completo está en
            errores_por_serie.csv (vista Artefactos).
          </Typography>
        </>
      )}

      {/* --- Figuras del documento ------------------------------------------ */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="subtitle2">
          Figuras del documento — tal como van a la tesis
        </Typography>
        <Tabs
          value={Math.min(figura, Math.max(FIGURAS.length - 1, 0))}
          onChange={(_, v) => setFigura(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ mb: 1 }}
        >
          {FIGURAS.map((nombre) => (
            <Tab key={nombre} label={nombre.replace(".png", "")} />
          ))}
        </Tabs>
        <Box
          component="img"
          src={urlDeArtefacto(
            corrida,
            FIGURAS[Math.min(figura, Math.max(FIGURAS.length - 1, 0))] ?? "",
          )}
          alt={FIGURAS[Math.min(figura, Math.max(FIGURAS.length - 1, 0))] ?? ""}
          sx={{
            maxWidth: "100%",
            border: "1px solid",
            borderColor: EJE,
            borderRadius: 1,
            bgcolor: "#fff",
          }}
        />
      </Box>
    </Seccion>
  );
}
