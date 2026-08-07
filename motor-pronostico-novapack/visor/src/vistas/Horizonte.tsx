/**
 * Horizonte — la curva de degradación D(h) del protocolo multihorizonte.
 * A un paso, un modelo que proyecta línea recta nunca queda expuesto; esta
 * vista muestra quién sostiene el horizonte de planificación (RN-4, protocolo
 * separado: la Tabla 8 y el contraste siguen siendo a un paso).
 */

import { useMemo, useState } from "react";
import {
  Grid2 as Grid,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";

import Grafico from "../componentes/Grafico";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import { benchmarksDe, useCsvGenerico, useManifiesto } from "../datos/hooks";
import { ejeBase } from "../tema/echarts";
import { colorDeModelo, entero, numero, rolDe } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "horizonte",
  titulo: "Horizonte",
  orden: 6.5,
  descripcion: "Degradación D(h) con orígenes rodantes (multihorizonte)",
};

interface FilaHorizonte {
  modelo: string;
  horizonte: number;
  wmape_val: number;
  bias_val: number;
  D: number;
  n_observaciones: number;
  n_series: number;
  n_origenes: number;
}

type Metrica = "D" | "wmape_val" | "bias_val";

const ETIQUETAS: Record<Metrica, string> = {
  D: "D = WMAPE val. + |Bias val.| (%)",
  wmape_val: "WMAPE valorizado (%)",
  bias_val: "Bias valorizado (%)",
};

export default function Horizonte() {
  const corrida = useCorridaActiva();
  const manifiesto = useManifiesto(corrida);
  const tabla = useCsvGenerico(corrida, "multihorizonte_horizonte.csv");
  const [metrica, setMetrica] = useState<Metrica>("D");

  const benchmarks = benchmarksDe(manifiesto.data);
  const notas = manifiesto.data?.resultados?.multihorizonte;

  const filas = useMemo(
    () => (tabla.data ?? []) as unknown as FilaHorizonte[],
    [tabla.data],
  );

  const { horizontes, series } = useMemo(() => {
    const porModelo = new Map<string, FilaHorizonte[]>();
    filas.forEach((fila) => {
      const lista = porModelo.get(fila.modelo) ?? [];
      lista.push(fila);
      porModelo.set(fila.modelo, lista);
    });
    const hs = [...new Set(filas.map((fila) => fila.horizonte))].sort(
      (a, b) => a - b,
    );
    // Orden estable: destacados primero (motor al final para que quede ARRIBA
    // en el z-order de ECharts), resto alfabético — el color ya codifica el ROL.
    const nombres = [...porModelo.keys()].sort();
    const destacados = ["motor", "lightgbm", benchmarks.promedioMovil, benchmarks.naive];
    const ordenados = [
      ...nombres.filter((n) => !destacados.includes(n)),
      ...destacados.filter((n) => nombres.includes(n)).reverse(),
    ];
    return {
      modelos: ordenados,
      horizontes: hs,
      series: ordenados.map((nombre) => {
        const esDestacado = destacados.includes(nombre);
        const porH = new Map(
          (porModelo.get(nombre) ?? []).map((fila) => [fila.horizonte, fila]),
        );
        return {
          type: "line" as const,
          name: nombre,
          symbol: esDestacado ? "circle" : "none",
          symbolSize: 5,
          lineStyle: {
            width: nombre === "motor" ? 3 : esDestacado ? 2 : 1,
            color: colorDeModelo(nombre, benchmarks),
            opacity: esDestacado ? 1 : 0.45,
          },
          itemStyle: { color: colorDeModelo(nombre, benchmarks) },
          emphasis: { focus: "series" as const },
          data: hs.map((h) => {
            const fila = porH.get(h);
            return fila ? [h, Number(fila[metrica])] : [h, null];
          }),
        };
      }),
    };
  }, [filas, benchmarks, metrica]);

  // Recorte declarado del eje (misma regla que la Figura 5): las variantes
  // _gf divergen a cientos de % y aplastarían la banda donde se decide la
  // comparación. Los destacados se ven SIEMPRE completos; los valores exactos
  // de todo lo demás están en la tabla.
  const limiteEje = useMemo(() => {
    if (metrica === "bias_val" || filas.length === 0) return undefined;
    const valores = filas
      .map((fila) => Number(fila[metrica]))
      .filter((v) => Number.isFinite(v))
      .sort((a, b) => a - b);
    const p90 = valores[Math.min(valores.length - 1, Math.floor(0.9 * valores.length))];
    const destacados = ["motor", "lightgbm", benchmarks.promedioMovil, benchmarks.naive];
    const maxDestacado = Math.max(
      ...filas
        .filter((fila) => destacados.includes(fila.modelo))
        .map((fila) => Number(fila[metrica])),
      0,
    );
    return Math.ceil(Math.max(p90, 1.1 * maxDestacado));
  }, [filas, metrica, benchmarks]);

  if (tabla.isPending || manifiesto.isPending) return <Cargando />;
  if (tabla.error) return <ErrorCarga error={tabla.error} />;
  if (filas.length === 0) {
    return (
      <Aviso>
        Esta corrida no tiene evaluación multihorizonte (etapa desactivada o
        sin maestro de costos).
      </Aviso>
    );
  }

  const origenesPorH = new Map<number, number>();
  filas.forEach((fila) => {
    origenesPorH.set(
      fila.horizonte,
      Math.max(origenesPorH.get(fila.horizonte) ?? 0, fila.n_origenes),
    );
  });
  const dGlobal = Object.entries(notas?.D_global_pct ?? {}).sort(
    (a, b) => a[1] - b[1],
  );

  return (
    <Seccion
      titulo="Degradación con el horizonte"
      detalle="Orígenes rodantes sobre el bloque de prueba: cada mes se re-proyecta todo el horizonte, como opera la empresa. La selección del motor queda CONGELADA de validación (RN-2). Protocolo separado del principal (RN-4): la Tabla 8 sigue siendo a un paso."
    >
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Orígenes"
            valor={entero(notas?.origenes?.length)}
            ayuda="Cada mes del bloque de prueba es un origen; h=1 los agrega todos, el horizonte máximo se apoya en uno solo."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Horizonte máximo"
            valor={`${entero(notas?.horizonte_maximo)} meses`}
            ayuda="El objetivo operativo es 18; el histórico permite medir hasta donde hay realidad con la que comparar."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Mejor D global"
            valor={dGlobal.length ? dGlobal[0][0] : "n/d"}
            detalle={
              dGlobal.length ? `${numero(dGlobal[0][1], 1)} %` : undefined
            }
            ayuda="Todos los horizontes y orígenes agregados, valorizado por costo."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Re-entrenos LightGBM"
            valor={entero(notas?.reentrenos_lightgbm)}
            ayuda="Uno por origen, con el número de árboles congelado del ajuste base (la parada temprana no se repite)."
          />
        </Grid>
      </Grid>

      <ToggleButtonGroup
        exclusive
        size="small"
        value={metrica}
        onChange={(_, valor: Metrica | null) => valor && setMetrica(valor)}
        sx={{ mb: 1 }}
      >
        <ToggleButton value="D">D (decisión)</ToggleButton>
        <ToggleButton value="wmape_val">WMAPE val.</ToggleButton>
        <ToggleButton value="bias_val">Bias val.</ToggleButton>
      </ToggleButtonGroup>

      <Grafico
        alto={420}
        opciones={{
          grid: { left: 8, right: 24, top: 40, bottom: 8, containLabel: true },
          legend: {
            top: 0,
            type: "scroll",
            textStyle: { fontSize: 11 },
          },
          xAxis: {
            type: "category",
            name: "h (meses desde el origen)",
            nameLocation: "middle",
            nameGap: 28,
            data: horizontes.map(String),
            ...ejeBase,
            splitLine: { show: false },
            boundaryGap: false,
          },
          yAxis: {
            type: "value",
            name: ETIQUETAS[metrica],
            nameLocation: "middle",
            nameGap: 44,
            max: limiteEje,
            ...ejeBase,
          },
          series: series.map((s) => ({
            ...s,
            data: s.data.map(([h, v]) => [String(h), v]),
          })),
          tooltip: {
            trigger: "axis",
            valueFormatter: (valor) =>
              typeof valor === "number" ? `${numero(valor, 1)} %` : "n/d",
          },
        }}
      />
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 3 }}>
        Orígenes que agrega cada horizonte:{" "}
        {horizontes
          .map((h) => `h=${h}: ${entero(origenesPorH.get(h))}`)
          .join(" · ")}
        . Los horizontes largos se apoyan en pocos orígenes y se leen con esa
        cautela. Color por ROL: azul motor, naranja LightGBM, gris oscuro
        benchmarks declarados, gris claro el resto.
        {limiteEje !== undefined &&
          ` Eje recortado en ${entero(limiteEje)} % (percentil 90; las variantes _gf divergen — valores completos en la tabla).`}
      </Typography>

      <Typography variant="subtitle2" gutterBottom>
        Tabla completa — modelo × horizonte
      </Typography>
      <TablaDatos
        filas={filas.map((fila) => ({
          modelo: fila.modelo,
          rol: rolDe(fila.modelo, benchmarks),
          horizonte: fila.horizonte,
          "WMAPE val (%)": Number(numero(fila.wmape_val, 1).replace(",", ".")),
          "Bias val (%)": Number(numero(fila.bias_val, 1).replace(",", ".")),
          "D (%)": Number(numero(fila.D, 1).replace(",", ".")),
          series: fila.n_series,
          "orígenes": fila.n_origenes,
          observaciones: fila.n_observaciones,
        }))}
        alto={520}
        decimales={1}
      />
    </Seccion>
  );
}
