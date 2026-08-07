/**
 * Comparar corridas — ablaciones lado a lado. Cambiar UNA decisión (--anular)
 * y ver qué movió; el hash de configuración distingue las corridas.
 */

import { useMemo, useState } from "react";
import {
  Alert,
  Autocomplete,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { useQueries } from "@tanstack/react-query";

import Grafico from "../componentes/Grafico";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridas } from "../datos/hooks";
import { leerCsv } from "../datos/csv";
import { urlDeArtefacto } from "../datos/indice";
import { ejeBase } from "../tema/echarts";
import { COLOR_ROL, numero } from "../tema/paleta";
import { METRICAS, type Agregado, type ClaveMetrica, type FilaResumen } from "../contrato";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "comparar",
  titulo: "Comparar",
  orden: 7,
  descripcion: "Corridas y ablaciones lado a lado, con Δ",
};

export default function Comparar() {
  const corridas = useCorridas();
  const [elegidas, setElegidas] = useState<string[]>([]);
  const [metrica, setMetrica] = useState<ClaveMetrica>("mase");
  const [agregado, setAgregado] = useState<Agregado>("mediana");

  const nombres = useMemo(
    () => corridas.data?.map((c) => c.nombre) ?? [],
    [corridas.data],
  );
  const activas = elegidas.length >= 2 ? elegidas : nombres.slice(0, 2);

  const resumenes = useQueries({
    queries: activas.map((nombre) => ({
      queryKey: [nombre, "resumen_metricas.csv"],
      queryFn: () =>
        leerCsv<FilaResumen>(urlDeArtefacto(nombre, "resumen_metricas.csv")),
      staleTime: Infinity,
      retry: 1,
    })),
  });

  const columna = `${metrica}_${agregado}`;

  const { filasPivote, protagonistas } = useMemo(() => {
    const porModelo = new Map<string, Record<string, unknown>>();
    resumenes.forEach((resultado, indice) => {
      const corrida = activas[indice];
      for (const fila of resultado.data ?? []) {
        const entrada = porModelo.get(fila.modelo) ?? { modelo: fila.modelo };
        entrada[corrida] = Number(fila[columna]);
        porModelo.set(fila.modelo, entrada);
      }
    });
    const base = activas[0];
    const filas = [...porModelo.values()]
      .map((fila) => {
        const conDelta = { ...fila };
        for (const otra of activas.slice(1)) {
          const valorBase = Number(fila[base]);
          const valorOtra = Number(fila[otra]);
          if (!Number.isNaN(valorBase) && !Number.isNaN(valorOtra)) {
            conDelta[`Δ ${otra}`] = valorOtra - valorBase;
          }
        }
        return conDelta;
      })
      .sort((a, b) => Number(a[base] ?? Infinity) - Number(b[base] ?? Infinity));

    const protagonistas = filas.filter(
      (fila) => fila.modelo === "motor" || fila.modelo === "lightgbm",
    );
    return { filasPivote: filas, protagonistas };
  }, [resumenes, activas, columna]);

  if (corridas.isPending) return <Cargando />;
  if (corridas.error) return <ErrorCarga error={corridas.error} />;
  if (nombres.length < 2) {
    return (
      <Aviso>
        Hace falta más de una corrida para comparar. Lanzá una ablación con
        --anular (por ejemplo modelos.motor_regla=mae_mas_bias) y aparece acá
        sola.
      </Aviso>
    );
  }

  const infoMetrica = METRICAS.find((m) => m.clave === metrica)!;
  const etiquetas = new Map(corridas.data!.map((c) => [c.nombre, c.etiqueta]));

  return (
    <Seccion
      titulo="Comparar corridas"
      detalle="Cada directorio salidas*/ es una corrida; las ablaciones aparecen solas. Δ = corrida − base (negativo = menos error que la base)."
    >
      <Stack direction="row" spacing={2} sx={{ mb: 2, flexWrap: "wrap" }}>
        <Autocomplete
          multiple
          size="small"
          options={nombres}
          getOptionLabel={(nombre) => etiquetas.get(nombre) ?? nombre}
          value={activas}
          onChange={(_, valor) => setElegidas(valor)}
          renderInput={(parametros) => (
            <TextField {...parametros} label="Corridas (la primera es la base)" />
          )}
          sx={{ minWidth: 420, flexGrow: 1 }}
        />
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

      {activas.some((_, i) => resumenes[i]?.error) && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Alguna corrida no tiene resumen_metricas.csv.
        </Alert>
      )}

      <TablaDatos filas={filasPivote} alto={520} decimales={3} />

      {protagonistas.length > 0 && (
        <>
          <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
            Los protagonistas, corrida por corrida
          </Typography>
          <Grafico
            alto={64 * activas.length + 80}
            opciones={{
              legend: { top: 0 },
              grid: { left: 8, right: 60, top: 34, bottom: 8, containLabel: true },
              xAxis: {
                type: "value",
                name: `${infoMetrica.titulo} — ${agregado}`,
                nameLocation: "middle",
                nameGap: 28,
                ...ejeBase,
              },
              yAxis: {
                type: "category",
                data: activas,
                ...ejeBase,
                splitLine: { show: false },
              },
              series: protagonistas.map((fila) => ({
                name: String(fila.modelo),
                type: "bar" as const,
                // Pares [x, y] explícitos: con eje categórico un escalar es
                // ambiguo y ECharts puede no dibujar la marca.
                data: activas.map((nombre) => [
                  Number(fila[nombre] ?? NaN),
                  nombre,
                ]),
                itemStyle: {
                  color:
                    fila.modelo === "motor"
                      ? COLOR_ROL["Motor (propuesta)"]
                      : COLOR_ROL.LightGBM,
                  borderRadius: [0, 3, 3, 0],
                },
                label: {
                  show: true,
                  position: "right" as const,
                  // p.value es el par [x, y]: el número está en la posición 0.
                  formatter: (p: any) =>
                    numero(p.value[0], metrica === "mase" ? 3 : 1),
                },
              })),
              tooltip: {
                formatter: (p: any) =>
                  `${p.name} · ${p.seriesName}: <b>${numero(p.value, 3)}</b>`,
              },
            }}
          />
        </>
      )}
    </Seccion>
  );
}
