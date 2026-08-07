/**
 * Artefactos — todo lo que la corrida produjo, sin vista a medida.
 * El mecanismo de aditividad «cero código»: cualquier archivo nuevo que el
 * pipeline escriba aparece acá solo, listado desde el autoindex de nginx.
 */

import { useMemo, useState } from "react";
import {
  Box,
  Chip,
  Grid2 as Grid,
  Stack,
  Typography,
} from "@mui/material";

import PanelJSON from "../componentes/PanelJSON";
import PanelTexto from "../componentes/PanelTexto";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import { Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import {
  useArtefactos,
  useCsvGenerico,
  useFlujo,
  useManifiesto,
} from "../datos/hooks";
import { urlDeArtefacto } from "../datos/indice";
import { EJE, entero } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "artefactos",
  titulo: "Artefactos",
  orden: 8,
  descripcion: "Todo lo que produjo la corrida; lo nuevo aparece solo",
};

export default function Artefactos() {
  const corrida = useCorridaActiva();
  const listado = useArtefactos(corrida);
  const flujo = useFlujo(corrida);
  const manifiesto = useManifiesto(corrida);
  const [csvActivo, setCsvActivo] = useState("");

  const { csvs, pngs, txts, declarados } = useMemo(() => {
    const archivos = (listado.data ?? []).filter((e) => e.type === "file");
    const declarados = new Set<string>();
    for (const etapa of flujo.data?.etapas ?? []) {
      for (const artefacto of etapa.artefactos) declarados.add(artefacto);
    }
    return {
      csvs: archivos.filter((e) => e.name.endsWith(".csv")).map((e) => e.name),
      pngs: archivos.filter((e) => e.name.endsWith(".png")).map((e) => e.name),
      txts: archivos.filter((e) => e.name.endsWith(".txt")).map((e) => e.name),
      declarados,
    };
  }, [listado.data, flujo.data]);

  const csvElegido = csvActivo || csvs[0] || "";
  const tabla = useCsvGenerico(corrida, csvElegido);
  const nuevos = csvs.filter((nombre) => !declarados.has(nombre));

  if (listado.isPending) return <Cargando />;
  if (listado.error) return <ErrorCarga error={listado.error} />;

  return (
    <Seccion
      titulo="Artefactos de la corrida"
      detalle={`Todo lo que hay en ${corrida}/. Lo nuevo aparece solo.`}
    >
      <Typography variant="subtitle2" gutterBottom>
        Tablas
      </Typography>
      {nuevos.length > 0 && (
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
          Sin etapa que los declare (probablemente recién agregados):{" "}
          {nuevos.join(" · ")}
        </Typography>
      )}
      <Stack direction="row" spacing={1} sx={{ mb: 1.5, flexWrap: "wrap", rowGap: 1 }}>
        {csvs.map((nombre) => (
          <Chip
            key={nombre}
            size="small"
            label={nombre}
            color={csvElegido === nombre ? "primary" : "default"}
            variant={csvElegido === nombre ? "filled" : "outlined"}
            onClick={() => setCsvActivo(nombre)}
          />
        ))}
      </Stack>
      {tabla.isPending && csvElegido !== "" ? (
        <Cargando />
      ) : tabla.data ? (
        <>
          <Typography variant="caption" color="text.secondary">
            {entero(tabla.data.length)} filas ×{" "}
            {entero(Object.keys(tabla.data[0] ?? {}).length)} columnas
          </Typography>
          <TablaDatos filas={tabla.data} alto={480} paginado={50} />
        </>
      ) : null}

      {pngs.length > 0 && (
        <>
          <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
            Figuras
          </Typography>
          <Grid container spacing={2}>
            {pngs.map((nombre) => (
              <Grid key={nombre} size={{ xs: 12, lg: 6 }}>
                <Typography variant="caption" color="text.secondary">
                  {nombre}
                </Typography>
                <Box
                  component="img"
                  src={urlDeArtefacto(corrida, nombre)}
                  alt={nombre}
                  sx={{
                    width: "100%",
                    border: "1px solid",
                    borderColor: EJE,
                    borderRadius: 1,
                    bgcolor: "#fff",
                  }}
                />
              </Grid>
            ))}
          </Grid>
        </>
      )}

      {txts.length > 0 && (
        <>
          <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
            Informes de texto
          </Typography>
          {txts.map((nombre) => (
            <PanelTexto key={nombre} corrida={corrida} archivo={nombre} titulo={nombre} />
          ))}
        </>
      )}

      <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
        Documentos JSON
      </Typography>
      {flujo.data && <PanelJSON titulo="flujo.json" datos={flujo.data} />}
      {manifiesto.data && (
        <PanelJSON titulo="manifiesto.json" datos={manifiesto.data} />
      )}
    </Seccion>
  );
}
