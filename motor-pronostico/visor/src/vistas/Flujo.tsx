/**
 * Flujo — el pipeline de punta a punta, renderizado GENÉRICAMENTE desde
 * flujo.json: esta vista no sabe cuántas etapas hay ni cómo se llaman. Una
 * etapa nueva registrada con reporte.etapa(...) aparece sola (el contrato).
 */

import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Chip,
  Grid2 as Grid,
  Stack,
  Step,
  StepContent,
  StepLabel,
  Stepper,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

import PanelJSON from "../componentes/PanelJSON";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import { useFlujo } from "../datos/hooks";
import { entero, numero } from "../tema/paleta";
import type { Etapa, MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "flujo",
  titulo: "Flujo",
  orden: 1,
  descripcion: "El experimento etapa por etapa",
};

function primeraMagnitud(dic: Record<string, unknown>): string | null {
  for (const [clave, valor] of Object.entries(dic)) {
    if (typeof valor === "number") {
      return `${clave.replace(/_/g, " ")}: ${entero(valor)}`;
    }
  }
  return null;
}

function Conteos({ conteos }: { conteos: Record<string, unknown> }) {
  const escalares: Record<string, unknown> = {};
  const bloques: { titulo: string; filas: Record<string, unknown>[] }[] = [];

  for (const [clave, valor] of Object.entries(conteos)) {
    if (Array.isArray(valor) && valor.length > 0 && typeof valor[0] === "object") {
      bloques.push({ titulo: clave, filas: valor as Record<string, unknown>[] });
    } else if (valor !== null && typeof valor === "object") {
      bloques.push({
        titulo: clave,
        filas: Object.entries(valor as Record<string, unknown>).map(
          ([k, v]) => ({ clave: k, valor: v }),
        ),
      });
    } else {
      escalares[clave] = valor;
    }
  }

  return (
    <Stack spacing={2}>
      {Object.keys(escalares).length > 0 && (
        <TablaDatos
          filas={Object.entries(escalares).map(([k, v]) => ({
            conteo: k.replace(/_/g, " "),
            valor: v,
          }))}
          alto={Math.min(300, 60 + 36 * Object.keys(escalares).length)}
        />
      )}
      {bloques.map((bloque) => (
        <Box key={bloque.titulo}>
          <Typography variant="caption" color="text.secondary">
            {bloque.titulo.replace(/_/g, " ")}
          </Typography>
          <TablaDatos
            filas={bloque.filas}
            alto={Math.min(340, 60 + 36 * bloque.filas.length)}
          />
        </Box>
      ))}
    </Stack>
  );
}

function DetalleEtapa({ etapa }: { etapa: Etapa }) {
  return (
    <Stack spacing={2}>
      <Grid container spacing={1}>
        {[...Object.entries(etapa.entrada), ...Object.entries(etapa.salida)].map(
          ([clave, valor]) => (
            <Grid key={clave} size={{ xs: 6, sm: 3 }}>
              <TarjetaKPI
                titulo={clave.replace(/_/g, " ")}
                valor={typeof valor === "number" ? entero(valor) : String(valor)}
              />
            </Grid>
          ),
        )}
      </Grid>
      {Object.keys(etapa.decisiones).length > 0 && (
        <PanelJSON titulo="Decisiones" datos={etapa.decisiones} />
      )}
      {Object.keys(etapa.conteos).length > 0 && <Conteos conteos={etapa.conteos} />}
      {etapa.artefactos.length > 0 && (
        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
          {etapa.artefactos.map((a) => (
            <Chip key={a} size="small" variant="outlined" label={a} />
          ))}
        </Stack>
      )}
      {etapa.notas.map((nota) => (
        <Typography key={nota} variant="caption" color="text.secondary">
          ↳ {nota}
        </Typography>
      ))}
    </Stack>
  );
}

export default function Flujo() {
  const corrida = useCorridaActiva();
  const { data: flujo, isPending, error } = useFlujo(corrida);

  if (isPending) return <Cargando />;
  if (error) return <ErrorCarga error={error} />;
  if (!flujo) {
    return (
      <Aviso>
        Esta corrida no tiene flujo.json (es anterior al contrato). Regenerála
        con `python main.py ejecutar`.
      </Aviso>
    );
  }

  return (
    <Seccion
      titulo="Flujo del experimento"
      detalle="De los datos crudos al contraste estadístico. Cada etapa declara qué entra, qué sale y qué decidió."
    >
      <Stepper orientation="vertical" nonLinear activeStep={-1}>
        {flujo.etapas.map((etapa) => {
          const magnitud =
            primeraMagnitud(etapa.salida) ?? primeraMagnitud(etapa.entrada);
          return (
            <Step key={etapa.id} expanded>
              <StepLabel
                optional={
                  <Typography variant="caption" color="text.secondary">
                    {[
                      etapa.rf,
                      magnitud,
                      etapa.duracion_s != null
                        ? `${numero(etapa.duracion_s, 1)} s`
                        : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </Typography>
                }
              >
                <Typography fontWeight={600}>{etapa.titulo}</Typography>
              </StepLabel>
              <StepContent>
                <Accordion disableGutters sx={{ my: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="body2">Detalle de la etapa</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <DetalleEtapa etapa={etapa} />
                  </AccordionDetails>
                </Accordion>
              </StepContent>
            </Step>
          );
        })}
      </Stepper>
    </Seccion>
  );
}
