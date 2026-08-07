/**
 * Corrida — la ficha de trazabilidad (RN-6): qué datos, qué configuración,
 * qué código y qué versiones produjeron estos números.
 */

import { Alert, Grid2 as Grid, Typography } from "@mui/material";

import PanelJSON from "../componentes/PanelJSON";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import { useManifiesto } from "../datos/hooks";
import { numero } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "corrida",
  titulo: "Corrida",
  orden: 9,
  descripcion: "Trazabilidad: hashes, commit, versiones (RN-6)",
};

export default function Corrida() {
  const corrida = useCorridaActiva();
  const { data: manifiesto, isPending, error } = useManifiesto(corrida);

  if (isPending) return <Cargando />;
  if (error) return <ErrorCarga error={error} />;
  if (!manifiesto) return null;

  const anulaciones = manifiesto.configuracion.contenido._anulaciones ?? [];

  return (
    <Seccion
      titulo="Corrida"
      detalle="Cada número del documento se rastrea hasta esta ficha (RN-6)."
    >
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Duración de la corrida"
            valor={`${numero(manifiesto.duracion_segundos, 0)} s`}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Datos (SHA-256)"
            valor={(manifiesto.datos.sha256 ?? "n/d").slice(0, 12)}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Configuración (SHA-256)"
            valor={manifiesto.configuracion.sha256.slice(0, 12)}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Commit"
            valor={(manifiesto.codigo.commit ?? "n/d").slice(0, 12)}
          />
        </Grid>
      </Grid>

      {manifiesto.codigo.arbol_limpio === false && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          El árbol de trabajo tenía cambios sin commitear al ejecutar. Los
          números de esta corrida no son citables hasta regenerarla desde un
          commit limpio.
        </Alert>
      )}
      {anulaciones.length > 0 && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Esta corrida es una <b>ablación</b>: {anulaciones.join("; ")}. El hash
          de configuración la distingue de la principal.
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid size={{ xs: 12, md: 5 }}>
          <Typography variant="subtitle2" gutterBottom>
            Versiones cargadas en la corrida
          </Typography>
          <TablaDatos
            filas={Object.entries(manifiesto.dependencias).map(
              ([paquete, version]) => ({ paquete, versión: version }),
            )}
            alto={440}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 7 }}>
          <Typography variant="subtitle2" gutterBottom>
            Salidas producidas (con su hash)
          </Typography>
          <TablaDatos
            filas={manifiesto.salidas.map((s) => ({
              archivo: s.archivo,
              bytes: s.bytes,
              sha256: `${s.sha256.slice(0, 16)}…`,
            }))}
            alto={440}
          />
        </Grid>
      </Grid>

      <PanelJSON
        titulo="Configuración completa de la corrida"
        datos={manifiesto.configuracion.contenido}
      />
    </Seccion>
  );
}
