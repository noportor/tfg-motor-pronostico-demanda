/**
 * Motor — la propuesta de la tesis: qué eligió, cuánto acertó.
 * Fuentes: notas del manifiesto + seleccion_motor.csv (evidencia por serie,
 * completa: el DataGrid virtualiza las 2.226 filas sin despeinarse).
 */

import { useMemo, useState } from "react";
import {
  Autocomplete,
  Grid2 as Grid,
  TextField,
  Typography,
} from "@mui/material";

import BarrasPorModelo from "../componentes/BarrasPorModelo";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { Aviso, Cargando, ErrorCarga } from "../componentes/Estado";
import { useCorridaActiva } from "../corrida";
import {
  benchmarksDe,
  useManifiesto,
  useSeleccionMotor,
} from "../datos/hooks";
import { entero, numero } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "motor",
  titulo: "Motor",
  orden: 5,
  descripcion: "La propuesta: selección por serie y tasa de acierto",
};

export default function Motor() {
  const corrida = useCorridaActiva();
  const manifiesto = useManifiesto(corrida);
  const seleccion = useSeleccionMotor(corrida);
  const [filtro, setFiltro] = useState<string[]>([]);

  const notas = manifiesto.data?.resultados?.motor;
  const benchmarks = benchmarksDe(manifiesto.data);

  const elegibles = useMemo(
    () =>
      seleccion.data
        ? [...new Set(seleccion.data.map((f) => f.modelo_elegido))].sort()
        : [],
    [seleccion.data],
  );

  const visibles = useMemo(() => {
    if (!seleccion.data) return [];
    if (filtro.length === 0) return seleccion.data;
    const conjunto = new Set(filtro);
    return seleccion.data.filter((f) => conjunto.has(f.modelo_elegido));
  }, [seleccion.data, filtro]);

  if (manifiesto.isPending) return <Cargando />;
  if (manifiesto.error) return <ErrorCarga error={manifiesto.error} />;
  if (!notas) return <Aviso>Esta corrida no tiene notas del motor.</Aviso>;

  const acierto = notas.acierto;

  return (
    <Seccion
      titulo="Motor de selección"
      detalle="Elige por serie el modelo con menor error EN VALIDACIÓN y lo aplica en prueba (RN-2). Nunca mira el bloque de prueba para decidir."
    >
      <Grid container spacing={1.5} sx={{ mb: 1.5 }}>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI titulo="Regla de selección" valor={notas.regla} />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Empates en validación"
            valor={entero(notas.empates)}
            ayuda="Se desempata alfabéticamente — declarado, no silencioso."
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Tasa de acierto en prueba"
            valor={`${numero(100 * acierto.tasa_acierto, 1)} %`}
            detalle={`azar: ${numero(100 * acierto.azar_esperado, 0)} %`}
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <TarjetaKPI
            titulo="Exceso de MAE mediano"
            valor={numero(acierto.exceso_mae_mediano, 3)}
            ayuda="Cuánto MAE de más se paga por no haber elegido el óptimo de prueba (el costo del winner's curse)."
          />
        </Grid>
      </Grid>

      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 3 }}>
        La brecha entre la tasa de acierto y el 100 % es el <i>winner's curse</i>:
        elegir el mínimo entre varios candidatos sobre 12 meses sobreestima al
        ganador. Es un resultado en sí mismo y se reporta tal cual (§12).
      </Typography>

      <Typography variant="subtitle2" gutterBottom>
        Qué modelo ganó la validación, y en cuántas series
      </Typography>
      <BarrasPorModelo
        filas={Object.entries(notas.reparto).map(([modelo, valor]) => ({
          modelo,
          valor,
        }))}
        tituloValor="series en que fue elegido"
        benchmarks={benchmarks}
        decimales={0}
        ascendente={false}
      />

      <Typography variant="subtitle2" sx={{ mt: 3 }} gutterBottom>
        La selección, serie por serie — seleccion_motor.csv
      </Typography>
      <Autocomplete
        multiple
        size="small"
        options={elegibles}
        value={filtro}
        onChange={(_, valor) => setFiltro(valor)}
        renderInput={(parametros) => (
          <TextField {...parametros} label="Filtrar por modelo elegido" />
        )}
        sx={{ maxWidth: 520, mb: 1.5 }}
      />
      {seleccion.isPending ? (
        <Cargando />
      ) : (
        <>
          <Typography variant="caption" color="text.secondary">
            {entero(visibles.length)} series. Las columnas de modelo son el MAE
            de cada candidato EN VALIDACIÓN; el motor eligió el mínimo.
          </Typography>
          <TablaDatos filas={visibles} alto={560} decimales={3} paginado={50} />
        </>
      )}
    </Seccion>
  );
}
