/** Informe .txt de la corrida, plegado; se baja recién al abrirlo. */

import { useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

import { Cargando, ErrorCarga } from "./Estado";
import { useTexto } from "../datos/hooks";

interface Props {
  corrida: string;
  archivo: string;
  titulo: string;
}

export default function PanelTexto({ corrida, archivo, titulo }: Props) {
  const [abierto, setAbierto] = useState(false);
  const texto = useTexto(abierto ? corrida : "", archivo);

  return (
    <Accordion
      disableGutters
      expanded={abierto}
      onChange={(_, expandido) => setAbierto(expandido)}
      sx={{ mt: 2 }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>{titulo}</AccordionSummary>
      <AccordionDetails>
        {texto.isPending && abierto && <Cargando />}
        {texto.error && <ErrorCarga error={texto.error} />}
        {texto.data && (
          <Box
            component="pre"
            sx={{
              m: 0,
              p: 1.5,
              fontSize: 12,
              bgcolor: "#f6f5f2",
              borderRadius: 1,
              overflow: "auto",
              maxHeight: 480,
            }}
          >
            {texto.data}
          </Box>
        )}
      </AccordionDetails>
    </Accordion>
  );
}
