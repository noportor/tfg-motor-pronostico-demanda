import { Accordion, AccordionDetails, AccordionSummary, Box } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

interface Props {
  titulo: string;
  datos: unknown;
}

export default function PanelJSON({ titulo, datos }: Props) {
  return (
    <Accordion disableGutters>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>{titulo}</AccordionSummary>
      <AccordionDetails>
        <Box
          component="pre"
          sx={{
            m: 0,
            p: 1.5,
            fontSize: 12,
            bgcolor: "#f6f5f2",
            borderRadius: 1,
            overflow: "auto",
            maxHeight: 420,
          }}
        >
          {JSON.stringify(datos, null, 2)}
        </Box>
      </AccordionDetails>
    </Accordion>
  );
}
