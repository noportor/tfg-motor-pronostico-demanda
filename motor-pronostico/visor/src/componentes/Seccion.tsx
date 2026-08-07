import { Box, Typography } from "@mui/material";
import type { ReactNode } from "react";

interface Props {
  titulo: string;
  detalle?: string;
  children: ReactNode;
}

export default function Seccion({ titulo, detalle, children }: Props) {
  return (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        {titulo}
      </Typography>
      {detalle && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          {detalle}
        </Typography>
      )}
      {children}
    </Box>
  );
}
