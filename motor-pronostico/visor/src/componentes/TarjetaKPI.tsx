import { Paper, Stack, Tooltip, Typography } from "@mui/material";

interface Props {
  titulo: string;
  valor: string;
  detalle?: string;
  ayuda?: string;
}

export default function TarjetaKPI({ titulo, valor, detalle, ayuda }: Props) {
  const contenido = (
    <Paper sx={{ p: 2, height: "100%" }}>
      <Stack spacing={0.5}>
        <Typography variant="caption" color="text.secondary">
          {titulo}
        </Typography>
        <Typography variant="h5" fontWeight={600}>
          {valor}
        </Typography>
        {detalle && (
          <Typography variant="caption" color="text.secondary">
            {detalle}
          </Typography>
        )}
      </Stack>
    </Paper>
  );
  return ayuda ? <Tooltip title={ayuda}>{contenido}</Tooltip> : contenido;
}
