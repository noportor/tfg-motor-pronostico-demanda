import { Alert, Box, CircularProgress } from "@mui/material";

export function Cargando() {
  return (
    <Box sx={{ display: "flex", justifyContent: "center", p: 6 }}>
      <CircularProgress size={28} />
    </Box>
  );
}

export function ErrorCarga({ error }: { error: unknown }) {
  return (
    <Alert severity="error" sx={{ my: 2 }}>
      {error instanceof Error ? error.message : String(error)}
    </Alert>
  );
}

export function Aviso({ children }: { children: string }) {
  return (
    <Alert severity="info" sx={{ my: 2 }}>
      {children}
    </Alert>
  );
}
