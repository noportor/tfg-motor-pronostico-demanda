/**
 * Shell del visor: barra superior con el selector de corrida, navegación
 * lateral con las vistas descubiertas del registro, y ruteo /#/:corrida/:vista.
 */

import {
  Alert,
  AppBar,
  Box,
  Chip,
  Drawer,
  FormControl,
  List,
  ListItemButton,
  ListItemText,
  MenuItem,
  Select,
  Toolbar,
  Typography,
} from "@mui/material";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from "react-router-dom";

import { Cargando, ErrorCarga } from "./componentes/Estado";
import { useCorridas } from "./datos/hooks";
import { VISTAS } from "./vistas/registro";

const ANCHO_MENU = 210;

function Shell() {
  const { corrida = "" } = useParams<{ corrida: string }>();
  const navegar = useNavigate();
  const ubicacion = useLocation();
  const { data: corridas, isPending, error } = useCorridas();

  if (isPending) return <Cargando />;
  if (error) return <ErrorCarga error={error} />;
  if (!corridas || corridas.length === 0) {
    return (
      <Alert severity="warning" sx={{ m: 4 }}>
        No hay ninguna corrida en salidas*/. Generá una con: docker compose run
        --rm tfg python main.py ejecutar
      </Alert>
    );
  }

  const activa = corridas.find((c) => c.nombre === corrida);
  if (!activa) {
    return <Navigate to={`/${corridas[0].nombre}/flujo`} replace />;
  }

  const vistaActual = ubicacion.pathname.split("/")[2] ?? VISTAS[0].ruta;

  return (
    <Box sx={{ display: "flex" }}>
      <AppBar
        position="fixed"
        color="default"
        elevation={0}
        sx={{ zIndex: (t) => t.zIndex.drawer + 1, borderBottom: "1px solid", borderColor: "divider", bgcolor: "background.paper" }}
      >
        <Toolbar variant="dense" sx={{ gap: 2 }}>
          <Typography variant="subtitle1" fontWeight={700} sx={{ flexShrink: 0 }}>
            Motor de pronóstico — NOVAPACK
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          {activa.anulaciones.length > 0 && (
            <Chip
              size="small"
              color="primary"
              variant="outlined"
              label={`Ablación: ${activa.anulaciones.join("; ")}`}
            />
          )}
          <FormControl size="small" sx={{ minWidth: 260 }}>
            <Select
              value={corrida}
              onChange={(evento) =>
                navegar(`/${evento.target.value}/${vistaActual}`)
              }
            >
              {corridas.map((c) => (
                <MenuItem key={c.nombre} value={c.nombre}>
                  {c.etiqueta}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Toolbar>
      </AppBar>

      <Drawer
        variant="permanent"
        sx={{
          width: ANCHO_MENU,
          flexShrink: 0,
          "& .MuiDrawer-paper": { width: ANCHO_MENU, boxSizing: "border-box" },
        }}
      >
        <Toolbar variant="dense" />
        <List dense>
          {VISTAS.map((v) => (
            <ListItemButton
              key={v.ruta}
              selected={vistaActual === v.ruta}
              onClick={() => navegar(`/${corrida}/${v.ruta}`)}
            >
              <ListItemText primary={v.titulo} />
            </ListItemButton>
          ))}
        </List>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, p: 3, minWidth: 0 }}>
        <Toolbar variant="dense" />
        <Routes>
          {VISTAS.map((v) => (
            <Route key={v.ruta} path={v.ruta} element={<v.Componente />} />
          ))}
          <Route path="*" element={<Navigate to={VISTAS[0].ruta} replace />} />
        </Routes>
      </Box>
    </Box>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/:corrida/*" element={<Shell />} />
      <Route path="*" element={<PortadaRedireccion />} />
    </Routes>
  );
}

function PortadaRedireccion() {
  const { data: corridas, isPending, error } = useCorridas();
  if (isPending) return <Cargando />;
  if (error) return <ErrorCarga error={error} />;
  if (!corridas || corridas.length === 0) {
    return (
      <Alert severity="warning" sx={{ m: 4 }}>
        No hay ninguna corrida en salidas*/.
      </Alert>
    );
  }
  return <Navigate to={`/${corridas[0].nombre}/flujo`} replace />;
}
