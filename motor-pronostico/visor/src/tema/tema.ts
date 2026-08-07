/** Tema MUI del visor: claro, sobrio, con el acento validado de la paleta. */

import { createTheme } from "@mui/material/styles";
import { esES as coreEsES } from "@mui/material/locale";
import { esES as gridEsES } from "@mui/x-data-grid/locales";

import { ACENTO, PLANO_PAGINA, TINTA, TINTA_SECUNDARIA } from "./paleta";

export const tema = createTheme(
  {
    palette: {
      mode: "light",
      primary: { main: ACENTO },
      background: { default: PLANO_PAGINA, paper: "#ffffff" },
      text: { primary: TINTA, secondary: TINTA_SECUNDARIA },
    },
    typography: {
      fontFamily:
        'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif',
      h6: { fontWeight: 600 },
    },
    components: {
      MuiPaper: {
        defaultProps: { elevation: 0 },
        styleOverrides: {
          root: { border: "1px solid rgba(11, 11, 11, 0.10)" },
        },
      },
    },
  },
  coreEsES,
  gridEsES,
);
