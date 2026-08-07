import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// El visor se sirve desde nginx junto con /datos/ (el montaje de salidas*/).
// En desarrollo (perfil `dev` del compose), vite proxya /datos al nginx del
// visor para que la SPA vea las mismas corridas con HMR instantáneo.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/datos": {
        target: process.env.VISOR_DATOS_URL ?? "http://visor:80",
        changeOrigin: true,
      },
    },
  },
  build: {
    // ECharts + MUI son grandes; separarlos deja el chunk de la app liviano y
    // el navegador cachea las librerías entre despliegues.
    rollupOptions: {
      output: {
        manualChunks: {
          mui: ["@mui/material", "@mui/x-data-grid", "@mui/icons-material"],
          echarts: ["echarts", "echarts-for-react"],
        },
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
