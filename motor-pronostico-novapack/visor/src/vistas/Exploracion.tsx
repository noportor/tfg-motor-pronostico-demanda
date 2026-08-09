/**
 * Exploración ML2 — la narrativa de las corridas post-documento (V1–V8 +
 * blindaje), contada en el orden en que la evidencia se construyó.
 *
 * La vista es TRANSVERSAL: no depende de la corrida activa — lee los números
 * directo de los artefactos de las once corridas selladas (nombres fijos de
 * carpeta). Si una corrida no está en el disco, sus números salen «n/d» y la
 * narrativa sigue en pie: el texto cuenta el arco, los artefactos lo
 * respaldan.
 */

import { useMemo } from "react";
import { Grid2 as Grid, Typography } from "@mui/material";

import Grafico from "../componentes/Grafico";
import Seccion from "../componentes/Seccion";
import TablaDatos from "../componentes/TablaDatos";
import TarjetaKPI from "../componentes/TarjetaKPI";
import { useCsvGenerico, useManifiesto } from "../datos/hooks";
import { ejeBase } from "../tema/echarts";
import { ACENTO, TINTA_SECUNDARIA, numero } from "../tema/paleta";
import type { MetaVista } from "./tipos";

export const vista: MetaVista = {
  ruta: "exploracion",
  titulo: "Exploración ML2",
  orden: 6.8,
  descripcion:
    "La narrativa de las corridas post-documento: del mezclador a la conmutación por régimen",
};

// Carpetas fijas de las corridas selladas (gitignoradas: viven en el disco).
const C = {
  v1: "salidas_ml2",
  v2: "salidas_ml2_v2",
  v3: "salidas_ml2_v3",
  v4: "salidas_ml2_v4",
  v5: "salidas_ml2_v5",
  v6: "salidas_ml2_v6",
  v7: "salidas_ml2_v7",
  v8: "salidas_ml2_v8",
  blindaje: "salidas_ml2_blindaje",
  v7b: "salidas_ml2_v7_blindaje",
  v8b: "salidas_ml2_v8_blindaje",
} as const;

const COLOR_G2026 = ACENTO;
const COLOR_G2025 = "#b8551c";

interface FilaValorizada {
  d: number;
  wmape: number;
  bias: number;
}

interface ResumenCorrida {
  unPaso: Map<string, FilaValorizada>;
  horizonte: Map<string, number>;
}

/** Los dos artefactos que la narrativa consume de cada corrida. */
function useResumenCorrida(nombre: string): ResumenCorrida {
  const tabla = useCsvGenerico(nombre, "tabla_valorizada.csv");
  const manifiesto = useManifiesto(nombre);
  return useMemo(() => {
    const unPaso = new Map<string, FilaValorizada>();
    (tabla.data ?? []).forEach((fila) => {
      unPaso.set(String(fila["Modelo"]), {
        d: Number(fila["D = WMAPE + |Bias| (%)"]),
        wmape: Number(fila["WMAPE valorizado (%)"]),
        bias: Number(fila["Bias valorizado (%)"]),
      });
    });
    const horizonte = new Map<string, number>(
      Object.entries(
        manifiesto.data?.resultados?.multihorizonte?.D_global_pct ?? {},
      ).map(([modelo, d]) => [modelo, Number(d)]),
    );
    return { unPaso, horizonte };
  }, [tabla.data, manifiesto.data]);
}

const d1 = (r: ResumenCorrida, modelo: string): number | undefined =>
  r.unPaso.get(modelo)?.d;
const dh = (r: ResumenCorrida, modelo: string): number | undefined =>
  r.horizonte.get(modelo);
const fmt = (v: number | undefined, decimales = 1): string =>
  v === undefined ? "n/d" : numero(v, decimales);

/** El mejor sistema a un paso de una corrida (mínimo de su tabla valorizada). */
function mejorUnPaso(r: ResumenCorrida): { modelo: string; d: number } | null {
  let mejor: { modelo: string; d: number } | null = null;
  r.unPaso.forEach((fila, modelo) => {
    if (Number.isFinite(fila.d) && (mejor === null || fila.d < mejor.d)) {
      mejor = { modelo, d: fila.d };
    }
  });
  return mejor;
}

interface FilaSenal {
  mes: string;
  sesgo_movil_pct: number;
  lambda: number;
}

export default function Exploracion() {
  // Orden FIJO de hooks: una llamada por corrida sellada.
  const v1 = useResumenCorrida(C.v1);
  const v2 = useResumenCorrida(C.v2);
  const v3 = useResumenCorrida(C.v3);
  const v4 = useResumenCorrida(C.v4);
  const v5 = useResumenCorrida(C.v5);
  const v6 = useResumenCorrida(C.v6);
  const v7 = useResumenCorrida(C.v7);
  const v8 = useResumenCorrida(C.v8);
  const blindaje = useResumenCorrida(C.blindaje);
  const v7b = useResumenCorrida(C.v7b);
  const v8b = useResumenCorrida(C.v8b);
  const senal26 = useCsvGenerico(C.v8, "conmutador_senal.csv");
  const senal25 = useCsvGenerico(C.v8b, "conmutador_senal.csv");

  // --- El arco: el mejor sistema a un paso, generación por generación -------
  const arco = useMemo(() => {
    const generaciones: [string, ResumenCorrida][] = [
      ["V1", v1], ["V2", v2], ["V3", v3], ["V4", v4],
      ["V5", v5], ["V6", v6], ["V7", v7], ["V8", v8],
    ];
    return generaciones.map(([nombre, r]) => {
      const mejor = mejorUnPaso(r);
      return { generacion: nombre, modelo: mejor?.modelo ?? "n/d", d: mejor?.d };
    });
  }, [v1, v2, v3, v4, v5, v6, v7, v8]);

  // --- El talón: el sesgo del menú curado en los dos años -------------------
  const MENU_CURADO = ["lightgbm", "chronos", "exp_smooth_opt", "holt_winters", "tft"];
  const sesgosMenu = useMemo(
    () =>
      MENU_CURADO.map((modelo) => ({
        modelo,
        g2026: v5.unPaso.get(modelo)?.bias,
        g2025: blindaje.unPaso.get(modelo)?.bias,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [v5, blindaje],
  );

  // --- La señal del conmutador en ambos años --------------------------------
  const filasSenal26 = (senal26.data ?? []) as unknown as FilaSenal[];
  const filasSenal25 = (senal25.data ?? []) as unknown as FilaSenal[];
  const mesesPrueba = Array.from({ length: 12 }, (_, i) => `m${i + 1}`);

  return (
    <>
      <Seccion
        titulo="La exploración posterior al documento"
        detalle="Once corridas selladas (V1–V8 sobre la prueba g2026; el protocolo desplazado a g2025 como blindaje, con dos gemelas). Cada capítulo enuncia lo que se midió y de qué corrida salen los números: esta vista los lee EN VIVO de los artefactos — elegí la corrida en el selector para auditar cualquiera en profundidad."
      >
        <Grid container spacing={1.5} sx={{ mb: 1 }}>
          <Grid size={{ xs: 6, md: 3 }}>
            <TarjetaKPI
              titulo="Mejor sistema a un paso (g2026)"
              valor={`mezcla_prom ${fmt(d1(v5, "mezcla_prom"))} %`}
              detalle={`récord ${fmt(d1(v6, "mezcla_prom"))} % con calendario (vía sesgo)`}
              ayuda="Promedio simple del menú curado {lightgbm, chronos, SES, HW, TFT}. El campeón individual previo (LightGBM) quedó en 77,0."
            />
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <TarjetaKPI
              titulo="Corona del horizonte"
              valor={`lightgbm ${fmt(dh(v5, "lightgbm"))} %`}
              detalle={`también en g2025: ${fmt(dh(blindaje, "lightgbm"))} %`}
              ayuda="D(h) global con orígenes rodantes. La única afirmación de podio que sobrevive intacta a los DOS años de prueba."
            />
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <TarjetaKPI
              titulo="Mejor en el año de quiebre"
              valor={`mezcla_pond·7 ${fmt(d1(v7b, "mezcla_pond"))} %`}
              detalle="el menú curado cayó a 90,2 ahí"
              ayuda="La póliza estática: el menú con diversidad de sesgo deliberada (V7) rescata ~10 puntos en g2025."
            />
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <TarjetaKPI
              titulo="Mejor peor-caso del horizonte"
              valor={`conmutada ${fmt(Math.max(dh(v8, "mezcla_conmutada") ?? NaN, dh(v8b, "mezcla_conmutada") ?? NaN))} %`}
              detalle="un solo sistema para ambos regímenes"
              ayuda="mezcla_conmutada (V8): detector causal de régimen sobre el sesgo móvil observado; minimax del horizonte entre g2026 y g2025."
            />
          </Grid>
        </Grid>

        <Grafico
          alto={300}
          opciones={{
            grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
            xAxis: {
              type: "category",
              data: arco.map((p) => p.generacion),
              ...ejeBase,
              splitLine: { show: false },
            },
            yAxis: {
              type: "value",
              name: "mejor D a un paso (%)",
              nameLocation: "middle",
              nameGap: 40,
              min: 60,
              ...ejeBase,
            },
            series: [
              {
                type: "line",
                name: "mejor sistema de la generación",
                symbol: "circle",
                symbolSize: 7,
                lineStyle: { width: 2, color: ACENTO },
                itemStyle: { color: ACENTO },
                data: arco.map((p) => p.d ?? null),
                markLine: {
                  silent: true,
                  symbol: "none",
                  lineStyle: { type: "dashed", color: TINTA_SECUNDARIA },
                  label: { formatter: "campeón individual (lightgbm 77,0)", position: "insideEndTop", fontSize: 11 },
                  data: [{ yAxis: 77.0 }],
                },
              },
            ],
            tooltip: {
              trigger: "axis",
              formatter: (params: unknown) => {
                const p = (params as { dataIndex: number }[])[0];
                const punto = arco[p.dataIndex];
                return `${punto.generacion}: ${punto.modelo} · D ${fmt(punto.d)} %`;
              },
            },
          }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
          El mejor sistema de cada generación (prueba g2026, D valorizada a un
          paso). La caída grande es la V4 (combinar en vez de seleccionar); el
          repunte de la V7 es el precio medido de la robustez; la V8 lo
          recupera conmutando.
        </Typography>
      </Seccion>

      <Seccion
        titulo="1 · Nuevos brazos: redes y fundacional (V1–V3)"
        detalle="Cuatro arquitecturas neuronales globales (DLinear, N-HiTS, DeepAR, TFT) y un fundacional cero-shot (Chronos-Bolt) bajo el MISMO protocolo congelado que los once brazos del documento."
      >
        <TablaDatos
          alto={300}
          filas={[
            { brazo: "chronos (calibrado por categoría, V3)", "D un paso (%)": fmt(d1(v3, "chronos")), lectura: "El mejor brazo individual del estudio a un paso: 5 factores de validación convierten al cero-shot en líder. El factor global (V2) sobre-corregía." },
            { brazo: "tft (Tweedie + exógenas, V2)", "D un paso (%)": fmt(d1(v2, "tft")), lectura: "De colapsado (107,0 univariante) a 2.º del horizonte (78,5): la pérdida y las exógenas estáticas lo rescatan." },
            { brazo: "dlinear (control lineal)", "D un paso (%)": fmt(d1(v1, "dlinear")), lectura: "El lineal le gana a las redes profundas univariantes (resultado Zeng): la arquitectura no es el cuello." },
            { brazo: "nhits", "D un paso (%)": fmt(d1(v1, "nhits")), lectura: "Discreto en g2026 — y LÍDER a un paso en el año de quiebre (72,6 en g2025): su sesgo negativo persistente es contrapeso." },
            { brazo: "deepar", "D un paso (%)": fmt(d1(v1, "deepar")), lectura: "Tres verosimilitudes, tres colapsos (sesgo −74 a −98): incompatible con esta demanda; se retira del menú (declarado)." },
          ]}
        />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Las tres lecciones de la etapa: (a) el <b>sesgo de la pérdida domina
          sobre la arquitectura</b> (la lección Tweedie del campeón, replicada);
          (b) la <b>búsqueda de hiperparámetros no transfiere</b> a través del
          quiebre cambiario (V3: ganadores del pliegue con sesgo ~0 dieron −27
          a −37 en prueba — maldición del ganador a nivel de búsqueda); (c) el
          fundacional compite sin entrenar, pero su calibración hereda el
          régimen de la ventana donde se estima.
        </Typography>
      </Seccion>

      <Seccion
        titulo="2 · Combinar en vez de seleccionar (V4)"
        detalle="El motor paga la maldición del ganador (acierto 10–21 %). La lección M4/M5: el promedio simple de pocos modelos razonables es extraordinariamente difícil de batir."
      >
        <Grid container spacing={1.5} sx={{ mb: 1 }}>
          <Grid size={{ xs: 12, md: 7 }}>
            <Grafico
              alto={280}
              opciones={{
                grid: { left: 8, right: 40, top: 12, bottom: 8, containLabel: true },
                xAxis: {
                  type: "value",
                  name: "sesgo valorizado a un paso (%)",
                  nameLocation: "middle",
                  nameGap: 28,
                  ...ejeBase,
                },
                yAxis: {
                  type: "category",
                  data: [...MENU_CURADO, "mezcla_prom"],
                  ...ejeBase,
                  splitLine: { show: false },
                },
                series: [
                  {
                    type: "bar",
                    barWidth: 16,
                    data: [...MENU_CURADO, "mezcla_prom"].map((modelo) => ({
                      value: v5.unPaso.get(modelo)?.bias ?? null,
                      itemStyle: {
                        color: modelo === "mezcla_prom" ? ACENTO : TINTA_SECUNDARIA,
                        opacity: modelo === "mezcla_prom" ? 1 : 0.55,
                      },
                    })),
                    label: {
                      show: true,
                      position: "right",
                      formatter: (p) =>
                        typeof p.value === "number" ? numero(p.value, 1) : "n/d",
                      fontSize: 11,
                    },
                  },
                ],
                tooltip: { trigger: "axis" },
              }}
            />
          </Grid>
          <Grid size={{ xs: 12, md: 5 }}>
            <Typography variant="body2" color="text.secondary">
              La mecánica del récord: los sesgos <b>opuestos</b> del menú se
              cancelan en el promedio (lightgbm sub-pronostica −14, TFT y HW
              sobre-pronostican +7/+6 → la mezcla queda en −1,5), una cobertura
              que la selección dura no puede dar. Resultado: <b>mezcla_prom
              D {fmt(d1(v4, "mezcla_prom"))} %</b> contra 74,3 del mejor
              individual — y la primera propuesta que vence significativamente
              al ingenuo a un paso. El promedio simple le ganó al ponderado por
              serie (V4) y al ponderado por horizonte (V5): tres confirmaciones
              M4 en estos datos.
            </Typography>
          </Grid>
        </Grid>
      </Seccion>

      <Seccion
        titulo="3 · Caminos explorados que enseñaron por qué no (V5–V6)"
        detalle="Dos generaciones de resultados negativos declarados — el material de la sección «decisiones descartadas con causa» del documento."
      >
        <TablaDatos
          alto={260}
          filas={[
            { experimento: "mezcla_h: pesos por horizonte (V5)", resultado: `${fmt(dh(v5, "mezcla_h"))} % vs ${fmt(dh(v5, "mezcla_prom"))} % del promedio`, leccion: "Los pesos inverso-D quedan casi planos y colapsan al promedio simple; a un paso inclinar hacia lightgbm rompe la cancelación de sesgos." },
            { experimento: "directo re-entrenado por origen (V5)", resultado: `${fmt(dh(v5, "lightgbm_directo"))} % vs ${fmt(dh(v5, "lightgbm"))} % del recursivo`, leccion: "Re-entrenar EMPEORA al directo: la ventaja del recursivo es su diseño (recursión con features simuladas), no el re-entrenamiento mensual. Confundido de la V4 resuelto." },
            { experimento: "calendario comercial como feature (V6)", resultado: `lightgbm ${fmt(d1(v6, "lightgbm"))} % (récord un paso ${fmt(d1(v6, "mezcla_prom"))} %)`, leccion: "Corrige el SESGO de campaña (−14,1→−10,7), no el error absoluto (que sube +1,4 %); al árbol le sirve la DISTANCIA a clases (meses_a_clases), no la bandera de temporada. Auditoría adversarial en docs/auditoria_ml2_v6.md." },
          ]}
        />
      </Seccion>

      <Seccion
        titulo="4 · La prueba de los dos años: el blindaje"
        detalle="El protocolo ÍNTEGRO desplazado un año (train g2018–2023, val g2024, prueba g2025 = arranque de la crisis cambiaria). Lo que sobrevive es del sistema; lo que no, era del año."
      >
        <Grid container spacing={1.5}>
          <Grid size={{ xs: 12, md: 7 }}>
            <Grafico
              alto={300}
              opciones={{
                grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
                legend: { top: 0, textStyle: { fontSize: 11 } },
                xAxis: {
                  type: "category",
                  data: MENU_CURADO,
                  ...ejeBase,
                  splitLine: { show: false },
                  axisLabel: { fontSize: 10, rotate: 20 },
                },
                yAxis: {
                  type: "value",
                  name: "sesgo valorizado (%)",
                  nameLocation: "middle",
                  nameGap: 36,
                  ...ejeBase,
                },
                series: [
                  {
                    type: "bar",
                    name: "g2026 (normal)",
                    barWidth: 14,
                    itemStyle: { color: COLOR_G2026 },
                    data: sesgosMenu.map((f) => f.g2026 ?? null),
                  },
                  {
                    type: "bar",
                    name: "g2025 (quiebre)",
                    barWidth: 14,
                    itemStyle: { color: COLOR_G2025 },
                    data: sesgosMenu.map((f) => f.g2025 ?? null),
                  },
                ],
                tooltip: {
                  trigger: "axis",
                  valueFormatter: (v) =>
                    typeof v === "number" ? `${numero(v, 1)} %` : "n/d",
                },
              }}
            />
          </Grid>
          <Grid size={{ xs: 12, md: 5 }}>
            <Typography variant="body2" color="text.secondary">
              <b>El talón de la mezcla, medido.</b> En el año normal el menú
              tiene sesgos de ambos signos y el promedio los cancela. En el año
              de quiebre <b>todo el menú sobre-pronostica</b> (+7 a +35): no
              queda nada que cancelar y la mezcla cae de 67,8 a 90,2 mientras
              el naive (que se re-ancla cada mes) casi lidera con 78,7.
              También sobrevive lo importante: la corona del horizonte de
              lightgbm ({fmt(dh(v5, "lightgbm"))} / {fmt(dh(blindaje, "lightgbm"))} %),
              las mezclas contra ma_12, la maldición del ganador del motor — y
              el colapso de ma_12/HW en crisis: el método actual de la empresa
              es el MÁS frágil ante quiebres.
            </Typography>
          </Grid>
        </Grid>
      </Seccion>

      <Seccion
        titulo="5 · La respuesta en dos actos: póliza estática (V7) y conmutación (V8)"
        detalle="V7: el menú con diversidad de sesgo deliberada (+nhits +naive_m1) rescata ~10 puntos en el quiebre pero paga prima en el año normal — ningún menú estático domina. V8: un detector causal (sesgo móvil de 3 meses YA observado, rampa declarada 10→20 %) conmuta entre menús."
      >
        <Typography variant="subtitle2" gutterBottom>
          Las cuatro esquinas — D valorizada (%), menor es mejor
        </Typography>
        <TablaDatos
          alto={260}
          filas={[
            {
              sistema: "menú curado (prom-5)",
              "g2026 un paso": fmt(d1(v5, "mezcla_prom")),
              "g2026 D(h)": fmt(dh(v5, "mezcla_prom")),
              "g2025 un paso": fmt(d1(blindaje, "mezcla_prom")),
              "g2025 D(h)": fmt(dh(blindaje, "mezcla_prom")),
            },
            {
              sistema: "menú diverso (prom-7)",
              "g2026 un paso": fmt(d1(v7, "mezcla_prom")),
              "g2026 D(h)": fmt(dh(v7, "mezcla_prom")),
              "g2025 un paso": fmt(d1(v7b, "mezcla_prom")),
              "g2025 D(h)": fmt(dh(v7b, "mezcla_prom")),
            },
            {
              sistema: "póliza ponderada (pond-7)",
              "g2026 un paso": fmt(d1(v7, "mezcla_pond")),
              "g2026 D(h)": fmt(dh(v7, "mezcla_pond")),
              "g2025 un paso": fmt(d1(v7b, "mezcla_pond")),
              "g2025 D(h)": fmt(dh(v7b, "mezcla_pond")),
            },
            {
              sistema: "mezcla_conmutada (V8)",
              "g2026 un paso": fmt(d1(v8, "mezcla_conmutada")),
              "g2026 D(h)": fmt(dh(v8, "mezcla_conmutada")),
              "g2025 un paso": fmt(d1(v8b, "mezcla_conmutada")),
              "g2025 D(h)": fmt(dh(v8b, "mezcla_conmutada")),
            },
          ]}
        />
        <Grid container spacing={1.5} sx={{ mt: 0.5 }}>
          <Grid size={{ xs: 12, md: 6 }}>
            <Grafico
              alto={280}
              opciones={{
                grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
                legend: { top: 0, textStyle: { fontSize: 11 } },
                xAxis: {
                  type: "category",
                  data: mesesPrueba,
                  name: "mes de prueba",
                  nameLocation: "middle",
                  nameGap: 26,
                  ...ejeBase,
                  splitLine: { show: false },
                },
                yAxis: {
                  type: "value",
                  name: "sesgo móvil 3 m (%)",
                  nameLocation: "middle",
                  nameGap: 38,
                  ...ejeBase,
                },
                series: [
                  {
                    type: "line",
                    name: "g2026 (normal)",
                    symbol: "circle",
                    symbolSize: 4,
                    lineStyle: { width: 2, color: COLOR_G2026 },
                    itemStyle: { color: COLOR_G2026 },
                    data: filasSenal26.map((f) => Number(f.sesgo_movil_pct)),
                  },
                  {
                    type: "line",
                    name: "g2025 (quiebre)",
                    symbol: "circle",
                    symbolSize: 4,
                    lineStyle: { width: 2, color: COLOR_G2025 },
                    itemStyle: { color: COLOR_G2025 },
                    data: filasSenal25.map((f) => Number(f.sesgo_movil_pct)),
                    markLine: {
                      silent: true,
                      symbol: "none",
                      lineStyle: { type: "dashed", color: TINTA_SECUNDARIA },
                      label: { fontSize: 10, position: "insideEndTop" },
                      data: [
                        { yAxis: 10, label: { formatter: "activación 10 %" } },
                        { yAxis: 20, label: { formatter: "pleno 20 %" } },
                        { yAxis: -10, label: { formatter: "" } },
                        { yAxis: -20, label: { formatter: "" } },
                      ],
                    },
                  },
                ],
                tooltip: {
                  trigger: "axis",
                  valueFormatter: (v) =>
                    typeof v === "number" ? `${numero(v, 1)} %` : "n/d",
                },
              }}
            />
          </Grid>
          <Grid size={{ xs: 12, md: 6 }}>
            <Grafico
              alto={280}
              opciones={{
                grid: { left: 8, right: 24, top: 36, bottom: 8, containLabel: true },
                legend: { top: 0, textStyle: { fontSize: 11 } },
                xAxis: {
                  type: "category",
                  data: mesesPrueba,
                  name: "mes de prueba",
                  nameLocation: "middle",
                  nameGap: 26,
                  ...ejeBase,
                  splitLine: { show: false },
                },
                yAxis: {
                  type: "value",
                  name: "λ (peso del menú diverso)",
                  nameLocation: "middle",
                  nameGap: 34,
                  min: 0,
                  max: 1,
                  ...ejeBase,
                },
                series: [
                  {
                    type: "line",
                    name: "g2026 (normal)",
                    step: "middle",
                    symbol: "circle",
                    symbolSize: 4,
                    lineStyle: { width: 2, color: COLOR_G2026 },
                    itemStyle: { color: COLOR_G2026 },
                    areaStyle: { color: COLOR_G2026, opacity: 0.08 },
                    data: filasSenal26.map((f) => Number(f.lambda)),
                  },
                  {
                    type: "line",
                    name: "g2025 (quiebre)",
                    step: "middle",
                    symbol: "circle",
                    symbolSize: 4,
                    lineStyle: { width: 2, color: COLOR_G2025 },
                    itemStyle: { color: COLOR_G2025 },
                    areaStyle: { color: COLOR_G2025, opacity: 0.12 },
                    data: filasSenal25.map((f) => Number(f.lambda)),
                  },
                ],
                tooltip: {
                  trigger: "axis",
                  valueFormatter: (v) =>
                    typeof v === "number" ? numero(v, 2) : "n/d",
                },
              }}
            />
          </Grid>
        </Grid>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          <b>La señal (izquierda) y la acción (derecha).</b> En g2025 el
          detector dispara al primer mes (+18 → +77 %), sostiene λ=1 durante el
          quiebre y <b>suelta solo</b> cuando el régimen se estabiliza (señal
          &lt; 6 % en los últimos meses): el ciclo de vida completo del
          quiebre, trazado. La conmutada domina al menú diverso estático en el
          año normal, logra el mejor peor-caso del horizonte del estudio y es
          significativa contra ma_12 en AMBOS regímenes — lo que ningún brazo
          individual consigue. Limitaciones declaradas: la señal de 3 meses es
          ruidosa en año normal (la prima de +1,1/+3,5 contra el menú curado),
          a un paso la póliza estática ponderada sigue siendo competidora
          seria, y el diseño del detector se informó en los dos años medidos —
          su validez plena espera gestiones futuras.
        </Typography>
      </Seccion>

      <Seccion
        titulo="Trazabilidad"
        detalle="Cada número de esta vista se lee en vivo del artefacto de su corrida (tabla_valorizada.csv y manifiesto). Las corridas están selladas con manifiesto y hash; las configs config_ml2_*.yaml declaran cada decisión, y los mensajes de commit del repositorio narran generación por generación. La auditoría adversarial de la V6 (números confirmados, tres interpretaciones corregidas) está en docs/auditoria_ml2_v6.md."
      >
        <Typography variant="caption" color="text.secondary">
          Corridas: {Object.values(C).join(" · ")}. Los mezcladores estándar de
          la V8 reproducen EXACTO los números de la V5 y del blindaje
          (mezcla_prom {fmt(d1(v8, "mezcla_prom"))} = {fmt(d1(v5, "mezcla_prom"))};
          control de determinismo doble, verificado también por la
          re-auditoría multi-agente).
        </Typography>
      </Seccion>
    </>
  );
}
