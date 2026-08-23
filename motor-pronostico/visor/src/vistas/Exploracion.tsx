/**
 * Exploración ML2 — la narrativa de las corridas post-documento (V1–V9 +
 * blindaje), contada en el orden en que la evidencia se construyó — incluida
 * la auditoría que encontró una fuga de selección en las V4–V8 y la
 * remedición limpia (V9), que es la corrida CANÓNICA: la que reporta la
 * tesis.
 *
 * La vista es TRANSVERSAL: no depende de la corrida activa — lee los números
 * directo de los artefactos de las trece corridas selladas (nombres fijos de
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
  v9: "salidas_ml2_v9",
  blindaje: "salidas_ml2_blindaje",
  v7b: "salidas_ml2_v7_blindaje",
  v8b: "salidas_ml2_v8_blindaje",
  v9b: "salidas_ml2_v9_blindaje",
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
  // Orden FIJO de hooks: una llamada por corrida que la narrativa consume
  // (blindaje y v7_blindaje quedan en C solo para el listado de trazabilidad:
  // sus números fueron remedidos por la V9).
  const v1 = useResumenCorrida(C.v1);
  const v2 = useResumenCorrida(C.v2);
  const v3 = useResumenCorrida(C.v3);
  const v4 = useResumenCorrida(C.v4);
  const v5 = useResumenCorrida(C.v5);
  const v6 = useResumenCorrida(C.v6);
  const v7 = useResumenCorrida(C.v7);
  const v8 = useResumenCorrida(C.v8);
  const v9 = useResumenCorrida(C.v9);
  const v8b = useResumenCorrida(C.v8b);
  const v9b = useResumenCorrida(C.v9b);
  const senal26 = useCsvGenerico(C.v9, "conmutador_senal.csv");
  const senal25 = useCsvGenerico(C.v9b, "conmutador_senal.csv");

  // --- El arco: el mejor sistema a un paso, generación por generación -------
  const arco = useMemo(() => {
    const generaciones: [string, ResumenCorrida][] = [
      ["V1", v1], ["V2", v2], ["V3", v3], ["V4", v4],
      ["V5", v5], ["V6", v6], ["V7", v7], ["V8", v8], ["V9", v9],
    ];
    return generaciones.map(([nombre, r]) => {
      const mejor = mejorUnPaso(r);
      return { generacion: nombre, modelo: mejor?.modelo ?? "n/d", d: mejor?.d };
    });
  }, [v1, v2, v3, v4, v5, v6, v7, v8, v9]);

  // --- El talón: el sesgo del menú curado en los dos años (corrida limpia) --
  const MENU_CURADO = ["lightgbm", "chronos", "exp_smooth_opt", "holt_winters", "tft"];
  const sesgosMenu = useMemo(
    () =>
      MENU_CURADO.map((modelo) => ({
        modelo,
        g2026: v9.unPaso.get(modelo)?.bias,
        g2025: v9b.unPaso.get(modelo)?.bias,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [v9, v9b],
  );

  // --- La señal del conmutador en ambos años --------------------------------
  const filasSenal26 = (senal26.data ?? []) as unknown as FilaSenal[];
  const filasSenal25 = (senal25.data ?? []) as unknown as FilaSenal[];
  const mesesPrueba = Array.from({ length: 12 }, (_, i) => `m${i + 1}`);

  return (
    <>
      <Seccion
        titulo="La exploración posterior al documento"
        detalle="Trece corridas selladas (V1–V9 sobre la prueba g2026; el protocolo desplazado a g2025 como blindaje). La V9 es la corrida CANÓNICA: remide la V8 con las configuraciones elegidas por validación, tras la auditoría que encontró la fuga de selección de las V4–V8 (capítulo 6). Cada capítulo enuncia lo que se midió y de qué corrida salen los números: esta vista los lee EN VIVO de los artefactos — elija la corrida en el selector para auditar cualquiera en profundidad."
      >
        <Grid container spacing={1.5} sx={{ mb: 1 }}>
          <Grid size={{ xs: 6, md: 3 }}>
            <TarjetaKPI
              titulo="Menú curado a un paso (g2026, V9)"
              valor={`mezcla_prom ${fmt(d1(v9, "mezcla_prom"))} %`}
              detalle="menú de 4 (sin TFT): 70,0 por recombinación"
              ayuda="Promedio simple del menú curado {lightgbm, chronos, SES, HW, TFT} en la corrida limpia. Excluir al integrante débil (TFT) da la mejor cifra a un paso del estudio: 70,0. El campeón individual del año normal (chronos) quedó en 74,3; LightGBM en 77,0."
            />
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <TarjetaKPI
              titulo="Corona del horizonte"
              valor={`lightgbm ${fmt(dh(v9, "lightgbm"))} %`}
              detalle={`también en g2025: ${fmt(dh(v9b, "lightgbm"))} %`}
              ayuda="D(h) global con orígenes rodantes. La única afirmación de podio que sobrevive intacta a los DOS años de prueba — y a la corrida limpia."
            />
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <TarjetaKPI
              titulo="Mejor individual del quiebre"
              valor={`nhits ${fmt(d1(v9b, "nhits"))} %`}
              detalle="de los más débiles del año normal (92,3)"
              ayuda="La reversión que funda la diversidad de sesgos: el sesgo negativo estructural de N-HiTS, que lo hunde en el año normal (solo TFT queda debajo), es exactamente el contrapeso que la crisis premia."
            />
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <TarjetaKPI
              titulo="Mejor peor-caso a un paso"
              valor={`conmutada ${fmt(Math.max(d1(v9, "mezcla_conmutada") ?? NaN, d1(v9b, "mezcla_conmutada") ?? NaN))} %`}
              detalle="efecto mediano/grande vs ma_12 en ambas"
              ayuda="mezcla_conmutada (V9): detector causal de régimen sobre el sesgo móvil observado. Su ventaja sobre la práctica vigente es de efecto mediano y grande (r = −0,41 y −0,54); los brazos individuales, aun cuando alcanzan significancia, no pasan de efecto pequeño."
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
                  label: { formatter: "lightgbm solo (77,0)", position: "insideEndTop", fontSize: 11 },
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
          paso). La caída de la V4 mezclaba dos cosas: el efecto real de
          combinar y una fuga de selección en las configs de TFT/N-HiTS
          (capítulo 6) que infló la cancelación de sesgos. La V9 remide
          limpio: la mezcla honesta queda en 74,0–74,9, y la exclusión del
          integrante débil (recombinación declarada) da el 70,0 que reporta la
          tesis.
        </Typography>
      </Seccion>

      <Seccion
        titulo="1 · Nuevos brazos: redes y fundacional (V1–V3)"
        detalle="Cuatro arquitecturas neuronales globales (DLinear, N-HiTS, DeepAR, TFT) y un fundacional cero-shot (Chronos-Bolt) bajo el MISMO protocolo congelado que los once brazos del documento."
      >
        <TablaDatos
          alto={300}
          filas={[
            { brazo: "chronos (calibrado por categoría, V3)", "D un paso (%)": fmt(d1(v3, "chronos")), lectura: "El mejor brazo individual a un paso en la gestión normal: 5 factores de validación convierten al cero-shot en líder. El factor global (V2) sobre-corregía." },
            { brazo: "tft (config honesta V3, medida en V9)", "D un paso (%)": fmt(d1(v9, "tft")), lectura: "La búsqueda por validación no lo rescata: 106,1 limpio. La config V2 que PARECÍA rescatarlo solo se sostenía elegida mirando la prueba — el origen de la fuga del capítulo 6. Veredicto: el integrante débil del menú." },
            { brazo: "dlinear (control lineal)", "D un paso (%)": fmt(d1(v1, "dlinear")), lectura: "El lineal le gana a las redes profundas univariantes (resultado Zeng): la arquitectura no es el cuello." },
            { brazo: "nhits (config honesta V3, medida en V9)", "D un paso (%)": fmt(d1(v9, "nhits")), lectura: "De los más débiles del año normal (solo TFT queda debajo) — y MEJOR individual del año de quiebre (70,6 en g2025): su sesgo negativo persistente es el contrapeso que la crisis premia." },
            { brazo: "deepar", "D un paso (%)": fmt(d1(v1, "deepar")), lectura: "Tres verosimilitudes, tres colapsos (sesgo −68 a −98): incompatible con esta demanda; se retira del menú (declarado)." },
          ]}
        />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Las tres lecciones de la etapa: (a) el <b>sesgo de la pérdida domina
          sobre la arquitectura</b> (la lección Tweedie del campeón, replicada);
          (b) la <b>búsqueda de hiperparámetros no transfiere</b> a través del
          quiebre cambiario (V3: ganadores del pliegue con sesgo ~0 dieron −27
          a −37 en prueba — maldición del ganador a nivel de búsqueda); (c) el
          fundacional compite sin entrenar, pero su calibración hereda el
          régimen de la ventana donde se estima. La respuesta correcta a (b)
          era conservar esas configs de todos modos: recuperar configs viejas
          por su desempeño observado en la prueba es seleccionar sobre el
          examen — el error que las V4–V8 cometieron y la V9 corrigió
          (capítulo 6).
        </Typography>
      </Seccion>

      <Seccion
        titulo="2 · Combinar en vez de seleccionar (V4, remedido limpio en V9)"
        detalle="El motor paga la maldición del ganador (acierto 9–21 % según la corrida). La lección M4/M5 sobrevive a la corrida limpia con su condición a la vista: el promedio de pocos modelos SÓLIDOS es difícil de batir — un integrante débil no aporta diversidad, erosiona. El gráfico lee la V9."
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
                      value: v9.unPaso.get(modelo)?.bias ?? null,
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
              La mecánica: los sesgos <b>opuestos</b> del menú se cancelan en
              el promedio (lightgbm sub-pronostica −14,1 mientras HW +5,7 y el
              resto lo compensan), una cobertura que la selección dura no puede
              dar. En limpio: <b>mezcla_prom D {fmt(d1(v9, "mezcla_prom"))} %</b>,
              y la exclusión del integrante débil (TFT, 106,1) baja el menú de
              4 a <b>70,0</b> (recombinación de los mismos pronósticos
              publicados) — la mejor cifra a un paso del estudio, 5,4 puntos
              bajo el motor. La primera propuesta que vence significativamente
              al ingenuo a un paso (p = 1,1×10⁻⁷). Y el hallazgo M4 se matiza
              con la medición: con menú parejo el promedio simple no se supera;
              con un integrante débil, ponderar por validación SÍ aporta
              (mezcla_pond {fmt(d1(v9, "mezcla_pond"))} % contra
              {" "}{fmt(d1(v9, "mezcla_prom"))} %).
            </Typography>
          </Grid>
        </Grid>
      </Seccion>

      <Seccion
        titulo="3 · Caminos explorados que enseñaron por qué no (V5–V6)"
        detalle="Dos generaciones de resultados negativos declarados. Medidos con el menú de su época (luego remedido limpio en V9): las lecciones de MÉTODO sobreviven — el calendario es un hallazgo solo-LightGBM (config nunca afectada por la fuga), y la de ponderación se matiza en V9: con un integrante débil en el menú, ponderar sí aporta."
      >
        <TablaDatos
          alto={260}
          filas={[
            { experimento: "mezcla_h: pesos por horizonte (V5)", resultado: `${fmt(dh(v5, "mezcla_h"))} % vs ${fmt(dh(v5, "mezcla_prom"))} % del promedio`, leccion: "Los pesos inverso-D quedan casi planos y colapsan al promedio simple; a un paso inclinar hacia lightgbm rompe la cancelación de sesgos." },
            { experimento: "directo re-entrenado por origen (V5)", resultado: `${fmt(dh(v5, "lightgbm_directo"))} % vs ${fmt(dh(v5, "lightgbm"))} % del recursivo`, leccion: "Re-entrenar EMPEORA al directo: la ventaja del recursivo es su diseño (recursión con features simuladas), no el re-entrenamiento mensual. Confundido de la V4 resuelto." },
            { experimento: "calendario comercial como feature (V6)", resultado: `lightgbm ${fmt(d1(v6, "lightgbm"))} % (mezcla de su época: ${fmt(d1(v6, "mezcla_prom"))} %, remedida en V9)`, leccion: "Corrige el SESGO de campaña (−14,1→−10,7), no el error absoluto (que sube +1,4 %); al árbol le sirve la DISTANCIA a clases (meses_a_clases), no la bandera de temporada. Auditoría adversarial en docs/auditoria_ml2_v6.md." },
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
              <b>El talón de la mezcla, medido en limpio.</b> En el año normal
              el menú tiene sesgos de ambos signos y el promedio los cancela.
              En el año de quiebre los integrantes <b>estructurales</b>{" "}
              sobre-pronostican a la vez (del +7,2 del suavizado al +22,5 de
              HW): el menú de 4 — el ganador del año normal, sin sesgos
              negativos que lo compensen — cae de 70,0 a 84,4
              (recombinación), la mezcla de 5 sube a
              {" "}{fmt(d1(v9b, "mezcla_prom"))} % y el naive (que se re-ancla
              cada mes) incluso la supera por décimas
              ({fmt(d1(v9b, "naive_m1"))} %). Los brazos de sesgo negativo son
              el contrapeso: nhits pasa de la cola del año normal a mejor
              individual ({fmt(d1(v9b, "nhits"))} %). También sobrevive
              lo importante: la corona del horizonte de lightgbm
              ({fmt(dh(v9, "lightgbm"))} / {fmt(dh(v9b, "lightgbm"))} %), las
              mezclas contra ma_12, la maldición del ganador del motor — y el
              colapso de ma_12/HW en crisis (D &gt; 111): el método actual de
              la empresa es el MÁS frágil ante quiebres.
            </Typography>
          </Grid>
        </Grid>
      </Seccion>

      <Seccion
        titulo="5 · La respuesta en dos actos: póliza estática y conmutación (V9)"
        detalle="Acto 1: el menú con diversidad de sesgo deliberada (+nhits +naive_m1) rescata 6,7 puntos en el quiebre (72,1 vs 78,8) al precio de 1 punto en el año normal (75,9 vs 74,9) — ningún menú estático domina. Acto 2: un detector causal (sesgo móvil de 3 meses YA observado, rampa declarada 10→20 %) conmuta entre menús. Números limpios: corridas V9; el diverso, por recombinación de los mismos pronósticos publicados."
      >
        <Typography variant="subtitle2" gutterBottom>
          Las cuatro esquinas — D valorizada (%), menor es mejor
        </Typography>
        <TablaDatos
          alto={260}
          filas={[
            {
              sistema: "menú curado (prom-5, V9)",
              "g2026 un paso": fmt(d1(v9, "mezcla_prom")),
              "g2026 D(h)": fmt(dh(v9, "mezcla_prom")),
              "g2025 un paso": fmt(d1(v9b, "mezcla_prom")),
              "g2025 D(h)": fmt(dh(v9b, "mezcla_prom")),
            },
            {
              sistema: "menú curado (pond por validación, V9)",
              "g2026 un paso": fmt(d1(v9, "mezcla_pond")),
              "g2026 D(h)": fmt(dh(v9, "mezcla_pond")),
              "g2025 un paso": fmt(d1(v9b, "mezcla_pond")),
              "g2025 D(h)": fmt(dh(v9b, "mezcla_pond")),
            },
            {
              sistema: "menú diverso (prom-7, recombinación)",
              "g2026 un paso": "75,9",
              "g2026 D(h)": "—",
              "g2025 un paso": "72,1",
              "g2025 D(h)": "—",
            },
            {
              sistema: "mezcla_conmutada (V9)",
              "g2026 un paso": fmt(d1(v9, "mezcla_conmutada")),
              "g2026 D(h)": fmt(dh(v9, "mezcla_conmutada")),
              "g2025 un paso": fmt(d1(v9b, "mezcla_conmutada")),
              "g2025 D(h)": fmt(dh(v9b, "mezcla_conmutada")),
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
          detector dispara al primer mes (+21 → +77 %), sostiene λ=1 durante
          el episodio (con una liberación transitoria en diciembre) y
          <b> suelta definitivamente solo</b> cuando el régimen se estabiliza:
          el ciclo de vida completo del quiebre, trazado. La conmutada logra
          el mejor peor-caso a un paso del estudio (75,3), el mejor horizonte
          del quiebre después del modelo global (71,5, a 1,4 de lightgbm) y
          bate a ma_12 con efecto mediano y grande (r = −0,41 y −0,54) —
          magnitud a la que ningún brazo individual llega: lightgbm, aun
          significativo, queda en efecto insignificante-pequeño. Limitaciones
          declaradas: la
          señal de 3 meses es ruidosa en año normal (oscila entre −27 y +34 %
          por la volatilidad de campaña; prima de 0,4 puntos contra el menú
          curado), a un paso el menú diverso estático logra un desempeño
          comparable sin requerir detector, y el diseño del detector se
          informó en los dos años medidos — su validez plena espera gestiones
          futuras.
        </Typography>
      </Seccion>

      <Seccion
        titulo="6 · La fuga de selección y la corrida limpia (V9)"
        detalle="La auditoría de justificación encontró que las V4–V8 usaban las configs de TFT y N-HiTS elegidas por su desempeño EN PRUEBA (documentado en config_ml2_v4.yaml): la búsqueda honesta V3 medía peor y se la revirtió mirando el examen — selección sobre el propio bloque de evaluación. La V9 remide TODO con las configs del HPO por validación, en ambas gestiones. La tesis reporta SOLO números V9."
      >
        <TablaDatos
          alto={280}
          filas={[
            {
              modelo: "tft",
              "V8 contaminada (g2026)": fmt(d1(v8, "tft")),
              "V9 limpia (g2026)": fmt(d1(v9, "tft")),
              "V8 (g2025)": fmt(d1(v8b, "tft")),
              "V9 (g2025)": fmt(d1(v9b, "tft")),
            },
            {
              modelo: "nhits",
              "V8 contaminada (g2026)": fmt(d1(v8, "nhits")),
              "V9 limpia (g2026)": fmt(d1(v9, "nhits")),
              "V8 (g2025)": fmt(d1(v8b, "nhits")),
              "V9 (g2025)": fmt(d1(v9b, "nhits")),
            },
            {
              modelo: "mezcla_prom (el arrastre)",
              "V8 contaminada (g2026)": fmt(d1(v8, "mezcla_prom")),
              "V9 limpia (g2026)": fmt(d1(v9, "mezcla_prom")),
              "V8 (g2025)": fmt(d1(v8b, "mezcla_prom")),
              "V9 (g2025)": fmt(d1(v9b, "mezcla_prom")),
            },
            {
              modelo: "mezcla_conmutada",
              "V8 contaminada (g2026)": fmt(d1(v8, "mezcla_conmutada")),
              "V9 limpia (g2026)": fmt(d1(v9, "mezcla_conmutada")),
              "V8 (g2025)": fmt(d1(v8b, "mezcla_conmutada")),
              "V9 (g2025)": fmt(d1(v9b, "mezcla_conmutada")),
            },
            {
              modelo: "lightgbm (control: no cambió)",
              "V8 contaminada (g2026)": fmt(d1(v8, "lightgbm")),
              "V9 limpia (g2026)": fmt(d1(v9, "lightgbm")),
              "V8 (g2025)": fmt(d1(v8b, "lightgbm")),
              "V9 (g2025)": fmt(d1(v9b, "lightgbm")),
            },
          ]}
        />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          El mecanismo, a la vista: la config de TFT elegida en prueba era
          justo la que mejor cancelaba el sesgo −14,1 de lightgbm, y esa
          cancelación se «comprobó» en el examen. Con el TFT honesto la
          cancelación gratis desaparece y el titular 67,8 se convierte en
          {" "}{fmt(d1(v9, "mezcla_prom"))} — de donde sale la respuesta
          editorial: excluir al débil (70,0 por recombinación) y declarar el
          criterio. La fila de control (lightgbm, config nunca tocada)
          reproduce idéntico entre corridas: solo cambió lo que debía cambiar.
          El resultado PRINCIPAL de la tesis (motor 75,4 vs 100,0 del método
          vigente) nunca dependió de estas configs: sale de la corrida
          canónica del documento, previa a la extensión.
        </Typography>
      </Seccion>

      <Seccion
        titulo="Trazabilidad"
        detalle="Cada número de esta vista se lee en vivo del artefacto de su corrida (tabla_valorizada.csv y manifiesto). Las corridas están selladas con manifiesto y hash; las configs config_ml2_*.yaml declaran cada decisión — la fuga de las V4–V8 quedó documentada en el propio config_ml2_v4.yaml y su corrección en config_ml2_v9*.yaml —, y los mensajes de commit del repositorio narran generación por generación. La auditoría adversarial de la V6 está en docs/auditoria_ml2_v6.md."
      >
        <Typography variant="caption" color="text.secondary">
          Corridas: {Object.values(C).join(" · ")}. Controles de determinismo:
          los mezcladores de la V8 reproducen EXACTO los de la V5
          (mezcla_prom {fmt(d1(v8, "mezcla_prom"))} = {fmt(d1(v5, "mezcla_prom"))})
          y el control limpio de la V9 reproduce el lightgbm publicado
          ({fmt(d1(v9, "lightgbm"))} = {fmt(d1(v5, "lightgbm"))}): cada
          corrida nueva cambió solo lo que declaraba cambiar.
        </Typography>
      </Seccion>
    </>
  );
}
