# Auditoría de la corrida V6 (calendario comercial) — 2026-08-09

Re-verificación adversarial de la V6 contra sus artefactos (`salidas_ml2_v5/`
vs `salidas_ml2_v6/`), posterior al commit `2b09322`. Cinco verificadores
independientes (un paso, horizonte, importancias, estratos/contraste,
determinismo) más un crítico buscando lecturas omitidas. Este documento
registra lo que se CONFIRMA, lo que se CORRIGE del veredicto original y lo
que queda abierto. Los números del commit son correctos; tres
interpretaciones no se sostienen y se corrigen acá.

## Confirmado

- **Control de determinismo: perfecto.** Los 9 brazos no tocados por el
  calendario reproducen sus números de V5 a precisión float completa, a un
  paso y en los 12 horizontes. Solo cambian lightgbm/lightgbm_directo y sus
  herederos (mezclas, motor). La sanidad `lightgbm_directo == lightgbm` a un
  paso se mantiene (74,5 idénticos en todas las columnas).
- **Los números de cabecera**: lightgbm 77,0→74,5 (bias −14,1→−10,7, WMAPE
  62,9→63,8); mezcla_prom 67,8→67,3 (mínimo de la tabla); horizonte lightgbm
  71,9→73,0, directo 84,0→83,4; `meses_a_clases` 503 divisiones (empatada
  con `mes`), `temporada_alta` 25.ª con 191.
- **La señal del calendario es real, no canibalismo**: `mes` cedió 481
  divisiones y el 65 % de su ganancia; las dos features nuevas capturan
  1,02 M de ganancia, más de lo que `mes` perdió. Por ganancia-por-división,
  `temporada_alta` es la más densa (2.065 por split) — la historia «solo
  trabajó meses_a_clases» vale por divisiones, se matiza por ganancia.

## Correcciones al veredicto original

1. **El récord a un paso es cancelación de sesgo, no reducción de error.**
   El error absoluto valorizado SUBE en todos los brazos tocados (lightgbm
   77,1→78,2 M Bs, +1,4 %; mezcla_prom 81,2→81,5 M, WMAPE 66,3→66,6). La D
   mejora porque |bias| cae (mezcla_prom −1,5→−0,8). Bajo la métrica de
   decisión declarada del estudio —que penaliza el sesgo porque el
   sub-pronóstico cuesta quiebres— el récord es legítimo, pero debe
   enunciarse como lo que es: el calendario corrige el sesgo de campaña,
   no achica el error.
2. **«La ganancia se concentra en estacionales de alto valor» es falso, dos
   veces.** Las recurrentes concentran 2,2× más demanda valorizada (83,8 vs
   38,7 M Bs) y la mejora de D es MAYOR en recurrentes (−1,7) y suaves
   (−2,3) que en estacionales (−1,5). Peor: el WMAPE estacional EMPEORA
   (78,4→82,1) con el bias cruzando a sobre-pronóstico (−6,5→+1,4). La
   divergencia D-valorizada vs Wilcoxon se explica por el punto 1: la D
   mejora vía sesgo agregado, mientras el error compuesto POR SERIE (lo que
   el Wilcoxon rankea) empeora levemente — las victorias por serie caen en
   todos los brazos tocados (lightgbm 85,7→80,0; mezcla_prom 86→83) y las
   recogen brazos no tocados (nhits, HW).
3. **El mecanismo «el empuje estacional se realimenta y compone» está
   refutado por la forma de los datos.** En el horizonte la degradación es
   100 % sesgo con WMAPE que MEJORA (63,2→62,9; el error valorizado total
   CAE 587,9→585,1 M Bs), pero el bias se hace MÁS negativo (−8,7→−10,2;
   h=6: −8,4→−11,2) — lo contrario de una amplificación del empuje. Además
   la degradación pica en h=5–8 y desaparece en h=11–12 (forma incompatible
   con realimentación compuesta) y es PAREJA entre estratos (estacional
   +1,5, suave +1,4, lumpy +1,4 — no un fenómeno estacional). Lo que queda
   en pie, y es el dato duro: con las MISMAS features el directo (sin
   recursión) mejora y el recursivo empeora — la interacción
   recursión × calendario es el locus, pero el mecanismo fino queda
   ABIERTO.
4. **Dos omisiones del veredicto.** (a) El estrato intermitente empeora
   (93,1→97,3, bias −16,5→−20,6; 38 series con costo). (b) El motor empeora
   (73,7→74,0, error absoluto +1,9 %): la selección migra +42 series a
   lightgbm_directo (183→225) porque el calendario lo hace ver mejor en la
   ventana de validación, y el motor resultante es peor fuera de muestra —
   una instancia más de la maldición del ganador que el estudio viene
   documentando.
5. **La mejora a un paso no es uniforme en el tiempo**: 8 de 12 orígenes del
   horizonte empeoran, y el último origen (2026-02, solo h=1) también
   (83,6→85,8).

## Lectura final

El calendario comercial mueve la métrica de decisión declarada en la
dirección correcta a un paso (récord D 67,3 del sistema recomendado,
sesgo −0,8) al costo de un leve aumento del error absoluto, una leve
degradación del horizonte por sesgo (mecanismo abierto) y un motor peor.
Para el sistema recomendado (mezcla_prom) el neto es marginalmente
positivo a un paso y neutro en el horizonte (72,9→73,1). Es una palanca
real pero chica y con contrapartidas — coherente con los rendimientos
decrecientes que cerraron la exploración de arquitectura.
