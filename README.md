# Trabajo Final de Grado — Nicolás Oporto Rojas

**«Diseño de un motor de pronóstico de demanda basado en series temporales para
mejorar la precisión de la planificación de la demanda en NOVAPACK S.A. para la
gestión 2026»**

Maestría en Ciencia de Datos e Inteligencia Artificial
UAGRM School of Engineering — Universidad Autónoma «Gabriel René Moreno»
Santa Cruz de la Sierra, Bolivia

---

## Qué hay en este repositorio

| Carpeta | Contenido |
|---|---|
| [`motor-pronostico-novapack/`](motor-pronostico-novapack/) | **El análisis cuantitativo.** Código, pruebas y entorno reproducible que generan la Tabla 8, las figuras y el contraste estadístico del Capítulo III. Es el material que enlaza el **Anexo G**. |
| [`documento/`](documento/) | El documento oficial de la tesis y sus anexos. |

Cada carpeta tiene su propio `README.md`. Para correr el análisis, empezá por
[`motor-pronostico-novapack/README.md`](motor-pronostico-novapack/README.md).

---

## Confidencialidad

El histórico de ventas que sustenta el análisis está cubierto por un acuerdo de
confidencialidad con la empresa (**Anexo B** del documento). En consecuencia:

- Los **datos** no se versionan. El análisis parte de un snapshot congelado cuyo
  SHA-256 queda registrado en el manifiesto de cada corrida; el archivo en sí
  nunca entra al repositorio.
- El **esquema del sistema de origen** —nombres de base, esquema, tabla y
  columnas— tampoco. Se declara en un archivo local que no se versiona.
- Los **códigos de producto** se reemplazan por identificadores estables al
  extraer, porque el detalle por serie se adjunta como anexo del documento.

Hay un control automático (`motor-pronostico-novapack/tests/test_confidencialidad.py`)
que inspecciona **todo lo que git rastrea en este repositorio** y hace fallar la
suite si aparece un identificador de la empresa, una credencial o una etiqueta
real de regional. No inspecciona el interior de los documentos ofimáticos de
`documento/`: eso hay que revisarlo a ojo antes de publicar.

---

## Trabajar sobre este repositorio

La identidad de los commits y el helper de credenciales están fijados **en la
configuración local de este repositorio**, no en la global, para no interferir
con otras cuentas de GitHub que se usen en la misma máquina:

```bash
git config --local user.name  noportor
git config --local user.email noportor@soe.uagrm.edu.bo

# Neutraliza los helpers heredados del gitconfig del sistema y usa el de gh.
# La primera entrada vacía es la que borra la cadena previa: sin ella, el helper
# del sistema se ejecuta primero, devuelve la credencial de otra cuenta y GitHub
# contesta «Repository not found» —404 en vez de 403, porque el repo es privado—,
# que parece un problema de URL y no lo es.
git config --local --replace-all credential.helper ""
git config --local --add         credential.helper "!gh auth git-credential"
```

`gh` usa la cuenta que tenga activa. Si se cambia con `gh auth switch`, los push
a este repositorio empiezan a fallar hasta volver a `noportor`.

---

## Reproducibilidad

Los números del documento se regeneran con un solo comando dentro de un entorno
con versiones fijadas. Cada corrida escribe un `manifiesto.json` con el hash del
archivo de datos, el hash de la configuración, el commit que generó los números y
las versiones de biblioteca realmente cargadas.

Comprobación hecha: la tabla de resultados sale **idéntica** ejecutando en
Python 3.11 con LightGBM 4.6 y en Python 3.13 con LightGBM 4.7.
