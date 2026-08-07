# Documento de la tesis

Acá van el documento oficial y sus anexos.

## Antes de agregar un archivo

El control automático de confidencialidad **no puede mirar dentro** de los
`.docx`, `.pdf` ni `.xlsx`: son binarios comprimidos, y el control salta esas
extensiones. Lo que se deje acá hay que revisarlo a ojo. En concreto, antes de
que el repositorio se haga público:

- Que la empresa aparezca únicamente con su nombre de fantasía y no con el real,
  en el texto y también en encabezados, pies, metadatos y propiedades del
  archivo.
- Que no haya códigos de producto, listas de clientes ni capturas del sistema
  interno.
- Que el **Anexo B** (acuerdo de confidencialidad) no exponga nombres, cargos ni
  firmas de terceros que no hayan consentido su publicación.
- Que las tablas y figuras coincidan con las salidas vigentes de
  `motor-pronostico-novapack/salidas/`. Si se vuelve a ejecutar el análisis, los
  números del documento hay que actualizarlos.

## Trazabilidad

Cada tabla y figura del Capítulo III se regenera desde
[`../motor-pronostico-novapack/`](../motor-pronostico-novapack/). El
`manifiesto.json` de la corrida indica con qué datos, qué configuración y qué
commit se produjo cada número, de modo que cualquier cifra del documento se puede
rastrear hasta su origen.
