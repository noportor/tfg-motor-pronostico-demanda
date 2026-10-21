# Documento de la tesis

Acá van el documento oficial y sus anexos.

## Antes de agregar un archivo

El control automático de confidencialidad **no puede mirar dentro** de los
`.docx`, `.pdf` ni `.xlsx`: son binarios comprimidos. Por eso la suite falla
si se versiona un `.docx`, `.xlsx`, `.pdf`, `.pptx` o `.zip`
(`test_no_se_versiona_ningun_documento_ofimatico_ni_comprimido`): las copias de
trabajo viven en `documento/privado/`, que git ignora. Si alguna vez se decide
publicar un archivo de ese tipo, hay que revisarlo a ojo y agregar en la prueba
una excepción con su motivo escrito. La revisión, en concreto:

- Que la empresa aparezca únicamente con su nombre de fantasía y no con el real,
  en el texto y también en encabezados, pies, metadatos y propiedades del
  archivo.
- Que no haya códigos de producto, listas de clientes ni capturas del sistema
  interno.
- Que el **Anexo 2** (acuerdo de confidencialidad) no exponga nombres, cargos ni
  firmas de terceros que no hayan consentido su publicación.
- Que las tablas y figuras coincidan con las salidas vigentes de
  `motor-pronostico/salidas/`. Si se vuelve a ejecutar el análisis, los
  números del documento hay que actualizarlos.

## Trazabilidad

Cada tabla y figura del Capítulo III se regenera desde
[`../motor-pronostico/`](../motor-pronostico/). El
`manifiesto.json` de la corrida indica con qué datos, qué configuración y qué
commit se produjo cada número, de modo que cualquier cifra del documento se puede
rastrear hasta su origen.
