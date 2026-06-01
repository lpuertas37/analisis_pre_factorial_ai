# Agente bibliografico de instrumentos existentes

## Objetivo

Antes de generar items nuevos, el sistema debe buscar si ya existen instrumentos,
escalas, adaptaciones o bancos de items para el constructo. La generacion con IA
debe partir de evidencia existente cuando la literatura la ofrece, especialmente
si hay adaptaciones al contexto, poblacion o idioma objetivo.

## Fuentes recomendadas

1. Consensus para busqueda inicial de papers revisados por pares.
2. Crossref, PubMed, Semantic Scholar u OpenAlex para verificar DOI y metadatos.
3. PsycINFO, ERIC, Scopus o Web of Science si se dispone de acceso institucional.
4. OSF, PsyArXiv o arXiv para preprints y materiales suplementarios.
5. Manuales tecnicos, anexos, suplementos y repositorios donde puedan aparecer
   items completos.

## Busquedas sugeridas en Consensus

Usar una combinacion por constructo, poblacion y formato:

```text
"nombre del constructo" scale items validation year:2010-2026
"nombre del constructo" questionnaire instrument psychometric year:2010-2026
"nombre del constructo" workplace scale validation year:2010-2026
"nombre del constructo" organizational assessment instrument year:2010-2026
"nombre del constructo" situational judgment test personality year:2010-2026
"nombre del constructo" Spanish adaptation scale items year:2010-2026
```

Para contextos Skillia/organizacionales:

```text
"constructo" employee scale validation
"constructo" workplace questionnaire psychometric
"constructo" organizational behavior measure items
"constructo" selection assessment situational judgment test
```

## Criterios de seleccion

Priorizar fuentes que tengan:

- DOI o identificador verificable;
- instrumento nombrado;
- definicion clara del constructo;
- dimensiones o facetas reportadas;
- poblacion y contexto de aplicacion;
- propiedades psicometricas;
- items completos en texto, anexos o suplementos;
- adaptacion al idioma, cultura o contexto objetivo.

## Extraccion minima

Registrar las fuentes en:

```text
templates/fuentes_instrumentos_template.csv
```

Columnas:

```csv
source_id,constructo,dimension,instrumento,referencia,doi_o_url,poblacion_original,contexto_original,idioma,adaptacion_contexto,item_id_original,item_text,escala_respuesta,notas
```

Si el paper reporta el instrumento pero no muestra los items, dejar `item_text`
vacio y explicar en `notas` donde podria estar el material.

## Preparar la revision

```powershell
python tools/preparar_revision_instrumentos.py `
  --input templates/fuentes_instrumentos_template.csv `
  --output-dir salidas_revision_instrumentos `
  --contexto-objetivo organizacion `
  --poblacion-objetivo trabajadores
```

Genera:

- `instrumentos_existentes.csv`: instrumentos detectados, trazabilidad, contexto
  y utilidad como punto de partida.
- `items_existentes.csv`: items textuales disponibles, decision inicial y motivo.
- `revision_instrumentos_resumen.json`: resumen de trazabilidad y utilidad.

## Uso dentro del flujo de agentes

```text
constructo
-> busqueda Consensus/literatura
-> fuentes_instrumentos_template.csv
-> preparar_revision_instrumentos.py
-> instrumentos_existentes.csv + items_existentes.csv
-> matriz teorica
-> adaptacion/generacion de items
-> revision psicometrica
-> PFA/embeddings
```

## Gate metodologico

No generar items desde cero si existe un instrumento con items disponibles y
buena trazabilidad. En ese caso, el banco nuevo debe justificar si adapta,
reformula, combina o excluye items existentes.
