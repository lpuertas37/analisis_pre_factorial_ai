# Arquitectura de agentes basada en literatura para generacion y validacion de items

## Proposito

Esta arquitectura convierte la ruta conceptual:

```text
AIG clasica -> LLMs para generacion de items -> agentes evaluadores
-> embeddings/PFA -> jueceo experto -> piloto -> analisis psicometrico real
```

en un flujo operativo para construir bancos de items organizacionales, tipo Skillia,
con trazabilidad teorica, control psicometrico y validacion semantica previa al
pilotaje.

## Base de literatura

| Fuente | Aporte al sistema | Uso en la arquitectura |
|---|---|---|
| Gierl & Haladyna (2012), DOI `10.4324/9780203803912` | Base clasica de AIG: modelos cognitivos, modelos de item, generacion sistematica y familias de items. | Agente de matriz teorica, agente de modelo de item y ensamblador. |
| Gierl, Lai & Turner (2012), DOI `10.1111/j.1365-2923.2012.04289.x` | Proceso AIG en tres etapas: modelo cognitivo, modelo de item, generacion por software. | Gate obligatorio antes de permitir generacion masiva. |
| Bezirhan & von Davier (2023), DOI `10.1016/j.caeai.2023.100161` | Uso practico de GPT con prompts, seleccion de salidas, revision humana y control de legibilidad/coherencia. | Agente de prompts, agente editor humano-asistido y agente de legibilidad. |
| Li et al. (2026), DOI `10.1016/j.chbr.2026.100964`; preprint `10.48550/arXiv.2412.12144` | Generacion automatica de SJTs de personalidad con prompts optimizados, temperatura, reproducibilidad entre modelos y validacion psicometrica. | Agente generador de SJTs, comparador de modelos y evaluador de reproducibilidad. |
| Laverghetta Jr. et al. (2024), DOI arXiv `10.48550/arXiv.2409.00202` | Pipeline CPIG con generadores y evaluadores LLM iterativos para mejorar items. | Orquestador multiagente, ciclo generar-evaluar-reformular. |
| Guenole et al. (2025), DOI OSF `10.31234/osf.io/vf3se_v2` | Pseudo factor analysis con matrices de similitud de embeddings para revisar estructura antes de datos reales. | Agente PFA y selector iterativo de items. |
| Guenole, Samo & Sun (2024), DOI OSF `10.31234/osf.io/9a4qx` | Pseudo-discriminacion desde embeddings para aproximar parametros antes de recolectar respuestas. | Agente de discriminacion semantica preliminar. |
| Suarez-Alvarez et al. (2026), DOI `10.70478/psicothema.2026.38.01` | Guia practica: calidad de datos, alineacion con uso previsto, comparacion de modelos, validacion humana, constructos, prompts, alineacion semantica, PFA y tecnicas model-free. | Marco de gobernanza, gates de validez, confiabilidad, equidad y privacidad. |

## Contrato de entrada

El sistema no debe generar items si faltan estos campos:

| Campo | Descripcion |
|---|---|
| `constructo` | Nombre del constructo focal. |
| `definicion_operacional` | Que conducta, juicio, actitud o habilidad se pretende medir. |
| `poblacion_objetivo` | Grupo para el que se disena el instrumento. |
| `contexto_uso` | Seleccion, diagnostico organizacional, formacion, investigacion, clima, etc. |
| `decision_prevista` | Que decision se tomara con los resultados. |
| `dimensiones_esperadas` | Subdimensiones teoricas o facetas. |
| `constructos_vecinos` | Constructos relacionados que deben diferenciarse. |
| `formato_item` | Likert, SJT, opcion multiple, respuesta abierta, forced choice. |
| `escala_respuesta` | Anclajes y numero de puntos. |
| `fuentes_base` | Papers, instrumentos, manuales, DOI, OSF, arXiv o notas expertas. |
| `restricciones` | Longitud, lenguaje, cultura, nivel educativo, terminos prohibidos, sesgo sensible. |

## Flujo de agentes

### 1. Agente bibliografico

**Base:** Suarez-Alvarez et al. (2026), Gierl & Haladyna (2012).

**Entrada:** constructo, contexto, fuentes base.

**Tareas:**

- verificar DOI, arXiv, OSF o fuente;
- extraer definiciones, dimensiones, instrumentos previos y evidencias de validez;
- separar evidencia fuerte, evidencia preliminar y supuestos;
- construir una matriz de literatura trazable.

**Salida:** `literatura_matriz.csv`

```csv
source_id,referencia,doi_o_url,constructo,dimension,definicion,evidencia,nota_uso
```

**Gate:** ningun constructo pasa a diseno si no tiene definicion y fuente trazable.

### 2. Agente de mapa nomologico

**Base:** Suarez-Alvarez et al. (2026), Guenole et al. (2025).

**Entrada:** definiciones del constructo focal y constructos vecinos.

**Tareas:**

- definir limites semanticos del constructo;
- declarar que NO mide el instrumento;
- comparar constructos vecinos mediante embeddings;
- marcar solapamientos teoricos antes de generar items.

**Salida:** `mapa_nomologico.csv`

```csv
constructo,definicion,constructo_vecino,similitud_semantica,riesgo_solapamiento,decision
```

**Gate:** si el constructo focal es indistinguible semanticamente de otro, se debe redefinir.

### 3. Agente constructor psicometrico

**Base:** Gierl & Haladyna (2012), Gierl et al. (2012).

**Entrada:** literatura validada y mapa nomologico.

**Tareas:**

- convertir dimensiones en indicadores observables;
- construir modelo cognitivo o modelo de respuesta;
- definir reglas de item por indicador;
- establecer dificultad, polaridad y restricciones.

**Salida:** `matriz_especificacion.csv`

```csv
constructo,dimension,indicador,conducta_observable,nivel,formato,reglas_item,fuente
```

**Gate:** ningun item puede generarse si no apunta a un indicador especifico.

### 4. Agente de modelo de item

**Base:** AIG clasica de Gierl & Haladyna.

**Entrada:** matriz de especificacion.

**Tareas:**

- crear plantillas o familias de items;
- separar elementos fijos y variables;
- definir restricciones de forma;
- documentar que partes pueden variar sin cambiar el constructo.

**Salida:** `modelos_item.csv`

```csv
modelo_id,dimension,indicador,stem_template,variables,restricciones,ejemplo_valido,ejemplo_invalido
```

**Gate:** un modelo de item debe poder generar multiples items equivalentes, no solo una frase bonita.

### 5. Agente generador Likert

**Base:** Suarez-Alvarez et al. (2026), Bezirhan & von Davier (2023).

**Entrada:** modelos de item, escala de respuesta, restricciones.

**Tareas:**

- generar varias versiones por indicador;
- variar redaccion, no el significado;
- controlar longitud, claridad y una sola idea por item;
- producir items directos e inversos solo si estan justificados.

**Salida:** `items_likert_candidatos.csv`

```csv
item_id,constructo,dimension,indicador,text,polarity,source_model,prompt_id,status
```

**Gate:** minimo 3 a 5 candidatos por indicador antes de filtrar.

### 6. Agente generador SJT

**Base:** Li et al. (2026).

**Entrada:** dimensiones, indicadores, contexto laboral y incidentes criticos.

**Tareas:**

- generar situacion realista;
- generar alternativas de respuesta;
- mapear cada alternativa a rasgo, faceta o nivel de efectividad;
- controlar cultura, rol, jerarquia y plausibilidad.

**Salida:** `items_sjt_candidatos.csv`

```csv
item_id,dimension,indicador,situacion,opcion_a,opcion_b,opcion_c,opcion_d,clave_o_rubrica,fuente
```

**Gate:** ningun SJT pasa si la situacion no exige juicio situado o si las opciones no son plausibles.

### 7. Agente comparador de modelos

**Base:** Suarez-Alvarez et al. (2026), Li et al. (2026).

**Entrada:** mismo prompt ejecutado en varios modelos o configuraciones.

**Tareas:**

- comparar modelos, temperaturas y prompts;
- medir estabilidad de dimensiones, calidad y diversidad;
- detectar dependencia de un solo modelo;
- recomendar el lote mas robusto.

**Salida:** `comparacion_generacion.csv`

```csv
modelo,prompt_id,temperatura,n_items,calidad_media,diversidad,reproducibilidad,observaciones
```

**Gate:** no usar un banco generado por un solo modelo sin revision comparativa si el uso es sensible.

### 8. Agente critico psicometrico

**Base:** Suarez-Alvarez et al. (2026), Gierl & Haladyna (2012).

**Entrada:** items candidatos.

**Tareas:**

- detectar doble contenido;
- detectar ambiguedad, generalidad, tautologia o deseabilidad social;
- verificar alineacion item-indicador;
- recomendar conservar, reformular, fusionar o eliminar.

**Salida:** `revision_psicometrica.csv`

```csv
item_id,problema,gravedad,decision,justificacion,reformulacion_sugerida
```

**Gate:** items con doble contenido o constructo ambiguo no pasan a PFA.

### 9. Agente de sesgo, equidad y sensibilidad cultural

**Base:** Suarez-Alvarez et al. (2026), Ho (2024), Hao et al. (2024).

**Entrada:** items filtrados y poblacion objetivo.

**Tareas:**

- detectar lenguaje excluyente o supuestos culturales;
- detectar dependencia innecesaria de rol, genero, edad, estatus, region o tecnologia;
- revisar privacidad y sensibilidad de datos;
- proponer redacciones equivalentes mas inclusivas.

**Salida:** `revision_equidad.csv`

```csv
item_id,riesgo_grupo,riesgo_contexto,gravedad,decision,reformulacion
```

**Gate:** items con riesgo alto requieren reformulacion o aprobacion experta explicita.

### 10. Agente editor y normalizador

**Base:** Bezirhan & von Davier (2023).

**Entrada:** items aprobados con observaciones.

**Tareas:**

- corregir gramatica;
- homogeneizar tono y longitud;
- preservar significado psicometrico;
- preparar version final candidata.

**Salida:** `items_maestros.csv`

```csv
item_id,construct,dimension,indicator,text,source,polarity,status,risk_level,review_notes
```

**Gate:** toda reformulacion debe conservar indicador, dimension y constructo.

### 11. Agente de alineacion semantica

**Base:** Suarez-Alvarez et al. (2026), Guenole et al. (2025).

**Entrada:** definiciones de constructos e items maestros.

**Tareas:**

- calcular similitud item-constructo;
- verificar que cada item sea mas similar a su constructo objetivo que a constructos vecinos;
- marcar items semanticamente fuera de lugar.

**Salida:** `alineacion_semantica.csv`

```csv
item_id,constructo_objetivo,sim_objetivo,constructo_mas_cercano,sim_vecino,margen,decision
```

**Gate:** items con margen bajo pasan a reformulacion antes de PFA.

### 12. Agente PFA y residuales

**Base:** Guenole et al. (2025), Suarez-Alvarez et al. (2026).

**Entrada:** CSV minimo `item_id,construct,text`.

**Tareas:**

- generar embeddings;
- construir matriz de similitud;
- ejecutar pseudo factor analysis;
- revisar cargas bajas, cargas cruzadas y baja diferenciacion;
- producir residuales, heatmaps y scree plot.

**Salida:** artefactos actuales de `analisis_pre_factorial.py`.

**Gate:** banco no pasa a jueceo experto si hay estructura semantica incoherente no explicada.

### 13. Agente de pseudo-discriminacion

**Base:** Guenole, Samo & Sun (2024).

**Entrada:** items candidatos y constructos/dimensiones.

**Tareas:**

- estimar discriminacion semantica preliminar item-constructo;
- detectar items demasiado genericos;
- priorizar items con mejor capacidad de diferenciar dimensiones.

**Salida:** `pseudo_discriminacion.csv`

```csv
item_id,dimension,pseudo_discriminacion,ranking_dimension,decision
```

**Gate:** items con pseudo-discriminacion baja deben justificarse o reformularse.

### 14. Agente de jueceo experto

**Base:** Bezirhan & von Davier (2023), Suarez-Alvarez et al. (2026).

**Entrada:** banco filtrado, PFA, alineacion semantica y revisiones.

**Tareas:**

- preparar paquete para jueces;
- calcular acuerdo por dimension, indicador y decision;
- consolidar comentarios;
- fijar banco piloto.

**Salida:** `jueceo_experto.csv`

```csv
item_id,juez,claridad,pertinencia,representatividad,sesgo,decision,comentarios
```

**Gate:** ningun instrumento organizacional sensible pasa a piloto sin revision humana.

### 15. Agente de piloto y psicometria real

**Base:** Suarez-Alvarez et al. (2026); estandares AERA, APA & NCME.

**Entrada:** banco piloto y respuestas reales.

**Tareas:**

- analizar confiabilidad;
- ejecutar EFA/CFA/IRT segun el caso;
- comparar estructura empirica contra PFA;
- revisar funcionamiento diferencial de items si aplica;
- emitir version final o nueva iteracion.

**Salida:** `reporte_psicometrico_real.html`

**Gate:** la PFA no reemplaza este paso. La evidencia empirica con personas es el criterio final.

## Orquestador recomendado

```text
Intake
-> Bibliografico
-> Mapa nomologico
-> Constructor psicometrico
-> Modelo de item
-> Generador Likert y/o SJT
-> Comparador de modelos
-> Critico psicometrico
-> Sesgo/equidad
-> Editor normalizador
-> Alineacion semantica
-> PFA/residuales
-> Pseudo-discriminacion
-> Jueceo experto
-> Piloto
-> Psicometria real
```

## Estados de decision por item

| Estado | Significado |
|---|---|
| `candidato` | Generado, aun sin revision. |
| `reformular` | Tiene valor teorico, pero requiere ajuste. |
| `fusionar` | Redundante con otro item. |
| `eliminar` | Problema no corregible o baja alineacion. |
| `preaprobado` | Pasa revision automatica y queda listo para jueceo experto. |
| `piloto` | Aprobado por expertos para aplicacion piloto. |
| `final` | Conservado luego de analisis psicometrico real. |

## Implementacion minima en el repo

Para que el repositorio cubra la idea completa, faltan estos componentes:

1. `generar_matriz_teorica.py`
2. `generar_items.py`
3. `revisar_items.py`
4. `evaluar_alineacion_semantica.py`
5. `estimar_pseudo_discriminacion.py`
6. `orquestar_banco_items.py`

El script existente `analisis_pre_factorial.py` corresponde principalmente a los
agentes 12 y parte del 11. La arquitectura actual del repo documenta el flujo,
pero todavia no implementa la generacion ni los agentes evaluadores previos.
