# Analisis pre-factorial con items y embeddings

Este script arma un flujo pre-factorial inspirado en Suarez-Alvarez et al. (2026),
Psicothema, 38(1), 1-12, DOI `10.70478/psicothema.2026.38.01`.

Sirve para depurar un banco de items antes del pilotaje con participantes:

- convierte textos de items en vectores;
- calcula una matriz de similitud semantica;
- extrae una solucion factorial exploratoria sobre esa matriz con rotacion `oblimin`, `varimax` o sin rotacion;
- marca items con carga baja, carga cruzada o poca diferenciacion;
- genera heatmaps y distribucion de residuales como diagnostico model-free.

## Instalacion base

```powershell
cd D:\proyectos\analisis_pre_factorial_ai
pip install -r requirements.txt
```

## Instalacion mas fiel al paper

Para usar embeddings semanticos tipo transformer:

```powershell
cd D:\proyectos\analisis_pre_factorial_ai
pip install -r requirements-embeddings.txt
```

El script usa `--backend auto` por defecto. Si `sentence-transformers` esta instalado,
usa embeddings semanticos; si no, cae a `TF-IDF` local.

## Uso rapido local

```powershell
python analisis_pre_factorial.py --items ejemplo_items.csv --n-factors 4 --backend tfidf --output-dir salidas_ejemplo
```

## Uso recomendado para ser mas fiel al paper

```powershell
python analisis_pre_factorial.py `
  --items ejemplo_items.csv `
  --n-factors 4 `
  --rotation oblimin `
  --backend sentence-transformers `
  --sentence-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --iterate `
  --output-dir salidas_semanticas
```

## Comparar varios modelos de embeddings

El paper recomienda no depender ciegamente de una sola salida de IA. Puedes comparar
varios modelos asi:

```powershell
python analisis_pre_factorial.py `
  --items ejemplo_items.csv `
  --n-factors 4 `
  --backend sentence-transformers `
  --compare-models "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,sentence-transformers/distiluse-base-multilingual-cased-v2" `
  --output-dir salidas_comparacion
```

Esto genera `comparacion_modelos.csv` y una carpeta `comparacion_modelos` con
las salidas completas de cada modelo.

## Formato del CSV

Columnas obligatorias:

- `item_id`: identificador unico del item.
- `text`: texto completo del item.

Columna recomendada:

- `construct`: dimension teorica esperada.

Cuando incluyes `construct`, el script compara el factor asignado contra la
dimension teorica esperada y genera `validacion_constructos.csv`.

## Plantillas para agentes

La carpeta `templates` contiene contratos estandar para construir bancos de
items antes del analisis:

- `constructos_template.csv`: matriz teorica para constructos, dimensiones,
  definiciones operacionales e indicadores.
- `items_maestros_template.csv`: matriz maestra con trazabilidad completa por
  item.
- `items_minimo_script_template.csv`: formato minimo que consume este script.

La arquitectura de agentes esta documentada en:

```text
docs/arquitectura_agentes.md
docs/arquitectura_agentes_literatura.md
```

Flujo recomendado:

```text
constructo -> matriz teorica -> items candidatos -> revision psicometrica
-> revision de sesgo -> matriz maestra -> CSV minimo -> analisis pre-factorial
```

Para exportar una matriz maestra al formato minimo:

```powershell
python tools/exportar_csv_pre_factorial.py `
  --input templates/items_maestros_template.csv `
  --output salidas_preparacion/items_para_pre_factorial.csv
```

## Generar matriz teorica e items candidatos

Para cubrir el flujo completo desde constructos hasta items, ahora hay dos
comandos previos al analisis pre-factorial.

Primero, convierte una matriz de constructos/dimensiones en una matriz teorica
trazable:

```powershell
python tools/generar_matriz_teorica.py `
  --input templates/constructos_template.csv `
  --output salidas_preparacion/matriz_teorica.csv `
  --formato likert
```

La entrada debe incluir, como minimo:

```text
constructo,dimension,definicion_conceptual,definicion_operacional,poblacion,contexto,fuente_teorica
```

El script conserva `indicador` y `conducta_observable` si ya existen; si faltan,
los deriva de la dimension y la definicion operacional. Tambien marca nivel de
evidencia y riesgo preliminar de solapamiento.

Luego genera items candidatos:

```powershell
python tools/generar_items.py `
  --input salidas_preparacion/matriz_teorica.csv `
  --output salidas_preparacion/items_maestros_generados.csv `
  --por-indicador 4 `
  --minimo-output salidas_preparacion/items_para_pre_factorial.csv
```

La salida maestra conserva trazabilidad por constructo, dimension, indicador,
fuente, polaridad, estado y notas de revision. El CSV minimo queda listo para el
analisis pre-factorial.

Luego puedes correr:

```powershell
python analisis_pre_factorial.py `
  --items salidas_preparacion/items_para_pre_factorial.csv `
  --backend sentence-transformers `
  --rotation oblimin `
  --iterate `
  --output-dir salidas_pre_factorial
```

## Salidas

- `loadings_y_decisiones.csv`: cargas rotadas, factor asignado, decision y KMO por item.
- `matriz_similitud.csv`: similitud semantica entre items.
- `matriz_residuales.csv`: diferencia entre similitud observada y reproducida.
- `autovalores.csv`: autovalores y varianza acumulada.
- `heatmap_similitud.png`, `heatmap_residuales.png`, `scree_plot.png`, `distribucion_residuales.png`.
- `resumen.json`: resumen ejecutivo del analisis.
- `validacion_constructos.csv`: acuerdo entre constructo esperado y factor asignado, si existe `construct`.
- `reporte_pre_factorial.html`: reporte visual con metricas, items a revisar y graficos.
- `resumen_iterativo.csv`: aparece si usas `--iterate`.
- `comparacion_modelos.csv`: aparece si usas `--compare-models`.

## Que lo hace mas fiel al paper

La version ampliada sigue mejor las guias 9 y 10 del articulo:

- usa embeddings semanticos de modelos transformer;
- analiza la matriz de similitud entre items;
- extrae una solucion pseudo-factorial;
- usa `oblimin` por defecto, porque en psicologia normalmente esperamos factores correlacionados;
- revisa cargas bajas, cargas cruzadas y baja diferenciacion;
- evalua residuales con tecnicas model-free;
- permite depuracion iterativa del banco de items;
- compara estabilidad entre modelos de embeddings;
- contrasta factores obtenidos con constructos teoricos esperados.

## Criterio de items a revisar

Con embeddings semanticos es normal que muchas cargas secundarias sean moderadas,
porque los items comparten vocabulario y un tema general. Por eso el script no
marca una carga cruzada solo por superar `0.30`; ahora exige que la carga
secundaria sea alta y que ademas este demasiado cerca de la carga primaria.

Reglas por defecto:

- revisar si la carga primaria es menor a `0.40`;
- revisar si la diferencia entre carga primaria y secundaria es menor a `0.15`;
- revisar por carga cruzada si la secundaria supera `0.30` y tambien supera el
  `75%` de la carga primaria.

Puedes ajustar esos criterios:

```powershell
python analisis_pre_factorial.py `
  --items ejemplo_items.csv `
  --backend sentence-transformers `
  --n-factors 4 `
  --min-primary-loading 0.40 `
  --max-cross-loading 0.30 `
  --max-cross-loading-ratio 0.75 `
  --min-loading-gap 0.15
```

## Rotacion factorial

Por defecto el script usa `oblimin`, una rotacion oblicua. Es recomendable cuando
las dimensiones teoricas pueden estar correlacionadas, por ejemplo autoeficacia,
uso critico, etica y aprendizaje.

Opciones disponibles:

- `--rotation oblimin`: factores correlacionados, opcion recomendada.
- `--rotation varimax`: factores ortogonales/no correlacionados.
- `--rotation none`: sin rotacion.

Ejemplo:

```powershell
python analisis_pre_factorial.py `
  --items ejemplo_items.csv `
  --backend sentence-transformers `
  --n-factors 4 `
  --rotation oblimin `
  --output-dir salidas_oblimin
```

## Nota metodologica

Este analisis no sustituye un EFA/CFA con datos de respuestas reales. Es una fase
pre-factorial para revisar coherencia semantica, redundancia y posibles cargas
cruzadas antes de aplicar el instrumento en campo.
