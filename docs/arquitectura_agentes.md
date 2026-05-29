# Arquitectura de agentes para construccion de items

Esta arquitectura estandariza el flujo para construir items tipo Likert a partir de un constructo y preparar el banco para el analisis pre-factorial semantico.

## Contrato de entrada

El usuario entrega, como minimo:

- constructo;
- dimensiones o subdimensiones esperadas;
- poblacion objetivo;
- contexto de aplicacion;
- proposito del instrumento;
- escala de respuesta prevista;
- papers o instrumentos base.

## Agentes

| Agente | Entrada | Salida | Gate |
|---|---|---|---|
| Bibliografico | Constructo y papers base | Fuentes, definiciones, dimensiones e instrumentos existentes | Las fuentes deben ser trazables |
| Constructor psicometrico | Fuentes y constructo | Matriz de especificacion: constructo, dimension, definicion operacional, indicador | Ningun item sin indicador |
| Generador AIG | Matriz de especificacion | Variantes de items por indicador | Cada item mide una sola idea |
| Critico psicometrico | Banco inicial | Alertas de doble contenido, ambiguedad, generalidad y deseabilidad social | Conservar, reformular o eliminar |
| Revisor de sesgo/equidad | Items y poblacion | Riesgos por grupo, rol, cultura o jerarquia | Ajustar lenguaje y supuestos |
| Reescritor | Items marcados | Version reformulada con el mismo indicador | No cambiar el constructo medido |
| Preparador CSV | Matriz maestra | CSV minimo `item_id,construct,text` | Compatible con `analisis_pre_factorial.py` |
| NLP/PFA | CSV minimo | Cargas, decisiones, residuales, validacion de constructos y reporte HTML | Revisar items marcados |
| Comparador | Salidas por modelo/rotacion | Robustez entre embeddings, oblimin y varimax | Senales consistentes tienen prioridad |
| Integrador | Todas las salidas | Banco piloto depurado | Decision final por item |

## Flujo recomendado

```text
papers
-> constructos
-> dimensiones
-> indicadores
-> items candidatos
-> revision psicometrica
-> revision de sesgo
-> matriz maestra
-> CSV minimo
-> embeddings + pseudo factor analysis
-> comparacion de modelos y rotaciones
-> banco piloto
-> pilotaje con respuestas reales
-> EFA/CFA/IRT
```

## Matriz maestra de items

Usar `templates/items_maestros_template.csv` como contrato principal:

```csv
item_id,construct,dimension,indicator,text,source,polarity,status,risk_level,review_notes
```

Para correr el script pre-factorial, exportar solo:

```csv
item_id,construct,text
```

## Criterios de decision

- Conservar: item claro, unidimensional, trazable y semanticamente alineado.
- Reformular: item ambiguo, demasiado general, con baja diferenciacion o riesgo de sesgo corregible.
- Fusionar: item redundante con otro de la misma dimension.
- Eliminar: item sin indicador claro, con carga semantica debil persistente o solapamiento no corregible.

## Nota metodologica

El analisis NLP/PFA es pre-factorial. Ayuda a revisar coherencia semantica antes del pilotaje, pero no reemplaza EFA/CFA/IRT con respuestas reales.
