# Inducción en IA

Material de formación para equipos médicos sobre cómo funcionan las redes neuronales aplicadas al diagnóstico por imagen.

## Dentro de una red neuronal (`web/`)

Animación interactiva en pixel art de una red neuronal real (784 → 16 → 16 → 10) entrenada con MNIST (95,4 % de aciertos en el conjunto de prueba). Permite:

- Dibujar un dígito y ver cómo la señal atraviesa la red hasta las 10 probabilidades de salida.
- Inspeccionar cualquier neurona y ver qué patrón de píxeles la activa o la apaga.
- Rebobinar el entrenamiento (desde pesos aleatorios hasta la red final) con la curva de error.
- Ver un mapa de calor de qué píxeles empujaron la decisión.
- Añadir ruido o tapar zonas para provocar errores.
- Seguir un recorrido guiado de 7 pasos con analogías clínicas y consultar un diccionario de términos.

Para usarla basta con abrir `web/index.html` en un navegador; no necesita servidor ni instalación.

## Pregúntale a la normativa (`rag/`)

Demo interactiva de un asistente RAG + LLM sobre la normativa y los datos de un hospital ficticio (todos los documentos y cifras son sintéticos). Muestra los cuatro pasos: pregunta, búsqueda de fragmentos (con puntuación visible), consulta que recibe el LLM y respuesta con citas clicables. Incluye control de acceso por perfil (médico/a, enfermería, administración), edición de documentos para ver cómo cambia la respuesta sin reentrenar, y comparación con el LLM sin documentos.

Publicada como artifact de Claude, responde en vivo con Claude (con permiso del usuario). Abierta como archivo local, usa respuestas grabadas para las preguntas sugeridas.

## Un paciente, tres sistemas (`fhir/`)

Demo interactiva de las ventajas de la interoperabilidad con HL7 FHIR, ambientada en Bolivia con datos sintéticos. María, de 67 años, llega a Urgencias con sus antecedentes repartidos entre un centro de salud, un laboratorio y el hospital. Un interruptor compara el caso sin interoperabilidad (reacción alérgica a la penicilina y una HbA1c repetida) con el caso con FHIR (alerta de alergia y prueba evitada). Incluye los "Rayos X" de cada dato (recursos FHIR R4 con códigos SNOMED CT, LOINC, CIE-10 y UCUM), el tráfico REST simulado entre sistemas, una comparación HL7 v2 frente a FHIR, un guion de 7 pasos y un botón para reiniciar el caso.

## Reentrenar la red (`training/`)

```bash
pip install numpy
mkdir -p training/data && cd training/data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -O "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz"
done
cd ../.. && python3 training/train.py   # genera web/model.js
```
