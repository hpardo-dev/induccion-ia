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

## Reentrenar la red (`training/`)

```bash
pip install numpy
mkdir -p training/data && cd training/data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -O "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz"
done
cd ../.. && python3 training/train.py   # genera web/model.js
```
