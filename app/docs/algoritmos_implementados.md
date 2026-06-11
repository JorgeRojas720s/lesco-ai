# Algoritmos implementados

Actualizado: 2026-06-07

Este documento resume con que trabaja cada algoritmo implementado en el
proyecto y para que se usa dentro del flujo de reconocimiento de senas LESCO.

## 1. Red neuronal

**Archivo principal:** `app/ml/neural_sign_classifier.py`

**Con que trabaja:**

- Secuencias de landmarks de manos extraidos con MediaPipe.
- Cada muestra tiene una forma fija de 60 frames.
- Cada frame contiene 126 valores: 2 manos x 21 puntos x 3 coordenadas.
- Los datos vienen del dataset HDF5, normalmente `data/signs_dataset.h5`.
- Las etiquetas son las senas grabadas, por ejemplo `HOLA`, `GRACIAS` o
  `BUENOS_DIAS`.

**Para que se usa:**

- Aprender patrones de movimiento y posicion de las manos a partir de ejemplos
  grabados.
- Predecir la sena mas probable para una secuencia capturada en vivo.
- Devolver probabilidades por etiqueta, no solo una respuesta final.

**Rol en el sistema:**

Es el modelo principal de clasificacion. Recibe una sena completa, ya segmentada
y remuestreada, y produce una distribucion de probabilidades sobre las clases
conocidas.

## 2. Filtro bayesiano

**Archivo principal:** `app/ml/bayesian_filter.py`

**Con que trabaja:**

- Probabilidades generadas por la red neuronal.
- Varias vistas temporales de una misma sena.
- Una distribucion posterior sobre las etiquetas conocidas.

**Para que se usa:**

- Estabilizar la decision de la red neuronal.
- Evitar que una prediccion aislada o ruidosa domine el resultado.
- Acumular evidencia antes de aceptar una sena como confiable.

**Rol en el sistema:**

Funciona como una capa probabilistica despues de la red neuronal. La red predice
varias veces sobre recortes de la misma sena, y el filtro combina esas
predicciones para obtener una decision mas estable.

## 3. Sistema basado en reglas

**Archivo principal:** `app/ml/rule_based_translator.py`

**Con que trabaja:**

- La etiqueta final estabilizada por red neuronal + filtro bayesiano.
- La confianza de esa prediccion.
- La cantidad de frames capturados para la sena.
- La presencia o ausencia de la mano en el tiempo.
- Estados de dedos cuando se quieran usar reglas simbolicas simples.

**Para que se usa:**

- Decidir si una prediccion debe agregarse al texto o ignorarse.
- Rechazar predicciones con confianza menor al umbral configurado.
- Exigir una cantidad minima de frames estables antes de aceptar una sena.
- Cerrar la palabra actual cuando la mano desaparece por mas de 1 segundo.
- Preparar reglas auxiliares por dedos, por ejemplo:
  - pulgar extendido y los demas dedos doblados -> posible `BIEN`
  - todos los dedos extendidos -> posible `ALTO`

**Rol en el sistema:**

No reemplaza al modelo. Su funcion es razonar sobre la salida del modelo para
convertir detecciones individuales en texto mas consistente.

## 4. Reconocimiento por templates con DTW

**Archivo principal:** `app/cli/recognize.py`

**Con que trabaja:**

- Templates promedio calculados desde el dataset HDF5.
- Secuencias de landmarks capturadas en vivo.
- Distancias entre secuencias de movimiento.

**Para que se usa:**

- Comparar una sena nueva contra ejemplos promedio del dataset.
- Medir que tan parecida es la secuencia capturada a cada clase conocida.
- Obtener un ranking de candidatos sin entrenar una red neuronal.

**Rol en el sistema:**

Es un reconocedor alternativo basado en comparacion de secuencias. Usa DTW
`Dynamic Time Warping` para alinear movimientos que pueden durar distinto tiempo
y calcular cual template se parece mas a la sena en vivo.

## Flujo principal actual

El flujo principal del reconocedor neuronal es:

```text
Camara
  -> MediaPipe Hands
  -> extraccion de landmarks
  -> segmentacion de movimiento
  -> red neuronal
  -> filtro bayesiano
  -> validacion contra referencias del dataset
  -> sistema basado en reglas
  -> texto final
```

En resumen:

- La **red neuronal** reconoce la sena.
- El **filtro bayesiano** estabiliza la prediccion.
- La **validacion por referencias del dataset** rechaza movimientos que no se
  parecen a la clase predicha.
- El **sistema basado en reglas** decide si la prediccion se agrega al texto.
- El reconocedor **DTW** queda como una alternativa por templates desde el CLI.
