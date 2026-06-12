PROYECTO 1 - LESCO-AI
Instrucciones basicas de ejecucion

1. Requisitos

- Python 3.10 o superior.
- uv, manejador de dependencias para Python.
- Camara web funcional.

Para instalar uv:

Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

Verificar instalacion:
uv --version

2. Entrar a la carpeta del codigo

Descomprimir el archivo ZIP y entrar a la carpeta del proyecto:

cd codigo/lesco-ai

Si el proyecto se esta ejecutando directamente desde esta entrega, la carpeta puede
llamarse solamente:

cd lesco-ai

3. Instalar dependencias

Ejecutar:

uv sync

Este comando crea el entorno virtual automaticamente e instala las dependencias
definidas en pyproject.toml y uv.lock.

4. Ejecutar la aplicacion web recomendada

Desde la carpeta lesco-ai, ejecutar:

uv run uvicorn app.api.server:app --port 8000

Si se desea ejecutar con umbrales ajustados para mejorar la deteccion desde la
camara del navegador, usar uno de estos comandos segun el sistema operativo:

Linux/macOS:

LESCO_MIN_DETECTED_RATIO=0.35 LESCO_MIN_CLASS_DISTANCE_THRESHOLD=2.5 uv run uvicorn app.api.server:app --port 8000 --no-access-log

Windows PowerShell:

$env:LESCO_MIN_DETECTED_RATIO="0.35"; $env:LESCO_MIN_CLASS_DISTANCE_THRESHOLD="2.5"; uv run uvicorn app.api.server:app --port 8000 --no-access-log

Luego abrir en el navegador:

http://localhost:8000/

Permitir el acceso a la camara cuando el navegador lo solicite.

La aplicacion web permite:

- Reconocer señas en vivo.
- Grabar muestras nuevas para el dataset.
- Entrenar nuevamente el modelo.
- Inspeccionar las etiquetas y muestras disponibles.

Flujo recomendado de uso:

1. Entrar a "Grabar muestras" y capturar varias tomas por seña.
2. Entrar a "Entrenar modelo" para actualizar el clasificador.
3. Entrar a "Reconocer" para probar el reconocimiento en vivo.

5. Ejecutar el reconocedor por consola/ventana OpenCV

Como alternativa a la interfaz web, ejecutar:

uv run python -m app.main

Se abrira una ventana usando la camara local. Para cerrar la ventana, presionar:

Q

6. Comandos opcionales

Recolectar muestras desde CLI:

uv run python -m app.cli.collect_data --label HOLA

Entrenar el modelo con el dataset incluido:

uv run python -m app.cli.train_neural_network --dataset data/signs_dataset.h5

Inspeccionar el dataset:

uv run python -m app.cli.inspect_dataset --dataset data/signs_dataset.h5

7. Archivos importantes

- data/signs_dataset.h5: dataset de señas.
- models/neural_sign_classifier.npz: modelo entrenado esperado por la aplicacion.
- app/api/server.py: servidor web FastAPI.
- app/frontend/index.html: interfaz web.
- app/main.py: reconocedor en tiempo real por OpenCV.

Nota:
Si el modelo no existe o se desea actualizar, primero grabar muestras y entrenar
desde la aplicacion web o desde los comandos indicados arriba.
