# LESCO-AI

> [!NOTE]
Detección de manos en tiempo real usando MediaPipe y OpenCV, orientado al reconocimiento de la Lengua de Señas Costarricense (LESCO).

## Requisitos

> [!IMPORTANT]
>- Python 3.10+
>- [uv](https://docs.astral.sh/uv/getting-started/installation/) — manejador de paquetes y entornos virtuales

### Instalar uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```
source $HOME/.local/bin/env
```

Verificá que quedó instalado:
```bash
uv --version
```

## Instalación
> [!TIP]
> Cloná el repositorio y entrá a la carpeta:
> ```bash
> git clone https://github.com/tu-usuario/lesco-ai.git
> cd lesco-ai
> ```
> 
> Instalá las dependencias:
> ```bash
> uv sync
> ```

Esto crea automáticamente el entorno virtual y descarga todo lo necesario según `pyproject.toml` y `uv.lock`.

## Estructura del proyecto

```text
app/
  cli/       comandos ejecutables: captura, inspeccion, entrenamiento, reconocimiento
  ml/        modelos y clasificadores de aprendizaje automatico
  storage/   lectura/escritura del dataset HDF5
  vision/    camara, preprocesamiento, deteccion de manos y landmarks
  utils/     helpers de dibujo e imagen
  api/       placeholder para una API futura
  core/      placeholder para logica central futura
  models/    placeholder para modelos de dominio futuros
  services/  placeholder para servicios futuros
```

## Correr la app

> [!IMPORTANT]
> ```bash
> uv run python -m app.main
> ```

Se abre la camara, detecta tus manos con MediaPipe y usa el modelo entrenado
`models/neural_sign_classifier.npz` para mostrar la palabra mas probable.

Presioná `Q` para cerrar la ventana.

## App web (recomendado, reemplaza al CLI)

Hay un hub web (`app/frontend`) servido por un servidor FastAPI que reutiliza el
MISMO pipeline de detección que el CLI (no cambia la lógica). Levantalo con:

```bash
uv run uvicorn app.api.server:app --port 8000
```

Luego abrí <http://localhost:8000/> y dale permiso a la cámara. El menú tiene
cuatro secciones:

- **Reconocer** — HUD en vivo (predicción, barras de confianza, historial, radar
  de movimiento, overlay de landmarks con MediaPipe, partículas, FPS).
- **Grabar muestras** — graba señas al dataset *desde la propia cámara web*, con
  cuenta regresiva. Como se graban por el mismo pipeline que reconoce, el dataset
  coincide con lo que ve el reconocedor (evita el desajuste del CLI).
- **Entrenar modelo** — re-entrena con el dataset actual y recarga el modelo en
  caliente, sin reiniciar el servidor.
- **Inspeccionar** — etiquetas y cantidad de muestras por clase.

Flujo recomendado: **Grabar** (15-30 tomas por seña) → **Entrenar** → **Reconocer**.

### Endpoints

`POST /predict`, `POST /collect/{start,frame,save,discard}`, `POST /train`,
`GET /train/status`, `GET /api/dataset`.

### Ajustes por variable de entorno (opcional)

```bash
LESCO_MODEL=models/neural_sign_classifier.npz   # ruta del modelo
LESCO_DATASET=data/signs_dataset.h5             # ruta del dataset
LESCO_MIN_CLASS_DISTANCE_THRESHOLD=4.0          # afloja validación de distancia
LESCO_MIN_DETECTED_RATIO=0.30                   # exige menos "mano detectada"
LESCO_CONFIDENCE_THRESHOLD=0.6                  # baja el umbral de confianza
```

> Si regrabás el dataset por la web y re-entrenás, normalmente no hace falta
> aflojar estos umbrales: las distancias dejan de estar infladas.

## Entrenar la red neuronal

Primero grabá varias muestras por etiqueta con el recolector:

```bash
uv run python -m app.cli.collect_data --label HOLA
uv run python -m app.cli.collect_data --label GRACIAS
```

Cada seña se guarda en `data/signs_dataset.h5` con forma fija `(60, 126)`:
60 frames remuestreados y 126 valores por frame. Los 126 valores representan
2 manos x 21 landmarks x 3 coordenadas. La duracion original de la sena puede
variar; queda registrada en `original_lengths`, pero la red recibe siempre una
secuencia del mismo tamano.

Entrená el modelo con:

```bash
uv run python -m app.cli.train_neural_network --dataset data/signs_dataset.h5
```

El modelo se guarda por defecto en:

```text
models/neural_sign_classifier.npz
```

Para un resultado evaluable, graba al menos 20-30 muestras por sena. Con muy
pocas muestras, la red puede memorizar las tomas y no generalizar bien en vivo.
