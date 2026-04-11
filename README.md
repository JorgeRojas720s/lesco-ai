
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

## Correr la app

> [!IMPORTANT]
> ```bash
> uv run python -m app.main
> ```

Se abre la cámara y empezás a ver los 21 puntos de la mano detectados en tiempo real.

Presioná `Q` para cerrar la ventana.
