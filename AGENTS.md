# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ project managed with `uv`. Application code lives in `app/`.

- `app/main.py` starts the real-time neural sign recognizer.
- `app/cli/` contains command-line entry points for collecting data, inspecting datasets, training, and recognition.
- `app/ml/` contains machine-learning classifiers and related model code.
- `app/storage/` contains the HDF5 dataset persistence layer.
- `app/vision/` contains camera capture, frame preprocessing, MediaPipe hand detection, and landmark extraction.
- `app/utils/` contains image drawing helpers.
- `app/api/`, `app/core/`, `app/models/`, and `app/services/` are placeholders for future API, domain, model, and service layers.
- `app/docs/` stores project reference documents, including the final project PDF.

There is no test directory yet. Add tests under `tests/` mirroring the source layout, for example `tests/vision/test_landmark_extractor.py`.

## Build, Test, and Development Commands

- `uv sync` installs dependencies from `pyproject.toml` and `uv.lock` into the local virtual environment.
- `uv run python -m app.main` runs the neural webcam recognizer. Press `q` in the video window to exit.
- `uv run python -m app.cli.collect_data --label HOLA` records sign samples into the HDF5 dataset.
- `uv run python -m app.cli.inspect_dataset data/signs_dataset.h5` summarizes the collected dataset.
- `uv run python -m app.cli.train_neural_network --dataset data/signs_dataset.h5` trains the neural classifier.
- `uv add <package>` adds a runtime dependency and updates project metadata.
- `uv lock` refreshes the lockfile after dependency changes.

No separate build step is currently required.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, type hints for public functions, and dataclasses for structured detection results when appropriate. Use `snake_case` for modules, functions, variables, and package names; use `PascalCase` for classes such as `HandDetector` and `DetectionResult`.

Keep OpenCV frames clearly labeled by color format in names or docstrings, for example `frame_bgr` and `frame_rgb`. Prefer small modules grouped by responsibility under `app/vision/` instead of adding unrelated logic to `main.py`.

No formatter or linter is configured yet. If one is introduced, document the exact command here and keep style-only changes separate from behavioral changes.

## Testing Guidelines

No testing framework is configured yet. When adding tests, prefer `pytest` and name files `test_*.py`. Focus first on deterministic logic such as landmark extraction, bounding boxes, preprocessing behavior, and utility drawing contracts. Avoid tests that require a physical webcam unless they are marked or separated as integration tests.

Run tests with `uv run pytest` once `pytest` is added to the project.

## Commit & Pull Request Guidelines

Recent history uses short Conventional Commit-style subjects such as `feat: add mediapipe hand tracking via webcam`. Continue using `feat:`, `fix:`, `docs:`, `test:`, and `refactor:` with imperative, specific descriptions.

Pull requests should include a concise summary, testing notes, and screenshots or short recordings for visible computer-vision changes. Link related issues or course tasks when available. Do not commit `.venv/`, generated caches, local camera captures, or large datasets/model artifacts unless explicitly required.
