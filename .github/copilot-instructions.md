# CVRecord Development Instructions

Apply these rules for all implementation work in this repository.

## Product Goal
Build a single-screen desktop GUI for webcam recording with:
- one text input for base filename
- control buttons: Record, Pause, Stop, Play, Revert, Save
- live webcam preview and replay preview visible on the same screen

## Control Semantics (Authoritative)
- `Record`: start recording, or resume recording if paused.
- `Pause`: pause recording only. It must not trigger playback.
- `Stop`: stop playback.
- `Play`: play the current recorded clip. If recording is in progress, pause recording before playback.
- `Revert`: cancel and discard the current recording.
- `Save`: persist the current clip.

## Save Naming Convention
- Output filename format: `<text_input>_<YYYYMMDD>_<HHMMSS>.mp4`
- Sanitize the text input for filesystem safety.

## Quality and Compression
- Prioritize high visual quality with efficient compression.
- Prefer H.264 in MP4 with CRF-style quality settings when available.
- If OpenCV-only encoding limits quality/size efficiency on the host system, add and use `imageio-ffmpeg` (or equivalent FFmpeg-backed path).

## Development Guidelines
### Code Style (Ruff / pyproject.toml)
- Max line length: 120
- Target Python version: py314
- Keep formatting compatible with Ruff format settings:
  - quote-style = preserve
  - skip-magic-trailing-comma = false
- Linting rules enabled:
  - E (pycodestyle errors)
  - F (pyflakes)
  - I (isort)
  - B (bugbear)
  - UP (pyupgrade)
  - SIM (simplify)
- Import sorting behavior:
  - combine-as-imports = true
- Ignore/exclude these folders when running checks:
  - .venv
  - venv
  - __pycache__
  - build
  - dist
  - AppIcons

### Docstrings (Google Style)
- Use Google-style docstrings for all public APIs (public modules, classes, methods, and functions).
- Include these sections when applicable:
  - Args:
  - Returns:
  - Raises:
- In Raises:, use this format for each exception:
  - ExceptionType: description
- Do not use parenthetical exception notes. Avoid forms like:
  - ExceptionType: description (if ...)
  - description (raises ExceptionType)

