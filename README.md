# CVRecord

CVRecord is a single-window desktop webcam recorder with live preview and replay preview on the same screen.
You can Record, Pause, Stop playback, Play the current clip, Revert (discard) the clip, and Save as MP4.

## Runtime Folders

- `records/`: output videos saved as `<name>_<YYYYMMDD>_<HHMMSS>.mp4`.
- `icons/`: cached button icon PNG files generated at runtime for faster startup and reuse.

When running the built app, these folders are expected beside `CVRecord.exe`.

## Development

- `src/main.py`: main GUI app, recording/playback flow, and save logic.
- `src/app_paths.py`: runtime path helpers for app root, `records`, and `icons`.
- `src/camera/camera_detection.py`: camera and mode discovery.
- `src/icon_creator/icon_creator.py`: icon creation and cache loading.

## Platform and Build
The actual source code has been tested under Windows 11 only.

Build on Windows with the PowerShell script: `./build.ps1`.
