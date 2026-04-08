"""Single-window webcam recorder with live and replay views."""

from __future__ import annotations

import contextlib
import re
import subprocess
import time
import tomllib
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
from pytablericons import outline_icon

from app_paths import get_records_dir, get_settings_path
from camera.camera_detection import CameraDevice, CameraMode, list_cameras as list_external_cameras
from icon_creator.icon_creator import create_icon

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360
PREVIEW_MAX_WIDTH = 640
PREVIEW_MAX_HEIGHT = 480
CAMERA_INDEX = 0
CAPTURE_FPS = 30
ACTIVE_RECORD_COLOR = "#C62828"
ACTIVE_PAUSE_COLOR = "#C62828"
ACTIVE_PLAY_COLOR = "#2E7D32"


_set_log_level = cv2.__dict__.get("setLogLevel")
if callable(_set_log_level):
    with contextlib.suppress(Exception):
        _set_log_level(2)  # Keep only error-level OpenCV logs.


def sanitize_filename_base(text: str) -> str:
    """Return a filesystem-safe base filename.

    Args:
        text: User-provided base filename text.

    Returns:
        A sanitized, non-empty filename stem.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "recording"


def build_output_path(base_text: str) -> Path:
    """Create a timestamped output path in the records directory.

    Args:
        base_text: User-provided base filename text.

    Returns:
        Output path with format <base>_<YYYYMMDD>_<HHMMSS>.mp4.
    """
    safe_base = sanitize_filename_base(base_text)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records_dir = get_records_dir()
    records_dir.mkdir(parents=True, exist_ok=True)
    return records_dir / f"{safe_base}_{timestamp}.mp4"


def open_camera_capture(camera_index: int):
    """Open a camera by index for frame capture."""
    return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)


def safe_camera_read(capture) -> tuple[bool, np.ndarray | None]:
    """Safely read a frame from a capture without raising OpenCV exceptions.

    Args:
        capture: OpenCV VideoCapture instance.

    Returns:
        Tuple of (success, frame). Frame is None when no valid frame is available.
    """
    if capture is None or not capture.isOpened():
        return False, None

    try:
        ok, frame = capture.read()
    except cv2.error:
        return False, None
    if not ok or frame is None:
        return False, None
    return True, frame


class H264FFmpegWriter:
    """Write video frames to H.264 MP4 via FFmpeg rawvideo input."""

    def __init__(self, output_path: Path, width: int, height: int, fps: int) -> None:
        """Initialize encoder settings.

        Args:
            output_path: Target MP4 path.
            width: Frame width.
            height: Frame height.
            fps: Frame rate.
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """Start an FFmpeg process configured for CRF-based H.264 encoding.

        Raises:
            RuntimeError: If ffmpeg executable cannot be found.
        """
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if not ffmpeg_exe:
            raise RuntimeError("Unable to locate ffmpeg executable.")

        command = [
            ffmpeg_exe,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr) -> None:
        """Write a single BGR frame to ffmpeg stdin.

        Args:
            frame_bgr: OpenCV BGR frame with shape (height, width, 3).

        Raises:
            RuntimeError: If writer has not been started.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Writer process is not started.")
        self._process.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        """Finalize the MP4 file and validate encoder exit status.

        Raises:
            RuntimeError: If ffmpeg exits with a non-zero status code.
        """
        if self._process is None:
            return

        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait()
        if self._process.returncode != 0:
            raise RuntimeError("FFmpeg failed to encode the output video.")
        self._process = None


class RecorderApp(ctk.CTk):
    """Desktop GUI for webcam recording, replay, revert, and save."""

    def __init__(self) -> None:
        """Create the UI, initialize camera, and start preview loops."""
        super().__init__()

        # Pre-create reusable icons
        self.icon_rec = create_icon(outline_icon.OutlineIcon.PLAYER_RECORD)
        self.icon_pause = create_icon(outline_icon.OutlineIcon.PLAYER_PAUSE)
        self.icon_revert = create_icon(outline_icon.OutlineIcon.ARROW_BACK_UP)
        self.icon_play = create_icon(outline_icon.OutlineIcon.PLAYER_PLAY)
        self.icon_stop = create_icon(outline_icon.OutlineIcon.PLAYER_STOP)

        self.title("CVRecord")
        self.geometry("1080x700")
        self.minsize(1080, 700)

        self.camera_index = CAMERA_INDEX
        self.capture_width = PREVIEW_WIDTH
        self.capture_height = PREVIEW_HEIGHT
        self.capture_fps = float(CAPTURE_FPS)
        self.capture_pixel_format = "unknown"
        self.settings_path = get_settings_path()
        saved_settings = self._load_settings()
        self.preview_width = PREVIEW_WIDTH
        self.preview_height = PREVIEW_HEIGHT
        self.camera_devices: list[CameraDevice] = []
        self._refresh_external_camera_info()
        if self.available_cameras and self.camera_index not in self.available_cameras:
            self.camera_index = self.available_cameras[0]

        best_mode = self._best_mode_for_index(self.camera_index) if self.available_cameras else None
        if best_mode is not None:
            self.capture_width = best_mode.width
            self.capture_height = best_mode.height
            self.capture_fps = best_mode.fps
            self.capture_pixel_format = best_mode.pixel_format

        self._apply_saved_settings(saved_settings)

        self._update_preview_dimensions(self.capture_width, self.capture_height)

        self.supported_sizes = self._sizes_for_index(self.camera_index) if self.available_cameras else []
        if (self.capture_width, self.capture_height) not in self.supported_sizes:
            self.supported_sizes.insert(0, (self.capture_width, self.capture_height))
        self.capture = (
            self._open_capture(self.camera_index, self.capture_width, self.capture_height, self.capture_fps)
            if self.available_cameras
            else cv2.VideoCapture()
        )

        self.record_state = "idle"  # idle, recording, paused
        self.is_playing = False
        self.play_index = 0
        self._play_job: str | None = None

        self.recorded_frames = []
        self.last_live_frame = None
        self.last_replay_frame = None
        self.live_photo = None
        self.replay_photo = None
        self.read_failures = 0
        self.record_elapsed_before_pause = 0.0
        self.record_started_at: float | None = None

        self.filename_var = ctk.StringVar(value="recording")
        self.status_var = ctk.StringVar(value="Ready")
        self._initial_focus_done = False

        self._build_ui()
        self._bind_shortcuts()
        self._show_empty_replay()
        self._update_transport_button_colors()
        if not self.capture.isOpened():
            self._set_status("Webcam unavailable. Open Settings to choose another device.")
        else:
            self._save_settings(
                camera_id=self.camera_index,
                width=self.capture_width,
                height=self.capture_height,
                fps=self.capture_fps,
            )
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(15, self._update_live_view)

    def _load_settings(self) -> dict[str, int | float] | None:
        """Load camera settings from TOML when available.

        Returns:
            Parsed camera settings dict, or None when unavailable/invalid.
        """
        if not self.settings_path.exists():
            return None

        try:
            data = tomllib.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return None

        def parse_int_field(value: object) -> int | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value) if value.is_integer() else None
            if isinstance(value, str):
                try:
                    return int(value.strip())
                except ValueError:
                    return None
            return None

        def parse_float_field(value: object) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int | float):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value.strip())
                except ValueError:
                    return None
            return None

        camera_id = parse_int_field(data.get("camera_id"))
        width = parse_int_field(data.get("width"))
        height = parse_int_field(data.get("height"))
        fps = parse_float_field(data.get("fps"))
        if camera_id is None or width is None or height is None or fps is None:
            return None

        parsed = {
            "camera_id": camera_id,
            "width": width,
            "height": height,
            "fps": fps,
        }

        if parsed["width"] <= 0 or parsed["height"] <= 0 or parsed["fps"] <= 0:
            return None
        return parsed

    def _apply_saved_settings(self, settings: dict[str, int | float] | None) -> None:
        """Apply persisted camera settings when valid and available.

        Args:
            settings: Persisted settings mapping or None.
        """
        if settings is None or not self.available_cameras:
            return

        saved_camera = int(settings["camera_id"])
        if saved_camera in self.available_cameras:
            self.camera_index = saved_camera

        saved_width = int(settings["width"])
        saved_height = int(settings["height"])
        saved_fps = float(settings["fps"])

        modes = self._modes_for_index(self.camera_index)
        matching_modes = [
            mode
            for mode in modes
            if mode.width == saved_width and mode.height == saved_height and abs(mode.fps - saved_fps) < 0.01
        ]
        if matching_modes:
            preferred = max(
                matching_modes,
                key=lambda mode: 1 if mode.pixel_format.lower() in {"mjpeg", "mjpg"} else 0,
            )
            self.capture_width = preferred.width
            self.capture_height = preferred.height
            self.capture_fps = preferred.fps
            self.capture_pixel_format = preferred.pixel_format
            return

        sizes = self._sizes_for_index(self.camera_index)
        if (saved_width, saved_height) in sizes:
            self.capture_width = saved_width
            self.capture_height = saved_height
            fps_values = self._fps_values_for(self.camera_index, saved_width, saved_height)
            if fps_values:
                self.capture_fps = min(fps_values, key=lambda value: abs(value - saved_fps))

    def _save_settings(self, camera_id: int, width: int, height: int, fps: float) -> None:
        """Persist current camera selection to settings.toml.

        Args:
            camera_id: Selected camera index.
            width: Selected capture width.
            height: Selected capture height.
            fps: Selected capture fps.
        """
        records_dir = self.settings_path.parent
        records_dir.mkdir(parents=True, exist_ok=True)
        content = (
            f"camera_id = {int(camera_id)}\nwidth = {int(width)}\nheight = {int(height)}\nfps = {float(fps):.2f}\n"
        )
        with contextlib.suppress(OSError):
            self.settings_path.write_text(content, encoding="utf-8")

    def _bind_shortcuts(self) -> None:
        """Register keyboard shortcuts for common actions."""
        self.bind("<Control-s>", self._on_save_shortcut)
        self.bind("<Control-S>", self._on_save_shortcut)

    def _on_save_shortcut(self, event) -> str | None:
        """Handle keyboard save shortcut.

        Args:
            event: Tk key event.

        Returns:
            ``"break"`` when shortcut is handled, otherwise ``None``.
        """
        focused_widget = self.focus_get()
        if isinstance(focused_widget, ctk.CTkEntry):
            return None

        self._on_save()
        return "break"

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        top.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(top, text="Filename Base").grid(row=0, column=0, padx=(12, 8), pady=10)
        self.name_entry = ctk.CTkEntry(top, textvariable=self.filename_var)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=10)
        ctk.CTkButton(top, text="Settings", command=self._open_settings).grid(
            row=0,
            column=2,
            padx=(0, 12),
            pady=10,
        )

        center = ctk.CTkFrame(self)
        center.grid(row=1, column=0, sticky="nsew", padx=14, pady=8)
        center.grid_columnconfigure((0, 1), weight=1)
        center.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(center, text="Live Webcam").grid(row=0, column=0, pady=(12, 6))
        ctk.CTkLabel(center, text="Replay").grid(row=0, column=1, pady=(12, 6))

        self.live_label = ctk.CTkLabel(center, text="", width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        self.live_label.grid(row=1, column=0, padx=10, pady=(0, 12), sticky="nsew")

        self.replay_label = ctk.CTkLabel(center, text="", width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        self.replay_label.grid(row=1, column=1, padx=10, pady=(0, 12), sticky="nsew")

        live_controls = ctk.CTkFrame(center)
        live_controls.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 12))
        self.record_button = ctk.CTkButton(live_controls, text="", image=self.icon_rec, command=self._on_record)
        self.record_button.pack(side="left", padx=8, pady=10)
        self.pause_button = ctk.CTkButton(live_controls, text="", image=self.icon_pause, command=self._on_pause)
        self.pause_button.pack(side="left", padx=8, pady=10)
        self.revert_button = ctk.CTkButton(live_controls, text="", image=self.icon_revert, command=self._on_revert)
        self.revert_button.pack(side="left", padx=8, pady=10)

        replay_controls = ctk.CTkFrame(center)
        replay_controls.grid(row=2, column=1, sticky="ew", padx=10, pady=(0, 12))
        self.play_button = ctk.CTkButton(replay_controls, text="", image=self.icon_play, command=self._on_play)
        self.play_button.pack(side="left", padx=8, pady=10)
        self.stop_button = ctk.CTkButton(replay_controls, text="", image=self.icon_stop, command=self._on_stop)
        self.stop_button.pack(side="left", padx=8, pady=10)
        self._apply_preview_dimensions()

        controls = ctk.CTkFrame(self)
        controls.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 8))
        ctk.CTkButton(controls, text="Save", command=self._on_save).pack(
            fill="both",
            expand=True,
            padx=8,
            pady=10,
        )

        status = ctk.CTkFrame(self)
        status.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))
        ctk.CTkLabel(status, textvariable=self.status_var, anchor="w").pack(
            fill="x",
            padx=12,
            pady=10,
        )

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _update_transport_button_colors(self) -> None:
        """Highlight the active transport action based on current state."""
        default_button_color = self.revert_button.cget("fg_color")
        default_hover_color = self.revert_button.cget("hover_color")

        self.record_button.configure(fg_color=default_button_color, hover_color=default_hover_color)
        self.pause_button.configure(fg_color=default_button_color, hover_color=default_hover_color)
        self.play_button.configure(fg_color=default_button_color, hover_color=default_hover_color)

        if self.record_state == "recording":
            self.record_button.configure(fg_color=ACTIVE_RECORD_COLOR, hover_color=ACTIVE_RECORD_COLOR)
        elif self.record_state == "paused":
            self.pause_button.configure(fg_color=ACTIVE_PAUSE_COLOR, hover_color=ACTIVE_PAUSE_COLOR)

        if self.is_playing:
            self.play_button.configure(fg_color=ACTIVE_PLAY_COLOR, hover_color=ACTIVE_PLAY_COLOR)

    def _clear_current_capture(self) -> None:
        """Reset the in-memory clip and replay state."""
        self._stop_playback()
        self.record_state = "idle"
        self.recorded_frames.clear()
        self._show_empty_replay()
        self._update_transport_button_colors()

    def _focus_name_entry(self) -> None:
        """Focus the filename entry and select its current text."""
        self.name_entry.focus_set()
        self.name_entry.select_range(0, ctk.END)

    def _refresh_external_camera_info(self) -> None:
        """Refresh camera and mode info from non-OpenCV provider."""
        try:
            self.camera_devices = list_external_cameras()
        except Exception:
            self.camera_devices = []
        self.available_cameras = list(range(len(self.camera_devices)))

    def _sizes_for_index(self, camera_index: int) -> list[tuple[int, int]]:
        """Return supported sizes for a camera index.

        Uses external provider data only.
        """
        if 0 <= camera_index < len(self.camera_devices):
            sizes = [(mode.width, mode.height) for mode in self.camera_devices[camera_index].modes]
            if sizes:
                return list(dict.fromkeys(sizes))
        return []

    def _modes_for_index(self, camera_index: int) -> list[CameraMode]:
        """Return detailed modes for a camera index when available."""
        if 0 <= camera_index < len(self.camera_devices):
            modes = list(self.camera_devices[camera_index].modes)
            if modes:
                return modes
        return []

    def _fps_values_for(self, camera_index: int, width: int, height: int) -> list[float]:
        """Return available FPS values for a camera and resolution."""
        fps_values = [
            mode.fps
            for mode in self._modes_for_index(camera_index)
            if mode.width == width and mode.height == height and mode.fps > 0
        ]
        if not fps_values:
            fps_values = [self.capture_fps or float(CAPTURE_FPS)]
        return sorted(set(fps_values))

    def _update_preview_dimensions(self, width: int, height: int) -> None:
        """Update preview panel size to best match active capture aspect ratio."""
        if width <= 0 or height <= 0:
            self.preview_width = PREVIEW_WIDTH
            self.preview_height = PREVIEW_HEIGHT
            return

        ratio = width / height
        if abs(ratio - (16 / 9)) <= abs(ratio - (4 / 3)):
            target_w, target_h = 16, 9
        else:
            target_w, target_h = 4, 3

        scaled_width = PREVIEW_MAX_WIDTH
        scaled_height = int(round(scaled_width * target_h / target_w))
        if scaled_height > PREVIEW_MAX_HEIGHT:
            scaled_height = PREVIEW_MAX_HEIGHT
            scaled_width = int(round(scaled_height * target_w / target_h))

        self.preview_width = scaled_width
        self.preview_height = scaled_height

    def _apply_preview_dimensions(self) -> None:
        """Apply current preview dimensions to live and replay labels."""
        self.live_label.configure(width=self.preview_width, height=self.preview_height)
        self.replay_label.configure(width=self.preview_width, height=self.preview_height)

    def _best_mode_for_index(self, camera_index: int) -> CameraMode | None:
        """Pick best mode using MJPEG, then resolution, then FPS priority."""
        modes = self._modes_for_index(camera_index)
        if not modes:
            return None

        def rank(mode: CameraMode) -> tuple[int, int, float]:
            is_mjpeg = 1 if mode.pixel_format.lower() in {"mjpeg", "mjpg"} else 0
            area = mode.width * mode.height
            return (is_mjpeg, area, mode.fps)

        return max(modes, key=rank)

    def _open_capture(self, camera_index: int, width: int, height: int, fps: float):
        capture = open_camera_capture(camera_index)

        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            capture.set(cv2.CAP_PROP_FPS, float(fps))
        return capture

    def _recover_capture(self) -> None:
        """Attempt to recover live capture using current camera settings."""
        if self.capture.isOpened():
            self.capture.release()
        self.capture = self._open_capture(self.camera_index, self.capture_width, self.capture_height, self.capture_fps)
        ok, _ = safe_camera_read(self.capture)
        if ok:
            self.read_failures = 0
            self._set_status(f"Recovered camera {self.camera_index}")
        else:
            self._set_status(
                f"Camera stream unavailable ({self.camera_index}, "
                f"{self.capture_width}x{self.capture_height}@{self.capture_fps:.2f})"
            )

    def _apply_camera_settings(self, camera_index: int, width: int, height: int, fps: float) -> bool:
        old_index = self.camera_index
        old_width = self.capture_width
        old_height = self.capture_height
        old_fps = self.capture_fps

        if self.capture.isOpened():
            self.capture.release()

        new_capture = self._open_capture(camera_index, width, height, fps)
        if not new_capture.isOpened():
            new_capture.release()
            self.capture = self._open_capture(old_index, old_width, old_height, old_fps)
            self._set_status(f"Unable to open camera {camera_index}")
            return False

        ok, frame = False, None
        for _ in range(3):
            ok, frame = safe_camera_read(new_capture)
            if ok:
                break

        if not ok:
            new_capture.release()
            self.capture = self._open_capture(old_index, old_width, old_height, old_fps)
            self._set_status(f"Camera {camera_index} did not return frames")
            return False

        self.capture = new_capture
        self.camera_index = camera_index
        self.capture_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.capture_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.capture_fps = actual_fps if actual_fps > 0 else fps

        matching_modes = [
            mode
            for mode in self._modes_for_index(camera_index)
            if mode.width == width and mode.height == height and abs(mode.fps - fps) < 0.01
        ]
        if matching_modes:
            preferred = max(
                matching_modes,
                key=lambda mode: 1 if mode.pixel_format.lower() in {"mjpeg", "mjpg"} else 0,
            )
            self.capture_pixel_format = preferred.pixel_format

        if frame is not None and hasattr(frame, "shape"):
            frame_h, frame_w = frame.shape[:2]
            if frame_w > 0 and frame_h > 0:
                self.capture_width = frame_w
                self.capture_height = frame_h

        self._update_preview_dimensions(self.capture_width, self.capture_height)
        self._apply_preview_dimensions()

        self.supported_sizes = self._sizes_for_index(self.camera_index)
        if (self.capture_width, self.capture_height) not in self.supported_sizes:
            self.supported_sizes.insert(0, (self.capture_width, self.capture_height))
        self.read_failures = 0

        if self.record_state == "recording":
            self.record_state = "paused"
        self._stop_playback()
        self._update_transport_button_colors()
        self._set_status(
            f"Camera {self.camera_index} set to {self.capture_width}x{self.capture_height} @ {self.capture_fps:.2f} fps"
        )
        self._save_settings(
            camera_id=self.camera_index,
            width=self.capture_width,
            height=self.capture_height,
            fps=self.capture_fps,
        )
        return True

    def _open_settings_dialog(self) -> None:
        self._refresh_external_camera_info()
        if not self.available_cameras:
            self._set_status("No webcam detected")
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Settings")
        dialog.geometry("460x290")
        dialog.resizable(False, False)
        dialog.grab_set()

        dialog.grid_columnconfigure(1, weight=1)

        camera_values = []
        for index in self.available_cameras:
            if 0 <= index < len(self.camera_devices):
                camera_values.append(f"{index}: {self.camera_devices[index].name}")
            else:
                camera_values.append(f"{index}: Camera {index}")
        default_camera = self.camera_index
        if default_camera not in self.available_cameras and self.available_cameras:
            default_camera = self.available_cameras[0]
        if 0 <= default_camera < len(self.camera_devices):
            camera_var = ctk.StringVar(value=f"{default_camera}: {self.camera_devices[default_camera].name}")
        else:
            camera_var = ctk.StringVar(value=f"{default_camera}: Camera {default_camera}")

        ctk.CTkLabel(dialog, text="Webcam").grid(
            row=0,
            column=0,
            padx=(16, 10),
            pady=(18, 10),
            sticky="w",
        )
        camera_menu = ctk.CTkOptionMenu(dialog, values=camera_values, variable=camera_var)
        camera_menu.grid(row=0, column=1, padx=(0, 16), pady=(18, 10), sticky="ew")

        ctk.CTkLabel(dialog, text="Frame Size").grid(
            row=1,
            column=0,
            padx=(16, 10),
            pady=10,
            sticky="w",
        )
        initial_size_values = [f"{w}x{h}" for w, h in self.supported_sizes]
        if not initial_size_values:
            initial_size_values = [f"{self.capture_width}x{self.capture_height}"]
        size_var = ctk.StringVar(value=f"{self.capture_width}x{self.capture_height}")
        if size_var.get() not in initial_size_values and initial_size_values:
            size_var.set(initial_size_values[0])
        size_menu = ctk.CTkOptionMenu(dialog, values=initial_size_values, variable=size_var)
        size_menu.grid(row=1, column=1, padx=(0, 16), pady=10, sticky="ew")

        ctk.CTkLabel(dialog, text="FPS").grid(
            row=2,
            column=0,
            padx=(16, 10),
            pady=10,
            sticky="w",
        )

        default_fps_values = self._fps_values_for(default_camera, self.capture_width, self.capture_height)
        fps_values = [f"{value:.2f}" for value in default_fps_values]
        fps_var = ctk.StringVar(value=f"{self.capture_fps:.2f}")
        if fps_var.get() not in fps_values and fps_values:
            fps_var.set(fps_values[0])

        fps_menu = ctk.CTkOptionMenu(dialog, values=fps_values, variable=fps_var)
        fps_menu.grid(row=2, column=1, padx=(0, 16), pady=10, sticky="ew")

        def parse_camera_index(text: str) -> int:
            prefix, _, _ = text.partition(":")
            return int(prefix.strip())

        def refresh_sizes(camera_text: str) -> None:
            camera_index = parse_camera_index(camera_text)
            sizes = self.supported_sizes if camera_index == self.camera_index else self._sizes_for_index(camera_index)
            if not sizes:
                sizes = [(self.capture_width, self.capture_height)]
            values = [f"{w}x{h}" for w, h in sizes]
            size_menu.configure(values=values)
            size_var.set(values[0])
            refresh_fps(camera_index, values[0])

        def refresh_fps(camera_index: int, size_text: str) -> None:
            width_text, height_text = size_text.split("x", maxsplit=1)
            width = int(width_text)
            height = int(height_text)
            fps_list = self._fps_values_for(camera_index, width, height)
            values = [f"{value:.2f}" for value in fps_list]
            fps_menu.configure(values=values)
            fps_var.set(values[0])

        def on_size_change(selected_size: str) -> None:
            camera_index = parse_camera_index(camera_var.get())
            refresh_fps(camera_index, selected_size)

        camera_menu.configure(command=refresh_sizes)
        size_menu.configure(command=on_size_change)

        buttons = ctk.CTkFrame(dialog, fg_color="transparent")
        buttons.grid(row=3, column=0, columnspan=2, padx=16, pady=(16, 12), sticky="e")

        def on_apply() -> None:
            camera_index = parse_camera_index(camera_var.get())
            width_text, height_text = size_var.get().split("x", maxsplit=1)
            width = int(width_text)
            height = int(height_text)
            fps = float(fps_var.get())
            if self._apply_camera_settings(camera_index, width, height, fps):
                dialog.destroy()

        ctk.CTkButton(buttons, text="Cancel", command=dialog.destroy).pack(
            side="right",
            padx=(10, 0),
        )
        ctk.CTkButton(buttons, text="Apply", command=on_apply).pack(side="right")

    def _frame_to_photo(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        return ctk.CTkImage(light_image=img, dark_image=img, size=(self.preview_width, self.preview_height))

    def _show_live(self, frame_bgr: np.ndarray) -> None:
        self.live_photo = self._frame_to_photo(frame_bgr)
        self.live_label.configure(image=self.live_photo)

    def _show_replay(self, frame_bgr: np.ndarray) -> None:
        self.replay_photo = self._frame_to_photo(frame_bgr)
        self.replay_label.configure(image=self.replay_photo)

    def _show_empty_replay(self) -> None:
        """Show a white placeholder when no clip is available for replay."""
        blank = np.full((self.preview_height, self.preview_width, 3), 255, dtype=np.uint8)
        self.last_replay_frame = None
        self._show_replay(blank)

    def _format_duration(self, seconds: float) -> str:
        """Format elapsed seconds as HH:MM:SS."""
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _overlay_timer(self, frame_bgr: np.ndarray, label: str) -> np.ndarray:
        """Draw a timer badge centered near the bottom of the preview."""
        frame = frame_bgr.copy()
        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return frame

        scale_ratio = max(0.5, frame_w / max(1.0, float(self.preview_width)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.6, 0.72 * scale_ratio)
        thickness = max(1, int(round(2 * scale_ratio)))
        margin_y = int(round(12 * scale_ratio))
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

        pad_x = int(round(8 * scale_ratio))
        pad_y = int(round(6 * scale_ratio))

        badge_w = text_w + (2 * pad_x)
        badge_h = text_h + baseline + (2 * pad_y)

        x1 = (frame_w - badge_w) // 2
        y2 = frame_h - margin_y
        x2 = x1 + badge_w
        y1 = y2 - badge_h

        # Clamp badge inside frame bounds to avoid off-screen drawing.
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(x1 + 1, min(x2, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(y1 + 1, min(y2, frame_h - 1))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        text_x = min(x2 - 1, x1 + pad_x)
        text_y = min(y2 - baseline - 1, y1 + pad_y + text_h)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def _current_record_elapsed(self) -> float:
        """Return total recorded time, including resumed segments."""
        if self.record_state == "recording" and self.record_started_at is not None:
            return self.record_elapsed_before_pause + (time.monotonic() - self.record_started_at)
        return self.record_elapsed_before_pause

    def _update_live_view(self) -> None:
        ok, frame = safe_camera_read(self.capture)
        if ok and frame is not None:
            self.read_failures = 0
            self.last_live_frame = frame

            live_frame = frame
            if self.record_state in {"recording", "paused"}:
                elapsed_text = self._format_duration(self._current_record_elapsed())
                prefix = "REC" if self.record_state == "recording" else "PAUSE"
                live_frame = self._overlay_timer(live_frame, f"{prefix} {elapsed_text}")

            self._show_live(live_frame)

            if not self._initial_focus_done:
                self._initial_focus_done = True
                self._focus_name_entry()

            if self.record_state == "recording":
                self.recorded_frames.append(frame.copy())
        else:
            self.read_failures += 1
            if self.read_failures >= 15:
                self._recover_capture()

        self.after(15, self._update_live_view)

    def _on_record(self) -> None:
        if self.record_state == "recording":
            self._set_status("Already recording")
            self._focus_name_entry()
            return

        if self.record_state == "idle":
            self.recorded_frames.clear()
            self.last_replay_frame = None
            self._stop_playback()
            self.record_elapsed_before_pause = 0.0
            self._set_status("Recording started")
        else:
            self._set_status("Recording resumed")

        self.record_started_at = time.monotonic()
        self.record_state = "recording"
        self._update_transport_button_colors()
        self._focus_name_entry()

    def _on_pause(self) -> None:
        if self.record_state != "recording":
            self._set_status("Pause ignored: recording is not active")
            self._focus_name_entry()
            return

        if self.record_started_at is not None:
            self.record_elapsed_before_pause += time.monotonic() - self.record_started_at
        self.record_started_at = None
        self.record_state = "paused"
        self._update_transport_button_colors()
        self._set_status("Recording paused")
        self._focus_name_entry()

    def _on_stop(self) -> None:
        self._stop_playback()
        self._update_transport_button_colors()
        self._set_status("Playback stopped")
        self._focus_name_entry()

    def _on_play(self) -> None:
        if not self.recorded_frames:
            self._set_status("Nothing to play")
            self._focus_name_entry()
            return

        if self.record_state == "recording":
            self.record_state = "paused"
            self._set_status("Recording paused for playback")

        self._stop_playback()
        self.is_playing = True
        self.play_index = 0
        self._update_transport_button_colors()
        self._set_status("Playback started")
        self._focus_name_entry()
        self._playback_tick()

    def _playback_tick(self) -> None:
        if not self.is_playing:
            return

        if self.play_index >= len(self.recorded_frames):
            self.is_playing = False
            self._play_job = None
            if self.recorded_frames:
                final_frame = self.recorded_frames[-1]
                self.last_replay_frame = final_frame
                self._show_replay(final_frame)
            self._update_transport_button_colors()
            self._set_status("Playback finished")
            return

        frame = self.recorded_frames[self.play_index]
        elapsed = self.play_index / max(1.0, self.capture_fps)
        replay_frame = self._overlay_timer(frame, f"PLAY {self._format_duration(elapsed)}")
        self.last_replay_frame = frame
        self._show_replay(replay_frame)
        self.play_index += 1
        delay_ms = max(1, round(1000 / max(1.0, self.capture_fps)))
        self._play_job = self.after(delay_ms, self._playback_tick)

    def _stop_playback(self) -> None:
        self.is_playing = False
        if self._play_job is not None:
            self.after_cancel(self._play_job)
            self._play_job = None

        if self.recorded_frames:
            frame_index = max(0, min(self.play_index - 1, len(self.recorded_frames) - 1))
            plain_frame = self.recorded_frames[frame_index]
            self.last_replay_frame = plain_frame
            self._show_replay(plain_frame)
        self._update_transport_button_colors()

    def _on_revert(self) -> None:
        self._clear_current_capture()
        self.record_elapsed_before_pause = 0.0
        self.record_started_at = None
        self._set_status("Current recording discarded")
        self._focus_name_entry()

    def _open_settings(self) -> None:
        """Revert pending clip before opening the settings dialog."""
        self._on_revert()
        self._focus_name_entry()
        self._open_settings_dialog()

    def _on_save(self) -> None:
        if not self.recorded_frames:
            self._set_status("Nothing to save")
            self._focus_name_entry()
            return

        output_path = build_output_path(self.filename_var.get())
        first_frame = self.recorded_frames[0]
        height, width = first_frame.shape[:2]

        writer = H264FFmpegWriter(
            output_path=output_path,
            width=width,
            height=height,
            fps=max(1, int(round(self.capture_fps))),
        )
        try:
            writer.start()
            for frame in self.recorded_frames:
                writer.write(frame)
            writer.close()
        except RuntimeError as exc:
            self._set_status(f"Save failed: {exc}")
            self._focus_name_entry()
            return

        self._clear_current_capture()
        self._set_status(f"Saved: {output_path.name}")
        self._focus_name_entry()

    def _on_close(self) -> None:
        self._stop_playback()
        if self.capture.isOpened():
            self.capture.release()
        self.destroy()


def main() -> None:
    """Launch the CVRecord GUI application."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = RecorderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
