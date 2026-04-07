"""Camera and mode detection using platform-native backends outside OpenCV.

This module is designed to be portable through a provider interface.
For now, only a Windows implementation is provided.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol

import imageio_ffmpeg


@dataclass(frozen=True)
class CameraMode:
    """Represents a supported camera output mode.

    Args:
        width: Mode width in pixels.
        height: Mode height in pixels.
        fps: Mode frame rate in frames per second.
        pixel_format: Pixel format label reported by FFmpeg.
    """

    width: int
    height: int
    fps: float
    pixel_format: str


@dataclass(frozen=True)
class CameraDevice:
    """Represents a detected camera and its available modes.

    Args:
        name: Human-readable camera name from system backend.
        modes: Supported resolution modes.
    """

    name: str
    modes: tuple[CameraMode, ...]


class CameraInfoProvider(Protocol):
    """Protocol for platform-specific camera information providers."""

    def list_cameras(self) -> list[CameraDevice]:
        """Return detected cameras and supported modes."""
        ...


class UnsupportedCameraInfoProvider:
    """Fallback provider for unsupported platforms."""

    def list_cameras(self) -> list[CameraDevice]:
        return []


class WindowsFFmpegCameraInfoProvider:
    """Windows camera provider based on FFmpeg DirectShow queries."""

    def __init__(self) -> None:
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def _run_ffmpeg(self, args: list[str]) -> str:
        command = [
            self.ffmpeg_exe,
            "-hide_banner",
            *args,
        ]
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        # FFmpeg typically writes listing output to stderr for probing commands.
        return f"{proc.stdout}\n{proc.stderr}"

    def _parse_device_names(self, output: str) -> list[str]:
        names: list[str] = []

        for raw_line in output.splitlines():
            line = raw_line.strip()
            if "(video)" not in line:
                continue

            match = re.search(r'"([^"]+)"', line)
            if not match:
                continue

            name = match.group(1)
            if name not in names:
                names.append(name)

        return names

    def _parse_modes(self, output: str) -> tuple[CameraMode, ...]:
        modes_set: set[tuple[int, int, float, str]] = set()

        def parse_fps_values(line: str) -> list[float]:
            values: list[float] = []
            for match in re.finditer(r"fps\s*=\s*([0-9]+(?:\.[0-9]+)?)", line, flags=re.IGNORECASE):
                values.append(float(match.group(1)))
            return values

        def parse_pixel_format(line: str) -> str:
            match = re.search(r"pixel_format\s*=\s*([A-Za-z0-9_]+)", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).lower()
            return "unknown"

        for line in output.splitlines():
            fps_values = parse_fps_values(line)
            if not fps_values:
                continue

            pixel_format = parse_pixel_format(line)

            for match in re.finditer(r"(\d{2,5})x(\d{2,5})", line):
                width = int(match.group(1))
                height = int(match.group(2))
                if width > 0 and height > 0:
                    for fps in fps_values:
                        if fps > 0:
                            modes_set.add((width, height, fps, pixel_format))

        if not modes_set:
            for line in output.splitlines():
                for match in re.finditer(r"(\d{2,5})x(\d{2,5})", line):
                    width = int(match.group(1))
                    height = int(match.group(2))
                    if width > 0 and height > 0:
                        modes_set.add((width, height, 30.0, "unknown"))

        sorted_modes = sorted(modes_set, key=lambda item: (item[0] * item[1], item[0], item[1], item[2], item[3]))
        return tuple(
            CameraMode(width=w, height=h, fps=fps, pixel_format=pixel_format)
            for w, h, fps, pixel_format in sorted_modes
        )

    def _list_modes_for_device(self, camera_name: str) -> tuple[CameraMode, ...]:
        escaped = camera_name.replace('"', r'\"')
        output = self._run_ffmpeg(["-f", "dshow", "-list_options", "true", "-i", f"video={escaped}"])
        return self._parse_modes(output)

    def list_cameras(self) -> list[CameraDevice]:
        output = self._run_ffmpeg(["-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        device_names = self._parse_device_names(output)

        devices: list[CameraDevice] = []
        for name in device_names:
            modes = self._list_modes_for_device(name)
            devices.append(CameraDevice(name=name, modes=modes))
        return devices


def get_camera_info_provider() -> CameraInfoProvider:
    """Return the appropriate camera provider for the current OS."""
    if sys.platform.startswith("win"):
        return WindowsFFmpegCameraInfoProvider()
    return UnsupportedCameraInfoProvider()


def list_cameras() -> list[CameraDevice]:
    """Convenience function to list cameras and their modes."""
    provider = get_camera_info_provider()
    return provider.list_cameras()
