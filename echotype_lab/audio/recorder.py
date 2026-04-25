"""
echotype_lab.audio.recorder
============================
Record short audio segments triggered by keystroke events.

The recorder captures a fixed-duration audio clip every time a key event is
detected (via a callback), and saves the clip as a 16-bit PCM WAV file.

Dependencies
------------
- pyaudio  (PortAudio bindings)
- wave     (stdlib)
- numpy

Usage example
-------------
>>> from echotype_lab.audio.recorder import KeystrokeRecorder
>>> rec = KeystrokeRecorder(output_dir="data/raw", label="a")
>>> rec.record_single_keystroke()   # saves data/raw/a_0001.wav
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE: int = 44_100       # Hz
DEFAULT_CHANNELS: int = 1              # mono
DEFAULT_CHUNK: int = 1_024             # frames per buffer read
DEFAULT_CLIP_DURATION: float = 0.5    # seconds captured per keystroke
DEFAULT_FORMAT: int = 16              # bits per sample (PCM-16)


class AudioRecorder:
    """Low-level audio recorder backed by PyAudio.

    Parameters
    ----------
    sample_rate:
        Recording sample rate in Hz.
    channels:
        Number of audio channels (1 = mono, 2 = stereo).
    chunk:
        Number of frames read per PortAudio buffer callback.
    clip_duration:
        Length of each recorded clip in seconds.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk: int = DEFAULT_CHUNK,
        clip_duration: float = DEFAULT_CLIP_DURATION,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.clip_duration = clip_duration
        self._pa = None  # lazy-import PyAudio

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def record_clip(self) -> np.ndarray:
        """Block for *clip_duration* seconds and return raw PCM as a
        1-D ``numpy.ndarray`` of ``int16`` samples.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples,)`` with dtype ``int16``.
        """
        import pyaudio  # noqa: PLC0415 – optional runtime dep

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        n_frames = int(self.sample_rate / self.chunk * self.clip_duration)
        frames: list[bytes] = []
        for _ in range(n_frames):
            frames.append(stream.read(self.chunk, exception_on_overflow=False))

        stream.stop_stream()
        stream.close()
        pa.terminate()

        raw = b"".join(frames)
        return np.frombuffer(raw, dtype=np.int16)

    def save_wav(self, samples: np.ndarray, path: Path | str) -> Path:
        """Write *samples* to a WAV file at *path*.

        Parameters
        ----------
        samples:
            1-D ``int16`` PCM array.
        path:
            Destination file path (parent dirs are created automatically).

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit → 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(samples.tobytes())

        logger.debug("Saved WAV: %s (%d samples)", path, len(samples))
        return path.resolve()


class KeystrokeRecorder:
    """High-level recorder that links keystroke labels with audio clips.

    Records a fixed-duration audio clip for each keystroke label and stores
    WAV files under *output_dir/<label>/<label>_<index>.wav*.

    Parameters
    ----------
    output_dir:
        Root directory for saving recorded clips.
    label:
        Key label being recorded (e.g. ``"a"``, ``"space"``).
    recorder:
        Underlying :class:`AudioRecorder` instance.  Created with defaults
        if not provided.
    """

    def __init__(
        self,
        output_dir: Path | str = Path("data/raw"),
        label: str = "unknown",
        recorder: Optional[AudioRecorder] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.label = label
        self.recorder = recorder or AudioRecorder()
        self._index: int = self._next_index()

    # ------------------------------------------------------------------

    def record_single_keystroke(self) -> Path:
        """Record one clip and save it to disk.

        Returns
        -------
        Path
            Path of the newly created WAV file.
        """
        logger.info("Recording clip for key '%s' (#%04d) …", self.label, self._index)
        samples = self.recorder.record_clip()
        dest = self._build_path()
        self.recorder.save_wav(samples, dest)
        self._index += 1
        return dest

    def record_n_keystrokes(self, n: int, *, pause: float = 1.0) -> list[Path]:
        """Record *n* clips with an optional *pause* between each.

        Parameters
        ----------
        n:
            Number of keystrokes / clips to record.
        pause:
            Seconds to sleep between recordings.

        Returns
        -------
        list[Path]
            Paths to all saved WAV files.
        """
        paths: list[Path] = []
        for i in range(n):
            print(f"  [{i + 1}/{n}] Press '{self.label}' now …")
            time.sleep(0.1)  # brief settle
            paths.append(self.record_single_keystroke())
            if i < n - 1:
                time.sleep(pause)
        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_path(self) -> Path:
        label_dir = self.output_dir / self.label
        label_dir.mkdir(parents=True, exist_ok=True)
        return label_dir / f"{self.label}_{self._index:04d}.wav"

    def _next_index(self) -> int:
        label_dir = self.output_dir / self.label
        if not label_dir.exists():
            return 1
        existing = list(label_dir.glob(f"{self.label}_*.wav"))
        return len(existing) + 1
