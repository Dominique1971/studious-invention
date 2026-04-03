"""
audio.py - Audio system for the city simulator.

Features
--------
• Engine audio: idle hum, acceleration, redline warning, braking squeal –
  generated procedurally with NumPy when no audio files are present
• 3D positional audio via Panda3D AudioManager (OpenAL backend)
• Ambient city sounds: traffic noise, wind
• Emergency siren
• Rain sound for weather
• Footstep sounds for pedestrians
• Master / per-category volume controls
"""

from __future__ import annotations

import math
import struct
import wave
import io
import os
import random
from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if TYPE_CHECKING:
    from direct.showbase.ShowBase import ShowBase

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AUDIO_DIR = Path("assets/audio")

# ---------------------------------------------------------------------------
# Procedural audio generation helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050

def _make_wav_bytes(samples: "np.ndarray", sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 [-1, 1] NumPy array to WAV bytes (16-bit mono)."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _sine(freq: float, dur: float, amp: float = 0.5,
           sr: int = SAMPLE_RATE) -> "np.ndarray":
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t) * amp


def _noise(dur: float, amp: float = 0.5, sr: int = SAMPLE_RATE) -> "np.ndarray":
    n = int(sr * dur)
    return (np.random.default_rng().random(n).astype(np.float32) * 2 - 1) * amp


def _generate_engine_idle(sr: int = SAMPLE_RATE) -> bytes:
    """Low rumble at ~80 Hz with harmonics."""
    if not HAS_NUMPY:
        return b""
    dur = 2.0
    t   = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig  = np.sin(2 * np.pi * 80 * t) * 0.35
    sig += np.sin(2 * np.pi * 160 * t) * 0.18
    sig += np.sin(2 * np.pi * 240 * t) * 0.08
    sig += _noise(dur, amp=0.03)
    return _make_wav_bytes(sig, sr)


def _generate_engine_rev(base_hz: float = 180, sr: int = SAMPLE_RATE) -> bytes:
    """Rising engine pitch from idle to redline."""
    if not HAS_NUMPY:
        return b""
    dur = 3.0
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n, endpoint=False)
    freq = np.linspace(base_hz, base_hz * 4.5, n)
    phase = np.cumsum(2 * np.pi * freq / sr)
    sig   = np.sin(phase) * 0.5
    sig  += np.sin(phase * 2) * 0.2
    sig  += _noise(dur, amp=0.05)
    # Envelope
    env = np.linspace(0.3, 1.0, n)
    return _make_wav_bytes(sig * env, sr)


def _generate_brake_squeal(sr: int = SAMPLE_RATE) -> bytes:
    if not HAS_NUMPY:
        return b""
    dur = 1.5
    t   = np.linspace(0, dur, int(sr * dur), endpoint=False)
    freq = np.linspace(900, 700, len(t))
    phase = np.cumsum(2 * np.pi * freq / sr)
    sig   = np.sin(phase) * 0.6
    env   = np.exp(-3 * t / dur)
    return _make_wav_bytes(sig * env, sr)


def _generate_ambient_traffic(sr: int = SAMPLE_RATE) -> bytes:
    if not HAS_NUMPY:
        return b""
    dur  = 5.0
    sig  = _noise(dur, amp=0.25)
    # Low-pass filter via cumulative sum trick
    sig  = np.convolve(sig, np.ones(16) / 16, mode="same")
    sig += _sine(60, dur, amp=0.08)
    return _make_wav_bytes(sig, sr)


def _generate_rain(sr: int = SAMPLE_RATE) -> bytes:
    if not HAS_NUMPY:
        return b""
    dur = 4.0
    sig  = _noise(dur, amp=0.4)
    sig  = np.convolve(sig, np.ones(8) / 8, mode="same")
    return _make_wav_bytes(sig, sr)


def _generate_siren(sr: int = SAMPLE_RATE) -> bytes:
    if not HAS_NUMPY:
        return b""
    dur = 2.0
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n, endpoint=False)
    freq = 800 + 400 * np.sin(2 * np.pi * 0.8 * t)
    phase = np.cumsum(2 * np.pi * freq / sr)
    sig   = np.sin(phase) * 0.7
    return _make_wav_bytes(sig, sr)


def _generate_footstep(sr: int = SAMPLE_RATE) -> bytes:
    if not HAS_NUMPY:
        return b""
    dur = 0.1
    n   = int(sr * dur)
    sig  = _noise(dur, amp=0.5)[:n]
    env  = np.exp(-30 * np.linspace(0, dur, n))
    return _make_wav_bytes(sig * env, sr)


# ---------------------------------------------------------------------------
# Audio manager wrapper
# ---------------------------------------------------------------------------

SOUND_NAMES = [
    ("engine_idle",    _generate_engine_idle),
    ("engine_rev",     _generate_engine_rev),
    ("brake_squeal",   _generate_brake_squeal),
    ("ambient_traffic",_generate_ambient_traffic),
    ("rain",           _generate_rain),
    ("siren",          _generate_siren),
    ("footstep",       _generate_footstep),
]


class AudioSystem:
    """
    Manages all game audio.

    Parameters
    ----------
    base   : ShowBase
    config : dict  (full game config)
    """

    def __init__(self, base: "ShowBase", config: dict):
        self.base   = base
        self.config = config

        audio_cfg = config.get("audio", {})
        self._master_vol  = float(audio_cfg.get("master_volume",  0.8))
        self._engine_vol  = float(audio_cfg.get("engine_volume",  1.0))
        self._ambient_vol = float(audio_cfg.get("ambient_volume", 0.6))
        self._music_vol   = float(audio_cfg.get("music_volume",   0.4))

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_proc_audio()

        self._sounds: Dict[str, object] = {}
        self._load_sounds()

        self._engine_playing  = False
        self._ambient_playing = False
        self._rain_playing    = False

        self._start_ambient()

    # ------------------------------------------------------------------
    # Asset generation
    # ------------------------------------------------------------------

    def _ensure_proc_audio(self):
        """Write procedurally generated WAV files if not present."""
        for name, gen_fn in SOUND_NAMES:
            path = AUDIO_DIR / f"{name}.wav"
            if not path.exists():
                data = gen_fn()
                if data:
                    path.write_bytes(data)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_sounds(self):
        """Load all WAV files. Falls back silently if audio backend absent."""
        mgr = self.base.sfxManagerList
        if not mgr:
            return
        for name, _ in SOUND_NAMES:
            path = str(AUDIO_DIR / f"{name}.wav")
            try:
                snd = self.base.loader.loadSfx(path)
                if snd:
                    self._sounds[name] = snd
            except Exception as exc:
                pass

    def _get(self, name: str):
        return self._sounds.get(name)

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------

    def _play_looping(self, name: str, volume: float):
        snd = self._get(name)
        if snd:
            snd.setLoop(True)
            snd.setVolume(volume * self._master_vol)
            snd.play()

    def _stop(self, name: str):
        snd = self._get(name)
        if snd:
            snd.stop()

    def _play_once(self, name: str, volume: float):
        snd = self._get(name)
        if snd:
            snd.setLoop(False)
            snd.setVolume(volume * self._master_vol)
            snd.play()

    # ------------------------------------------------------------------
    # Engine audio
    # ------------------------------------------------------------------

    def _start_ambient(self):
        self._play_looping("ambient_traffic", self._ambient_vol)
        self._ambient_playing = True

    def update_engine(self, rpm: float, speed_kph: float, braking: bool):
        """Call every frame with current engine RPM and speed."""
        # Idle vs rev
        if rpm < 1200:
            if not self._engine_playing:
                self._play_looping("engine_idle", self._engine_vol * 0.6)
                self._engine_playing = True
            idle = self._get("engine_idle")
            if idle:
                idle.setVolume(0.5 * self._engine_vol * self._master_vol)
        else:
            rev = self._get("engine_rev")
            if rev:
                # Pitch shift via playback rate (Panda3D supports this)
                try:
                    rate = 0.5 + (rpm / 7500.0) * 1.0
                    rev.setPlayRate(rate)
                    rev.setVolume(self._engine_vol * self._master_vol * 0.8)
                    if not rev.status() == rev.PLAYING:
                        rev.setLoop(True)
                        rev.play()
                except Exception:
                    pass

        # Brake squeal
        squeal = self._get("brake_squeal")
        if squeal:
            if braking and speed_kph > 20:
                if squeal.status() != getattr(squeal, "PLAYING", None):
                    squeal.setLoop(True)
                    squeal.setVolume(0.4 * self._master_vol)
                    squeal.play()
            else:
                squeal.stop()

    # ------------------------------------------------------------------
    # Weather audio
    # ------------------------------------------------------------------

    def set_rain(self, active: bool, intensity: float = 1.0):
        if active and not self._rain_playing:
            snd = self._get("rain")
            if snd:
                snd.setLoop(True)
                snd.setVolume(intensity * self._ambient_vol * self._master_vol)
                snd.play()
                self._rain_playing = True
        elif not active and self._rain_playing:
            self._stop("rain")
            self._rain_playing = False

    # ------------------------------------------------------------------
    # Positional 3D audio (Panda3D Audio3DManager)
    # ------------------------------------------------------------------

    def attach_siren_to_vehicle(self, vehicle_np):
        """Start a looping siren sound attached to *vehicle_np* in 3D space."""
        try:
            from direct.showbase import Audio3DManager
            mgr = Audio3DManager.Audio3DManager(self.base.sfxManagerList[0],
                                                 self.base.camera)
            snd = mgr.loadSfx(str(AUDIO_DIR / "siren.wav"))
            if snd:
                mgr.attachSoundToObject(snd, vehicle_np)
                mgr.setSoundVelocityAuto(snd)
                snd.setLoop(True)
                snd.play()
                return snd
        except Exception:
            pass
        return None

    def play_footstep(self):
        """Play a single footstep sound (non-positional)."""
        self._play_once("footstep", 0.15)

    # ------------------------------------------------------------------
    # Volume controls
    # ------------------------------------------------------------------

    def set_master_volume(self, vol: float):
        self._master_vol = max(0.0, min(1.0, vol))
        self.base.musicManager.setVolume(self._master_vol * self._music_vol)

    def set_music_volume(self, vol: float):
        self._music_vol = max(0.0, min(1.0, vol))

    def set_ambient_volume(self, vol: float):
        self._ambient_vol = vol
        snd = self._get("ambient_traffic")
        if snd:
            snd.setVolume(vol * self._master_vol)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        for snd in self._sounds.values():
            try:
                snd.stop()
            except Exception:
                pass
        self._sounds.clear()
