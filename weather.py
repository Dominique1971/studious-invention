"""
weather.py - Weather system for the city simulator.

Features
--------
• WeatherState enum: CLEAR, CLOUDY, RAIN, HEAVY_RAIN, THUNDER
• Smooth transitions between states driven by a Markov chain
• Panda3D particle-system rain
• Panda3D exponential fog (density scales with weather)
• Wet-road shader parameter update
• Lightning flash via directional-light intensity spike
• Wind affecting scene objects (simple NodePath sway)
• Time-based weather probability (more rain at night/early morning)
"""

from __future__ import annotations

import math
import random
import enum
from typing import Optional, Dict, List, Callable, TYPE_CHECKING

from panda3d.core import (
    Vec3, Vec4, LColor, NodePath,
    Fog, AmbientLight,
)

if TYPE_CHECKING:
    from direct.showbase.ShowBase import ShowBase
    from renderer import CityRenderer
    from audio import AudioSystem

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class WeatherState(enum.Enum):
    CLEAR       = "clear"
    CLOUDY      = "cloudy"
    RAIN        = "rain"
    HEAVY_RAIN  = "heavy_rain"
    THUNDER     = "thunder"


# Markov-chain transition matrix (row = current, col = next state)
# Order: CLEAR, CLOUDY, RAIN, HEAVY_RAIN, THUNDER
_TRANSITIONS: Dict[WeatherState, Dict[WeatherState, float]] = {
    WeatherState.CLEAR:      {WeatherState.CLEAR: 0.92, WeatherState.CLOUDY:     0.08,
                               WeatherState.RAIN:  0.00, WeatherState.HEAVY_RAIN: 0.00,
                               WeatherState.THUNDER:0.00},
    WeatherState.CLOUDY:     {WeatherState.CLEAR: 0.10, WeatherState.CLOUDY:     0.60,
                               WeatherState.RAIN:  0.28, WeatherState.HEAVY_RAIN: 0.02,
                               WeatherState.THUNDER:0.00},
    WeatherState.RAIN:       {WeatherState.CLEAR: 0.02, WeatherState.CLOUDY:     0.15,
                               WeatherState.RAIN:  0.65, WeatherState.HEAVY_RAIN: 0.15,
                               WeatherState.THUNDER:0.03},
    WeatherState.HEAVY_RAIN: {WeatherState.CLEAR: 0.00, WeatherState.CLOUDY:     0.05,
                               WeatherState.RAIN:  0.40, WeatherState.HEAVY_RAIN: 0.45,
                               WeatherState.THUNDER:0.10},
    WeatherState.THUNDER:    {WeatherState.CLEAR: 0.00, WeatherState.CLOUDY:     0.10,
                               WeatherState.RAIN:  0.40, WeatherState.HEAVY_RAIN: 0.40,
                               WeatherState.THUNDER:0.10},
}

# Weather parameters per state
_WEATHER_PARAMS: Dict[WeatherState, dict] = {
    WeatherState.CLEAR:      {"fog_density": 0.0005, "wetness": 0.0,
                               "rain_rate": 0,    "ambient_mult": 1.0,
                               "sky_tint": Vec3(0.40, 0.70, 1.00)},
    WeatherState.CLOUDY:     {"fog_density": 0.0015, "wetness": 0.1,
                               "rain_rate": 0,    "ambient_mult": 0.75,
                               "sky_tint": Vec3(0.55, 0.55, 0.60)},
    WeatherState.RAIN:       {"fog_density": 0.004,  "wetness": 0.6,
                               "rain_rate": 200,  "ambient_mult": 0.50,
                               "sky_tint": Vec3(0.30, 0.35, 0.45)},
    WeatherState.HEAVY_RAIN: {"fog_density": 0.010,  "wetness": 0.95,
                               "rain_rate": 600,  "ambient_mult": 0.35,
                               "sky_tint": Vec3(0.20, 0.22, 0.30)},
    WeatherState.THUNDER:    {"fog_density": 0.012,  "wetness": 1.0,
                               "rain_rate": 800,  "ambient_mult": 0.25,
                               "sky_tint": Vec3(0.15, 0.18, 0.25)},
}

# How many real seconds until we roll for next state transition
_TRANSITION_INTERVAL = 120.0   # every 2 minutes


# ---------------------------------------------------------------------------
# Rain particle helper
# ---------------------------------------------------------------------------

class RainParticleSystem:
    """
    Simulates rain using a pool of small NodePaths falling from above
    the camera. Falls back gracefully if Panda3D particles are unavailable.
    """

    MAX_DROPS = 1000
    DROP_SPEED = 40.0   # m/s
    SPREAD     = 30.0   # horizontal spawn radius

    def __init__(self, base: "ShowBase", parent_np: NodePath):
        self.base      = base
        self._drops: List[NodePath] = []
        self._active   = False
        self._rate     = 0
        self._timer    = 0.0
        self._parent   = parent_np
        self._spawn_interval = 0.05

        try:
            from direct.particles.ParticleEffect import ParticleEffect
            from direct.particles.Particles import Particles
            from panda3d.core import Point3
            self._use_particles = True
            self._peffect = ParticleEffect()
        except Exception:
            self._use_particles = False

    def set_rate(self, rate: int):
        self._rate   = rate
        self._active = rate > 0
        if not self._active:
            self._clear_drops()
        self._spawn_interval = max(0.001, 1.0 / max(rate, 1))

    def update(self, dt: float, camera_pos: Vec3):
        if not self._active:
            return

        self._timer += dt
        while self._timer >= self._spawn_interval and len(self._drops) < self.MAX_DROPS:
            self._timer -= self._spawn_interval
            self._spawn_drop(camera_pos)

        # Move drops
        dead = []
        for np_ in self._drops:
            pos = np_.getPos()
            np_.setPos(pos.x, pos.y, pos.z - self.DROP_SPEED * dt)
            if pos.z < camera_pos.z - 5.0:
                dead.append(np_)
        for np_ in dead:
            np_.removeNode()
            self._drops.remove(np_)

    def _spawn_drop(self, camera_pos: Vec3):
        from panda3d.core import LineSegs
        import random as _r
        ls = LineSegs("rain_drop")
        ls.setColor(0.5, 0.6, 0.8, 0.5)
        ls.setThickness(1.0)
        ls.moveTo(0, 0, 0)
        ls.drawTo(0, 0, -0.6)
        np_ = self._parent.attachNewNode(ls.create())
        ox = _r.uniform(-self.SPREAD, self.SPREAD) + camera_pos.x
        oy = _r.uniform(-self.SPREAD, self.SPREAD) + camera_pos.y
        np_.setPos(ox, oy, camera_pos.z + _r.uniform(5, 20))
        self._drops.append(np_)

    def _clear_drops(self):
        for np_ in self._drops:
            np_.removeNode()
        self._drops.clear()

    def cleanup(self):
        self._clear_drops()


# ---------------------------------------------------------------------------
# Wind sway for foliage / flags
# ---------------------------------------------------------------------------

class WindSystem:
    """Applies sinusoidal sway to registered NodePaths."""

    def __init__(self):
        self._targets: List[NodePath] = []
        self._base_hprs: List[Vec3]   = []
        self._phases:   List[float]   = []
        self._strength  = 0.0
        self._time      = 0.0

    def register(self, np_: NodePath, phase_offset: float = 0.0):
        self._targets.append(np_)
        self._base_hprs.append(Vec3(np_.getHpr()))
        self._phases.append(phase_offset)

    def set_strength(self, strength: float):
        self._strength = strength

    def update(self, dt: float):
        self._time += dt
        for i, np_ in enumerate(self._targets):
            base_hpr = self._base_hprs[i]
            phase    = self._phases[i]
            sway     = math.sin(self._time * 1.5 + phase) * self._strength * 5.0
            np_.setHpr(base_hpr.x + sway, base_hpr.y, base_hpr.z)


# ---------------------------------------------------------------------------
# Main weather system
# ---------------------------------------------------------------------------

class WeatherSystem:
    """
    Manages all weather effects and transitions.

    Parameters
    ----------
    base     : ShowBase
    renderer : CityRenderer
    audio    : AudioSystem
    config   : dict
    """

    def __init__(self, base: "ShowBase", renderer, audio, config: dict):
        self.base     = base
        self.renderer = renderer
        self.audio    = audio
        self.config   = config
        self._rng     = random.Random()

        self._state      = WeatherState.CLEAR
        self._target     = WeatherState.CLEAR
        self._blend       = 1.0        # 0→1 interpolation towards target
        self._blend_speed = 0.1        # blend units per second
        self._trans_timer = 0.0

        self._wetness    = 0.0
        self._fog_density= 0.0005

        self._rain   = RainParticleSystem(base, base.render)
        self._wind   = WindSystem()

        self._lightning_timer    = 0.0
        self._lightning_active   = False
        self._lightning_duration = 0.0

        self._setup_fog()

    # ------------------------------------------------------------------

    def _setup_fog(self):
        self._fog = Fog("weather_fog")
        self._fog.setColor(0.7, 0.7, 0.8)
        self._fog.setExpDensity(0.0005)
        self.base.render.attachNewNode(self._fog)
        self.base.render.setFog(self._fog)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float, hour: float):
        # State transition roll
        self._trans_timer += dt
        if self._trans_timer >= _TRANSITION_INTERVAL:
            self._trans_timer = 0.0
            self._roll_transition(hour)

        # Blend towards target
        self._blend = min(self._blend + dt * self._blend_speed, 1.0)
        self._apply_blend()

        # Rain
        params = _WEATHER_PARAMS[self._state]
        self._rain.update(dt, Vec3(self.base.camera.getPos()))

        # Lightning
        if self._state == WeatherState.THUNDER:
            self._update_lightning(dt)

        # Wind
        wind_strength = {
            WeatherState.CLEAR: 0.5, WeatherState.CLOUDY: 1.0,
            WeatherState.RAIN: 2.0, WeatherState.HEAVY_RAIN: 3.5,
            WeatherState.THUNDER: 5.0,
        }.get(self._state, 1.0)
        self._wind.set_strength(wind_strength)
        self._wind.update(dt)

        # Audio
        if self.audio:
            self.audio.set_rain(params["rain_rate"] > 0,
                                 intensity=params["rain_rate"] / 800.0)

    def _roll_transition(self, hour: float):
        """Choose next weather state based on Markov chain."""
        matrix = _TRANSITIONS[self._state]
        states = list(matrix.keys())
        weights = list(matrix.values())

        # Bias towards rain at night / early morning
        if hour < 6 or hour > 22:
            idx_rain = states.index(WeatherState.RAIN) if WeatherState.RAIN in states else -1
            if idx_rain >= 0:
                weights[idx_rain] *= 1.5

        total = sum(weights)
        weights = [w / total for w in weights]
        r = self._rng.random()
        cumul = 0.0
        for s, w in zip(states, weights):
            cumul += w
            if r <= cumul:
                if s != self._target:
                    self._target = s
                    self._blend  = 0.0
                return

    def _apply_blend(self):
        t = self._blend
        cur_p = _WEATHER_PARAMS[self._state]
        tgt_p = _WEATHER_PARAMS[self._target]

        fog   = cur_p["fog_density"] + (tgt_p["fog_density"] - cur_p["fog_density"]) * t
        wet   = cur_p["wetness"]     + (tgt_p["wetness"]     - cur_p["wetness"])     * t
        rate  = int(cur_p["rain_rate"] + (tgt_p["rain_rate"] - cur_p["rain_rate"]) * t)

        sky_c = Vec3(
            cur_p["sky_tint"].x + (tgt_p["sky_tint"].x - cur_p["sky_tint"].x) * t,
            cur_p["sky_tint"].y + (tgt_p["sky_tint"].y - cur_p["sky_tint"].y) * t,
            cur_p["sky_tint"].z + (tgt_p["sky_tint"].z - cur_p["sky_tint"].z) * t,
        )

        self._fog_density = fog
        self._wetness     = wet
        self._fog.setExpDensity(fog)

        if self.renderer:
            self.renderer.set_fog_density(fog, sky_c)
            self.base.render.setShaderInput("wetness", wet)

        self._rain.set_rate(rate)

        if t >= 1.0:
            self._state  = self._target

    # ------------------------------------------------------------------
    # Lightning
    # ------------------------------------------------------------------

    def _update_lightning(self, dt: float):
        self._lightning_timer += dt
        interval = self._rng.uniform(8.0, 25.0)
        if self._lightning_timer >= interval:
            self._lightning_timer = 0.0
            self._trigger_lightning()

        if self._lightning_active:
            self._lightning_duration -= dt
            if self._lightning_duration <= 0:
                self._lightning_active = False
                # Restore normal ambient
                self.base.render.setShaderInput("sunIntensity", 0.0)

    def _trigger_lightning(self):
        self._lightning_active   = True
        self._lightning_duration = self._rng.uniform(0.05, 0.2)
        # Bright flash via shader input
        self.base.render.setShaderInput("sunIntensity", 8.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, state: WeatherState):
        """Force-set weather state immediately."""
        self._target = state
        self._blend  = 0.0

    def register_foliage(self, np_: NodePath, phase: float = 0.0):
        """Register a tree/flag NodePath for wind sway."""
        self._wind.register(np_, phase)

    @property
    def state(self) -> WeatherState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.value

    @property
    def wetness(self) -> float:
        return self._wetness

    @property
    def fog_density(self) -> float:
        return self._fog_density

    def cleanup(self):
        self._rain.cleanup()
        self.base.render.clearFog()
