"""
lighting.py - Dynamic time-of-day lighting system.

• Sun / moon directional light tracking
• Sky colour, ambient light, and fog that change smoothly with hour
• Street light PointLights activated at dusk
• Per-district colour temperature hints
• Headlight helpers for vehicles
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, TYPE_CHECKING

from panda3d.core import (
    AmbientLight, DirectionalLight, PointLight, Spotlight,
    NodePath, Vec3, Vec4, LColor,
    PandaNode,
)

if TYPE_CHECKING:
    from direct.showbase.ShowBase import ShowBase
    from renderer import CityRenderer

# ---------------------------------------------------------------------------
# Sky gradient colour stops (hour → RGB)
# ---------------------------------------------------------------------------

_SKY_COLOURS: List[Tuple[float, Vec3]] = [
    (0.0,  Vec3(0.02, 0.02, 0.08)),   # midnight
    (5.0,  Vec3(0.05, 0.04, 0.12)),   # pre-dawn deep blue
    (6.0,  Vec3(0.55, 0.27, 0.15)),   # dawn pink/orange
    (7.5,  Vec3(0.70, 0.58, 0.45)),   # early morning
    (9.0,  Vec3(0.53, 0.81, 1.00)),   # morning blue
    (12.0, Vec3(0.40, 0.70, 1.00)),   # noon
    (16.0, Vec3(0.45, 0.72, 1.00)),   # afternoon
    (18.5, Vec3(0.90, 0.45, 0.10)),   # sunset orange
    (19.5, Vec3(0.55, 0.20, 0.10)),   # dusk red
    (20.5, Vec3(0.10, 0.08, 0.18)),   # early night
    (24.0, Vec3(0.02, 0.02, 0.08)),   # midnight again
]

_SUN_COLOURS: List[Tuple[float, Vec3]] = [
    (0.0,  Vec3(0.0,  0.0,  0.0 )),
    (5.8,  Vec3(0.0,  0.0,  0.0 )),
    (6.0,  Vec3(1.0,  0.5,  0.1 )),   # dawn orange sun
    (8.0,  Vec3(1.0,  0.9,  0.7 )),
    (12.0, Vec3(1.0,  1.0,  0.95)),   # noon white
    (16.0, Vec3(1.0,  0.95, 0.80)),
    (18.5, Vec3(1.0,  0.55, 0.15)),   # sunset
    (19.5, Vec3(0.6,  0.2,  0.05)),
    (20.0, Vec3(0.0,  0.0,  0.0 )),
    (24.0, Vec3(0.0,  0.0,  0.0 )),
]

_SUN_INTENSITY: List[Tuple[float, float]] = [
    (0.0,  0.0),
    (5.8,  0.0),
    (6.0,  0.3),
    (8.0,  0.8),
    (12.0, 1.2),
    (16.0, 1.0),
    (18.5, 0.6),
    (19.5, 0.2),
    (20.0, 0.0),
    (24.0, 0.0),
]

_AMBIENT_INTENSITY: List[Tuple[float, float]] = [
    (0.0,  0.05),
    (6.0,  0.10),
    (8.0,  0.25),
    (12.0, 0.40),
    (18.5, 0.20),
    (20.0, 0.08),
    (24.0, 0.05),
]

_FOG_DENSITY: List[Tuple[float, float]] = [
    (0.0,  0.003),
    (5.0,  0.006),   # dawn fog
    (9.0,  0.001),
    (18.0, 0.001),
    (20.0, 0.003),
    (24.0, 0.003),
]

# Hours when street lights are on
_STREET_LIGHT_ON_HOUR  = 19.5
_STREET_LIGHT_OFF_HOUR =  6.5


def _lerp_colour_stops(stops: list, hour: float) -> Vec3:
    """Linearly interpolate between colour stops at *hour*."""
    # Clamp to [0, 24]
    h = hour % 24.0
    for i in range(len(stops) - 1):
        h0, c0 = stops[i]
        h1, c1 = stops[i + 1]
        if h0 <= h <= h1:
            t = (h - h0) / max(h1 - h0, 1e-6)
            return Vec3(
                c0.x + (c1.x - c0.x) * t,
                c0.y + (c1.y - c0.y) * t,
                c0.z + (c1.z - c0.z) * t,
            )
    return stops[-1][1]


def _lerp_scalar_stops(stops: list, hour: float) -> float:
    h = hour % 24.0
    for i in range(len(stops) - 1):
        h0, v0 = stops[i]
        h1, v1 = stops[i + 1]
        if h0 <= h <= h1:
            t = (h - h0) / max(h1 - h0, 1e-6)
            return v0 + (v1 - v0) * t
    return stops[-1][1]


# ---------------------------------------------------------------------------
# Street light data class
# ---------------------------------------------------------------------------

class StreetLight:
    def __init__(self, pos: Vec3, parent: NodePath, base):
        self._light = PointLight("street_light")
        self._light.setColor(LColor(1.0, 0.85, 0.55, 1.0))
        self._light.setAttenuation(Vec3(0.5, 0.02, 0.002))
        self._np = parent.attachNewNode(self._light)
        self._np.setPos(pos)
        base.render.setLight(self._np)
        self._active = False
        self._base   = base
        self.set_active(False)

    def set_active(self, active: bool):
        if active == self._active:
            return
        self._active = active
        if active:
            self._base.render.setLight(self._np)
        else:
            self._base.render.clearLight(self._np)

    def cleanup(self):
        self._base.render.clearLight(self._np)
        self._np.removeNode()


# ---------------------------------------------------------------------------
# Main lighting system
# ---------------------------------------------------------------------------

class LightingSystem:
    """
    Manages the full lighting environment for a given time of day.

    Parameters
    ----------
    base : ShowBase
        The running Panda3D application.
    renderer : CityRenderer
        Used to configure shadow maps on the sun light.
    config : dict
        Full game configuration.
    """

    def __init__(self, base, renderer, config: dict):
        self.base     = base
        self.renderer = renderer
        self.config   = config
        self._hour    = 12.0      # start at noon
        self._time_scale = 60.0   # real seconds per game hour (1 min = 1 game hour)

        self._street_lights: List[StreetLight] = []

        self._setup_ambient()
        self._setup_sun()
        self._setup_moon()
        self._apply_hour(self._hour)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ambient(self):
        self._ambient_light = AmbientLight("ambient")
        self._ambient_np    = self.base.render.attachNewNode(self._ambient_light)
        self.base.render.setLight(self._ambient_np)

    def _setup_sun(self):
        self._sun = DirectionalLight("sun")
        self._sun.setColor(LColor(1, 1, 0.9, 1))
        # Enable shadow casting via renderer
        if self.renderer:
            self.renderer.configure_directional_shadow(self._sun, film_size=1200.0)
        self._sun_np = self.base.render.attachNewNode(self._sun)
        self._sun_np.setHpr(45, -60, 0)   # default noon angle
        self.base.render.setLight(self._sun_np)

    def _setup_moon(self):
        self._moon = DirectionalLight("moon")
        self._moon.setColor(LColor(0.15, 0.15, 0.25, 1))
        self._moon_np = self.base.render.attachNewNode(self._moon)
        self._moon_np.setHpr(200, -40, 0)
        # Moon is always active but intensity near-zero at day
        self.base.render.setLight(self._moon_np)

    # ------------------------------------------------------------------
    # Street lights
    # ------------------------------------------------------------------

    def register_street_lights(self, positions: List[Vec3]):
        """
        Create PointLight nodes for each street lamp position.
        Call after city geometry is loaded.
        """
        root = self.base.render
        for pos in positions:
            sl = StreetLight(pos, root, self.base)
            self._street_lights.append(sl)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, dt: float):
        """Advance time and update all lights. Call once per frame with dt in seconds."""
        self._hour = (self._hour + dt / self._time_scale) % 24.0
        self._apply_hour(self._hour)

    def set_hour(self, hour: float):
        """Jump directly to a specific time of day (0–24)."""
        self._hour = hour % 24.0
        self._apply_hour(self._hour)

    @property
    def hour(self) -> float:
        return self._hour

    def _apply_hour(self, hour: float):
        sky_col   = _lerp_colour_stops(_SKY_COLOURS, hour)
        sun_col   = _lerp_colour_stops(_SUN_COLOURS, hour)
        sun_int   = _lerp_scalar_stops(_SUN_INTENSITY, hour)
        amb_int   = _lerp_scalar_stops(_AMBIENT_INTENSITY, hour)
        fog_dens  = _lerp_scalar_stops(_FOG_DENSITY, hour)

        # Sky / background colour
        self.base.win.setClearColor(LColor(sky_col.x, sky_col.y, sky_col.z, 1.0))

        # Ambient
        a = LColor(sky_col.x * amb_int, sky_col.y * amb_int, sky_col.z * amb_int, 1.0)
        self._ambient_light.setColor(a)

        # Sun direction (simple elevation based on hour)
        elevation_deg = _sun_elevation(hour)
        azimuth_deg   = (hour / 24.0) * 360.0 - 90.0
        self._sun_np.setHpr(azimuth_deg, -elevation_deg, 0)
        sun_c = LColor(sun_col.x * sun_int, sun_col.y * sun_int,
                        sun_col.z * sun_int, 1.0)
        self._sun.setColor(sun_c)

        # Moon (opposite direction, dim)
        moon_int = max(0.0, 1.0 - sun_int * 3)
        self._moon_np.setHpr(azimuth_deg + 180, -(-elevation_deg), 0)
        self._moon.setColor(LColor(0.15 * moon_int, 0.15 * moon_int,
                                    0.25 * moon_int, 1.0))

        # Street lights
        lights_on = hour >= _STREET_LIGHT_ON_HOUR or hour < _STREET_LIGHT_OFF_HOUR
        for sl in self._street_lights:
            sl.set_active(lights_on)

        # Update renderer fog
        if self.renderer:
            self.renderer.set_fog_density(fog_dens, sky_col)
            # Push uniforms to PBR shader
            sun_dir = _compute_sun_direction(azimuth_deg, elevation_deg)
            self.renderer.update_shader_inputs(
                sun_dir, sun_col, sun_int,
                Vec3(self.base.camera.getPos()),
                0.0, hour * 3600.0
            )

    # ------------------------------------------------------------------
    # Headlight factory
    # ------------------------------------------------------------------

    def create_headlights(self, vehicle_np: NodePath) -> Tuple[NodePath, NodePath]:
        """
        Attach a pair of Spotlight headlights to *vehicle_np*.
        Returns (left_np, right_np).
        """
        results = []
        for side, ox in [("left", -0.7), ("right", 0.7)]:
            spot = Spotlight(f"headlight_{side}")
            spot.setColor(LColor(1.0, 0.98, 0.9, 1.0))
            lens = spot.getLens()
            lens.setFov(25, 20)
            lens.setNearFar(0.5, 80.0)
            np_ = vehicle_np.attachNewNode(spot)
            np_.setPos(ox, 2.1, 0.55)
            np_.setHpr(0, -5, 0)
            self.base.render.setLight(np_)
            results.append(np_)
        return tuple(results)

    def create_taillights(self, vehicle_np: NodePath) -> Tuple[NodePath, NodePath]:
        """Attach red PointLight taillights to *vehicle_np*."""
        results = []
        for side, ox in [("left", -0.6), ("right", 0.6)]:
            pl = PointLight(f"taillight_{side}")
            pl.setColor(LColor(1.0, 0.05, 0.05, 1.0))
            pl.setAttenuation(Vec3(1, 0.05, 0.01))
            np_ = vehicle_np.attachNewNode(pl)
            np_.setPos(ox, -2.0, 0.5)
            self.base.render.setLight(np_)
            results.append(np_)
        return tuple(results)

    # ------------------------------------------------------------------
    # District colour temperature
    # ------------------------------------------------------------------

    def get_district_colour_temperature(self, district: str) -> Vec3:
        """Return a warm/cool tint Vec3 for neon/industrial districts."""
        mapping = {
            "downtown":   Vec3(1.0, 0.85, 0.60),   # warm neon
            "midtown":    Vec3(0.95, 0.95, 1.00),
            "suburbs":    Vec3(1.00, 0.95, 0.85),
            "industrial": Vec3(0.80, 0.90, 1.00),   # cool halogen
            "park":       Vec3(1.00, 1.00, 0.95),
        }
        return mapping.get(district, Vec3(1, 1, 1))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        for sl in self._street_lights:
            sl.cleanup()
        self._street_lights.clear()
        self.base.render.clearLight(self._sun_np)
        self.base.render.clearLight(self._moon_np)
        self.base.render.clearLight(self._ambient_np)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sun_elevation(hour: float) -> float:
    """Return sun elevation angle (degrees) for a given hour."""
    if hour < 5.5 or hour > 19.0:
        return -20.0
    t = (hour - 5.5) / (19.0 - 5.5)     # 0→1 over the day
    return math.sin(t * math.pi) * 75.0   # peak 75° at noon


def _compute_sun_direction(azimuth_deg: float, elevation_deg: float) -> Vec3:
    az  = math.radians(azimuth_deg)
    el  = math.radians(elevation_deg)
    x   = math.cos(el) * math.sin(az)
    y   = math.cos(el) * math.cos(az)
    z   = math.sin(el)
    return Vec3(x, y, z)
