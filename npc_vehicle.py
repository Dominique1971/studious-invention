"""
npc_vehicle.py - AI traffic vehicles.

Features
--------
• NPCVehicle – path-following along road segments, traffic-light obedience,
  lane keeping, collision avoidance, turn signals
• EmergencyVehicle subclass – red/blue flashing lights + siren
• Dynamic traffic density (rush-hour scaling at 8 am / 5 pm)
• Simple but stable physics using BulletVehicle
"""

from __future__ import annotations

import math
import random
import enum
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

from panda3d.core import (
    Vec3, Vec4, LPoint3f, NodePath, BitMask32,
    LColor, PointLight, LineSegs,
)
from panda3d.bullet import (
    BulletWorld, BulletRigidBodyNode, BulletBoxShape,
    BulletVehicle, ZUp, BulletRayHit,
)

if TYPE_CHECKING:
    from city_generator import RoadSegment, Intersection
    from direct.showbase.ShowBase import ShowBase

# ---------------------------------------------------------------------------
# Enums / constants
# ---------------------------------------------------------------------------

class NPCVehicleType(enum.Enum):
    CAR       = "car"
    TRUCK     = "truck"
    BUS       = "bus"
    EMERGENCY = "emergency"


# Speed limits by road type (km/h)
SPEED_LIMITS: Dict[str, float] = {
    "arterial": 60.0,
    "local":    40.0,
    "highway": 100.0,
    "path":     20.0,
}

COLOURS = [
    Vec4(0.7, 0.1, 0.1, 1), Vec4(0.1, 0.2, 0.8, 1), Vec4(0.9, 0.9, 0.2, 1),
    Vec4(0.1, 0.6, 0.2, 1), Vec4(0.5, 0.5, 0.5, 1), Vec4(0.9, 0.5, 0.0, 1),
    Vec4(0.2, 0.6, 0.8, 1), Vec4(0.8, 0.8, 0.8, 1),
]

# ---------------------------------------------------------------------------
# Traffic light state tracker (shared singleton)
# ---------------------------------------------------------------------------

class TrafficLightSystem:
    """Very simple 4-way traffic light cycle. Shared across all NPCs."""

    CYCLE_SECONDS = 30.0      # full red+green cycle

    def __init__(self):
        self._phase = 0.0      # 0 – CYCLE_SECONDS

    def update(self, dt: float):
        self._phase = (self._phase + dt) % self.CYCLE_SECONDS

    def is_green_ns(self) -> bool:
        """True when North–South traffic has green light."""
        return self._phase < self.CYCLE_SECONDS * 0.45

    def is_green_ew(self) -> bool:
        return not self.is_green_ns()

    def seconds_until_green_ns(self) -> float:
        if self.is_green_ns():
            return 0.0
        return self.CYCLE_SECONDS - self._phase

    def get_phase_fraction(self) -> float:
        return self._phase / self.CYCLE_SECONDS


_GLOBAL_TRAFFIC_LIGHTS = TrafficLightSystem()


def get_traffic_light_system() -> TrafficLightSystem:
    return _GLOBAL_TRAFFIC_LIGHTS


# ---------------------------------------------------------------------------
# Waypoint path
# ---------------------------------------------------------------------------

class RoadPath:
    """Sequence of (x, z) waypoints describing a drive path along roads."""

    def __init__(self, waypoints: List[Tuple[float, float]], loop: bool = True):
        self.waypoints = waypoints
        self.loop = loop
        self._idx = 0

    @classmethod
    def from_road_segment(cls, seg: "RoadSegment", reverse: bool = False) -> "RoadPath":
        steps = max(2, int(math.hypot(seg.x2 - seg.x1, seg.z2 - seg.z1) / 20))
        xs = [seg.x1 + (seg.x2 - seg.x1) * i / (steps - 1) for i in range(steps)]
        zs = [seg.z1 + (seg.z2 - seg.z1) * i / (steps - 1) for i in range(steps)]
        pts = list(zip(xs, zs))
        if reverse:
            pts = pts[::-1]
        return cls(pts, loop=True)

    def current(self) -> Tuple[float, float]:
        return self.waypoints[self._idx]

    def advance(self) -> bool:
        """Move to next waypoint. Returns True when looped."""
        self._idx += 1
        if self._idx >= len(self.waypoints):
            if self.loop:
                self._idx = 0
                return True
            else:
                self._idx = len(self.waypoints) - 1
        return False

    def distance_to_current(self, x: float, z: float) -> float:
        tx, tz = self.current()
        return math.hypot(x - tx, z - tz)


# ---------------------------------------------------------------------------
# NPC Vehicle
# ---------------------------------------------------------------------------

class NPCVehicle:
    """
    AI-controlled traffic vehicle.

    Parameters
    ----------
    base : ShowBase
    bullet_world : BulletWorld
    road_path : RoadPath
    npc_type : NPCVehicleType
    spawn_offset : float   lateral lane offset in metres
    rng : random.Random
    """

    WAYPOINT_REACH_DIST = 6.0
    RAY_LENGTH          = 15.0
    AVOID_FORCE         = 3.0

    def __init__(
        self,
        base: "ShowBase",
        bullet_world: BulletWorld,
        road_path: RoadPath,
        npc_type: NPCVehicleType = NPCVehicleType.CAR,
        spawn_offset: float = 0.0,
        rng: Optional[random.Random] = None,
    ):
        self.base         = base
        self.bullet_world = bullet_world
        self.path         = road_path
        self.npc_type     = npc_type
        self._rng         = rng or random.Random()
        self._colour      = self._rng.choice(COLOURS)
        self._lane_offset = spawn_offset

        # State
        self._speed_kph   = 0.0
        self._target_speed= SPEED_LIMITS.get("local", 40.0)
        self._stopped     = False
        self._stopped_for = 0.0   # seconds stopped at light
        self._indicator   = 0     # -1 left, 0 none, +1 right
        self._indicator_timer = 0.0
        self._blink_on    = False
        self._blink_timer = 0.0

        self._build(road_path)
        self._load_model()
        self._setup_lights()

    # ------------------------------------------------------------------

    def _build(self, road_path: RoadPath):
        wx, wz = road_path.current()
        shape  = BulletBoxShape(Vec3(0.9, 2.0, 0.4))
        body   = BulletRigidBodyNode(f"npc_{id(self)}")
        body.setMass(1400.0)
        body.addShape(shape)
        body.setDeactivationEnabled(False)
        self._chassis_np = self.base.render.attachNewNode(body)
        self._chassis_np.setPos(wx, wz, 1.0)
        self._chassis_np.setCollideMask(BitMask32.allOn())
        self.bullet_world.attachRigidBody(body)
        self._body_node = body

        self._vehicle = BulletVehicle(self.bullet_world, body)
        self._vehicle.setCoordinateSystem(ZUp)
        self.bullet_world.attachVehicle(self._vehicle)

        for ox, oy in [(-0.9, 1.4), (0.9, 1.4), (-0.9, -1.4), (0.9, -1.4)]:
            w = self._vehicle.createWheel()
            w.setChassisConnectionPointCs(Vec3(ox, oy, 0.0))
            w.setFrontWheel(oy > 0)
            w.setWheelDirectionCs(Vec3(0, 0, -1))
            w.setWheelAxleCs(Vec3(1, 0, 0))
            w.setWheelRadius(0.32)
            w.setSuspensionStiffness(40.0)
            w.setWheelsDampingRelaxation(2.3)
            w.setWheelsDampingCompression(4.4)
            w.setFrictionSlip(1.2)
            w.setRollInfluence(0.12)

    def _load_model(self):
        path = f"assets/kenney/car_kit/models/{self.npc_type.value}.egg"
        try:
            self._model_np = self.base.loader.loadModel(path)
        except Exception:
            from renderer import CityRenderer
            self._model_np = CityRenderer.make_colored_box(
                self.base, Vec3(1.8, 4.0, 1.4), self._colour, f"npc_body_{id(self)}"
            )
        self._model_np.reparentTo(self._chassis_np)

    def _setup_lights(self):
        # Brake lights (red point lights at rear)
        self._brake_light_np = []
        for ox in (-0.6, 0.6):
            pl = PointLight(f"npc_brake_{id(self)}_{ox}")
            pl.setColor(LColor(0.8, 0.05, 0.05, 1))
            pl.setAttenuation(Vec3(1, 0.1, 0.02))
            np_ = self._chassis_np.attachNewNode(pl)
            np_.setPos(ox, -2.1, 0.5)
            self._brake_light_np.append(np_)
            self.base.render.setLight(np_)

        # Indicator PointLights
        self._indicator_left_np  = self._make_indicator_light(-0.9)
        self._indicator_right_np = self._make_indicator_light( 0.9)

    def _make_indicator_light(self, ox: float) -> NodePath:
        pl = PointLight(f"npc_ind_{id(self)}_{ox}")
        pl.setColor(LColor(1.0, 0.6, 0.0, 1))
        pl.setAttenuation(Vec3(1, 0.2, 0.1))
        np_ = self._chassis_np.attachNewNode(pl)
        np_.setPos(ox, 2.0 if ox > 0 else -2.0, 0.5)
        self.base.render.clearLight(np_)
        return np_

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float, hour: float, intersections):
        tl = get_traffic_light_system()
        pos = self._chassis_np.getPos()
        wx, wz = self.path.current()

        # Reached waypoint?
        dist = math.hypot(pos.x - wx, pos.y - wz)
        if dist < self.WAYPOINT_REACH_DIST:
            self.path.advance()
            wx, wz = self.path.current()
            # Set indicator if turning
            self._update_indicator(wx - pos.x, wz - pos.y)

        # Steer towards target
        target_dir  = Vec3(wx - pos.x, wz - pos.y, 0)
        if target_dir.length() > 0.01:
            target_dir.normalize()
        heading  = self._chassis_np.getQuat().getForward()
        cross_z  = heading.x * target_dir.y - heading.y * target_dir.x
        steer    = max(-0.4, min(0.4, cross_z * 2.5))

        # Traffic light check (stop if red at intersection)
        at_light = self._near_intersection(intersections, pos, 12.0)
        must_stop = False
        if at_light:
            go_ns = tl.is_green_ns()
            is_ns = abs(heading.y) > abs(heading.x)
            must_stop = not (go_ns == is_ns)

        # Rush-hour speed multiplier
        density_mult = _rush_hour_multiplier(hour)
        target_kph   = self._target_speed * density_mult

        # Collision avoidance ray
        avoid = self._cast_forward_ray(heading)

        if must_stop or avoid:
            self._stopped_for += dt
            engine = 0.0
            brake  = 60.0
            self._stopped = True
        else:
            self._stopped_for = 0.0
            self._stopped = False
            if self._speed_kph < target_kph:
                engine = 1200.0
                brake  = 0.0
            elif self._speed_kph > target_kph + 5:
                engine = 0.0
                brake  = 20.0
            else:
                engine = 400.0
                brake  = 0.0

        for i in range(4):
            self._vehicle.applyBrakes(brake, i)
        for i in (2, 3):
            self._vehicle.applyEngineForce(engine, i)
        for i in (0, 1):
            self._vehicle.setSteeringValue(steer, i)

        # Speed
        vel = self._body_node.getLinearVelocity()
        self._speed_kph = vel.length() * 3.6

        # Indicator blink
        self._blink_timer += dt
        if self._blink_timer > 0.5:
            self._blink_timer = 0.0
            self._blink_on = not self._blink_on
            self._apply_indicator_blink()

        # Indicator timeout
        self._indicator_timer += dt
        if self._indicator_timer > 3.0:
            self._set_indicator(0)
            self._indicator_timer = 0.0

    # ------------------------------------------------------------------

    def _near_intersection(self, intersections, pos: Vec3, radius: float) -> bool:
        for inter in intersections:
            if inter.has_traffic_light:
                if math.hypot(inter.x - pos.x, inter.z - pos.y) < radius:
                    return True
        return False

    def _cast_forward_ray(self, heading: Vec3) -> bool:
        """Return True if an obstacle is within RAY_LENGTH ahead."""
        origin = self._chassis_np.getPos() + Vec3(0, 0, 0.5)
        end    = origin + heading * self.RAY_LENGTH
        result = self.bullet_world.rayTestClosest(origin, end)
        if result.hasHit():
            node = result.getNode()
            if node is not self._body_node:
                return True
        return False

    def _update_indicator(self, dx: float, dz: float):
        heading = self._chassis_np.getQuat().getForward()
        right   = Vec3(heading.y, -heading.x, 0)
        dot     = right.x * dx + right.y * dz
        if dot > 0.3:
            self._set_indicator(1)
        elif dot < -0.3:
            self._set_indicator(-1)
        else:
            self._set_indicator(0)
        self._indicator_timer = 0.0

    def _set_indicator(self, val: int):
        self._indicator = val
        if val == 0:
            self.base.render.clearLight(self._indicator_left_np)
            self.base.render.clearLight(self._indicator_right_np)

    def _apply_indicator_blink(self):
        if self._indicator == 0:
            return
        if self._blink_on:
            if self._indicator < 0:
                self.base.render.setLight(self._indicator_left_np)
            else:
                self.base.render.setLight(self._indicator_right_np)
        else:
            self.base.render.clearLight(self._indicator_left_np)
            self.base.render.clearLight(self._indicator_right_np)

    # ------------------------------------------------------------------

    @property
    def speed_kph(self) -> float:
        return self._speed_kph

    @property
    def position(self) -> Vec3:
        return Vec3(self._chassis_np.getPos())

    def cleanup(self):
        self.bullet_world.removeVehicle(self._vehicle)
        self.bullet_world.removeRigidBody(self._body_node)
        for np_ in self._brake_light_np:
            self.base.render.clearLight(np_)
        self.base.render.clearLight(self._indicator_left_np)
        self.base.render.clearLight(self._indicator_right_np)
        self._chassis_np.removeNode()


# ---------------------------------------------------------------------------
# Emergency vehicle subclass
# ---------------------------------------------------------------------------

class EmergencyVehicle(NPCVehicle):
    """Police / ambulance / fire engine with flashing lights and siren."""

    FLASH_PERIOD = 0.4

    def __init__(self, base, bullet_world, road_path, rng=None):
        super().__init__(base, bullet_world, road_path,
                         NPCVehicleType.EMERGENCY, rng=rng)
        self._target_speed = 80.0
        self._flash_timer  = 0.0
        self._flash_state  = False
        self._setup_emergency_lights()

    def _setup_emergency_lights(self):
        colours = [(1, 0, 0), (0, 0, 1)]
        self._emerg_lights = []
        for i, (r, g, b) in enumerate(colours):
            pl = PointLight(f"emerg_{id(self)}_{i}")
            pl.setColor(LColor(r, g, b, 1))
            pl.setAttenuation(Vec3(0.3, 0.02, 0.005))
            np_ = self._chassis_np.attachNewNode(pl)
            np_.setPos((-0.6 if i == 0 else 0.6), 0, 1.2)
            self._emerg_lights.append(np_)
        self.base.render.setLight(self._emerg_lights[0])

    def update(self, dt, hour, intersections):
        # Emergency vehicles ignore traffic lights – pass empty list
        super().update(dt, hour, [])
        self._flash_timer += dt
        if self._flash_timer > self.FLASH_PERIOD:
            self._flash_timer = 0.0
            self._flash_state = not self._flash_state
            on_idx  = 0 if self._flash_state else 1
            off_idx = 1 if self._flash_state else 0
            self.base.render.setLight(self._emerg_lights[on_idx])
            self.base.render.clearLight(self._emerg_lights[off_idx])

    def cleanup(self):
        for np_ in self._emerg_lights:
            self.base.render.clearLight(np_)
        super().cleanup()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def spawn_traffic(
    base,
    bullet_world: BulletWorld,
    roads,
    count: int,
    rng: Optional[random.Random] = None,
) -> List[NPCVehicle]:
    """
    Spawn *count* NPC vehicles spread across the road network.
    Returns list of NPCVehicle instances.
    """
    _rng = rng or random.Random()
    vehicles: List[NPCVehicle] = []
    # Filter driveable roads
    driveable = [r for r in roads if r.road_type in ("arterial", "local")]
    if not driveable:
        return vehicles
    for _ in range(count):
        seg = _rng.choice(driveable)
        rev = _rng.random() < 0.5
        # Lane offset: half the road width / 4 (one lane)
        offset = (seg.width / 4) * (1 if not rev else -1)
        path   = RoadPath.from_road_segment(seg, reverse=rev)
        vtype  = _rng.choices(
            [NPCVehicleType.CAR, NPCVehicleType.TRUCK, NPCVehicleType.BUS,
             NPCVehicleType.EMERGENCY],
            weights=[75, 15, 8, 2]
        )[0]
        if vtype == NPCVehicleType.EMERGENCY:
            v = EmergencyVehicle(base, bullet_world, path, rng=_rng)
        else:
            v = NPCVehicle(base, bullet_world, path, vtype, offset, _rng)
        vehicles.append(v)
    return vehicles


# ---------------------------------------------------------------------------
# Rush-hour density helper
# ---------------------------------------------------------------------------

def _rush_hour_multiplier(hour: float) -> float:
    """Return speed multiplier (lower = more congestion = slower)."""
    # Morning rush 7:30–9:30, evening rush 16:30–18:30
    peaks = [(8.0, 0.6), (17.5, 0.55)]
    for ph, low in peaks:
        dist = abs(hour - ph)
        if dist < 1.5:
            t = 1.0 - dist / 1.5
            return 1.0 - t * (1.0 - low)
    return 1.0
