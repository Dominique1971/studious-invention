"""
npc_pedestrian.py - Pedestrian AI for the city simulator.

Features
--------
• Walks along sidewalk waypoints
• Waits at crosswalks when traffic light is red
• Steps back when a vehicle gets close
• Simple bob / walking-cycle animation via NodePath Z offset
• Multiple pedestrian "types" via different colour tints
"""

from __future__ import annotations

import math
import enum
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

from panda3d.core import (
    Vec3, Vec4, LColor, NodePath, BitMask32,
    CollisionSphere, CollisionNode,
    TransparencyAttrib,
)
from panda3d.bullet import (
    BulletWorld, BulletRigidBodyNode, BulletCapsuleShape, ZUp,
    BulletGhostNode,
)

if TYPE_CHECKING:
    from direct.showbase.ShowBase import ShowBase
    from npc_vehicle import TrafficLightSystem

# ---------------------------------------------------------------------------
# Pedestrian appearance variants
# ---------------------------------------------------------------------------

class PedestrianType(enum.Enum):
    ADULT_MALE   = "adult_male"
    ADULT_FEMALE = "adult_female"
    CHILD        = "child"
    ELDERLY      = "elderly"
    CYCLIST      = "cyclist"


PEDESTRIAN_CONFIG: Dict[PedestrianType, dict] = {
    PedestrianType.ADULT_MALE:   {"height": 1.8, "speed": 1.4, "colour": Vec4(0.3, 0.3, 0.8, 1)},
    PedestrianType.ADULT_FEMALE: {"height": 1.65,"speed": 1.3, "colour": Vec4(0.8, 0.3, 0.5, 1)},
    PedestrianType.CHILD:        {"height": 1.2, "speed": 1.6, "colour": Vec4(0.8, 0.7, 0.1, 1)},
    PedestrianType.ELDERLY:      {"height": 1.6, "speed": 0.7, "colour": Vec4(0.5, 0.5, 0.5, 1)},
    PedestrianType.CYCLIST:      {"height": 1.8, "speed": 4.0, "colour": Vec4(0.2, 0.7, 0.2, 1)},
}

# Distance at which a pedestrian reacts to an approaching vehicle
VEHICLE_AVOID_DIST  = 8.0
VEHICLE_FLEE_DIST   = 4.0
CROSSWALK_WAIT_DIST = 3.0

# ---------------------------------------------------------------------------
# Waypoint route
# ---------------------------------------------------------------------------

class SidewalkRoute:
    """Closed polygon of (x, z) sidewalk waypoints."""

    def __init__(self, waypoints: List[Tuple[float, float]]):
        self.waypoints = waypoints
        self._idx = 0

    def current(self) -> Tuple[float, float]:
        return self.waypoints[self._idx]

    def advance(self):
        self._idx = (self._idx + 1) % len(self.waypoints)

    def distance_to_current(self, x: float, z: float) -> float:
        tx, tz = self.current()
        return math.hypot(x - tx, z - tz)

    def next_waypoint(self) -> Tuple[float, float]:
        nxt = (self._idx + 1) % len(self.waypoints)
        return self.waypoints[nxt]


# ---------------------------------------------------------------------------
# Pedestrian
# ---------------------------------------------------------------------------

class NPCPedestrian:
    """
    A single AI pedestrian.

    Parameters
    ----------
    base        : ShowBase
    bullet_world: BulletWorld
    route       : SidewalkRoute
    ped_type    : PedestrianType
    rng         : random.Random
    """

    REACH_DIST    = 1.5
    BOB_AMP       = 0.04   # metres
    BOB_FREQ      = 2.8    # Hz (steps per second)

    def __init__(
        self,
        base: "ShowBase",
        bullet_world: BulletWorld,
        route: SidewalkRoute,
        ped_type: PedestrianType = PedestrianType.ADULT_MALE,
        rng: Optional[random.Random] = None,
    ):
        self.base         = base
        self.bullet_world = bullet_world
        self.route        = route
        self.ped_type     = ped_type
        self._rng         = rng or random.Random()
        self._cfg         = PEDESTRIAN_CONFIG[ped_type]

        self._speed       = self._cfg["speed"] * self._rng.uniform(0.85, 1.15)
        self._bob_phase   = self._rng.uniform(0, math.pi * 2)
        self._fleeing     = False
        self._flee_dir    = Vec3(0, 0, 0)
        self._flee_timer  = 0.0
        self._waiting     = False
        self._wait_timer  = 0.0

        wx, wz = route.current()
        self._build_physics(wx, wz)
        self._load_model()

    # ------------------------------------------------------------------

    def _build_physics(self, wx: float, wz: float):
        h = self._cfg["height"]
        shape = BulletCapsuleShape(0.25, h - 0.5, ZUp)
        body  = BulletRigidBodyNode(f"ped_{id(self)}")
        body.setMass(70.0)
        body.addShape(shape)
        body.setAngularFactor(Vec3(0, 0, 0))   # no tipping
        body.setDeactivationEnabled(False)
        self._np = self.base.render.attachNewNode(body)
        self._np.setPos(wx, wz, 0.0)
        self._np.setCollideMask(BitMask32.bit(2))
        self.bullet_world.attachRigidBody(body)
        self._body = body

    def _load_model(self):
        asset_root = Path(__file__).resolve().parent / "assets" / "kenney" / "pedestrians"
        path = asset_root / f"{self.ped_type.value}.egg"
        self._model = None

        if path.exists():
            try:
                self._model = self.base.loader.loadModel(str(path))
            except Exception:
                self._model = None

        if self._model is None or self._model.isEmpty():
            # Capsule placeholder using a thin box
            from renderer import CityRenderer
            h = self._cfg["height"]
            self._model = CityRenderer.make_colored_box(
                self.base, Vec3(0.5, 0.5, h), self._cfg["colour"],
                f"ped_body_{id(self)}"
            )
        self._model.reparentTo(self._np)
        self._model.setPos(0, 0, 0)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        vehicle_positions: List[Vec3],
        traffic_light: "TrafficLightSystem",
        crosswalk_positions: Optional[List[Tuple[float, float]]] = None,
    ):
        pos = Vec3(self._np.getPos())

        # 1 – Vehicle avoidance
        flee_vec = self._check_vehicle_proximity(pos, vehicle_positions)
        if flee_vec.length() > 0.01:
            self._fleeing  = True
            self._flee_dir = flee_vec
            self._flee_timer = 2.0

        if self._fleeing:
            self._flee_timer -= dt
            if self._flee_timer <= 0:
                self._fleeing = False
            move = self._flee_dir * self._speed * 1.5 * dt
            self._np.setPos(pos + Vec3(move.x, move.y, 0))
            self._animate_bob(dt)
            return

        # 2 – Traffic light check at crosswalk
        if crosswalk_positions and self._near_crosswalk(pos, crosswalk_positions):
            # Check if heading NS or EW to pick correct light phase
            tx, tz = self.route.current()
            going_ns = abs(tz - pos.y) > abs(tx - pos.x)
            green = (traffic_light.is_green_ns() if going_ns
                     else traffic_light.is_green_ew())
            if not green:
                self._waiting = True
                self._wait_timer += dt
                # Idle animation even while waiting
                self._animate_bob(dt, idle=True)
                return
        self._waiting = False

        # 3 – Walk toward waypoint
        tx, tz = self.route.current()
        dx, dz = tx - pos.x, tz - pos.y
        dist   = math.hypot(dx, dz)
        if dist < self.REACH_DIST:
            self.route.advance()
            tx, tz = self.route.current()
            dx, dz = tx - pos.x, tz - pos.y
            dist   = math.hypot(dx, dz) + 1e-6

        step = self._speed * dt
        nx   = pos.x + (dx / dist) * step
        nz   = pos.y + (dz / dist) * step
        self._np.setPos(nx, nz, pos.z)

        # Face direction of travel
        if dist > 0.1:
            heading_deg = math.degrees(math.atan2(dx, dz))
            self._np.setH(heading_deg)

        # Walking bob
        self._animate_bob(dt)

    def _check_vehicle_proximity(
        self, pos: Vec3, vehicle_positions: List[Vec3]
    ) -> Vec3:
        flee = Vec3(0, 0, 0)
        for vp in vehicle_positions:
            d = math.hypot(vp.x - pos.x, vp.y - pos.y)
            if d < VEHICLE_AVOID_DIST:
                strength = 1.0 - d / VEHICLE_AVOID_DIST
                flee.x  += (pos.x - vp.x) / max(d, 0.1) * strength * 2.0
                flee.y  += (pos.y - vp.y) / max(d, 0.1) * strength * 2.0
        return flee

    def _near_crosswalk(self, pos: Vec3, crosswalks: List[Tuple[float, float]]) -> bool:
        for cx, cz in crosswalks:
            if math.hypot(cx - pos.x, cz - pos.y) < CROSSWALK_WAIT_DIST:
                return True
        return False

    def _animate_bob(self, dt: float, idle: bool = False):
        """Vertical head-bob walking animation."""
        freq = self.BOB_FREQ if not idle else 0.5
        self._bob_phase = (self._bob_phase + dt * freq * math.pi * 2) % (math.pi * 2)
        bob_z = math.sin(self._bob_phase) * self.BOB_AMP
        self._model.setZ(bob_z)

    # ------------------------------------------------------------------

    @property
    def position(self) -> Vec3:
        return Vec3(self._np.getPos())

    @property
    def is_waiting(self) -> bool:
        return self._waiting

    def cleanup(self):
        self.bullet_world.removeRigidBody(self._body)
        self._np.removeNode()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def spawn_pedestrians(
    base,
    bullet_world: BulletWorld,
    sidewalk_waypoints: List[Tuple[float, float]],
    count: int,
    rng: Optional[random.Random] = None,
) -> List[NPCPedestrian]:
    """Spawn *count* pedestrians spread across sidewalk waypoints."""
    _rng = rng or random.Random()
    if not sidewalk_waypoints:
        return []

    pedestrians: List[NPCPedestrian] = []
    ped_types = list(PedestrianType)
    weights   = [35, 35, 10, 15, 5]

    # Group waypoints into routes of ~8-16 points each
    route_size = 10
    routes: List[SidewalkRoute] = []
    pts = list(sidewalk_waypoints)
    _rng.shuffle(pts)
    for i in range(0, len(pts), route_size):
        chunk = pts[i : i + route_size]
        if len(chunk) >= 2:
            routes.append(SidewalkRoute(chunk))

    if not routes:
        return []

    for _ in range(count):
        route = _rng.choice(routes)
        ptype = _rng.choices(ped_types, weights=weights)[0]
        ped   = NPCPedestrian(base, bullet_world, SidewalkRoute(list(route.waypoints)),
                               ptype, _rng)
        pedestrians.append(ped)

    return pedestrians
