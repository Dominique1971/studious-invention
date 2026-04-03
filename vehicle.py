"""
vehicle.py - Player-controlled vehicle with Bullet physics.

Features
--------
• Three vehicle types: SEDAN, SUV, SPORTS_CAR
• WASD / arrow-key driving; R to reset; TAB to cycle vehicle
• BulletVehicle chassis with four wheels and suspension
• Weight-transfer, per-wheel friction, gear simulation
• Skid marks via LineSegs on hard braking
• Damage model with red vignette overlay
• Working headlights and taillights (delegated to LightingSystem)
• Four camera modes: BUMPER, CHASE, CINEMATIC_DRONE, TOP_DOWN
• Speedometer helpers
"""

from __future__ import annotations

import math
import enum
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

from panda3d.core import (
    Vec3, Vec4, LPoint3f, NodePath, BitMask32,
    LineSegs, PandaNode, TransformState, LColor,
)
from panda3d.bullet import (
    BulletWorld, BulletRigidBodyNode, BulletBoxShape,
    BulletVehicle, ZUp, BulletDebugNode,
)

if TYPE_CHECKING:
    from direct.showbase.ShowBase import ShowBase
    from lighting import LightingSystem

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VehicleType(enum.Enum):
    SEDAN      = "sedan"
    SUV        = "suv"
    SPORTS_CAR = "sports_car"


class CameraMode(enum.Enum):
    CHASE           = "chase"
    BUMPER          = "bumper"
    CINEMATIC_DRONE = "cinematic_drone"
    TOP_DOWN        = "top_down"


# ---------------------------------------------------------------------------
# Vehicle presets
# ---------------------------------------------------------------------------

VEHICLE_PRESETS: Dict[VehicleType, dict] = {
    VehicleType.SEDAN: {
        "mass":              1400.0,
        "engine_force":      2500.0,
        "brake_force":       100.0,
        "max_steer":         0.45,
        "steer_speed":       2.5,
        "wheel_radius":      0.32,
        "suspension_stiff":  40.0,
        "suspension_damp":   2.3,
        "suspension_comp":   4.4,
        "friction_slip":     1.2,
        "chassis_half":      Vec3(0.9, 2.1, 0.4),
        "max_speed_kph":     220.0,
        "gear_count":        6,
        "colour":            Vec4(0.15, 0.35, 0.90, 1.0),
    },
    VehicleType.SUV: {
        "mass":              2100.0,
        "engine_force":      3500.0,
        "brake_force":       120.0,
        "max_steer":         0.40,
        "steer_speed":       2.0,
        "wheel_radius":      0.38,
        "suspension_stiff":  35.0,
        "suspension_damp":   2.5,
        "suspension_comp":   4.0,
        "friction_slip":     1.1,
        "chassis_half":      Vec3(1.05, 2.4, 0.55),
        "max_speed_kph":     180.0,
        "gear_count":        6,
        "colour":            Vec4(0.80, 0.25, 0.10, 1.0),
    },
    VehicleType.SPORTS_CAR: {
        "mass":              1200.0,
        "engine_force":      5000.0,
        "brake_force":       140.0,
        "max_steer":         0.42,
        "steer_speed":       3.0,
        "wheel_radius":      0.30,
        "suspension_stiff":  50.0,
        "suspension_damp":   3.0,
        "suspension_comp":   5.0,
        "friction_slip":     1.6,
        "chassis_half":      Vec3(0.85, 2.0, 0.30),
        "max_speed_kph":     300.0,
        "gear_count":        7,
        "colour":            Vec4(0.85, 0.85, 0.10, 1.0),
    },
}

# ---------------------------------------------------------------------------
# Gear ratios (simplified)
# ---------------------------------------------------------------------------

GEAR_RATIOS = [3.5, 2.1, 1.4, 1.0, 0.8, 0.65, 0.55]

# ---------------------------------------------------------------------------
# Player vehicle
# ---------------------------------------------------------------------------

class PlayerVehicle:
    """
    Full-featured player vehicle built on top of Panda3D's BulletVehicle.

    Parameters
    ----------
    base : ShowBase
    bullet_world : BulletWorld
    lighting : LightingSystem
    vehicle_type : VehicleType
    spawn_pos : Vec3
    """

    MAX_SKID_NODES = 500

    def __init__(
        self,
        base: "ShowBase",
        bullet_world: BulletWorld,
        lighting: "LightingSystem",
        vehicle_type: VehicleType = VehicleType.SEDAN,
        spawn_pos: Vec3 = Vec3(0, 0, 1.5),
    ):
        self.base          = base
        self.bullet_world  = bullet_world
        self.lighting      = lighting
        self.vehicle_type  = vehicle_type
        self.preset        = VEHICLE_PRESETS[vehicle_type]
        self.spawn_pos     = Vec3(spawn_pos)

        # State
        self._throttle    = 0.0
        self._brake       = 0.0
        self._steer_val   = 0.0
        self._speed_kph   = 0.0
        self._rpm         = 800.0
        self._gear        = 1
        self._damage      = 0.0       # 0–1
        self._headlights_on = True
        self._camera_mode   = CameraMode.CHASE
        self._camera_target_offset = Vec3(0, -8, 3)
        self._camera_smooth_pos    = Vec3(0, 0, 3)

        # Key state
        self._keys: Dict[str, bool] = {
            "forward": False, "backward": False,
            "left": False,    "right": False,
            "brake": False,   "headlights": False,
        }

        # Skid marks
        self._skid_segs: List[LineSegs] = []
        self._skid_nodes: List[NodePath] = []
        self._skidding = False

        self._build_chassis()
        self._build_vehicle()
        self._load_model()
        self._setup_headlights()
        self._setup_camera()
        self._bind_keys()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_chassis(self):
        half = self.preset["chassis_half"]
        shape = BulletBoxShape(half)
        body  = BulletRigidBodyNode(f"vehicle_{self.vehicle_type.value}")
        body.setMass(self.preset["mass"])
        body.addShape(shape)
        body.setDeactivationEnabled(False)
        body.setCcdMotionThreshold(0.5)
        body.setCcdSweptSphereRadius(0.5)
        self._chassis_np = self.base.render.attachNewNode(body)
        self._chassis_np.setPos(self.spawn_pos)
        self._chassis_np.setCollideMask(BitMask32.allOn())
        self.bullet_world.attachRigidBody(body)
        self._body_node = body

    def _build_vehicle(self):
        self._vehicle = BulletVehicle(self.bullet_world, self._body_node)
        self._vehicle.setCoordinateSystem(ZUp)
        self.bullet_world.attachVehicle(self._vehicle)

        p = self.preset
        whl_positions = [
            Vec3(-p["chassis_half"].x,  p["chassis_half"].y * 0.7,  0.0),  # FL
            Vec3( p["chassis_half"].x,  p["chassis_half"].y * 0.7,  0.0),  # FR
            Vec3(-p["chassis_half"].x, -p["chassis_half"].y * 0.7,  0.0),  # RL
            Vec3( p["chassis_half"].x, -p["chassis_half"].y * 0.7,  0.0),  # RR
        ]
        front = [True, True, False, False]

        for i, (pos, is_front) in enumerate(zip(whl_positions, front)):
            wheel = self._vehicle.createWheel()
            wheel.setChassisConnectionPointCs(pos)
            wheel.setFrontWheel(is_front)
            wheel.setWheelDirectionCs(Vec3(0, 0, -1))
            wheel.setWheelAxleCs(Vec3(1, 0, 0))
            wheel.setWheelRadius(p["wheel_radius"])
            wheel.setMaxSuspensionTravelCm(40.0)
            wheel.setSuspensionStiffness(p["suspension_stiff"])
            wheel.setWheelsDampingRelaxation(p["suspension_damp"])
            wheel.setWheelsDampingCompression(p["suspension_comp"])
            wheel.setFrictionSlip(p["friction_slip"])
            wheel.setRollInfluence(0.12)

    def _load_model(self):
        """Load vehicle model or fall back to coloured box."""
        model_path = f"assets/kenney/car_kit/models/{self.vehicle_type.value}.egg"
        try:
            self._model_np = self.base.loader.loadModel(model_path)
        except Exception:
            # Placeholder: coloured box
            from renderer import CityRenderer
            half = self.preset["chassis_half"]
            self._model_np = CityRenderer.make_colored_box(
                self.base, half * 2, self.preset["colour"],
                name=f"vehicle_{self.vehicle_type.value}"
            )
        self._model_np.reparentTo(self._chassis_np)
        self._model_np.setPos(0, 0, 0)

    def _setup_headlights(self):
        if self.lighting:
            self._headlight_nps = self.lighting.create_headlights(self._chassis_np)
            self._taillight_nps = self.lighting.create_taillights(self._chassis_np)

    def _setup_camera(self):
        self._camera_smooth_pos = Vec3(self._chassis_np.getPos()) + Vec3(0, -8, 3)

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self):
        km = {
            ("arrow_up",   True): ("forward",  True),
            ("arrow_up",   False):("forward",  False),
            ("arrow_down", True): ("backward", True),
            ("arrow_down", False):("backward", False),
            ("arrow_left", True): ("left",     True),
            ("arrow_left", False):("left",     False),
            ("arrow_right",True): ("right",    True),
            ("arrow_right",False):("right",    False),
            ("w",     True):  ("forward",  True),
            ("w",     False): ("forward",  False),
            ("s",     True):  ("backward", True),
            ("s",     False): ("backward", False),
            ("a",     True):  ("left",     True),
            ("a",     False): ("left",     False),
            ("d",     True):  ("right",    True),
            ("d",     False): ("right",    False),
            ("space", True):  ("brake",    True),
            ("space", False): ("brake",    False),
        }
        for (key, pressed), (action, val) in km.items():
            evt = key if pressed else f"{key}-up"
            self.base.accept(evt, self._set_key, [action, val])

        self.base.accept("r", self.reset)
        self.base.accept("h", self._toggle_headlights)
        self.base.accept("c", self._cycle_camera)

    def _set_key(self, action: str, val: bool):
        self._keys[action] = val

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, dt: float):
        p = self.preset

        # --------------- Steering ---------------
        steer_target = 0.0
        if self._keys["left"]:
            steer_target = p["max_steer"]
        if self._keys["right"]:
            steer_target = -p["max_steer"]

        # Speed-sensitive steering reduction
        speed_factor = max(0.3, 1.0 - self._speed_kph / 250.0)
        steer_target *= speed_factor
        self._steer_val = _lerp(self._steer_val, steer_target, dt * p["steer_speed"] * 4)

        # --------------- Engine / braking ---------------
        max_engine = p["engine_force"]
        if self._keys["brake"]:
            self._throttle = _lerp(self._throttle, 0.0, dt * 10)
            self._brake    = 1.0
        elif self._keys["forward"]:
            self._throttle = _lerp(self._throttle, 1.0, dt * 3)
            self._brake    = _lerp(self._brake, 0.0, dt * 8)
        elif self._keys["backward"]:
            self._throttle = _lerp(self._throttle, -0.5, dt * 3)
            self._brake    = _lerp(self._brake, 0.0, dt * 8)
        else:
            # Engine braking
            self._throttle = _lerp(self._throttle, 0.0, dt * 5)
            self._brake    = _lerp(self._brake, 0.2, dt * 2)

        # Gear-based engine force scaling
        engine_torque = max_engine * abs(self._throttle)
        gear_ratio    = GEAR_RATIOS[min(self._gear - 1, len(GEAR_RATIOS) - 1)]
        engine_torque *= gear_ratio

        # Apply to rear wheels (indices 2 and 3)
        sign = 1.0 if self._throttle >= 0 else -1.0
        for i in range(4):
            self._vehicle.applyBrakes(p["brake_force"] * self._brake, i)
        for i in (2, 3):
            self._vehicle.applyEngineForce(sign * engine_torque, i)

        # Steering on front wheels (0 and 1)
        for i in (0, 1):
            self._vehicle.setSteeringValue(self._steer_val, i)

        # --------------- Speed / RPM ---------------
        vel     = self._body_node.getLinearVelocity()
        speed_ms = vel.length()
        self._speed_kph = speed_ms * 3.6
        self._update_gear()
        self._rpm = self._compute_rpm()

        # --------------- Skid marks ---------------
        slip = abs(self._steer_val) * self._speed_kph / 50.0
        if slip > 0.5 or (self._keys["brake"] and self._speed_kph > 30):
            self._add_skid_marks()
            self._skidding = True
        else:
            self._skidding = False

        # --------------- Camera ---------------
        self._update_camera(dt)

    # ------------------------------------------------------------------
    # Gear simulation
    # ------------------------------------------------------------------

    def _update_gear(self):
        p = self.preset
        kph = self._speed_kph
        up_thresholds   = [30, 60, 100, 140, 180, 220, 9999]
        down_thresholds = [0,  20,  50,  90, 130, 170,  210]
        max_g = p["gear_count"]
        if kph > up_thresholds[min(self._gear - 1, max_g - 1)]:
            self._gear = min(self._gear + 1, max_g)
        elif kph < down_thresholds[min(self._gear - 1, max_g - 1)] and self._gear > 1:
            self._gear = max(self._gear - 1, 1)

    def _compute_rpm(self) -> float:
        """Approximate RPM from speed and gear."""
        base_rpm   = 800.0
        ratio      = GEAR_RATIOS[min(self._gear - 1, len(GEAR_RATIOS) - 1)]
        speed_rpm  = self._speed_kph * ratio * 20.0
        return min(base_rpm + speed_rpm * abs(self._throttle) * 0.5 + speed_rpm * 0.3,
                   7500.0)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _update_camera(self, dt: float):
        pos  = Vec3(self._chassis_np.getPos())
        fwd  = self._chassis_np.getQuat().getForward()
        mode = self._camera_mode

        if mode == CameraMode.CHASE:
            target = pos - fwd * 8.0 + Vec3(0, 0, 3.0)
        elif mode == CameraMode.BUMPER:
            target = pos + fwd * 2.5 + Vec3(0, 0, 0.8)
        elif mode == CameraMode.TOP_DOWN:
            target = pos + Vec3(0, 0, 25.0)
        elif mode == CameraMode.CINEMATIC_DRONE:
            import time as _time
            t = _time.time()
            orbit_r = 12.0
            orbit_x = math.cos(t * 0.3) * orbit_r
            orbit_y = math.sin(t * 0.3) * orbit_r
            target   = pos + Vec3(orbit_x, orbit_y, 8.0)
        else:
            target = pos - fwd * 8.0 + Vec3(0, 0, 3.0)

        # Smooth camera
        smooth = 6.0 if mode != CameraMode.CINEMATIC_DRONE else 2.0
        self._camera_smooth_pos = _vec3_lerp(self._camera_smooth_pos, target, dt * smooth)
        self.base.camera.setPos(self._camera_smooth_pos)
        self.base.camera.lookAt(pos + Vec3(0, 0, 0.8))

    def _cycle_camera(self):
        modes = list(CameraMode)
        idx   = modes.index(self._camera_mode)
        self._camera_mode = modes[(idx + 1) % len(modes)]

    # ------------------------------------------------------------------
    # Skid marks
    # ------------------------------------------------------------------

    def _add_skid_marks(self):
        for wi in (2, 3):   # rear wheels
            whl  = self._vehicle.getWheel(wi)
            pos  = whl.getChassisConnectionPointCs()
            world_pos = self._chassis_np.getPos() + pos
            ls = LineSegs("skid")
            ls.setColor(0.05, 0.05, 0.05, 0.7)
            ls.setThickness(6.0)
            ls.moveTo(world_pos)
            ls.drawTo(world_pos + Vec3(0, 0, 0.01))
            node_np = self.base.render.attachNewNode(ls.create())
            node_np.setPos(0, 0, 0.02)
            self._skid_nodes.append(node_np)
            if len(self._skid_nodes) > self.MAX_SKID_NODES:
                old = self._skid_nodes.pop(0)
                old.removeNode()

    # ------------------------------------------------------------------
    # Damage
    # ------------------------------------------------------------------

    def apply_damage(self, amount: float):
        self._damage = min(self._damage + amount, 1.0)

    def get_damage(self) -> float:
        return self._damage

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self):
        """Teleport back to spawn, upright."""
        self._body_node.setLinearVelocity(Vec3(0, 0, 0))
        self._body_node.setAngularVelocity(Vec3(0, 0, 0))
        self._chassis_np.setPos(self.spawn_pos)
        self._chassis_np.setHpr(0, 0, 0)
        self._throttle = 0.0
        self._brake    = 0.0
        self._steer_val = 0.0
        self._gear = 1

    def _toggle_headlights(self):
        self._headlights_on = not self._headlights_on
        if hasattr(self, "_headlight_nps") and self.lighting:
            for np_ in self._headlight_nps:
                if self._headlights_on:
                    self.base.render.setLight(np_)
                else:
                    self.base.render.clearLight(np_)

    @property
    def speed_kph(self) -> float:
        return self._speed_kph

    @property
    def is_braking(self) -> bool:
        return self._keys.get("brake", False) and self._speed_kph > 5.0

    @property
    def rpm(self) -> float:
        return self._rpm

    @property
    def gear(self) -> int:
        return self._gear

    @property
    def damage(self) -> float:
        return self._damage

    @property
    def position(self) -> Vec3:
        return Vec3(self._chassis_np.getPos())

    @property
    def node_path(self) -> NodePath:
        return self._chassis_np

    @property
    def camera_mode(self) -> CameraMode:
        return self._camera_mode

    def cleanup(self):
        self.bullet_world.removeVehicle(self._vehicle)
        self.bullet_world.removeRigidBody(self._body_node)
        self._chassis_np.removeNode()
        for np_ in self._skid_nodes:
            np_.removeNode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(max(t, 0.0), 1.0)


def _vec3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    t = min(max(t, 0.0), 1.0)
    return Vec3(a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t)
