"""
main.py - Entry point for the GPU-accelerated City Driving Simulator.

Initialises all subsystems and runs the Panda3D main loop.

Usage
-----
    python main.py
    python main.py --quality Ultra
    python main.py --hour 18.5
"""

from __future__ import annotations

import sys
import os
import argparse
import math
import subprocess
from pathlib import Path
from typing import Optional, List

# ---------------------------------------------------------------------------
# Panda3D imports
# ---------------------------------------------------------------------------
from panda3d.core import (
    loadPrcFileData, Vec3, Vec4, LColor, NodePath,
    AmbientLight, DirectionalLight,
    BitMask32, GeomNode,
    CollisionTraverser, CollisionHandlerEvent,
    TransformState,
    WindowProperties,
    TextNode,
)
from panda3d.bullet import BulletWorld, BulletDebugNode, BulletPlaneShape, BulletRigidBodyNode
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText

# ---------------------------------------------------------------------------
# Argument parsing (before window opens)
# ---------------------------------------------------------------------------

DEFAULT_START_HOUR: float = 12.0   # noon


def _parse_args():
    p = argparse.ArgumentParser(description="City Driving Simulator")
    p.add_argument("--quality", default=None,
                   choices=["Low", "Medium", "High", "Ultra"])
    p.add_argument("--hour",    type=float, default=None,
                   help=f"Starting hour 0-24 (default: {DEFAULT_START_HOUR}, noon)")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--debug-physics", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pre-window Panda3D config
# ---------------------------------------------------------------------------

def _configure_panda(config: dict, args):
    gfx = config.get("graphics", {})
    res = gfx.get("resolution", [1920, 1080])
    loadPrcFileData("", f"win-size {res[0]} {res[1]}")
    loadPrcFileData("", f"fullscreen {'#t' if gfx.get('fullscreen') else '#f'}")
    loadPrcFileData("", f"sync-video {'#t' if gfx.get('vsync', True) else '#f'}")
    loadPrcFileData("", "show-frame-rate-meter #f")
    loadPrcFileData("", "gl-version 4 3")
    loadPrcFileData("", "hardware-animated-vertices #t")
    loadPrcFileData("", "basic-shaders-only #f")
    if args.no_audio:
        loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "window-title City Simulator")


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu() -> str:
    # Try nvidia-smi first
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    # Fall back to Panda3D pipe info (available after ShowBase init)
    return "Unknown GPU (nvidia-smi unavailable)"


# ---------------------------------------------------------------------------
# Ground plane
# ---------------------------------------------------------------------------

def _create_ground(base: ShowBase, bullet_world: BulletWorld,
                   width: float, height: float):
    shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
    body  = BulletRigidBodyNode("ground")
    body.addShape(shape)
    np_ = base.render.attachNewNode(body)
    np_.setPos(0, 0, 0)
    bullet_world.attachRigidBody(body)

    # Visual ground mesh (large flat quad)
    from panda3d.core import CardMaker
    cm = CardMaker("ground_visual")
    hw, hh = width / 2, height / 2
    cm.setFrame(-hw, hw, -hh, hh)
    gnd = base.render.attachNewNode(cm.generate())
    gnd.setP(-90)
    gnd.setPos(0, 0, -0.01)
    gnd.setColor(0.22, 0.22, 0.22, 1)   # asphalt grey
    return np_


# ---------------------------------------------------------------------------
# City geometry builder
# ---------------------------------------------------------------------------

def _build_city_geometry(base: ShowBase, bullet_world: BulletWorld,
                          city_data, renderer):
    """
    Instantiate city buildings and roads from CityData.
    Uses asset models when available, falls back to coloured boxes.
    """
    from panda3d.bullet import BulletBoxShape

    DISTRICT_COLOURS = {
        "downtown":   Vec4(0.3, 0.3, 0.6, 1),
        "midtown":    Vec4(0.4, 0.5, 0.7, 1),
        "suburbs":    Vec4(0.6, 0.7, 0.4, 1),
        "industrial": Vec4(0.6, 0.5, 0.3, 1),
        "park":       Vec4(0.3, 0.6, 0.3, 1),
    }

    building_root = base.render.attachNewNode("buildings")

    for bld in city_data.buildings:
        colour = DISTRICT_COLOURS.get(bld.district.value, Vec4(0.5, 0.5, 0.5, 1))

        # Attempt to load model
        model_name = bld.model_type.lower().replace(" ", "_")
        model_path = f"assets/kenney/city_commercial/Models/OBJ/{model_name}.egg"
        loaded = False
        if Path(model_path).exists():
            try:
                m = base.loader.loadModel(model_path)
                m.reparentTo(building_root)
                m.setPos(bld.x, bld.z, 0)
                m.setH(bld.rotation)
                loaded = True
            except Exception:
                pass

        if not loaded:
            # Placeholder box
            from renderer import CityRenderer
            box = CityRenderer.make_colored_box(
                base, Vec3(bld.width, bld.depth, bld.height), colour,
                name=f"bld_{model_name}"
            )
            box.reparentTo(building_root)
            box.setPos(bld.x, bld.z, bld.height / 2)
            box.setH(bld.rotation)

            # Bullet collision box
            shape = BulletBoxShape(Vec3(bld.width / 2, bld.depth / 2, bld.height / 2))
            rb    = BulletRigidBodyNode(f"bld_{id(bld)}")
            rb.addShape(shape)
            rb.setMass(0)
            rb_np = base.render.attachNewNode(rb)
            rb_np.setPos(bld.x, bld.z, bld.height / 2)
            bullet_world.attachRigidBody(rb)

    # Roads – flat quads
    road_root = base.render.attachNewNode("roads")
    from panda3d.core import CardMaker
    for road in city_data.roads:
        length = math.hypot(road.x2 - road.x1, road.z2 - road.z1)
        cm = CardMaker("road_seg")
        cm.setFrame(-road.width / 2, road.width / 2, 0, length)
        rn = road_root.attachNewNode(cm.generate())
        rn.setP(-90)
        cx = (road.x1 + road.x2) / 2
        cz = (road.z1 + road.z2) / 2
        rn.setPos(cx, cz, 0.005)
        angle = math.degrees(math.atan2(road.x2 - road.x1, road.z2 - road.z1))
        rn.setH(angle)
        colour_val = 0.18 if road.road_type == "arterial" else 0.22
        rn.setColor(colour_val, colour_val, colour_val, 1)

    # Street lights at intersections
    light_positions = []
    for inter in city_data.intersections:
        if inter.has_traffic_light:
            light_positions.append(Vec3(inter.x, inter.z, 6.0))

    return building_root, road_root, light_positions


# ---------------------------------------------------------------------------
# Application class
# ---------------------------------------------------------------------------

class CitySimulator(ShowBase):
    """Main game application."""

    def __init__(self, config: dict, args):
        super().__init__()
        self.sim_config = config
        self.args   = args

        if args.quality:
            config["quality_preset"] = args.quality

        gfx_cfg = config.get("graphics", {})
        self.camLens.setFov(gfx_cfg.get("fov", 75))
        self.camLens.setNearFar(0.5, 3000.0)

        # Disable default mouse camera control
        self.disableMouse()

        self._target_fps   = gfx_cfg.get("target_fps", 60)
        self._show_fps     = config.get("debug", {}).get("show_fps", False)
        self._wireframe    = config.get("debug", {}).get("show_wireframe", False)
        self._paused       = False
        self._gpu_name     = _detect_gpu()

        print(f"[main] GPU detected: {self._gpu_name}")
        print(f"[main] Quality preset: {config.get('quality_preset', 'High')}")

        self._init_physics()
        self._init_renderer()
        self._init_city()
        self._init_lighting()
        self._init_player_vehicle()
        self._init_npc_traffic()
        self._init_pedestrians()
        self._init_weather()
        self._init_audio()
        self._init_hud()
        self._bind_global_keys()
        self._register_tasks()

        if self._show_fps:
            self.setFrameRateMeter(True)
        if self._wireframe:
            self.toggleWireframe()

        start_hour = args.hour if args.hour is not None else DEFAULT_START_HOUR
        self._lighting.set_hour(start_hour)

        print("[main] City Simulator started successfully.")
        print(f"[main] City: {len(self._city_data.buildings)} buildings, "
              f"{len(self._city_data.roads)} road segments")

    # ------------------------------------------------------------------
    # Subsystem initialisation
    # ------------------------------------------------------------------

    def _init_physics(self):
        self._bullet_world = BulletWorld()
        self._bullet_world.setGravity(Vec3(0, 0, -9.81))

        if self.args.debug_physics:
            debug_node = BulletDebugNode("bullet_debug")
            debug_node.showWireframe(True)
            debug_np = self.render.attachNewNode(debug_node)
            debug_np.show()
            self._bullet_world.setDebugNode(debug_node)

    def _init_renderer(self):
        from renderer import CityRenderer
        self._renderer = CityRenderer(self, self.sim_config)

    def _init_city(self):
        from city_generator import CityGenerator
        gen = CityGenerator(self.sim_config)
        self._city_data = gen.generate()

        # Ground plane
        _create_ground(self, self._bullet_world,
                        self._city_data.city_width,
                        self._city_data.city_height)

        # City geometry
        b_root, r_root, light_pos = _build_city_geometry(
            self, self._bullet_world, self._city_data, self._renderer
        )
        self._building_root   = b_root
        self._road_root       = r_root
        self._street_light_positions = light_pos

        # Sidewalk waypoints for pedestrians
        from city_generator import CityGenerator
        _gen = CityGenerator(self.sim_config)
        self._sidewalk_waypoints = _gen.get_sidewalk_waypoints(self._city_data)

    def _init_lighting(self):
        from lighting import LightingSystem
        self._lighting = LightingSystem(self, self._renderer, self.sim_config)
        self._lighting.register_street_lights(self._street_light_positions)

    def _init_player_vehicle(self):
        from vehicle import PlayerVehicle, VehicleType
        self._player = PlayerVehicle(
            self,
            self._bullet_world,
            self._lighting,
            vehicle_type=VehicleType.SEDAN,
            spawn_pos=Vec3(50, 50, 2.0),
        )

    def _init_npc_traffic(self):
        from npc_vehicle import spawn_traffic, get_traffic_light_system
        count = self.sim_config.get("gameplay", {}).get("npc_vehicles", 50)
        import random
        rng = random.Random(self.sim_config.get("city", {}).get("seed", 42))
        self._npc_vehicles = spawn_traffic(
            self, self._bullet_world, self._city_data.roads, count, rng
        )
        self._traffic_lights = get_traffic_light_system()
        print(f"[main] Spawned {len(self._npc_vehicles)} NPC vehicles.")

    def _init_pedestrians(self):
        from npc_pedestrian import spawn_pedestrians
        count = self.sim_config.get("gameplay", {}).get("npc_pedestrians", 200)
        import random
        rng = random.Random(self.sim_config.get("city", {}).get("seed", 42) + 1)
        self._pedestrians = spawn_pedestrians(
            self, self._bullet_world, self._sidewalk_waypoints, count, rng
        )
        print(f"[main] Spawned {len(self._pedestrians)} pedestrians.")

    def _init_weather(self):
        from weather import WeatherSystem
        self._weather = WeatherSystem(
            self, self._renderer, None, self.sim_config
        )

    def _init_audio(self):
        if self.args.no_audio:
            self._audio = None
            return
        try:
            from audio import AudioSystem
            self._audio = AudioSystem(self, self.sim_config)
            # Wire audio into weather
            self._weather.audio = self._audio
        except Exception as exc:
            print(f"[main] Audio init failed: {exc}")
            self._audio = None

    def _init_hud(self):
        from hud import HUD
        self._hud = HUD(
            self,
            self._city_data,
            on_resume=self._resume,
            on_quit=self.userExit,
            on_vehicle_select=self._on_vehicle_select,
        )

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_global_keys(self):
        self.accept("f2",      self.toggleWireframe)
        self.accept("f3",      self._toggle_collision_vis)
        self.accept("f4",      self._cycle_time_of_day)
        self.accept("f5",      self._toggle_pause)
        self.accept("page_up",   self._time_forward)
        self.accept("page_down", self._time_backward)

    def _toggle_collision_vis(self):
        debug_np = self.render.find("bullet_debug")
        if not debug_np.isEmpty():
            if debug_np.isHidden():
                debug_np.show()
            else:
                debug_np.hide()

    def _cycle_time_of_day(self):
        hour_steps = [0, 6, 8, 12, 17, 19, 22]
        cur = self._lighting.hour
        for h in hour_steps:
            if h > cur + 0.5:
                self._lighting.set_hour(float(h))
                print(f"[main] Time set to {h:02d}:00")
                return
        self._lighting.set_hour(0.0)

    def _time_forward(self):
        self._lighting.set_hour(self._lighting.hour + 1.0)

    def _time_backward(self):
        self._lighting.set_hour(self._lighting.hour - 1.0)

    def _toggle_pause(self):
        self._paused = not self._paused
        self._hud.pause_menu.toggle()

    def _resume(self):
        self._paused = False
        self._hud.pause_menu.toggle()

    def _on_vehicle_select(self, vehicle_name: str):
        from vehicle import VehicleType
        type_map = {
            "sedan":      VehicleType.SEDAN,
            "suv":        VehicleType.SUV,
            "sports_car": VehicleType.SPORTS_CAR,
        }
        vtype = type_map.get(vehicle_name, VehicleType.SEDAN)
        old_pos = self._player.position
        self._player.cleanup()
        from vehicle import PlayerVehicle
        self._player = PlayerVehicle(
            self, self._bullet_world, self._lighting, vtype, old_pos
        )
        self._hud.vehicle_sel.toggle()

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def _register_tasks(self):
        self.taskMgr.add(self._game_loop_task, "game_loop")
        self.taskMgr.add(self._physics_task,   "physics")

    def _physics_task(self, task):
        dt = globalClock.getDt()
        self._bullet_world.doPhysics(dt, 5, 1.0 / 180.0)
        return task.cont

    def _game_loop_task(self, task):
        if self._paused:
            return task.cont

        dt   = globalClock.getDt()
        hour = self._lighting.hour

        # Update traffic light system
        self._traffic_lights.update(dt)

        # Player vehicle
        self._player.update(dt)

        # NPC vehicles
        for npc in self._npc_vehicles:
            npc.update(dt, hour, self._city_data.intersections)

        # Pedestrians
        npc_positions = [npc.position for npc in self._npc_vehicles]
        npc_positions.append(self._player.position)
        crosswalk_pts = [(inter.x, inter.z) for inter in self._city_data.intersections
                         if inter.has_traffic_light]
        for ped in self._pedestrians:
            ped.update(dt, npc_positions, self._traffic_lights, crosswalk_pts)

        # Lighting
        self._lighting.update(dt)

        # Weather
        self._weather.update(dt, hour)

        # Audio
        if self._audio:
            self._audio.update_engine(
                self._player.rpm, self._player.speed_kph,
                braking=self._player.is_braking
            )

        # HUD
        self._hud.update(
            speed_kph=self._player.speed_kph,
            gear=self._player.gear,
            rpm=self._player.rpm,
            damage=self._player.damage,
            hour=hour,
            weather=self._weather.state_name,
            player_pos=self._player.position,
        )

        return task.cont

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        self._player.cleanup()
        for npc in self._npc_vehicles:
            npc.cleanup()
        for ped in self._pedestrians:
            ped.cleanup()
        self._lighting.cleanup()
        self._weather.cleanup()
        if self._audio:
            self._audio.cleanup()
        self._hud.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # Load config
    config_path = Path("config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print("[main] config.yaml not found – using defaults")
        config = {
            "quality_preset": "High",
            "city":     {"width_km": 2.0, "height_km": 2.0, "seed": 42},
            "graphics": {"resolution": [1920, 1080], "fullscreen": False,
                         "vsync": True, "fov": 75, "target_fps": 60},
            "rendering":{"shadows": True, "ssao": True, "bloom": True,
                         "dof": True, "motion_blur": True, "ssr": True,
                         "hdr": True, "volumetric_fog": True},
            "audio":    {"master_volume": 0.8, "engine_volume": 1.0,
                         "ambient_volume": 0.6, "music_volume": 0.4},
            "gameplay": {"npc_vehicles": 50, "npc_pedestrians": 200,
                         "traffic_density_multiplier": 1.0},
            "debug":    {"show_fps": False, "show_wireframe": False,
                         "show_collision": False},
        }

    _configure_panda(config, args)

    print("=" * 60)
    print("  City Driving Simulator")
    print("  Python", sys.version.split()[0])
    gpu = _detect_gpu()
    print(f"  GPU: {gpu}")
    print("=" * 60)

    app = CitySimulator(config, args)
    app.run()


if __name__ == "__main__":
    main()
