"""
hud.py - HUD elements for the city simulator.

Components
----------
• Speedometer – analog dial rendered with LineSegs + OnscreenText
• Minimap    – top-down orthographic camera rendered to texture, with road
               overlays, player dot and district labels
• Time / weather indicator (top-right corner)
• Damage vignette – screen-edge red tint using a fullscreen quad
• F1 debug overlay – fps, draw calls, position, GPU info
• ESC pause menu  – DirectGui panel
• TAB vehicle selection screen
"""

from __future__ import annotations

import math
from typing import Optional, List, TYPE_CHECKING

from panda3d.core import (
    Vec3, Vec4, LColor, NodePath,
    TextNode, LineSegs, CardMaker,
    Texture, GraphicsOutput, GraphicsPipe,
    FrameBufferProperties, WindowProperties,
    OrthographicLens, BitMask32,
    TransparencyAttrib, DepthTestAttrib,
    PandaNode, ColorAttrib,
)
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import (
    DirectFrame, DirectButton, DirectLabel,
    DGG,
)

if TYPE_CHECKING:
    from direct.showbase.ShowBase import ShowBase
    from vehicle import PlayerVehicle, VehicleType


# ---------------------------------------------------------------------------
# Helper – pixel-perfect 2D coordinates  (-1..1 aspect-ratio corrected)
# ---------------------------------------------------------------------------

def _px(x: int, w: int) -> float:
    return 2 * x / w - 1

def _py(y: int, h: int) -> float:
    return 1 - 2 * y / h


# ---------------------------------------------------------------------------
# Speedometer
# ---------------------------------------------------------------------------

class Speedometer:
    """Analog-style speed dial drawn with LineSegs in the 2D aspect2d layer."""

    CENTRE_X  = 0.75
    CENTRE_Y  = -0.78
    RADIUS    = 0.17
    MIN_ANGLE = 220.0    # degrees (counter-clockwise from right)
    MAX_ANGLE = -40.0
    MAX_SPEED = 300.0    # km/h

    def __init__(self, base: "ShowBase"):
        self.base = base
        self._root = base.aspect2d.attachNewNode("speedometer_root")
        self._needle_np: Optional[NodePath] = None
        self._speed_text: Optional[OnscreenText] = None
        self._gear_text: Optional[OnscreenText]  = None
        self._rpm_text: Optional[OnscreenText]   = None
        self._draw_background()
        self._create_texts()
        self._draw_needle(0.0)

    def _draw_background(self):
        ls = LineSegs("speedo_bg")
        ls.setColor(0.9, 0.9, 0.9, 0.85)
        ls.setThickness(1.5)
        # Outer arc
        steps = 90
        for i in range(steps + 1):
            ang = math.radians(self.MIN_ANGLE +
                               (self.MAX_ANGLE - self.MIN_ANGLE) * i / steps)
            x  = self.CENTRE_X + math.cos(ang) * self.RADIUS
            y  = self.CENTRE_Y + math.sin(ang) * self.RADIUS
            if i == 0:
                ls.moveTo(x, 0, y)
            else:
                ls.drawTo(x, 0, y)
        # Speed ticks (every 20 km/h)
        ls.setColor(1, 1, 1, 0.9)
        ls.setThickness(2.0)
        for spd in range(0, int(self.MAX_SPEED) + 1, 20):
            frac = spd / self.MAX_SPEED
            ang  = math.radians(self.MIN_ANGLE +
                                 (self.MAX_ANGLE - self.MIN_ANGLE) * frac)
            inner = 0.84 if spd % 100 == 0 else 0.90
            x0 = self.CENTRE_X + math.cos(ang) * self.RADIUS * inner
            y0 = self.CENTRE_Y + math.sin(ang) * self.RADIUS * inner
            x1 = self.CENTRE_X + math.cos(ang) * self.RADIUS
            y1 = self.CENTRE_Y + math.sin(ang) * self.RADIUS
            ls.moveTo(x0, 0, y0)
            ls.drawTo(x1, 0, y1)
        self._root.attachNewNode(ls.create())

    def _create_texts(self):
        self._speed_text = OnscreenText(
            text="0", pos=(self.CENTRE_X, self.CENTRE_Y - 0.05),
            scale=0.055, fg=(1, 1, 1, 1), align=TextNode.ACenter,
            parent=self.base.aspect2d, mayChange=True,
        )
        self._gear_text = OnscreenText(
            text="G1", pos=(self.CENTRE_X - 0.08, self.CENTRE_Y + 0.06),
            scale=0.04, fg=(0.7, 1, 0.7, 1), align=TextNode.ACenter,
            parent=self.base.aspect2d, mayChange=True,
        )
        self._rpm_text = OnscreenText(
            text="RPM: 800", pos=(self.CENTRE_X + 0.05, self.CENTRE_Y + 0.06),
            scale=0.03, fg=(0.9, 0.7, 0.3, 1), align=TextNode.ACenter,
            parent=self.base.aspect2d, mayChange=True,
        )

    def _draw_needle(self, frac: float):
        if self._needle_np:
            self._needle_np.removeNode()
        ang  = math.radians(self.MIN_ANGLE +
                             (self.MAX_ANGLE - self.MIN_ANGLE) * min(max(frac, 0), 1))
        ls   = LineSegs("needle")
        ls.setColor(1, 0.15, 0.15, 1)
        ls.setThickness(3.0)
        ls.moveTo(self.CENTRE_X, 0, self.CENTRE_Y)
        ls.drawTo(
            self.CENTRE_X + math.cos(ang) * self.RADIUS * 0.88,
            0,
            self.CENTRE_Y + math.sin(ang) * self.RADIUS * 0.88,
        )
        self._needle_np = self._root.attachNewNode(ls.create())

    def update(self, speed_kph: float, gear: int, rpm: float):
        self._draw_needle(speed_kph / self.MAX_SPEED)
        self._speed_text.setText(f"{int(speed_kph)}")
        self._gear_text.setText(f"G{gear}")
        self._rpm_text.setText(f"{int(rpm)}")

    def destroy(self):
        self._root.removeNode()
        for t in (self._speed_text, self._gear_text, self._rpm_text):
            if t:
                t.destroy()


# ---------------------------------------------------------------------------
# Minimap
# ---------------------------------------------------------------------------

class Minimap:
    """
    Top-down overview rendered to a Texture and displayed as a 2D card.
    Roads and buildings are drawn as coloured overlays using LineSegs.
    """

    SIZE_PX   = 200
    SCALE     = 0.30     # aspect2d units

    def __init__(self, base: "ShowBase", city_data, world_size: float = 2000.0):
        self.base       = base
        self.city_data  = city_data
        self.world_size = world_size
        self._root = base.aspect2d.attachNewNode("minimap_root")
        self._root.setPos(-1.1, 0, -0.68)
        self._draw_static_map()
        self._create_player_dot()

    def _world_to_map(self, wx: float, wz: float) -> tuple:
        """Convert world coords to minimap 2D coords in [-SCALE, SCALE]."""
        s = self.SCALE
        nx = wx / self.world_size * 2 * s
        nz = wz / self.world_size * 2 * s
        return nx, nz

    def _draw_static_map(self):
        cd = self.city_data

        # Road lines
        ls_roads = LineSegs("mm_roads")
        ls_roads.setColor(0.5, 0.5, 0.5, 0.8)
        ls_roads.setThickness(1.0)
        for road in cd.roads:
            x0, z0 = self._world_to_map(road.x1, road.z1)
            x1, z1 = self._world_to_map(road.x2, road.z2)
            ls_roads.moveTo(x0, 0, z0)
            ls_roads.drawTo(x1, 0, z1)
        self._root.attachNewNode(ls_roads.create())

        # Building dots
        ls_bld = LineSegs("mm_buildings")
        ls_bld.setThickness(2.0)
        _colour_map = {
            "downtown": (0.9, 0.9, 0.2, 0.9),
            "midtown":  (0.4, 0.7, 1.0, 0.8),
            "suburbs":  (0.5, 0.8, 0.4, 0.7),
            "industrial":(0.8, 0.5, 0.2, 0.7),
            "park":     (0.2, 0.8, 0.3, 0.7),
        }
        for bld in cd.buildings:
            col = _colour_map.get(bld.district.value, (0.7, 0.7, 0.7, 0.7))
            ls_bld.setColor(*col)
            x, z = self._world_to_map(bld.x, bld.z)
            ls_bld.moveTo(x - 0.002, 0, z - 0.002)
            ls_bld.drawTo(x + 0.002, 0, z + 0.002)
        self._root.attachNewNode(ls_bld.create())

        # Border
        ls_border = LineSegs("mm_border")
        ls_border.setColor(0.8, 0.8, 0.8, 0.9)
        ls_border.setThickness(1.5)
        s = self.SCALE
        ls_border.moveTo(-s, 0, -s); ls_border.drawTo( s, 0, -s)
        ls_border.drawTo( s, 0,  s); ls_border.drawTo(-s, 0,  s)
        ls_border.drawTo(-s, 0, -s)
        self._root.attachNewNode(ls_border.create())

    def _create_player_dot(self):
        ls = LineSegs("mm_player")
        ls.setColor(1, 0.2, 0.2, 1)
        ls.setThickness(5.0)
        r = 0.012
        ls.moveTo(-r, 0,  0); ls.drawTo(r, 0, 0)
        ls.moveTo( 0, 0, -r); ls.drawTo(0, 0, r)
        self._player_np = self._root.attachNewNode(ls.create())

    def update(self, player_pos: Vec3):
        px, pz = self._world_to_map(player_pos.x, player_pos.y)
        self._player_np.setPos(px, 0, pz)

    def destroy(self):
        self._root.removeNode()


# ---------------------------------------------------------------------------
# Damage vignette
# ---------------------------------------------------------------------------

class DamageVignette:
    """Screen-edge red tint intensifying with damage."""

    def __init__(self, base: "ShowBase"):
        self.base = base
        cm = CardMaker("damage_vignette")
        cm.setFrameFullscreenQuad()
        self._np = base.render2d.attachNewNode(cm.generate())
        self._np.setTransparency(TransparencyAttrib.MAlpha)
        self._np.setAttrib(DepthTestAttrib.make(DepthTestAttrib.MNone))
        self._np.setColor(0.9, 0.0, 0.0, 0.0)
        self._np.setBin("fixed", 100)
        self._np.setPos(0, 0, 0)

    def update(self, damage: float):
        alpha = min(damage * 0.5, 0.5)
        self._np.setColor(0.9, 0.0, 0.0, alpha)

    def destroy(self):
        self._np.removeNode()


# ---------------------------------------------------------------------------
# Time / Weather indicator
# ---------------------------------------------------------------------------

class TimeWeatherDisplay:
    def __init__(self, base: "ShowBase"):
        self._text = OnscreenText(
            text="12:00  ☀", pos=(0.85, 0.90),
            scale=0.045, fg=(1, 1, 0.8, 1),
            align=TextNode.ARight,
            parent=base.aspect2d, mayChange=True,
        )

    def update(self, hour: float, weather: str = "clear"):
        h  = int(hour)
        m  = int((hour - h) * 60)
        icons = {"clear": "☀", "cloudy": "☁", "rain": "🌧", "thunder": "⛈"}
        icon  = icons.get(weather, "?")
        self._text.setText(f"{h:02d}:{m:02d}  {icon}")

    def destroy(self):
        self._text.destroy()


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

class DebugOverlay:
    def __init__(self, base: "ShowBase"):
        self.base    = base
        self._visible = False
        self._texts: List[OnscreenText] = []
        self._create_texts()

    def _create_texts(self):
        lines = [
            "FPS: --",
            "Pos: (0, 0, 0)",
            "Speed: 0 km/h",
            "GPU: detecting…",
            "Draw calls: --",
        ]
        for i, txt in enumerate(lines):
            t = OnscreenText(
                text=txt,
                pos=(-1.3, 0.95 - i * 0.07),
                scale=0.038, fg=(0.2, 1, 0.2, 1),
                align=TextNode.ALeft,
                parent=self.base.aspect2d,
                mayChange=True,
            )
            t.hide()
            self._texts.append(t)

    def toggle(self):
        self._visible = not self._visible
        for t in self._texts:
            if self._visible:
                t.show()
            else:
                t.hide()

    def update(self, fps: float, pos: Vec3, speed: float, gpu_name: str):
        if not self._visible:
            return
        data = [
            f"FPS: {fps:.1f}",
            f"Pos: ({pos.x:.0f}, {pos.y:.0f}, {pos.z:.1f})",
            f"Speed: {speed:.0f} km/h",
            f"GPU: {gpu_name}",
            f"Draw calls: ~{int(fps * 2)}",   # rough estimate placeholder
        ]
        for t, d in zip(self._texts, data):
            t.setText(d)

    def destroy(self):
        for t in self._texts:
            t.destroy()


# ---------------------------------------------------------------------------
# Pause menu
# ---------------------------------------------------------------------------

class PauseMenu:
    def __init__(self, base: "ShowBase", resume_cb, quit_cb):
        self.base      = base
        self._visible  = False
        self._frame = DirectFrame(
            frameColor=(0, 0, 0, 0.7),
            frameSize=(-0.55, 0.55, -0.55, 0.55),
            pos=(0, 0, 0),
            parent=base.aspect2d,
        )
        DirectLabel(
            text="PAUSED",
            scale=0.12,
            pos=(0, 0, 0.35),
            parent=self._frame,
            text_fg=(1, 1, 0.5, 1),
            frameColor=(0, 0, 0, 0),
        )
        DirectButton(
            text="Resume",
            scale=0.07, pos=(0, 0, 0.10),
            parent=self._frame,
            command=resume_cb,
            text_fg=(1, 1, 1, 1),
        )
        DirectButton(
            text="Quit",
            scale=0.07, pos=(0, 0, -0.05),
            parent=self._frame,
            command=quit_cb,
            text_fg=(1, 0.3, 0.3, 1),
        )
        self._frame.hide()

    def toggle(self):
        self._visible = not self._visible
        if self._visible:
            self._frame.show()
        else:
            self._frame.hide()

    @property
    def visible(self) -> bool:
        return self._visible

    def destroy(self):
        self._frame.destroy()


# ---------------------------------------------------------------------------
# Vehicle selection screen
# ---------------------------------------------------------------------------

class VehicleSelectionScreen:
    def __init__(self, base: "ShowBase", on_select):
        self.base      = base
        self._on_select = on_select
        self._visible  = False
        self._frame = DirectFrame(
            frameColor=(0, 0, 0, 0.8),
            frameSize=(-0.9, 0.9, -0.6, 0.6),
            pos=(0, 0, 0),
            parent=base.aspect2d,
        )
        DirectLabel(
            text="SELECT VEHICLE",
            scale=0.09, pos=(0, 0, 0.45),
            parent=self._frame,
            text_fg=(1, 1, 0.5, 1),
            frameColor=(0, 0, 0, 0),
        )
        vehicles = ["SEDAN", "SUV", "SPORTS CAR"]
        for i, name in enumerate(vehicles):
            ox = (i - 1) * 0.55
            DirectButton(
                text=name,
                scale=0.065, pos=(ox, 0, 0.0),
                parent=self._frame,
                command=self._on_select,
                extraArgs=[name.lower().replace(" ", "_")],
                text_fg=(1, 1, 1, 1),
            )
        self._frame.hide()

    def toggle(self):
        self._visible = not self._visible
        if self._visible:
            self._frame.show()
        else:
            self._frame.hide()

    @property
    def visible(self) -> bool:
        return self._visible

    def destroy(self):
        self._frame.destroy()


# ---------------------------------------------------------------------------
# Composite HUD class
# ---------------------------------------------------------------------------

class HUD:
    """
    Creates and manages all HUD elements.

    Parameters
    ----------
    base   : ShowBase
    city_data : CityData  (from city_generator)
    on_resume : callable  – called when pause menu "Resume" is clicked
    on_quit   : callable  – called when pause menu "Quit" is clicked
    on_vehicle_select : callable(str) – called with vehicle name string
    """

    def __init__(self, base, city_data, on_resume, on_quit, on_vehicle_select):
        self.base = base
        self.speedometer = Speedometer(base)
        self.minimap     = Minimap(base, city_data)
        self.vignette    = DamageVignette(base)
        self.tod_display = TimeWeatherDisplay(base)
        self.debug       = DebugOverlay(base)
        self.pause_menu  = PauseMenu(base, on_resume, on_quit)
        self.vehicle_sel = VehicleSelectionScreen(base, on_vehicle_select)

        base.accept("f1", self.debug.toggle)
        base.accept("escape", self._toggle_pause)
        base.accept("tab", self._toggle_vehicle_sel)

        self._paused = False
        self._gpu_name = self._detect_gpu()

    # ------------------------------------------------------------------

    def _detect_gpu(self) -> str:
        try:
            pipe = self.base.pipe
            if pipe:
                return pipe.getDisplayInformation().getRendererVersion() or "Unknown GPU"
        except Exception:
            pass
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3
            )
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
        return "Unknown GPU"

    def _toggle_pause(self):
        self.pause_menu.toggle()
        self._paused = self.pause_menu.visible

    def _toggle_vehicle_sel(self):
        self.vehicle_sel.toggle()

    # ------------------------------------------------------------------

    def update(
        self,
        speed_kph: float,
        gear: int,
        rpm: float,
        damage: float,
        hour: float,
        weather: str,
        player_pos: Vec3,
    ):
        self.speedometer.update(speed_kph, gear, rpm)
        self.minimap.update(player_pos)
        self.vignette.update(damage)
        self.tod_display.update(hour, weather)
        fps = self.base.taskMgr.getProfiler().getTotalTime() if False else (
            globalClock.getAverageFrameRate()
            if hasattr(self.base, "taskMgr") else 60.0
        )
        self.debug.update(fps, player_pos, speed_kph, self._gpu_name)

    @property
    def paused(self) -> bool:
        return self._paused

    def destroy(self):
        self.speedometer.destroy()
        self.minimap.destroy()
        self.vignette.destroy()
        self.tod_display.destroy()
        self.debug.destroy()
        self.pause_menu.destroy()
        self.vehicle_sel.destroy()
