"""
city_generator.py - Procedural city layout generator.

Generates a 2 km × 2 km city grid containing multiple districts, roads, buildings,
intersections and traffic-light positions. Pure Python / NumPy – no Panda3D required.
"""

from __future__ import annotations

import enum
import math
import random
from typing import List, NamedTuple, Optional, Dict, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DistrictType(enum.Enum):
    DOWNTOWN   = "downtown"
    MIDTOWN    = "midtown"
    SUBURBS    = "suburbs"
    INDUSTRIAL = "industrial"
    PARK       = "park"


class Building(NamedTuple):
    x: float               # centre X (metres, East)
    z: float               # centre Z (metres, North)
    width: float           # footprint X
    depth: float           # footprint Z
    height: float          # building height (metres)
    district: DistrictType
    model_type: str        # e.g. "skyscraper", "office", "house", "warehouse"
    rotation: float        # Y-axis rotation in degrees


class RoadSegment(NamedTuple):
    x1: float
    z1: float
    x2: float
    z2: float
    width: float
    road_type: str         # "arterial", "local", "highway", "path"
    lanes: int


class Intersection(NamedTuple):
    x: float
    z: float
    has_traffic_light: bool
    district: DistrictType


class DistrictBoundary(NamedTuple):
    district: DistrictType
    min_x: float
    min_z: float
    max_x: float
    max_z: float


class CityData(NamedTuple):
    buildings: List[Building]
    roads: List[RoadSegment]
    intersections: List[Intersection]
    district_boundaries: List[DistrictBoundary]
    city_width: float
    city_height: float
    seed: int


# ---------------------------------------------------------------------------
# District configuration
# ---------------------------------------------------------------------------

DISTRICT_CONFIG: Dict[DistrictType, dict] = {
    DistrictType.DOWNTOWN: {
        "height_min": 60,   "height_max": 300,
        "street_width": 20, "block_size": 80,
        "density": 0.90,    "setback": 2,
        "model_types": ["skyscraper", "office_tower", "glass_tower"],
    },
    DistrictType.MIDTOWN: {
        "height_min": 20,   "height_max": 80,
        "street_width": 18, "block_size": 100,
        "density": 0.75,    "setback": 3,
        "model_types": ["office", "apartment_high", "mixed_use"],
    },
    DistrictType.SUBURBS: {
        "height_min": 5,    "height_max": 15,
        "street_width": 14, "block_size": 120,
        "density": 0.50,    "setback": 5,
        "model_types": ["house", "bungalow", "townhouse"],
    },
    DistrictType.INDUSTRIAL: {
        "height_min": 8,    "height_max": 25,
        "street_width": 22, "block_size": 150,
        "density": 0.60,    "setback": 4,
        "model_types": ["warehouse", "factory", "logistics"],
    },
    DistrictType.PARK: {
        "height_min": 2,    "height_max": 5,
        "street_width": 10, "block_size": 200,
        "density": 0.10,    "setback": 10,
        "model_types": ["pavilion", "fountain", "bench"],
    },
}


# ---------------------------------------------------------------------------
# Layout zones (fraction of city dimensions)
# ---------------------------------------------------------------------------

DISTRICT_ZONES: List[Tuple[DistrictType, float, float, float, float]] = [
    # (type,  x_min, x_max, z_min, z_max)  – all fractions 0–1
    (DistrictType.DOWNTOWN,   0.35, 0.65, 0.35, 0.65),
    (DistrictType.MIDTOWN,    0.20, 0.80, 0.20, 0.80),   # ring around downtown
    (DistrictType.PARK,       0.70, 0.95, 0.70, 0.95),
    (DistrictType.INDUSTRIAL, 0.00, 0.30, 0.65, 1.00),
    (DistrictType.SUBURBS,    0.00, 1.00, 0.00, 1.00),   # fills remainder
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CityGenerator:
    """
    Procedurally generates a city layout from a configuration dict.

    Config keys used:
        city.width_km  (float)
        city.height_km (float)
        city.seed      (int)
        districts.*    (per-district overrides)
    """

    def __init__(self, config: dict):
        city_cfg = config.get("city", {})
        self.width  = city_cfg.get("width_km",  2.0) * 1000   # metres
        self.height = city_cfg.get("height_km", 2.0) * 1000
        self.seed   = city_cfg.get("seed", 42)
        self.rng    = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

        # Allow per-district config overrides
        dist_cfg = config.get("districts", {})
        self._dist_cfg: Dict[DistrictType, dict] = {}
        for dt in DistrictType:
            base = dict(DISTRICT_CONFIG[dt])
            override = dist_cfg.get(dt.value, {})
            base.update(override)
            self._dist_cfg[dt] = base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> CityData:
        boundaries     = self._generate_district_boundaries()
        district_grid  = self._build_district_grid(boundaries)
        roads, inters  = self._generate_road_network(district_grid)
        buildings      = self._generate_buildings(roads, district_grid)

        return CityData(
            buildings=buildings,
            roads=roads,
            intersections=inters,
            district_boundaries=boundaries,
            city_width=self.width,
            city_height=self.height,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # District boundaries
    # ------------------------------------------------------------------

    def _generate_district_boundaries(self) -> List[DistrictBoundary]:
        w, h = self.width, self.height
        boundaries = []

        # Assign priorities so later zones do NOT overwrite earlier (more specific) ones
        seen: List[DistrictBoundary] = []

        for dt, fx0, fx1, fz0, fz1 in DISTRICT_ZONES:
            x0 = -w / 2 + fx0 * w
            x1 = -w / 2 + fx1 * w
            z0 = -h / 2 + fz0 * h
            z1 = -h / 2 + fz1 * h
            seen.append(DistrictBoundary(dt, x0, z0, x1, z1))

        # Build final non-overlapping list (downtown, then midtown ring, etc.)
        # For simplicity we return all; the grid lookup resolves precedence.
        return seen

    def _classify_point(self, x: float, z: float) -> DistrictType:
        """Return the highest-priority district for a world-space point."""
        for dt, fx0, fx1, fz0, fz1 in DISTRICT_ZONES:
            x0 = -self.width / 2  + fx0 * self.width
            x1 = -self.width / 2  + fx1 * self.width
            z0 = -self.height / 2 + fz0 * self.height
            z1 = -self.height / 2 + fz1 * self.height
            if x0 <= x <= x1 and z0 <= z <= z1:
                return dt
        return DistrictType.SUBURBS

    # ------------------------------------------------------------------
    # Grid helper (numpy based)
    # ------------------------------------------------------------------

    def _build_district_grid(self, _boundaries) -> np.ndarray:
        """2D numpy array mapping (row, col) → DistrictType index."""
        resolution = 200   # cells along each axis
        grid = np.zeros((resolution, resolution), dtype=np.int8)
        # DistrictType index map
        idx = {dt: i for i, dt in enumerate(DistrictType)}
        xs = np.linspace(-self.width  / 2, self.width  / 2, resolution)
        zs = np.linspace(-self.height / 2, self.height / 2, resolution)
        for ri, z in enumerate(zs):
            for ci, x in enumerate(xs):
                grid[ri, ci] = idx[self._classify_point(x, z)]
        return grid

    # ------------------------------------------------------------------
    # Road network
    # ------------------------------------------------------------------

    def _generate_road_network(
        self, district_grid: np.ndarray
    ) -> Tuple[List[RoadSegment], List[Intersection]]:
        roads: List[RoadSegment] = []
        inters: List[Intersection] = []

        half_w = self.width  / 2
        half_h = self.height / 2

        # Major arterials – evenly spaced grid at 200 m intervals
        arterial_spacing = 200.0
        arterial_xs = np.arange(-half_w, half_w + 1, arterial_spacing)
        arterial_zs = np.arange(-half_h, half_h + 1, arterial_spacing)

        for ax in arterial_xs:
            roads.append(RoadSegment(ax, -half_h, ax, half_h, 20, "arterial", 4))
        for az in arterial_zs:
            roads.append(RoadSegment(-half_w, az, half_w, az, 20, "arterial", 4))

        # Collect arterial intersections
        for ax in arterial_xs:
            for az in arterial_zs:
                dt = self._classify_point(ax, az)
                has_tl = dt in (DistrictType.DOWNTOWN, DistrictType.MIDTOWN,
                                DistrictType.INDUSTRIAL)
                inters.append(Intersection(ax, az, has_tl, dt))

        # Local streets – denser grid within each district block
        for ax_start, ax_end in zip(arterial_xs[:-1], arterial_xs[1:]):
            for az_start, az_end in zip(arterial_zs[:-1], arterial_zs[1:]):
                cx = (ax_start + ax_end) / 2
                cz = (az_start + az_end) / 2
                dt = self._classify_point(cx, cz)
                cfg = self._dist_cfg[dt]
                block = cfg["block_size"]
                sw    = cfg["street_width"]

                # Vertical locals
                local_xs = np.arange(ax_start + block, ax_end, block)
                for lx in local_xs:
                    roads.append(RoadSegment(lx, az_start, lx, az_end, sw, "local", 2))
                    for az in arterial_zs:
                        if az_start < az < az_end:
                            inters.append(Intersection(lx, az, False, dt))

                # Horizontal locals
                local_zs = np.arange(az_start + block, az_end, block)
                for lz in local_zs:
                    roads.append(RoadSegment(ax_start, lz, ax_end, lz, sw, "local", 2))

                # Sub-intersections at local × local
                for lx in local_xs:
                    for lz in local_zs:
                        inters.append(Intersection(lx, lz, False, dt))

        # Park paths
        park_bounds = next(
            (b for b in self._generate_district_boundaries()
             if b.district == DistrictType.PARK), None
        )
        if park_bounds:
            px0, pz0 = park_bounds.min_x, park_bounds.min_z
            px1, pz1 = park_bounds.max_x, park_bounds.max_z
            # Winding path
            step = 60.0
            xs_path = np.arange(px0 + step, px1, step)
            for lx in xs_path:
                roads.append(RoadSegment(lx, pz0, lx, pz1, 6, "path", 1))
            zs_path = np.arange(pz0 + step, pz1, step)
            for lz in zs_path:
                roads.append(RoadSegment(px0, lz, px1, lz, 6, "path", 1))

        return roads, inters

    # ------------------------------------------------------------------
    # Building placement
    # ------------------------------------------------------------------

    def _generate_buildings(
        self, roads: List[RoadSegment], district_grid: np.ndarray
    ) -> List[Building]:
        buildings: List[Building] = []
        half_w = self.width  / 2
        half_h = self.height / 2

        # Build a coarse occupancy grid to avoid overlap (10 m resolution)
        res    = 10
        occ_w  = int(self.width  / res) + 1
        occ_h  = int(self.height / res) + 1
        occupied = np.zeros((occ_h, occ_w), dtype=bool)

        def world_to_grid(x, z):
            ci = int((x + half_w) / res)
            ri = int((z + half_h) / res)
            return min(max(ri, 0), occ_h - 1), min(max(ci, 0), occ_w - 1)

        def mark_occupied(x, z, bw, bd):
            r0, c0 = world_to_grid(x - bw / 2 - 1, z - bd / 2 - 1)
            r1, c1 = world_to_grid(x + bw / 2 + 1, z + bd / 2 + 1)
            occupied[r0:r1+1, c0:c1+1] = True

        def is_free(x, z, bw, bd) -> bool:
            r0, c0 = world_to_grid(x - bw / 2, z - bd / 2)
            r1, c1 = world_to_grid(x + bw / 2, z + bd / 2)
            return not occupied[r0:r1+1, c0:c1+1].any()

        # Pre-mark road corridors as occupied
        for road in roads:
            hw = road.width / 2 + 2
            if abs(road.x2 - road.x1) < 0.1:   # vertical
                mark_occupied(road.x1, (road.z1 + road.z2) / 2,
                               hw * 2, abs(road.z2 - road.z1))
            else:                               # horizontal
                mark_occupied((road.x1 + road.x2) / 2, road.z1,
                               abs(road.x2 - road.x1), hw * 2)

        # Attempt placements on a jittered grid
        spacing = 30.0
        xs = np.arange(-half_w + spacing, half_w - spacing, spacing)
        zs = np.arange(-half_h + spacing, half_h - spacing, spacing)

        for base_x in xs:
            for base_z in zs:
                jx = base_x + self.rng.uniform(-spacing * 0.3, spacing * 0.3)
                jz = base_z + self.rng.uniform(-spacing * 0.3, spacing * 0.3)
                dt  = self._classify_point(jx, jz)
                cfg = self._dist_cfg[dt]

                # Respect density
                if self.rng.random() > cfg["density"]:
                    continue

                h_range = cfg["height_max"] - cfg["height_min"]
                bh = cfg["height_min"] + self.rng.random() * h_range
                # Footprint scales with height
                scale = math.sqrt(bh / 20) * 10
                bw = self.rng.uniform(scale * 0.6, scale * 1.2)
                bd = self.rng.uniform(scale * 0.6, scale * 1.2)
                bw = max(bw, 6.0)
                bd = max(bd, 6.0)

                sb = cfg["setback"]
                px, pz = jx, jz

                if is_free(px, pz, bw + sb * 2, bd + sb * 2):
                    model = self.rng.choice(cfg["model_types"])
                    rot   = self.rng.choice([0.0, 90.0, 180.0, 270.0])
                    buildings.append(Building(px, pz, bw, bd, bh, dt, model, rot))
                    mark_occupied(px, pz, bw + sb * 2, bd + sb * 2)

        return buildings

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_sidewalk_waypoints(self, city_data: CityData) -> List[Tuple[float, float]]:
        """Return a list of (x, z) waypoints along road edges for pedestrians."""
        waypoints = []
        for road in city_data.roads:
            if road.road_type == "path":
                continue
            offset = road.width / 2 + 1.5
            # Vertical road – add waypoints along both sides
            if abs(road.x2 - road.x1) < 0.1:
                num = max(2, int(abs(road.z2 - road.z1) / 20))
                zs  = np.linspace(road.z1, road.z2, num)
                for z in zs:
                    waypoints.append((road.x1 - offset, z))
                    waypoints.append((road.x1 + offset, z))
            else:
                num = max(2, int(abs(road.x2 - road.x1) / 20))
                xs  = np.linspace(road.x1, road.x2, num)
                for x in xs:
                    waypoints.append((x, road.z1 - offset))
                    waypoints.append((x, road.z1 + offset))
        return waypoints


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml, pathlib

    cfg_path = pathlib.Path(__file__).parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"city": {"width_km": 2.0, "height_km": 2.0, "seed": 42}}

    gen  = CityGenerator(cfg)
    data = gen.generate()
    print(f"Buildings  : {len(data.buildings)}")
    print(f"Roads      : {len(data.roads)}")
    print(f"Junctions  : {len(data.intersections)}")
    print(f"Districts  : {len(data.district_boundaries)}")
    print("Sample building:", data.buildings[0] if data.buildings else "none")
