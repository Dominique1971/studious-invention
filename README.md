# City Driving Simulator

A GPU-accelerated 3D open-world city driving simulator built in Python with **Panda3D** and **custom GLSL PBR shaders**.

---

## Features

- **Procedurally generated 2 km × 2 km city** – downtown skyscrapers, midtown offices, suburbs, industrial districts, and parks
- **PBR rendering** (metallic/roughness workflow) with GLSL shaders
- **Dynamic time of day** – sun arc, sky colour gradient, street lights activating at dusk
- **Weather system** – clear → cloudy → rain → thunder with particle rain, lightning flashes, and wet-road reflections
- **Player vehicle** with Bullet physics – weight transfer, gear simulation, skid marks, damage model
- **50 NPC traffic vehicles** with traffic-light obedience, lane keeping, turn signals, and emergency vehicles
- **200 NPC pedestrians** walking along sidewalks, waiting at crosswalks, reacting to vehicles
- **Full HUD** – analog speedometer, minimap, damage vignette, time/weather display, F1 debug overlay, pause menu, vehicle selector
- **Procedural audio** generated with NumPy (engine hum, acceleration, braking squeal, ambient traffic, rain, siren)

---

## Prerequisites

| Requirement | Minimum version |
|-------------|-----------------|
| Python | 3.10+ |
| pip | 22+ |
| NVIDIA driver | 520+ (for OpenGL 4.3) |
| CUDA | Optional |
| RAM | 8 GB |
| VRAM | 4 GB |
| OS | Linux, Windows 10+, macOS 12+ |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/studious-invention.git
cd studious-invention

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## First Run – Download / Generate Assets

```bash
python asset_downloader.py
```

This script:
1. Attempts to download Kenney.nl CC0 asset packs (city buildings, cars, nature).
2. Downloads AmbientCG PBR textures (concrete, asphalt, bricks, etc.).
3. Creates **placeholder assets** if downloads fail (the game will still run using coloured boxes).
4. Generates `assets/manifest.json` listing all available models and textures.

> **Note:** Kenney.nl asset packs may require manual download from <https://kenney.nl>. Place the ZIP files in `assets/_cache/` using the filenames shown in `asset_downloader.py`.

---

## Launch

```bash
python main.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--quality Low\|Medium\|High\|Ultra` | `High` | Rendering quality preset |
| `--hour 0-24` | `12.0` | Starting time of day |
| `--no-audio` | off | Disable audio system |
| `--debug-physics` | off | Show Bullet physics wireframe |

---

## Controls

| Key | Action |
|-----|--------|
| **W / ↑** | Accelerate |
| **S / ↓** | Reverse |
| **A / ←** | Steer left |
| **D / →** | Steer right |
| **Space** | Hand brake |
| **R** | Reset vehicle (respawn at spawn point) |
| **H** | Toggle headlights |
| **C** | Cycle camera modes (Chase → Bumper → Cinematic → Top-down) |
| **TAB** | Open vehicle selection screen |
| **ESC** | Pause menu |
| **F1** | Debug overlay (fps, position, GPU) |
| **F2** | Toggle wireframe |
| **F3** | Toggle physics collision boxes |
| **F4** | Cycle time of day (sunrise → noon → sunset → night) |
| **Page Up** | Advance time by 1 hour |
| **Page Down** | Rewind time by 1 hour |

---

## Camera Modes

| Mode | Description |
|------|-------------|
| **Chase** | Third-person follow camera, smooth lag |
| **Bumper** | First-person hood camera |
| **Cinematic Drone** | Orbiting aerial camera |
| **Top Down** | Overhead map view |

---

## Configuration (`config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `quality_preset` | `High` | Global quality: Low / Medium / High / Ultra |
| `city.width_km` | `2.0` | City width in kilometres |
| `city.height_km` | `2.0` | City height in kilometres |
| `city.seed` | `42` | Procedural generation seed |
| `rendering.shadows` | `true` | Cascaded shadow maps |
| `rendering.ssao` | `true` | Screen-space ambient occlusion |
| `rendering.bloom` | `true` | Bloom / lens flare |
| `rendering.dof` | `true` | Depth of field |
| `rendering.motion_blur` | `true` | Camera motion blur |
| `rendering.ssr` | `true` | Screen-space reflections (wet roads) |
| `rendering.hdr` | `true` | HDR tone-mapping (ACES Filmic) |
| `rendering.volumetric_fog` | `true` | Exponential fog |
| `graphics.resolution` | `[1920, 1080]` | Window resolution |
| `graphics.fullscreen` | `false` | Fullscreen mode |
| `graphics.vsync` | `true` | Vertical sync |
| `graphics.fov` | `75` | Horizontal field of view (degrees) |
| `graphics.target_fps` | `60` | Target frame rate |
| `audio.master_volume` | `0.8` | Master volume (0-1) |
| `gameplay.npc_vehicles` | `50` | Number of AI traffic vehicles |
| `gameplay.npc_pedestrians` | `200` | Number of AI pedestrians |

---

## Quality Presets

| Preset | Shadow Map | SSAO Samples | Bloom Passes | MSAA |
|--------|-----------|--------------|--------------|------|
| Low | 512 px | 16 | 1 | Off |
| Medium | 1024 px | 32 | 2 | 2× |
| High | 2048 px | 64 | 3 | 4× |
| Ultra | 4096 px | 128 | 5 | 8× |

---

## Directory Structure

```
.
├── main.py               # Entry point
├── city_generator.py     # Procedural city layout
├── renderer.py           # PBR rendering pipeline
├── lighting.py           # Dynamic lighting / time of day
├── vehicle.py            # Player vehicle (Bullet physics)
├── npc_vehicle.py        # AI traffic vehicles
├── npc_pedestrian.py     # AI pedestrians
├── hud.py                # HUD elements
├── audio.py              # Audio system
├── weather.py            # Weather effects
├── asset_downloader.py   # Asset fetcher / generator
├── config.yaml           # All configuration options
├── requirements.txt      # Python dependencies
├── assets/
│   ├── kenney/           # Kenney.nl CC0 model packs
│   ├── textures/         # AmbientCG PBR textures
│   ├── audio/            # Sound files (generated if absent)
│   ├── shaders/          # GLSL shaders (written at startup)
│   └── manifest.json     # Asset manifest
└── ASSETS.md             # Asset credits and licences
```

---

## Known Limitations

- Kenney.nl asset packs cannot be downloaded automatically (no stable CDN URL). Coloured placeholder boxes are used as fallback.
- The PBR shader requires OpenGL 4.3. On older hardware, `setShaderAuto()` is used as fallback.
- SSAO and full post-processing chain require a Panda3D build with `FilterManager` support.
- macOS users may need to set `gl-version 4 1` in Panda3D settings.
- Physics simulation can become unstable at high speeds (> 250 km/h) – reset with **R**.

---

## Screenshots

> *Screenshots will be added once the GPU rendering environment is available.*

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: panda3d` | Run `pip install panda3d==1.10.14` |
| Blank window / no geometry | Check GPU supports OpenGL 4.3; try `--quality Low` |
| No audio | Ensure OpenAL is installed, or use `--no-audio` |
| Very low FPS | Reduce `quality_preset` to `Low` or `Medium` |
| Vehicle sinks through ground | Reset with **R** |
| Missing model assets | Run `python asset_downloader.py` first |
| `yaml.safe_load` error | Run `pip install PyYAML==6.0.2` |

---

## Licence

Source code: MIT. Asset packs: CC0 (see [ASSETS.md](ASSETS.md)).
