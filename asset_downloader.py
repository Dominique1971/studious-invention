"""
asset_downloader.py - Downloads and manages game assets from Kenney.nl and AmbientCG.

Handles download failures gracefully. If a pack cannot be fetched automatically,
placeholder marker files are created so the game can run with procedural fallbacks.
"""

import os
import sys
import json
import zipfile
import hashlib
import shutil
import struct
import time
import logging
from pathlib import Path
from typing import Optional

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[WARN] requests not installed – download disabled.")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).parent
ASSETS_DIR = REPO_ROOT / "assets"
CACHE_DIR  = ASSETS_DIR / "_cache"
MANIFEST_FILE = ASSETS_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Asset definitions
# ---------------------------------------------------------------------------
KENNEY_PACKS = [
    {
        "name": "city-kit-commercial",
        "url":  "https://kenney.nl/assets/city-kit-commercial",
        "cdn":  None,   # No stable direct-download CDN URL – see note below
        "local_zip": "city_kit_commercial.zip",
        "dest_dir": "kenney/city_commercial",
    },
    {
        "name": "city-kit-roads",
        "url":  "https://kenney.nl/assets/city-kit-roads",
        "cdn":  None,
        "local_zip": "city_kit_roads.zip",
        "dest_dir": "kenney/city_roads",
    },
    {
        "name": "city-kit-suburban",
        "url":  "https://kenney.nl/assets/city-kit-suburban",
        "cdn":  None,
        "local_zip": "city_kit_suburban.zip",
        "dest_dir": "kenney/city_suburban",
    },
    {
        "name": "car-kit",
        "url":  "https://kenney.nl/assets/car-kit",
        "cdn":  None,
        "local_zip": "car_kit.zip",
        "dest_dir": "kenney/car_kit",
    },
    {
        "name": "nature-kit",
        "url":  "https://kenney.nl/assets/nature-kit",
        "cdn":  None,
        "local_zip": "nature_kit.zip",
        "dest_dir": "kenney/nature_kit",
    },
]

AMBIENTCG_TEXTURES = [
    {
        "name": "concrete_floor",
        "url":  "https://ambientcg.com/get?file=Concrete012_1K-PNG.zip",
        "dest_dir": "textures/concrete_floor",
    },
    {
        "name": "asphalt",
        "url":  "https://ambientcg.com/get?file=Asphalt012_1K-PNG.zip",
        "dest_dir": "textures/asphalt",
    },
    {
        "name": "brick_wall",
        "url":  "https://ambientcg.com/get?file=Bricks059_1K-PNG.zip",
        "dest_dir": "textures/brick_wall",
    },
    {
        "name": "glass_panels",
        "url":  "https://ambientcg.com/get?file=Glass01_1K-PNG.zip",
        "dest_dir": "textures/glass",
    },
    {
        "name": "metal_plates",
        "url":  "https://ambientcg.com/get?file=MetalPlates002_1K-PNG.zip",
        "dest_dir": "textures/metal_plates",
    },
    {
        "name": "grass",
        "url":  "https://ambientcg.com/get?file=Ground037_1K-PNG.zip",
        "dest_dir": "textures/grass",
    },
    {
        "name": "pavement",
        "url":  "https://ambientcg.com/get?file=PavingStones070_1K-PNG.zip",
        "dest_dir": "textures/pavement",
    },
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _make_session(retries: int = 3, backoff: float = 1.0) -> "requests.Session":
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://",  adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "CitySimAssetDownloader/1.0"})
    return session


def _download_file(session, url: str, dest: Path, chunk_size: int = 65536) -> bool:
    """Download *url* to *dest* with a simple progress bar. Returns True on success."""
    try:
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            bar  = "#" * (pct // 4)
                            sys.stdout.write(f"\r  [{bar:<25}] {pct:3d}%  {downloaded//1024} KB")
                            sys.stdout.flush()
            if total:
                print()
            return True
    except Exception as exc:
        log.warning("  Download failed: %s", exc)
        return False


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_zip(zip_path: Path, dest_dir: Path) -> list:
    """Extract zip to dest_dir; returns list of extracted file paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
        extracted = z.namelist()
    return extracted


# ---------------------------------------------------------------------------
# Placeholder generation helpers
# ---------------------------------------------------------------------------

def _write_1x1_png(path: Path, r: int, g: int, b: int):
    """Write a minimal 1×1 RGB PNG for placeholder textures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Raw IDAT data for 1×1 image (filter byte 0x00 + RGB)
    import zlib
    raw = zlib.compress(bytes([0, r, g, b]))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return c + crc
    png  = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", raw)
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _create_kenney_placeholder(pack: dict):
    dest = ASSETS_DIR / pack["dest_dir"]
    dest.mkdir(parents=True, exist_ok=True)
    # Write a marker so the game knows this pack was expected but not downloaded
    marker = dest / "PLACEHOLDER.txt"
    marker.write_text(
        f"Asset pack '{pack['name']}' was not downloaded automatically.\n"
        f"Please visit {pack['url']} and place the extracted contents here.\n"
        f"The game will use procedural geometry as fallback.\n"
    )
    # Create stub PBR textures so the renderer doesn't crash
    for name, rgb in [("albedo", (180, 180, 180)), ("normal", (127, 127, 255)),
                      ("roughness", (128, 128, 128)), ("metallic", (0, 0, 0))]:
        _write_1x1_png(dest / f"placeholder_{name}.png", *rgb)
    log.info("  Created placeholder for '%s' in %s", pack["name"], dest)


def _create_texture_placeholder(tex: dict):
    dest = ASSETS_DIR / tex["dest_dir"]
    dest.mkdir(parents=True, exist_ok=True)
    for name, rgb in [("Color", (180, 180, 180)), ("NormalGL", (127, 127, 255)),
                      ("Roughness", (128, 128, 128)), ("Metalness", (0, 0, 0)),
                      ("AmbientOcclusion", (255, 255, 255))]:
        _write_1x1_png(dest / f"placeholder_{name}.png", *rgb)
    marker = dest / "PLACEHOLDER.txt"
    marker.write_text(
        f"Texture '{tex['name']}' was not downloaded automatically.\n"
        f"Source: {tex['url']}\n"
        "Placeholder 1×1 textures have been created.\n"
    )
    log.info("  Created placeholder textures for '%s'", tex["name"])


# ---------------------------------------------------------------------------
# OBJ → EGG conversion
# ---------------------------------------------------------------------------

def _try_convert_obj_to_egg(obj_path: Path) -> Optional[Path]:
    """Attempt to convert an .obj model to .egg using Panda3D's obj2egg."""
    egg_path = obj_path.with_suffix(".egg")
    if egg_path.exists():
        return egg_path
    import subprocess
    try:
        result = subprocess.run(
            ["obj2egg", "-o", str(egg_path), str(obj_path)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            log.info("    Converted %s -> %s", obj_path.name, egg_path.name)
            return egg_path
        else:
            log.debug("    obj2egg failed: %s", result.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.debug("    obj2egg not available or timed out.")
    return None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest() -> dict:
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "models": [], "textures": []}
    for p in sorted(ASSETS_DIR.rglob("*.egg")):
        manifest["models"].append({"path": str(p.relative_to(ASSETS_DIR)), "format": "egg"})
    for p in sorted(ASSETS_DIR.rglob("*.obj")):
        manifest["models"].append({"path": str(p.relative_to(ASSETS_DIR)), "format": "obj"})
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in sorted(ASSETS_DIR.rglob(ext)):
            if "_cache" not in str(p):
                manifest["textures"].append({"path": str(p.relative_to(ASSETS_DIR))})
    return manifest


def _save_manifest(manifest: dict):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest saved → %s  (%d models, %d textures)",
             MANIFEST_FILE, len(manifest["models"]), len(manifest["textures"]))


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_kenney_packs(session):
    log.info("=== Kenney.nl Asset Packs ===")
    log.info("NOTE: Kenney.nl does not expose stable direct-download CDN URLs.")
    log.info("      Attempting to detect download links from pack pages…")

    for pack in KENNEY_PACKS:
        dest_dir = ASSETS_DIR / pack["dest_dir"]
        sentinel = dest_dir / ".downloaded"
        if sentinel.exists():
            log.info("  [cached] %s", pack["name"])
            continue

        log.info("  Trying '%s' …", pack["name"])
        cache_zip = CACHE_DIR / pack["local_zip"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        downloaded = False

        # Step 1 – try to scrape the direct download link from the pack page
        if not cache_zip.exists():
            try:
                page = session.get(pack["url"], timeout=15)
                page.raise_for_status()
                # Kenney pack pages contain a download link in the form:
                # https://kenney.nl/media/pages/assets/…/…-*.zip
                import re
                links = re.findall(
                    r'href=["\']([^"\']*\.zip)["\']', page.text, re.IGNORECASE
                )
                if links:
                    direct = links[0]
                    if not direct.startswith("http"):
                        direct = "https://kenney.nl" + direct
                    log.info("    Found link: %s", direct)
                    downloaded = _download_file(session, direct, cache_zip)
                else:
                    log.warning("    No .zip link found on pack page.")
            except Exception as exc:
                log.warning("    Could not fetch pack page: %s", exc)

        # Step 2 – if we have a zip, extract it
        if cache_zip.exists():
            log.info("    Extracting %s …", cache_zip.name)
            try:
                files = _extract_zip(cache_zip, dest_dir)
                # Convert any OBJ files to EGG
                for rel in files:
                    p = dest_dir / rel
                    if p.suffix.lower() == ".obj":
                        _try_convert_obj_to_egg(p)
                sentinel.touch()
                log.info("    ✓ Extracted %d files.", len(files))
                downloaded = True
            except Exception as exc:
                log.warning("    Extraction failed: %s", exc)

        if not downloaded:
            _create_kenney_placeholder(pack)


def download_ambientcg_textures(session):
    log.info("=== AmbientCG PBR Textures ===")
    for tex in AMBIENTCG_TEXTURES:
        dest_dir = ASSETS_DIR / tex["dest_dir"]
        sentinel  = dest_dir / ".downloaded"
        if sentinel.exists():
            log.info("  [cached] %s", tex["name"])
            continue

        log.info("  Downloading '%s' …", tex["name"])
        cache_zip = CACHE_DIR / (tex["name"] + ".zip")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not cache_zip.exists():
            ok = _download_file(session, tex["url"], cache_zip)
        else:
            ok = True

        if ok and cache_zip.exists():
            try:
                _extract_zip(cache_zip, dest_dir)
                sentinel.touch()
                log.info("    ✓ Done.")
            except Exception as exc:
                log.warning("    Extraction failed: %s – using placeholder.", exc)
                _create_texture_placeholder(tex)
        else:
            _create_texture_placeholder(tex)


def create_directory_structure():
    """Ensure assets directory tree exists with a helpful README."""
    dirs = [
        ASSETS_DIR,
        ASSETS_DIR / "kenney",
        ASSETS_DIR / "textures",
        ASSETS_DIR / "audio",
        ASSETS_DIR / "shaders",
        ASSETS_DIR / "fonts",
        CACHE_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    readme = ASSETS_DIR / "README.txt"
    readme.write_text(
        "ASSETS DIRECTORY\n"
        "================\n\n"
        "Kenney.nl packs (CC0):\n"
        "  Download manually from https://kenney.nl and place ZIP files under\n"
        "  assets/_cache/ with the names listed in asset_downloader.py.\n\n"
        "AmbientCG textures (CC0):\n"
        "  Downloaded automatically when internet is available.\n\n"
        "Audio:\n"
        "  Place .ogg / .wav files under assets/audio/\n"
        "  The engine generates procedural audio if files are missing.\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    log.info("City Simulator – Asset Downloader")
    log.info("===================================")
    create_directory_structure()

    if not HAS_REQUESTS:
        log.error("requests is not installed.  Run: pip install requests")
        log.info("Creating placeholders for all packs…")
        for pack in KENNEY_PACKS:
            _create_kenney_placeholder(pack)
        for tex in AMBIENTCG_TEXTURES:
            _create_texture_placeholder(tex)
    else:
        session = _make_session()
        download_kenney_packs(session)
        download_ambientcg_textures(session)

    manifest = _build_manifest()
    _save_manifest(manifest)
    log.info("Done. %d models and %d textures available.",
             len(manifest["models"]), len(manifest["textures"]))


if __name__ == "__main__":
    main()
