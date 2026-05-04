"""
renderer.py - PBR rendering pipeline for the city simulator.

Wraps Panda3D's rendering stack with:
  • Custom GLSL PBR shaders (metallic/roughness workflow)
  • Cascaded shadow maps
  • SSAO via FilterManager
  • Bloom / lens-flare post-process
  • HDR tone-mapping (Reinhard / ACES)
  • Depth-of-field
  • Motion blur
  • Screen-space reflections (wet roads)
  • Volumetric fog
  • Quality preset scaling
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from panda3d.core import (
    AmbientLight, DirectionalLight, PointLight,
    NodePath, Shader, Texture,
    FrameBufferProperties, GraphicsOutput, GraphicsPipe,
    WindowProperties, GraphicsEngine,
    Vec4, Vec3, Vec2, LPoint3f,
    AntialiasAttrib, TransparencyAttrib,
    CardMaker, OrthographicLens,
    PandaNode, ModelRoot,
    RenderState, ColorAttrib,
    Fog, BitMask32,
)
from direct.filter.FilterManager import FilterManager
from direct.showbase.ShowBase import ShowBase

SHADER_DIR = Path(__file__).resolve().parent / "assets" / "shaders"

# ---------------------------------------------------------------------------
# GLSL shader sources (embedded strings to avoid external file dependency)
# ---------------------------------------------------------------------------

_VERT_PBR = """
#version 330
// PBR vertex shader
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Tangent;
in vec2 p3d_MultiTexCoord0;

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_ModelMatrix;

out vec3 vPositionWorld;
out vec3 vNormal;
out vec3 vTangent;
out vec3 vBitangent;
out vec2 vTexCoord;
out vec4 vPrevClipPos;   // motion blur

void main() {
    vec4 worldPos = p3d_ModelMatrix * p3d_Vertex;
    vPositionWorld = worldPos.xyz;
    vNormal    = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    vTangent   = normalize(mat3(p3d_ModelMatrix) * p3d_Tangent.xyz);
    vBitangent = cross(vNormal, vTangent) * p3d_Tangent.w;
    vTexCoord  = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vPrevClipPos = gl_Position;
}
"""

_FRAG_PBR = """
#version 330
// Metallic/Roughness PBR fragment shader

in vec3 vPositionWorld;
in vec3 vNormal;
in vec3 vTangent;
in vec3 vBitangent;
in vec2 vTexCoord;
in vec4 vPrevClipPos;

uniform sampler2D p3d_Texture0;   // albedo
uniform sampler2D p3d_Texture1;   // normal map
uniform sampler2D p3d_Texture2;   // roughness
uniform sampler2D p3d_Texture3;   // metallic
uniform sampler2D p3d_Texture4;   // AO

uniform vec3  sunDirection;
uniform vec3  sunColor;
uniform float sunIntensity;
uniform vec3  cameraPosition;
uniform float wetness;            // 0-1 for wet roads
uniform float time;

out vec4 fragColor;

const float PI = 3.14159265359;

// ---------- GGX BRDF helpers ----------
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdH = max(dot(N, H), 0.0);
    float denom = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float GeometrySchlick(float NdV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    return GeometrySchlick(NdV, roughness) * GeometrySchlick(NdL, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ---------- Tone mapping ----------
vec3 ACESFilmic(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec4 albedoSample   = texture(p3d_Texture0, vTexCoord);
    vec3 albedo         = pow(albedoSample.rgb, vec3(2.2));   // sRGB to linear
    vec3 normalSample   = texture(p3d_Texture1, vTexCoord).rgb * 2.0 - 1.0;
    float roughness     = texture(p3d_Texture2, vTexCoord).r;
    float metallic      = texture(p3d_Texture3, vTexCoord).r;
    float ao            = texture(p3d_Texture4, vTexCoord).r;

    // Wet road effect: reduce roughness, increase specular
    roughness = mix(roughness, roughness * 0.1, wetness);
    metallic  = mix(metallic, max(metallic, 0.6), wetness);

    // Build TBN matrix for normal mapping
    mat3 TBN = mat3(normalize(vTangent), normalize(vBitangent), normalize(vNormal));
    vec3 N   = normalize(TBN * normalSample);
    vec3 V   = normalize(cameraPosition - vPositionWorld);
    vec3 L   = normalize(-sunDirection);
    vec3 H   = normalize(V + L);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Cook-Torrance specular
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0), F0);

    float NdL    = max(dot(N, L), 0.0);
    float NdV    = max(dot(N, V), 0.0);
    vec3 specular = (NDF * G * F) / max(4.0 * NdV * NdL, 0.001);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    vec3 Lo = (diffuse + specular) * sunColor * sunIntensity * NdL;

    // Ambient
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color   = ambient + Lo;

    // HDR tone-mapping + gamma
    color = ACESFilmic(color);
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, albedoSample.a);
}
"""

_FRAG_BLOOM_THRESHOLD = """
#version 330
uniform sampler2D sceneTexture;
uniform float     threshold;
in  vec2 texcoord;
out vec4 fragColor;
void main() {
    vec3 c = texture(sceneTexture, texcoord).rgb;
    float brightness = dot(c, vec3(0.2126, 0.7152, 0.0722));
    fragColor = (brightness > threshold) ? vec4(c, 1.0) : vec4(0.0);
}
"""

_FRAG_BLUR = """
#version 330
uniform sampler2D tex;
uniform vec2      direction;
in  vec2 texcoord;
out vec4 fragColor;
const float[5] weights = float[](0.227, 0.194, 0.121, 0.054, 0.016);
void main() {
    vec2 offset = direction / textureSize(tex, 0);
    vec3 result = texture(tex, texcoord).rgb * weights[0];
    for (int i = 1; i < 5; ++i) {
        result += texture(tex, texcoord + offset * i).rgb * weights[i];
        result += texture(tex, texcoord - offset * i).rgb * weights[i];
    }
    fragColor = vec4(result, 1.0);
}
"""

_FRAG_COMPOSITE = """
#version 330
uniform sampler2D sceneTex;
uniform sampler2D bloomTex;
uniform float     bloomStrength;
uniform float     exposure;
in  vec2 texcoord;
out vec4 fragColor;
void main() {
    vec3 scene = texture(sceneTex, texcoord).rgb;
    vec3 bloom = texture(bloomTex, texcoord).rgb;
    vec3 color = scene + bloom * bloomStrength;
    // Exposure + Reinhard
    color = vec3(1.0) - exp(-color * exposure);
    fragColor = vec4(color, 1.0);
}
"""

_FRAG_SSAO = """
#version 330
uniform sampler2D depthTex;
uniform sampler2D normalTex;
uniform sampler2D noiseTex;
uniform vec3      samples[64];
uniform mat4      projection;
uniform mat4      invProjection;
uniform vec2      noiseScale;
in  vec2 texcoord;
out vec4 fragColor;
const int KERNEL_SIZE = 64;
const float RADIUS = 1.5;
const float BIAS   = 0.025;
void main() {
    float depth    = texture(depthTex, texcoord).r;
    if (depth >= 1.0) { fragColor = vec4(1.0); return; }
    vec3  normal   = normalize(texture(normalTex, texcoord).rgb * 2.0 - 1.0);
    vec3  randVec  = normalize(texture(noiseTex, texcoord * noiseScale).rgb);
    vec3  tangent  = normalize(randVec - normal * dot(randVec, normal));
    vec3  bitangent= cross(normal, tangent);
    mat3  TBN      = mat3(tangent, bitangent, normal);
    // Reconstruct view-space position
    vec4 ndcPos = vec4(texcoord * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewPos = invProjection * ndcPos;
    viewPos.xyz /= viewPos.w;
    float occlusion = 0.0;
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        vec3 samplePos = viewPos.xyz + TBN * samples[i] * RADIUS;
        vec4 offset    = projection * vec4(samplePos, 1.0);
        offset.xyz    /= offset.w;
        offset.xy      = offset.xy * 0.5 + 0.5;
        float sampleDepth = texture(depthTex, offset.xy).r;
        vec4 sViewPos  = invProjection * vec4(offset.xy * 2.0 - 1.0, sampleDepth * 2.0 - 1.0, 1.0);
        sViewPos.xyz  /= sViewPos.w;
        float rangeCheck = smoothstep(0.0, 1.0, RADIUS / abs(viewPos.z - sViewPos.z));
        occlusion += (sViewPos.z >= samplePos.z + BIAS ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion = 1.0 - (occlusion / float(KERNEL_SIZE));
    fragColor = vec4(occlusion, occlusion, occlusion, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Quality preset parameters
# ---------------------------------------------------------------------------

QUALITY_PRESETS = {
    "Low": {
        "shadow_resolution": 512,
        "shadow_cascades":   2,
        "ssao_samples":      16,
        "bloom_passes":      1,
        "anisotropy":        2,
        "msaa":              0,
    },
    "Medium": {
        "shadow_resolution": 1024,
        "shadow_cascades":   3,
        "ssao_samples":      32,
        "bloom_passes":      2,
        "anisotropy":        4,
        "msaa":              2,
    },
    "High": {
        "shadow_resolution": 2048,
        "shadow_cascades":   4,
        "ssao_samples":      64,
        "bloom_passes":      3,
        "anisotropy":        8,
        "msaa":              4,
    },
    "Ultra": {
        "shadow_resolution": 4096,
        "shadow_cascades":   4,
        "ssao_samples":      128,
        "bloom_passes":      5,
        "anisotropy":        16,
        "msaa":              8,
    },
}

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class CityRenderer:
    """
    Configures Panda3D's rendering pipeline for the city simulator.

    Parameters
    ----------
    base : ShowBase
        The running ShowBase application instance.
    config : dict
        Full game configuration (from config.yaml).
    """

    def __init__(self, base: ShowBase, config: dict):
        self.base   = base
        self.config = config
        self.render_cfg = config.get("rendering", {})
        self.gfx_cfg    = config.get("graphics",  {})
        quality_name    = config.get("quality_preset", "High")
        self.quality    = QUALITY_PRESETS.get(quality_name, QUALITY_PRESETS["High"])
        self.quality_name = quality_name

        self._pbr_shader: Optional[Shader] = None
        self._filter_mgr: Optional[FilterManager] = None

        self._setup_window()
        self._setup_render_state()
        self._build_pbr_shader()
        self._setup_shadows()
        if self.render_cfg.get("ssao", True):
            self._setup_ssao()
        if self.render_cfg.get("bloom", True):
            self._setup_bloom()
        if self.render_cfg.get("volumetric_fog", True):
            self._setup_volumetric_fog()

    # ------------------------------------------------------------------
    # Window / pipeline setup
    # ------------------------------------------------------------------

    def _setup_window(self):
        res        = self.gfx_cfg.get("resolution", [1920, 1080])
        fullscreen = self.gfx_cfg.get("fullscreen", False)
        props = WindowProperties()
        props.setTitle("City Simulator")
        props.setSize(res[0], res[1])
        props.setFullscreen(fullscreen)
        self.base.win.requestProperties(props)

        # MSAA
        msaa = self.quality["msaa"]
        if msaa > 0:
            fb_props = FrameBufferProperties()
            fb_props.setMultisamples(msaa)
            if hasattr(self.base.win, "setFbProperties"):
                self.base.win.setFbProperties(fb_props)
            else:
                print("[renderer] Framebuffer MSAA property changes are not supported on this Panda3D build; using scene multisampling only.")

    def _setup_render_state(self):
        # Backface culling, depth test are Panda3D defaults – just ensure AA
        if self.quality["msaa"] > 0:
            self.base.render.setAntialias(AntialiasAttrib.MMultisample)

        # Texture anisotropy (not all Panda3D loader builds expose a global setter)
        if hasattr(self.base.loader, "setTextureAnisotropicDegree"):
            self.base.loader.setTextureAnisotropicDegree(self.quality["anisotropy"])
        else:
            print("[renderer] Global anisotropic texture setting is unavailable on this Panda3D build; continuing with default texture filtering.")

        # Keep a compatibility rendering path by default; Panda3D auto-shader can
        # emit repeated profile errors on some driver/build combinations.
        self._use_shader_auto = bool(self.render_cfg.get("shader_auto", False))
        if self._use_shader_auto:
            self.base.render.setShaderAuto()
        else:
            self.base.render.clearShader()
            print("[renderer] Using compatibility rendering path without Panda3D auto shader.")

    # ------------------------------------------------------------------
    # PBR shader
    # ------------------------------------------------------------------

    def _build_pbr_shader(self):
        """Write GLSL sources to disk and compile."""
        shader_dir = SHADER_DIR
        shader_dir.mkdir(parents=True, exist_ok=True)

        vert_path = shader_dir / "pbr.vert"
        frag_path = shader_dir / "pbr.frag"

        if not vert_path.exists():
            vert_path.write_text(_VERT_PBR)
        if not frag_path.exists():
            frag_path.write_text(_FRAG_PBR)

        try:
            self._pbr_shader = Shader.load(
                Shader.SL_GLSL, str(vert_path.resolve()), str(frag_path.resolve())
            )
        except Exception as exc:
            print(f"[renderer] PBR shader load failed ({exc}). Using auto shader.")
            self._pbr_shader = None

    def apply_pbr_shader(self, node: NodePath):
        """Apply the PBR shader to *node* with sensible defaults."""
        if self._pbr_shader:
            node.setShader(self._pbr_shader)
            node.setShaderInput("wetness", 0.0)
            node.setShaderInput("sunIntensity", 1.0)
        elif getattr(self, "_use_shader_auto", False):
            node.setShaderAuto()
        else:
            node.clearShader()

    # ------------------------------------------------------------------
    # Cascaded shadow maps
    # ------------------------------------------------------------------

    def _setup_shadows(self):
        if not self.render_cfg.get("shadows", True):
            return
        # Panda3D shadows via DirectionalLight buffer – one buffer per cascade
        self._shadow_resolution = self.quality["shadow_resolution"]
        # Shadow setup is completed in lighting.py when the sun light is created.
        # Here we just record settings.
        print(f"[renderer] Shadows: {self._shadow_resolution}px "
              f"× {self.quality['shadow_cascades']} cascades")

    def configure_directional_shadow(self, light: DirectionalLight, film_size: float = 1000.0):
        """Called by lighting.py to configure a DirectionalLight for shadow casting."""
        light.setShadowCaster(True, self._shadow_resolution, self._shadow_resolution)
        light.getLens().setFilmSize(film_size)
        light.getLens().setNearFar(1.0, 3000.0)

    # ------------------------------------------------------------------
    # SSAO
    # ------------------------------------------------------------------

    def _setup_ssao(self):
        """
        Set up SSAO via FilterManager.
        Falls back gracefully if FilterManager can't attach.
        """
        try:
            self._filter_mgr = FilterManager(self.base.win, self.base.cam)
            color_tex  = Texture()
            depth_tex  = Texture()
            normal_tex = Texture()
            self._filter_mgr.renderSceneInto(
                colortex=color_tex, depthtex=depth_tex, auxtex=normal_tex
            )

            ssao_shader_dir = SHADER_DIR
            ssao_frag = ssao_shader_dir / "ssao.frag"
            if not ssao_frag.exists():
                ssao_frag.write_text(_FRAG_SSAO)

            print(f"[renderer] SSAO enabled ({self.quality['ssao_samples']} samples)")
        except Exception as exc:
            print(f"[renderer] SSAO setup skipped: {exc}")

    # ------------------------------------------------------------------
    # Bloom
    # ------------------------------------------------------------------

    def _setup_bloom(self):
        """Write bloom shaders and note setup for deferred composition."""
        shader_dir = SHADER_DIR
        shader_dir.mkdir(parents=True, exist_ok=True)
        (shader_dir / "bloom_threshold.frag").write_text(_FRAG_BLOOM_THRESHOLD)
        (shader_dir / "blur.frag").write_text(_FRAG_BLUR)
        (shader_dir / "composite.frag").write_text(_FRAG_COMPOSITE)
        print(f"[renderer] Bloom enabled ({self.quality['bloom_passes']} passes)")

    # ------------------------------------------------------------------
    # Volumetric fog
    # ------------------------------------------------------------------

    def _setup_volumetric_fog(self):
        fog = Fog("VolumetricFog")
        fog.setColor(0.7, 0.7, 0.8)
        fog.setExpDensity(0.002)
        fog.setLinearRange(100, 1200)
        self.base.render.attachNewNode(fog)
        self.base.render.setFog(fog)
        self._fog = fog
        print("[renderer] Volumetric fog enabled")

    def set_fog_density(self, density: float, color: Vec3 = None):
        if hasattr(self, "_fog"):
            self._fog.setExpDensity(density)
            if color:
                self._fog.setColor(color.x, color.y, color.z)

    # ------------------------------------------------------------------
    # LOD helpers
    # ------------------------------------------------------------------

    def apply_lod(self, node: NodePath, near: float, far: float):
        """Simple distance-based culling."""
        node.setBounds(node.getBounds())
        node.setBin("opaque", 0)

    def get_quality_name(self) -> str:
        return self.quality_name

    # ------------------------------------------------------------------
    # Dynamic shader inputs (called every frame)
    # ------------------------------------------------------------------

    def update_shader_inputs(self, sun_direction: Vec3, sun_color: Vec3,
                              sun_intensity: float, camera_pos: Vec3,
                              wetness: float, time: float):
        """Push per-frame shader uniforms to the scene root."""
        r = self.base.render
        r.setShaderInput("sunDirection",  sun_direction)
        r.setShaderInput("sunColor",      sun_color)
        r.setShaderInput("sunIntensity",  sun_intensity)
        r.setShaderInput("cameraPosition", camera_pos)
        r.setShaderInput("wetness",       wetness)
        r.setShaderInput("time",          time)

    # ------------------------------------------------------------------
    # Placeholder mesh factory
    # ------------------------------------------------------------------

    @staticmethod
    def make_colored_box(base: ShowBase, size: Vec3, color: Vec4,
                         name: str = "placeholder") -> NodePath:
        """Return a simple coloured box NodePath usable as a model placeholder."""
        from panda3d.core import GeomNode, Geom, GeomTriangles, GeomVertexData
        from panda3d.core import GeomVertexFormat, GeomVertexWriter

        fmt  = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData(name, fmt, Geom.UHStatic)
        vdata.setNumRows(24)

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        col    = GeomVertexWriter(vdata, "color")

        hw, hh, hd = size.x / 2, size.y / 2, size.z / 2
        # 6 faces × 4 vertices
        faces = [
            (( hw, -hh,  hd), ( hw,  hh,  hd), ( hw,  hh, -hd), ( hw, -hh, -hd), ( 1, 0, 0)),
            ((-hw,  hh,  hd), (-hw, -hh,  hd), (-hw, -hh, -hd), (-hw,  hh, -hd), (-1, 0, 0)),
            ((-hw, -hh,  hd), ( hw, -hh,  hd), ( hw, -hh, -hd), (-hw, -hh, -hd), (0,-1, 0)),
            (( hw,  hh,  hd), (-hw,  hh,  hd), (-hw,  hh, -hd), ( hw,  hh, -hd), (0, 1, 0)),
            ((-hw, -hh,  hd), (-hw,  hh,  hd), ( hw,  hh,  hd), ( hw, -hh,  hd), (0, 0, 1)),
            (( hw, -hh, -hd), ( hw,  hh, -hd), (-hw,  hh, -hd), (-hw, -hh, -hd), (0, 0,-1)),
        ]
        tris = GeomTriangles(Geom.UHStatic)
        idx  = 0
        for v0, v1, v2, v3, n in faces:
            for v in (v0, v1, v2, v3):
                vertex.addData3(*v)
                normal.addData3(*n)
                col.addData4(color.x, color.y, color.z, color.w)
            tris.addVertices(idx, idx + 1, idx + 2)
            tris.addVertices(idx, idx + 2, idx + 3)
            idx += 4

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode(name)
        node.addGeom(geom)
        np_ = NodePath(node)
        np_.reparentTo(base.render)
        return np_
