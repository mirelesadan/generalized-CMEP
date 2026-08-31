"""Standalone Three.js GPU-instanced viewer for prepared CMEP volumes."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import webbrowser

import numpy as np

from cmep_visualization import select_threshold_points


_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "three-r128"
_THREE_PATH = _VENDOR_DIR / "three.min.js"
_CONTROLS_PATH = _VENDOR_DIR / "OrbitControls.js"


def write_gpu_volume_viewer(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    output_path: str | Path,
    threshold: float = 0.70,
    color: str = "red",
    colormap: str | Sequence[Any] | None = None,
    minimum_voxel_alpha: float = 0.30,
    background_color: str = "black",
    max_voxels: int | None = None,
    voxel_scale: float = 0.92,
    open_browser: bool = True,
) -> Path:
    """Write and optionally open a self-contained GPU-instanced voxel viewer.

    ``max_voxels`` is entirely user controlled.  ``None`` includes every voxel
    at or above the threshold; no internal upper display limit is imposed.
    The numerical volume is never modified.
    """
    instances = prepare_gpu_instances(
        volume,
        metadata,
        threshold=threshold,
        color=color,
        colormap=colormap,
        minimum_voxel_alpha=minimum_voxel_alpha,
        background_color=background_color,
        max_voxels=max_voxels,
        voxel_scale=voxel_scale,
    )
    three_source, controls_source = _load_three_sources()
    html = _build_html(instances, metadata, three_source, controls_source)

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8", newline="\n")
    temporary.replace(destination)

    if open_browser:
        webbrowser.open_new_tab(destination.as_uri())
    return destination


def prepare_gpu_instances(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    threshold: float,
    color: str,
    minimum_voxel_alpha: float,
    background_color: str,
    max_voxels: int | None,
    voxel_scale: float,
    colormap: str | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Select physical voxel centers and encode GPU instance attributes."""
    data = np.asarray(volume)
    if data.ndim != 3:
        raise ValueError(f"volume must be 3D in (x, y, z) order; got {data.shape}.")

    threshold_value = _normalized_scalar(threshold, "threshold")
    alpha_minimum = _normalized_scalar(
        minimum_voxel_alpha, "minimum_voxel_alpha"
    )
    scale = float(voxel_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("voxel_scale must be a positive finite number.")

    if max_voxels is None:
        selection_limit = int(data.size)
    else:
        if isinstance(max_voxels, (bool, np.bool_)):
            raise TypeError("max_voxels must be a positive integer or None.")
        selection_limit = int(max_voxels)
        if selection_limit < 1 or selection_limit != max_voxels:
            raise ValueError("max_voxels must be a positive integer or None.")

    points = select_threshold_points(
        data,
        metadata,
        threshold=threshold_value,
        max_points=selection_limit,
    )
    centers = np.column_stack((points["x"], points["y"], points["z"])).astype(
        np.float32, copy=False
    )
    intensities = np.asarray(points["intensity"], dtype=np.float32)
    if threshold_value >= 1.0:
        alphas = np.ones_like(intensities, dtype=np.float32)
    else:
        fraction = np.clip(
            (intensities - threshold_value) / (1.0 - threshold_value),
            0.0,
            1.0,
        )
        alphas = (
            alpha_minimum + fraction * (1.0 - alpha_minimum)
        ).astype(np.float32, copy=False)

    coordinates = metadata.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("metadata must contain x, y, z coordinate vectors.")
    coordinate_arrays = {axis: np.asarray(coordinates[axis]) for axis in "xyz"}
    voxel_size = np.array(
        [_axis_spacing(coordinate_arrays[axis]) * scale for axis in "xyz"],
        dtype=np.float32,
    )
    bounds = np.array(
        [
            [float(np.min(coordinate_arrays[axis])), float(np.max(coordinate_arrays[axis]))]
            for axis in "xyz"
        ],
        dtype=np.float32,
    )

    base_rgb = _parse_rgb(color)
    background_rgb = _parse_rgb(background_color)
    foreground_rgb = tuple(255 - component for component in background_rgb)
    colormap_lut, colormap_name = _colormap_lut(colormap)
    return {
        "positions_b64": _float32_base64(centers.reshape(-1)),
        "alphas_b64": _float32_base64(alphas),
        "intensities_b64": _float32_base64(intensities),
        "instance_count": int(len(centers)),
        "threshold_count": int(points["threshold_count"]),
        "visualization_downsampled": bool(points["visualization_downsampled"]),
        "voxel_size": voxel_size.tolist(),
        "bounds": bounds.tolist(),
        "base_color": [component / 255.0 for component in base_rgb],
        "use_colormap": colormap_lut is not None,
        "colormap_name": colormap_name,
        "colormap_rgb_b64": (
            _uint8_base64(colormap_lut) if colormap_lut is not None else None
        ),
        "background_css": _rgb_css(background_rgb),
        "foreground_css": _rgb_css(foreground_rgb),
        "threshold": threshold_value,
        "minimum_alpha": alpha_minimum,
    }


def _build_html(
    instances: Mapping[str, Any],
    metadata: Mapping[str, Any],
    three_source: str,
    controls_source: str,
) -> str:
    configuration = {
        **instances,
        "dataset_name": str(metadata.get("dataset_kind", "volume"))
        .replace("_", " ")
        .title(),
        "length_unit": str(metadata.get("length_unit", "")),
    }
    config_json = json.dumps(configuration, separators=(",", ":"))
    three_source = three_source.replace("</script", "<\\/script")
    controls_source = controls_source.replace("</script", "<\\/script")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{configuration['dataset_name']} GPU voxel viewer</title>
<style>
html,body{{width:100%;height:100%;margin:0;overflow:hidden;background:{instances['background_css']};}}
body{{font-family:Arial,sans-serif;color:{instances['foreground_css']};}}
#viewer{{position:fixed;z-index:0;inset:0;overflow:hidden;background:{instances['background_css']};}}
#viewer canvas{{display:block;width:100%;height:100%;touch-action:none;}}
#title{{position:fixed;z-index:10;left:16px;right:16px;top:12px;font-size:16px;line-height:1.35;white-space:normal;overflow-wrap:anywhere;pointer-events:none;}}
#status{{position:fixed;z-index:10;left:16px;right:16px;bottom:12px;font-size:12px;line-height:1.35;white-space:normal;overflow-wrap:anywhere;opacity:.88;pointer-events:none;}}
#error{{display:none;position:fixed;z-index:3;inset:16px;padding:16px;border:1px solid currentColor;white-space:pre-wrap;overflow:auto;}}
</style>
</head>
<body>
<div id="viewer"></div><div id="title"></div><div id="status">Initializing GPU viewer...</div><div id="error"></div>
<script>{three_source}</script>
<script>{controls_source}</script>
<script>
"use strict";
const config={config_json};

function decodeFloat32(encoded){{
  const binary=atob(encoded), bytes=new Uint8Array(binary.length);
  for(let start=0;start<binary.length;start+=1048576){{
    const stop=Math.min(start+1048576,binary.length);
    for(let i=start;i<stop;i++) bytes[i]=binary.charCodeAt(i);
  }}
  return new Float32Array(bytes.buffer);
}}

function decodeUint8(encoded){{
  const binary=atob(encoded), bytes=new Uint8Array(binary.length);
  for(let start=0;start<binary.length;start+=1048576){{
    const stop=Math.min(start+1048576,binary.length);
    for(let i=start;i<stop;i++) bytes[i]=binary.charCodeAt(i);
  }}
  return bytes;
}}

function labelSprite(text,color){{
  const canvas=document.createElement("canvas"); canvas.width=256; canvas.height=64;
  const context=canvas.getContext("2d"); context.font="28px Arial";
  context.fillStyle=color; context.textAlign="center"; context.textBaseline="middle";
  context.fillText(text,128,32);
  const texture=new THREE.CanvasTexture(canvas);
  const sprite=new THREE.Sprite(new THREE.SpriteMaterial({{map:texture,depthTest:false,transparent:true}}));
  sprite.scale.set(1.6,.4,1); return sprite;
}}

try{{
  if(!window.WebGLRenderingContext) throw new Error("WebGL is unavailable in this browser.");
  const host=document.getElementById("viewer");
  const renderer=new THREE.WebGLRenderer({{antialias:true,powerPreference:"high-performance"}});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  renderer.setSize(window.innerWidth,window.innerHeight); renderer.setClearColor(config.background_css,1);
  renderer.outputEncoding=THREE.sRGBEncoding; host.appendChild(renderer.domElement);

  const scene=new THREE.Scene(); scene.background=new THREE.Color(config.background_css);
  const bounds=config.bounds, center=new THREE.Vector3(
    (bounds[0][0]+bounds[0][1])/2,(bounds[1][0]+bounds[1][1])/2,(bounds[2][0]+bounds[2][1])/2);
  const extent=new THREE.Vector3(bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],bounds[2][1]-bounds[2][0]);
  const diagonal=Math.max(extent.length(),1e-3), camera=new THREE.PerspectiveCamera(42,window.innerWidth/window.innerHeight,diagonal/10000,diagonal*100);
  camera.up.set(0,0,1);
  const verticalFov=camera.fov*Math.PI/180;
  const horizontalFov=2*Math.atan(Math.tan(verticalFov/2)*camera.aspect);
  const limitingFov=Math.min(verticalFov,horizontalFov);
  const azimuth=-38*Math.PI/180,elevation=35*Math.PI/180,distance=diagonal*.58/Math.sin(limitingFov/2);
  camera.position.set(center.x+distance*Math.cos(elevation)*Math.cos(azimuth),center.y+distance*Math.cos(elevation)*Math.sin(azimuth),center.z+distance*Math.sin(elevation));
  camera.lookAt(center);

  const controls=new THREE.OrbitControls(camera,renderer.domElement); controls.target.copy(center);
  controls.enableDamping=true; controls.dampingFactor=.08; controls.screenSpacePanning=true;

  const positions=decodeFloat32(config.positions_b64), alphas=decodeFloat32(config.alphas_b64), intensities=decodeFloat32(config.intensities_b64);
  const cube=new THREE.BoxGeometry(config.voxel_size[0],config.voxel_size[1],config.voxel_size[2]);
  const geometry=new THREE.InstancedBufferGeometry(); geometry.index=cube.index;
  geometry.setAttribute("position",cube.attributes.position); geometry.setAttribute("normal",cube.attributes.normal);
  geometry.setAttribute("instanceOffset",new THREE.InstancedBufferAttribute(positions,3));
  geometry.setAttribute("instanceAlpha",new THREE.InstancedBufferAttribute(alphas,1));
  geometry.setAttribute("instanceIntensity",new THREE.InstancedBufferAttribute(intensities,1)); geometry.instanceCount=config.instance_count;

  const fallbackMap=new Uint8Array([255,255,255]);
  const mapBytes=config.use_colormap?decodeUint8(config.colormap_rgb_b64):fallbackMap;
  const mapWidth=config.use_colormap?256:1;
  const colorMap=new THREE.DataTexture(mapBytes,mapWidth,1,THREE.RGBFormat,THREE.UnsignedByteType);
  colorMap.minFilter=THREE.LinearFilter;colorMap.magFilter=THREE.LinearFilter;colorMap.wrapS=THREE.ClampToEdgeWrapping;colorMap.needsUpdate=true;

  const material=new THREE.ShaderMaterial({{
    uniforms:{{baseColor:{{value:new THREE.Color(config.base_color[0],config.base_color[1],config.base_color[2])}},colorMap:{{value:colorMap}},useColorMap:{{value:config.use_colormap?1.0:0.0}}}},
    vertexShader:`attribute vec3 instanceOffset; attribute float instanceAlpha; attribute float instanceIntensity; varying vec3 vNormal; varying float vAlpha; varying float vIntensity;
      void main(){{vNormal=normalize(normalMatrix*normal);vAlpha=instanceAlpha;vIntensity=instanceIntensity;vec3 p=position+instanceOffset;gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.0);}}`,
    fragmentShader:`precision highp float; uniform vec3 baseColor; uniform sampler2D colorMap; uniform float useColorMap; varying vec3 vNormal; varying float vAlpha; varying float vIntensity;
      void main(){{vec3 mapped=texture2D(colorMap,vec2(clamp(vIntensity,0.0,1.0),0.5)).rgb;vec3 voxelColor=mix(baseColor,mapped,step(0.5,useColorMap));vec3 light=normalize(vec3(.45,.65,1.0));float shade=.42+.58*abs(dot(normalize(vNormal),light));gl_FragColor=vec4(voxelColor*shade,vAlpha);}}`,
    transparent:true,depthTest:true,depthWrite:false,side:THREE.FrontSide
  }});
  const voxels=new THREE.Mesh(geometry,material); voxels.frustumCulled=false; scene.add(voxels);

  const origin=new THREE.Vector3(bounds[0][0],bounds[1][0],bounds[2][0]);
  const axisPoints=new Float32Array([
    origin.x,origin.y,origin.z,bounds[0][1],origin.y,origin.z,
    origin.x,origin.y,origin.z,origin.x,bounds[1][1],origin.z,
    origin.x,origin.y,origin.z,origin.x,origin.y,bounds[2][1]]);
  const axisGeometry=new THREE.BufferGeometry(); axisGeometry.setAttribute("position",new THREE.BufferAttribute(axisPoints,3));
  scene.add(new THREE.LineSegments(axisGeometry,new THREE.LineBasicMaterial({{color:config.foreground_css}})));
  const unit=config.length_unit?` (${{config.length_unit}})`:"";
  const labels=[
    ["x"+unit,new THREE.Vector3(bounds[0][1],origin.y,origin.z)],
    ["y"+unit,new THREE.Vector3(origin.x,bounds[1][1],origin.z)],
    ["z"+unit,new THREE.Vector3(origin.x,origin.y,bounds[2][1])]];
  labels.forEach(([text,position])=>{{const sprite=labelSprite(text,config.foreground_css);sprite.position.copy(position);sprite.scale.multiplyScalar(diagonal*.12);scene.add(sprite);}});

  const mapLabel=config.colormap_name?` | colormap: ${{config.colormap_name}}`:"";
  document.getElementById("title").textContent=`${{config.dataset_name}}: intensity >= ${{config.threshold.toFixed(2)}}${{mapLabel}}`;
  const gl=renderer.getContext(), debug=gl.getExtension("WEBGL_debug_renderer_info");
  const gpu=debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
  const countText=config.visualization_downsampled?`${{config.instance_count.toLocaleString()}} of ${{config.threshold_count.toLocaleString()}} voxels`:`${{config.instance_count.toLocaleString()}} voxels`;
  document.getElementById("status").textContent=`${{countText}} | GPU: ${{gpu}}`;

  function resize(){{camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);}}
  window.addEventListener("resize",resize);
  let diagnosticFrame=0;
  function animate(){{
    requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);
    renderer.domElement.dataset.cameraPosition=camera.position.toArray().map(value=>value.toFixed(6)).join(",");
    if(diagnosticFrame++===4){{
      const pixels=new Uint8Array(renderer.domElement.width*renderer.domElement.height*4);
      gl.readPixels(0,0,renderer.domElement.width,renderer.domElement.height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
      let nonblack=0;for(let i=0;i<pixels.length;i+=4)if(pixels[i]||pixels[i+1]||pixels[i+2])nonblack++;
      renderer.domElement.dataset.nonblackPixels=String(nonblack);
      renderer.domElement.dataset.totalPixels=String(renderer.domElement.width*renderer.domElement.height);
    }}
  }}
  animate();
}}catch(error){{
  const box=document.getElementById("error");box.style.display="block";box.textContent=String(error&&error.stack||error);
  document.getElementById("status").textContent="GPU viewer failed to initialize.";
}}
</script>
</body>
</html>
"""


def _load_three_sources() -> tuple[str, str]:
    missing = [path for path in (_THREE_PATH, _CONTROLS_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing vendored Three.js viewer assets: "
            + ", ".join(str(path) for path in missing)
        )
    return (
        _THREE_PATH.read_text(encoding="utf-8"),
        _CONTROLS_PATH.read_text(encoding="utf-8"),
    )


def _float32_base64(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def _uint8_base64(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.uint8)
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def _colormap_lut(
    colormap: str | Sequence[Any] | None,
    size: int = 256,
) -> tuple[np.ndarray | None, str | None]:
    if colormap is None:
        return None, None
    try:
        from plotly.colors import get_colorscale
    except ImportError as exc:
        raise ImportError(
            "Named GPU colormaps require plotly from requirements-cmep.txt."
        ) from exc

    if isinstance(colormap, str):
        name = colormap.strip()
        if not name:
            raise ValueError("colormap must not be empty.")
        try:
            scale = get_colorscale(name)
        except Exception as exc:
            raise ValueError(f"Unknown Plotly colormap: {colormap!r}.") from exc
        label = name
    else:
        values = list(colormap)
        if len(values) < 2:
            raise ValueError("A custom colormap requires at least two colors.")
        if all(isinstance(value, str) for value in values):
            positions = np.linspace(0.0, 1.0, len(values))
            scale = list(zip(positions.tolist(), values))
        else:
            scale = values
        label = "custom"

    try:
        stop_positions = np.asarray([float(stop[0]) for stop in scale])
        stop_colors = np.asarray([_parse_rgb(str(stop[1])) for stop in scale])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "A custom colormap must be colors or [position, color] pairs."
        ) from exc
    if (
        stop_positions.ndim != 1
        or len(stop_positions) < 2
        or not np.isfinite(stop_positions).all()
        or np.any(np.diff(stop_positions) < 0.0)
        or stop_positions[0] > 0.0
        or stop_positions[-1] < 1.0
    ):
        raise ValueError("Colormap positions must increase and span 0 through 1.")
    targets = np.linspace(0.0, 1.0, size)
    channels = [
        np.interp(targets, stop_positions, stop_colors[:, channel])
        for channel in range(3)
    ]
    lut = np.rint(np.column_stack(channels)).astype(np.uint8)
    return lut, label


def _axis_spacing(coordinates: np.ndarray) -> float:
    if coordinates.ndim != 1 or len(coordinates) < 2:
        return 1.0
    differences = np.abs(np.diff(coordinates.astype(np.float64)))
    positive = differences[differences > 0.0]
    return float(np.median(positive)) if len(positive) else 1.0


def _normalized_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return result


def _parse_rgb(color: str) -> tuple[int, int, int]:
    if not isinstance(color, str) or not color.strip():
        raise TypeError("Viewer colors must be non-empty strings.")
    try:
        from PIL import ImageColor

        parsed = ImageColor.getrgb(color.strip())
    except (ImportError, ValueError) as exc:
        raise ValueError(f"Unsupported color value: {color!r}.") from exc
    return tuple(int(component) for component in parsed[:3])


def _rgb_css(rgb: tuple[int, int, int]) -> str:
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
