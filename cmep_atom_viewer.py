"""Standalone GPU/CPU viewers for localized CMEP centers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import webbrowser

import numpy as np

from cmep_atom_localization import AtomLocalizationResult, load_atom_localization
from cmep_gpu_viewer import (
    _colormap_lut,
    _float32_base64,
    _load_three_sources,
    _parse_rgb,
    _rgb_css,
    _uint8_base64,
)


def write_gpu_atom_viewer(
    localization: str | Path | AtomLocalizationResult,
    *,
    output_path: str | Path,
    sphere_radius: float,
    score_threshold: float | None = None,
    color: str = "white",
    colormap: str | Sequence[Any] | None = "magma",
    minimum_sphere_alpha: float = 0.30,
    background_color: str = "black",
    open_browser: bool = True,
) -> Path:
    """Write and optionally open a self-contained localized-center viewer.

    Every localized center meeting ``score_threshold`` is rendered. There is
    deliberately no display-count cap or visualization subsampling.
    ``sphere_radius`` affects display only; it does not alter localization.
    """
    result = _as_localization_result(localization)
    instances = prepare_gpu_atom_instances(
        result,
        sphere_radius=sphere_radius,
        score_threshold=score_threshold,
        color=color,
        colormap=colormap,
        minimum_sphere_alpha=minimum_sphere_alpha,
        background_color=background_color,
    )
    three_source, controls_source = _load_three_sources()
    html = _build_atom_html(instances, result.metadata, three_source, controls_source)

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("output_path must end in .html or .htm.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    if open_browser:
        webbrowser.open_new_tab(destination.as_uri())
    return destination


def write_gpu_atom_marker_viewer(
    localization: str | Path | AtomLocalizationResult,
    *,
    output_path: str | Path,
    marker_size: float = 3.0,
    score_threshold: float | None = None,
    color: str = "white",
    colormap: str | Sequence[Any] | None = "magma",
    minimum_marker_alpha: float = 0.30,
    background_color: str = "black",
    initial_view_padding: float = 0.08,
    initial_zoom_factor: float = 0.78,
    open_browser: bool = True,
) -> Path:
    """Write a WebGL point-marker viewer with no display-count cap.

    Markers remain ``marker_size`` screen pixels wide while the camera moves.
    Every localized center meeting ``score_threshold`` is sent to the GPU.
    """
    result = _as_localization_result(localization)
    markers = prepare_gpu_atom_markers(
        result,
        marker_size=marker_size,
        score_threshold=score_threshold,
        color=color,
        colormap=colormap,
        minimum_marker_alpha=minimum_marker_alpha,
        background_color=background_color,
        initial_view_padding=initial_view_padding,
        initial_zoom_factor=initial_zoom_factor,
    )
    three_source, controls_source = _load_three_sources()
    html = _build_atom_html(markers, result.metadata, three_source, controls_source)

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("output_path must end in .html or .htm.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    if open_browser:
        webbrowser.open_new_tab(destination.as_uri())
    return destination


def write_cpu_atom_marker_viewer(
    localization: str | Path | AtomLocalizationResult,
    *,
    output_path: str | Path,
    marker_size: float = 3.0,
    score_threshold: float | None = None,
    color: str = "white",
    colormap: str | Sequence[Any] | None = "magma",
    minimum_marker_alpha: float = 0.30,
    background_color: str = "black",
    initial_view_padding: float = 0.08,
    initial_zoom_factor: float = 0.78,
    open_browser: bool = True,
) -> Path:
    """Write a Canvas 2D marker viewer that does not require WebGL.

    Every localized center meeting ``score_threshold`` is projected and
    rendered as a screen-space marker. There is no display-count cap or
    visualization subsampling. Rotation, zoom, and pan are performed with
    CPU-side JavaScript and the standard Canvas 2D API.
    """
    result = _as_localization_result(localization)
    markers = prepare_cpu_atom_markers(
        result,
        marker_size=marker_size,
        score_threshold=score_threshold,
        color=color,
        colormap=colormap,
        minimum_marker_alpha=minimum_marker_alpha,
        background_color=background_color,
        initial_view_padding=initial_view_padding,
        initial_zoom_factor=initial_zoom_factor,
    )
    html = _build_cpu_marker_html(markers, result.metadata)

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("output_path must end in .html or .htm.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    if open_browser:
        webbrowser.open_new_tab(destination.as_uri())
    return destination


def prepare_cpu_atom_markers(
    localization: AtomLocalizationResult,
    *,
    marker_size: float,
    score_threshold: float | None,
    color: str,
    colormap: str | Sequence[Any] | None,
    minimum_marker_alpha: float,
    background_color: str,
    initial_view_padding: float,
    initial_zoom_factor: float,
) -> dict[str, Any]:
    """Filter centers and encode attributes for the Canvas 2D viewer."""
    positions = np.asarray(localization.positions_xyz, dtype=np.float64)
    scores = np.asarray(localization.scores, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Localized positions must have shape (N, 3).")
    if scores.shape != (len(positions),):
        raise ValueError("Localized scores must contain one value per position.")

    size = float(marker_size)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("marker_size must be positive and finite.")
    threshold = (
        float(localization.metadata.get("score_threshold", 0.0))
        if score_threshold is None
        else float(score_threshold)
    )
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("score_threshold must be finite and lie in [0, 1].")
    minimum_alpha = float(minimum_marker_alpha)
    if not np.isfinite(minimum_alpha) or not 0.0 <= minimum_alpha <= 1.0:
        raise ValueError("minimum_marker_alpha must be finite and lie in [0, 1].")
    padding = float(initial_view_padding)
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("initial_view_padding must be nonnegative and finite.")
    zoom = float(initial_zoom_factor)
    if not np.isfinite(zoom) or zoom <= 0.0:
        raise ValueError("initial_zoom_factor must be positive and finite.")

    keep = np.isfinite(scores) & (scores >= threshold) & np.all(
        np.isfinite(positions), axis=1
    )
    displayed_positions = positions[keep].astype(np.float32, copy=False)
    displayed_scores = scores[keep].astype(np.float32, copy=False)
    if not len(displayed_positions):
        raise ValueError("No localized centers meet score_threshold.")
    if threshold >= 1.0:
        alphas = np.ones(len(displayed_scores), dtype=np.float32)
    else:
        fraction = np.clip(
            (displayed_scores - threshold) / (1.0 - threshold), 0.0, 1.0
        )
        alphas = (
            minimum_alpha + fraction * (1.0 - minimum_alpha)
        ).astype(np.float32, copy=False)

    bounds = _metadata_bounds(localization.metadata, positions)
    base_rgb = _parse_rgb(color)
    background_rgb = _parse_rgb(background_color)
    foreground_rgb = tuple(255 - component for component in background_rgb)
    colormap_lut, colormap_name = _colormap_lut(colormap)
    return {
        "positions_b64": _float32_base64(displayed_positions.reshape(-1)),
        "scores_b64": _float32_base64(displayed_scores),
        "alphas_b64": _float32_base64(alphas),
        "localized_count": int(len(positions)),
        "marker_count": int(len(displayed_positions)),
        "marker_size": size,
        "score_threshold": threshold,
        "bounds": bounds.tolist(),
        "base_color": list(base_rgb),
        "use_colormap": colormap_lut is not None,
        "colormap_name": colormap_name,
        "colormap_rgb_b64": (
            _uint8_base64(colormap_lut) if colormap_lut is not None else None
        ),
        "minimum_alpha": minimum_alpha,
        "initial_view_padding": padding,
        "initial_zoom_factor": zoom,
        "background_rgb": list(background_rgb),
        "foreground_rgb": list(foreground_rgb),
        "background_css": _rgb_css(background_rgb),
        "foreground_css": _rgb_css(foreground_rgb),
    }


def prepare_gpu_atom_markers(
    localization: AtomLocalizationResult,
    *,
    marker_size: float,
    score_threshold: float | None,
    color: str,
    colormap: str | Sequence[Any] | None,
    minimum_marker_alpha: float,
    background_color: str,
    initial_view_padding: float,
    initial_zoom_factor: float,
) -> dict[str, Any]:
    """Filter centers and encode point attributes for the WebGL viewer."""
    markers = prepare_cpu_atom_markers(
        localization,
        marker_size=marker_size,
        score_threshold=score_threshold,
        color=color,
        colormap=colormap,
        minimum_marker_alpha=minimum_marker_alpha,
        background_color=background_color,
        initial_view_padding=initial_view_padding,
        initial_zoom_factor=initial_zoom_factor,
    )
    markers["base_color"] = [component / 255.0 for component in markers["base_color"]]
    markers["instance_count"] = markers["marker_count"]
    markers["render_mode"] = "markers"
    return markers


def _build_cpu_marker_html(
    markers: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    configuration = {
        **markers,
        "length_unit": str(metadata.get("length_unit", "")),
        "camera_azimuth_degrees": -38.0,
        "camera_elevation_degrees": 35.0,
    }
    config_json = json.dumps(configuration, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Localized Atomic Centers: CPU markers</title>
<style>
html,body{{width:100%;height:100%;margin:0;overflow:hidden;background:{markers['background_css']};}}
body{{font-family:Arial,sans-serif;color:{markers['foreground_css']};}}
#viewer{{position:fixed;inset:0;overflow:hidden;background:{markers['background_css']};}}
#viewer canvas{{display:block;width:100%;height:100%;touch-action:none;cursor:grab;}}
#viewer canvas.dragging{{cursor:grabbing;}}
#title{{position:fixed;z-index:2;left:16px;right:16px;top:12px;font-size:16px;line-height:1.35;overflow-wrap:anywhere;pointer-events:none;}}
#status{{position:fixed;z-index:2;left:16px;right:16px;bottom:12px;font-size:12px;line-height:1.35;overflow-wrap:anywhere;opacity:.88;pointer-events:none;}}
#error{{display:none;position:fixed;z-index:3;inset:16px;padding:16px;border:1px solid currentColor;white-space:pre-wrap;overflow:auto;}}
</style>
</head>
<body>
<div id="viewer"><canvas id="canvas"></canvas></div><div id="title"></div><div id="status">Initializing CPU marker viewer...</div><div id="error"></div>
<script>
"use strict";
const config={config_json};

function decodeFloat32(encoded){{
  const binary=atob(encoded),bytes=new Uint8Array(binary.length);
  for(let start=0;start<binary.length;start+=1048576){{
    const stop=Math.min(start+1048576,binary.length);
    for(let i=start;i<stop;i++)bytes[i]=binary.charCodeAt(i);
  }}
  return new Float32Array(bytes.buffer);
}}
function decodeUint8(encoded){{
  const binary=atob(encoded),bytes=new Uint8Array(binary.length);
  for(let start=0;start<binary.length;start+=1048576){{
    const stop=Math.min(start+1048576,binary.length);
    for(let i=start;i<stop;i++)bytes[i]=binary.charCodeAt(i);
  }}
  return bytes;
}}
function dot(a,b){{return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}}
function cross(a,b){{return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}}
function normalize(v){{const n=Math.hypot(v[0],v[1],v[2])||1;return [v[0]/n,v[1]/n,v[2]/n];}}
function clamp(value,minimum,maximum){{return Math.max(minimum,Math.min(maximum,value));}}

try{{
  const canvas=document.getElementById("canvas"),context=canvas.getContext("2d",{{alpha:false,desynchronized:true}});
  if(!context)throw new Error("Canvas 2D is unavailable in this browser.");
  const positions=decodeFloat32(config.positions_b64),scores=decodeFloat32(config.scores_b64),alphas=decodeFloat32(config.alphas_b64);
  const colorMap=config.use_colormap?decodeUint8(config.colormap_rgb_b64):null;
  const bounds=config.bounds,center=[
    (bounds[0][0]+bounds[0][1])/2,(bounds[1][0]+bounds[1][1])/2,(bounds[2][0]+bounds[2][1])/2];
  const extent=[bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],bounds[2][1]-bounds[2][0]];
  const radius=Math.max(.5*Math.hypot(extent[0],extent[1],extent[2]),1e-6);
  const originalState={{
    azimuth:config.camera_azimuth_degrees*Math.PI/180,
    elevation:config.camera_elevation_degrees*Math.PI/180,
    zoom:config.initial_zoom_factor,
    panX:0,
    panY:0
  }};
  const state={{...originalState}};
  let width=1,height=1,pixelRatio=1,framePending=false,dragging=false,dragMode="rotate",lastX=0,lastY=0;
  const screenX=new Float32Array(config.marker_count),screenY=new Float32Array(config.marker_count),depth=new Float32Array(config.marker_count);
  const order=Array.from({{length:config.marker_count}},(_,index)=>index);
  let sprites=[];

  function makeSprites(){{
    const diameter=Math.max(config.marker_size*pixelRatio,1),spriteExtent=Math.ceil(diameter+4*pixelRatio),spriteRadius=diameter/2;
    sprites=Array.from({{length:256}},(_,index)=>{{
      const sprite=document.createElement("canvas");sprite.width=spriteExtent;sprite.height=spriteExtent;
      const spriteContext=sprite.getContext("2d"),score=index/255;
      const alpha=config.score_threshold>=1?1:config.minimum_alpha+clamp((score-config.score_threshold)/(1-config.score_threshold),0,1)*(1-config.minimum_alpha);
      const red=config.use_colormap?colorMap[index*3]:config.base_color[0];
      const green=config.use_colormap?colorMap[index*3+1]:config.base_color[1];
      const blue=config.use_colormap?colorMap[index*3+2]:config.base_color[2];
      spriteContext.fillStyle=`rgba(${{red}},${{green}},${{blue}},${{alpha}})`;
      spriteContext.beginPath();spriteContext.arc(spriteExtent/2,spriteExtent/2,spriteRadius,0,Math.PI*2);spriteContext.fill();
      return sprite;
    }});
  }}

  function cameraBasis(){{
    const cosine=Math.cos(state.elevation),direction=[cosine*Math.cos(state.azimuth),cosine*Math.sin(state.azimuth),Math.sin(state.elevation)];
    let right=normalize(cross([0,0,1],direction));
    if(Math.abs(state.elevation)>Math.PI/2-1e-5)right=[-Math.sin(state.azimuth),Math.cos(state.azimuth),0];
    return {{direction,right,up:normalize(cross(direction,right))}};
  }}

  function projectPoint(point,basis,distance,focal){{
    const relative=[point[0]-center[0],point[1]-center[1],point[2]-center[2]];
    const toward=dot(relative,basis.direction),cameraDepth=Math.max(distance-toward,1e-6),scale=state.zoom*focal/cameraDepth;
    return [width/2+state.panX+dot(relative,basis.right)*scale,height/2+state.panY-dot(relative,basis.up)*scale,toward];
  }}

  function drawAxes(basis,distance,focal){{
    const origin=[bounds[0][0],bounds[1][0],bounds[2][0]],ends=[
      [bounds[0][1],origin[1],origin[2]],[origin[0],bounds[1][1],origin[2]],[origin[0],origin[1],bounds[2][1]]];
    const labels=["x","y","z"],originScreen=projectPoint(origin,basis,distance,focal),unit=config.length_unit?` (${{config.length_unit}})`:"";
    context.strokeStyle=config.foreground_css;context.fillStyle=config.foreground_css;context.lineWidth=Math.max(pixelRatio,1);context.font=`${{12*pixelRatio}}px Arial`;
    ends.forEach((endpoint,index)=>{{
      const projected=projectPoint(endpoint,basis,distance,focal);
      context.beginPath();context.moveTo(originScreen[0],originScreen[1]);context.lineTo(projected[0],projected[1]);context.stroke();
      context.fillText(labels[index]+unit,projected[0]+5*pixelRatio,projected[1]-5*pixelRatio);
    }});
  }}

  function render(){{
    const started=performance.now(),basis=cameraBasis(),verticalFov=42*Math.PI/180;
    const horizontalFov=2*Math.atan(Math.tan(verticalFov/2)*(width/height)),limitingFov=Math.min(verticalFov,horizontalFov);
    const distance=radius*(1+2*config.initial_view_padding)/Math.sin(limitingFov/2),focal=(height/2)/Math.tan(verticalFov/2);
    context.fillStyle=config.background_css;context.fillRect(0,0,width,height);
    for(let index=0;index<config.marker_count;index++){{
      const offset=index*3,projected=projectPoint([positions[offset],positions[offset+1],positions[offset+2]],basis,distance,focal);
      screenX[index]=projected[0];screenY[index]=projected[1];depth[index]=projected[2];
    }}
    order.sort((left,right)=>depth[left]-depth[right]);
    for(const index of order){{
      const mapIndex=clamp(Math.round(scores[index]*255),0,255),sprite=sprites[mapIndex],half=sprite.width/2;
      context.drawImage(sprite,screenX[index]-half,screenY[index]-half);
    }}
    drawAxes(basis,distance,focal);
    const elapsed=performance.now()-started;
    canvas.dataset.renderedMarkers=String(config.marker_count);canvas.dataset.renderer="canvas2d-cpu";
    canvas.dataset.cameraState=[state.azimuth,state.elevation,state.zoom,state.panX,state.panY].map(value=>value.toFixed(6)).join(",");
    document.getElementById("status").textContent=`${{config.marker_count.toLocaleString()}} of ${{config.localized_count.toLocaleString()}} localized centers | marker size: ${{config.marker_size.toFixed(2)}} px | CPU Canvas 2D | draw: ${{elapsed.toFixed(1)}} ms`;
  }}

  function requestRender(){{
    if(framePending)return;framePending=true;requestAnimationFrame(()=>{{framePending=false;render();}});
  }}
  function resize(){{
    pixelRatio=Math.min(window.devicePixelRatio||1,2);width=Math.max(1,Math.round(window.innerWidth*pixelRatio));height=Math.max(1,Math.round(window.innerHeight*pixelRatio));
    canvas.width=width;canvas.height=height;makeSprites();requestRender();
  }}
  canvas.addEventListener("pointerdown",event=>{{
    event.preventDefault();dragging=true;dragMode=(event.button!==0||event.shiftKey||event.ctrlKey||event.metaKey)?"pan":"rotate";lastX=event.clientX;lastY=event.clientY;canvas.classList.add("dragging");canvas.setPointerCapture(event.pointerId);
  }});
  canvas.addEventListener("pointermove",event=>{{
    if(!dragging)return;const deltaX=event.clientX-lastX,deltaY=event.clientY-lastY;lastX=event.clientX;lastY=event.clientY;
    if(dragMode==="rotate"){{state.azimuth-=deltaX*.006;state.elevation=clamp(state.elevation+deltaY*.006,-Math.PI/2+.01,Math.PI/2-.01);}}
    else{{state.panX+=deltaX*pixelRatio;state.panY+=deltaY*pixelRatio;}}
    requestRender();
  }});
  function endDrag(event){{dragging=false;canvas.classList.remove("dragging");if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);}}
  canvas.addEventListener("pointerup",endDrag);canvas.addEventListener("pointercancel",endDrag);
  canvas.addEventListener("wheel",event=>{{event.preventDefault();state.zoom=clamp(state.zoom*Math.exp(-event.deltaY*.001),.05,20);requestRender();}},{{passive:false}});
  canvas.addEventListener("dblclick",()=>{{Object.assign(state,originalState);requestRender();}});
  canvas.addEventListener("contextmenu",event=>event.preventDefault());window.addEventListener("resize",resize);

  const mapLabel=config.colormap_name?` | colormap: ${{config.colormap_name}}`:"";
  document.getElementById("title").textContent=`Localized Atomic Centers: score >= ${{config.score_threshold.toFixed(3)}} | CPU markers${{mapLabel}}`;
  resize();
}}catch(error){{
  const box=document.getElementById("error");box.style.display="block";box.textContent=String(error&&error.stack||error);
  document.getElementById("status").textContent="CPU marker viewer failed to initialize.";
}}
</script>
</body>
</html>
"""


def prepare_gpu_atom_instances(
    localization: AtomLocalizationResult,
    *,
    sphere_radius: float,
    score_threshold: float | None,
    color: str,
    colormap: str | Sequence[Any] | None,
    minimum_sphere_alpha: float,
    background_color: str,
) -> dict[str, Any]:
    """Filter localized centers by score and encode GPU instance attributes."""
    positions = np.asarray(localization.positions_xyz, dtype=np.float64)
    scores = np.asarray(localization.scores, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Localized positions must have shape (N, 3).")
    if scores.shape != (len(positions),):
        raise ValueError("Localized scores must contain one value per position.")

    radius = float(sphere_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("sphere_radius must be positive and finite.")
    threshold = (
        float(localization.metadata.get("score_threshold", 0.0))
        if score_threshold is None
        else float(score_threshold)
    )
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("score_threshold must be finite and lie in [0, 1].")
    minimum_alpha = float(minimum_sphere_alpha)
    if not np.isfinite(minimum_alpha) or not 0.0 <= minimum_alpha <= 1.0:
        raise ValueError("minimum_sphere_alpha must be finite and lie in [0, 1].")

    keep = np.isfinite(scores) & (scores >= threshold) & np.all(
        np.isfinite(positions), axis=1
    )
    displayed_positions = positions[keep].astype(np.float32, copy=False)
    displayed_scores = scores[keep].astype(np.float32, copy=False)
    if threshold >= 1.0:
        alphas = np.ones(len(displayed_scores), dtype=np.float32)
    else:
        fraction = np.clip(
            (displayed_scores - threshold) / (1.0 - threshold), 0.0, 1.0
        )
        alphas = (
            minimum_alpha + fraction * (1.0 - minimum_alpha)
        ).astype(np.float32, copy=False)

    bounds = _metadata_bounds(localization.metadata, positions)
    base_rgb = _parse_rgb(color)
    background_rgb = _parse_rgb(background_color)
    foreground_rgb = tuple(255 - component for component in background_rgb)
    colormap_lut, colormap_name = _colormap_lut(colormap)
    return {
        "positions_b64": _float32_base64(displayed_positions.reshape(-1)),
        "scores_b64": _float32_base64(displayed_scores),
        "alphas_b64": _float32_base64(alphas),
        "localized_count": int(len(positions)),
        "instance_count": int(len(displayed_positions)),
        "render_mode": "spheres",
        "sphere_radius": radius,
        "initial_view_padding": 0.08,
        "initial_zoom_factor": 1.0,
        "score_threshold": threshold,
        "bounds": bounds.tolist(),
        "base_color": [component / 255.0 for component in base_rgb],
        "use_colormap": colormap_lut is not None,
        "colormap_name": colormap_name,
        "colormap_rgb_b64": (
            _uint8_base64(colormap_lut) if colormap_lut is not None else None
        ),
        "minimum_alpha": minimum_alpha,
        "background_css": _rgb_css(background_rgb),
        "foreground_css": _rgb_css(foreground_rgb),
    }


def _as_localization_result(
    value: str | Path | AtomLocalizationResult,
) -> AtomLocalizationResult:
    if isinstance(value, AtomLocalizationResult) or (
        hasattr(value, "positions_xyz")
        and hasattr(value, "scores")
        and isinstance(getattr(value, "metadata", None), Mapping)
    ):
        return value
    return load_atom_localization(value)


def _metadata_bounds(
    metadata: Mapping[str, Any], positions: np.ndarray
) -> np.ndarray:
    ranges = metadata.get("coordinate_ranges")
    if isinstance(ranges, Mapping) and all(axis in ranges for axis in "xyz"):
        bounds = np.asarray([ranges[axis] for axis in "xyz"], dtype=np.float64)
    elif len(positions):
        bounds = np.column_stack((np.min(positions, axis=0), np.max(positions, axis=0)))
    else:
        raise ValueError("Localization metadata must contain coordinate_ranges.")
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("coordinate_ranges must define finite x, y, z bounds.")
    if np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("Every coordinate range must increase.")
    zero_extent = bounds[:, 1] == bounds[:, 0]
    bounds[zero_extent, 0] -= 0.5
    bounds[zero_extent, 1] += 0.5
    return bounds.astype(np.float32)


def _build_atom_html(
    instances: Mapping[str, Any],
    metadata: Mapping[str, Any],
    three_source: str,
    controls_source: str,
) -> str:
    configuration = {
        **instances,
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
<title>Localized Atomic Centers</title>
<style>
html,body{{width:100%;height:100%;margin:0;overflow:hidden;background:{instances['background_css']};}}
body{{font-family:Arial,sans-serif;color:{instances['foreground_css']};}}
#viewer{{position:fixed;z-index:0;inset:0;overflow:hidden;background:{instances['background_css']};}}
#viewer canvas{{display:block;width:100%;height:100%;touch-action:none;}}
#title{{position:fixed;z-index:10;left:16px;right:16px;top:12px;font-size:16px;line-height:1.35;overflow-wrap:anywhere;pointer-events:none;}}
#status{{position:fixed;z-index:10;left:16px;right:16px;bottom:12px;font-size:12px;line-height:1.35;overflow-wrap:anywhere;opacity:.88;pointer-events:none;}}
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
  const binary=atob(encoded),bytes=new Uint8Array(binary.length);
  for(let start=0;start<binary.length;start+=1048576){{
    const stop=Math.min(start+1048576,binary.length);
    for(let i=start;i<stop;i++)bytes[i]=binary.charCodeAt(i);
  }}
  return new Float32Array(bytes.buffer);
}}
function decodeUint8(encoded){{
  const binary=atob(encoded),bytes=new Uint8Array(binary.length);
  for(let start=0;start<binary.length;start+=1048576){{
    const stop=Math.min(start+1048576,binary.length);
    for(let i=start;i<stop;i++)bytes[i]=binary.charCodeAt(i);
  }}
  return bytes;
}}
function labelSprite(text,color){{
  const canvas=document.createElement("canvas");canvas.width=256;canvas.height=64;
  const context=canvas.getContext("2d");context.font="28px Arial";
  context.fillStyle=color;context.textAlign="center";context.textBaseline="middle";context.fillText(text,128,32);
  const texture=new THREE.CanvasTexture(canvas);
  const sprite=new THREE.Sprite(new THREE.SpriteMaterial({{map:texture,depthTest:false,transparent:true}}));
  sprite.scale.set(1.6,.4,1);return sprite;
}}

try{{
  if(!window.WebGLRenderingContext)throw new Error("WebGL is unavailable in this browser.");
  const host=document.getElementById("viewer");
  const renderer=new THREE.WebGLRenderer({{antialias:true,powerPreference:"high-performance",alpha:false}});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  renderer.setSize(window.innerWidth,window.innerHeight);renderer.setClearColor(config.background_css,1);
  renderer.outputEncoding=THREE.sRGBEncoding;host.appendChild(renderer.domElement);

  const scene=new THREE.Scene();scene.background=new THREE.Color(config.background_css);
  const bounds=config.bounds,center=new THREE.Vector3(
    (bounds[0][0]+bounds[0][1])/2,(bounds[1][0]+bounds[1][1])/2,(bounds[2][0]+bounds[2][1])/2);
  const extent=new THREE.Vector3(bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],bounds[2][1]-bounds[2][0]);
  const primitiveExtent=config.render_mode==="spheres"?config.sphere_radius*4:1e-3;
  const diagonal=Math.max(extent.length(),primitiveExtent,1e-3);
  const camera=new THREE.PerspectiveCamera(42,window.innerWidth/window.innerHeight,Math.max(diagonal/10000,1e-6),diagonal*100);
  camera.up.set(0,0,1);
  const verticalFov=camera.fov*Math.PI/180,horizontalFov=2*Math.atan(Math.tan(verticalFov/2)*camera.aspect);
  const limitingFov=Math.min(verticalFov,horizontalFov),azimuth=-38*Math.PI/180,elevation=35*Math.PI/180;
  const distance=diagonal*.5*(1+2*config.initial_view_padding)/(Math.sin(limitingFov/2)*config.initial_zoom_factor);
  camera.position.set(center.x+distance*Math.cos(elevation)*Math.cos(azimuth),center.y+distance*Math.cos(elevation)*Math.sin(azimuth),center.z+distance*Math.sin(elevation));
  camera.lookAt(center);
  const controls=new THREE.OrbitControls(camera,renderer.domElement);controls.target.copy(center);
  controls.enableDamping=true;controls.dampingFactor=.08;controls.screenSpacePanning=true;

  const positions=decodeFloat32(config.positions_b64),scores=decodeFloat32(config.scores_b64),alphas=decodeFloat32(config.alphas_b64);
  const fallbackMap=new Uint8Array([255,255,255]),mapBytes=config.use_colormap?decodeUint8(config.colormap_rgb_b64):fallbackMap;
  const mapWidth=config.use_colormap?256:1,colorMap=new THREE.DataTexture(mapBytes,mapWidth,1,THREE.RGBFormat,THREE.UnsignedByteType);
  colorMap.minFilter=THREE.LinearFilter;colorMap.magFilter=THREE.LinearFilter;colorMap.wrapS=THREE.ClampToEdgeWrapping;colorMap.needsUpdate=true;
  let atomMaterial,atoms;
  if(config.render_mode==="markers"){{
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute("position",new THREE.BufferAttribute(positions,3));
    geometry.setAttribute("pointScore",new THREE.BufferAttribute(scores,1));
    geometry.setAttribute("pointAlpha",new THREE.BufferAttribute(alphas,1));
    atomMaterial=new THREE.ShaderMaterial({{
      uniforms:{{baseColor:{{value:new THREE.Color(config.base_color[0],config.base_color[1],config.base_color[2])}},colorMap:{{value:colorMap}},useColorMap:{{value:config.use_colormap?1.0:0.0}},markerSize:{{value:config.marker_size}},pixelRatio:{{value:renderer.getPixelRatio()}}}},
      vertexShader:`attribute float pointScore;attribute float pointAlpha;uniform float markerSize;uniform float pixelRatio;varying float vScore;varying float vAlpha;
        void main(){{vScore=pointScore;vAlpha=pointAlpha;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=max(markerSize*pixelRatio,1.0);}}`,
      fragmentShader:`precision highp float;uniform vec3 baseColor;uniform sampler2D colorMap;uniform float useColorMap;varying float vScore;varying float vAlpha;
        void main(){{float radial=length(gl_PointCoord-vec2(.5));if(radial>.5)discard;float edgeAlpha=1.0-smoothstep(.40,.5,radial);vec3 mapped=texture2D(colorMap,vec2(clamp(vScore,0.0,1.0),0.5)).rgb;vec3 markerColor=mix(baseColor,mapped,step(.5,useColorMap));gl_FragColor=vec4(markerColor,vAlpha*edgeAlpha);}}`,
      transparent:true,depthTest:true,depthWrite:false
    }});
    atoms=new THREE.Points(geometry,atomMaterial);
  }}else{{
    const sphere=new THREE.SphereGeometry(config.sphere_radius,16,12);
    const geometry=new THREE.InstancedBufferGeometry();geometry.index=sphere.index;
    geometry.setAttribute("position",sphere.attributes.position);geometry.setAttribute("normal",sphere.attributes.normal);
    geometry.setAttribute("instanceOffset",new THREE.InstancedBufferAttribute(positions,3));
    geometry.setAttribute("instanceScore",new THREE.InstancedBufferAttribute(scores,1));
    geometry.setAttribute("instanceAlpha",new THREE.InstancedBufferAttribute(alphas,1));geometry.instanceCount=config.instance_count;
    atomMaterial=new THREE.ShaderMaterial({{
      uniforms:{{baseColor:{{value:new THREE.Color(config.base_color[0],config.base_color[1],config.base_color[2])}},colorMap:{{value:colorMap}},useColorMap:{{value:config.use_colormap?1.0:0.0}}}},
      vertexShader:`attribute vec3 instanceOffset;attribute float instanceScore;attribute float instanceAlpha;varying vec3 vNormal;varying float vScore;varying float vAlpha;
        void main(){{vNormal=normalize(normalMatrix*normal);vScore=instanceScore;vAlpha=instanceAlpha;gl_Position=projectionMatrix*modelViewMatrix*vec4(position+instanceOffset,1.0);}}`,
      fragmentShader:`precision highp float;uniform vec3 baseColor;uniform sampler2D colorMap;uniform float useColorMap;varying vec3 vNormal;varying float vScore;varying float vAlpha;
        void main(){{vec3 mapped=texture2D(colorMap,vec2(clamp(vScore,0.0,1.0),0.5)).rgb;vec3 sphereColor=mix(baseColor,mapped,step(0.5,useColorMap));vec3 light=normalize(vec3(.45,.65,1.0));float diffuse=max(dot(normalize(vNormal),light),0.0);float shade=.34+.66*diffuse;gl_FragColor=vec4(sphereColor*shade,vAlpha);}}`,
      transparent:true,depthTest:true,depthWrite:false,side:THREE.FrontSide
    }});
    atoms=new THREE.Mesh(geometry,atomMaterial);
  }}
  atoms.frustumCulled=false;scene.add(atoms);

  const origin=new THREE.Vector3(bounds[0][0],bounds[1][0],bounds[2][0]);
  const axisPoints=new Float32Array([
    origin.x,origin.y,origin.z,bounds[0][1],origin.y,origin.z,
    origin.x,origin.y,origin.z,origin.x,bounds[1][1],origin.z,
    origin.x,origin.y,origin.z,origin.x,origin.y,bounds[2][1]]);
  const axisGeometry=new THREE.BufferGeometry();axisGeometry.setAttribute("position",new THREE.BufferAttribute(axisPoints,3));
  scene.add(new THREE.LineSegments(axisGeometry,new THREE.LineBasicMaterial({{color:config.foreground_css}})));
  const unit=config.length_unit?` (${{config.length_unit}})`:"";
  const labels=[
    ["x"+unit,new THREE.Vector3(bounds[0][1],origin.y,origin.z)],
    ["y"+unit,new THREE.Vector3(origin.x,bounds[1][1],origin.z)],
    ["z"+unit,new THREE.Vector3(origin.x,origin.y,bounds[2][1])]];
  labels.forEach(([text,position])=>{{const sprite=labelSprite(text,config.foreground_css);sprite.position.copy(position);sprite.scale.multiplyScalar(diagonal*.12);scene.add(sprite);}});

  const mapLabel=config.colormap_name?` | colormap: ${{config.colormap_name}}`:"";
  const modeLabel=config.render_mode==="markers"?"GPU markers":"GPU spheres";
  document.getElementById("title").textContent=`Localized Atomic Centers: score >= ${{config.score_threshold.toFixed(3)}} | ${{modeLabel}}${{mapLabel}}`;
  const gl=renderer.getContext(),debug=gl.getExtension("WEBGL_debug_renderer_info");
  const gpu=debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
  const sizeLabel=config.render_mode==="markers"?`marker size: ${{config.marker_size.toFixed(2)}} px`:`radius: ${{config.sphere_radius.toPrecision(4)}} ${{config.length_unit}}`;
  document.getElementById("status").textContent=`${{config.instance_count.toLocaleString()}} of ${{config.localized_count.toLocaleString()}} localized centers | ${{sizeLabel}} | GPU: ${{gpu}}`;

  function resize(){{renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);if(config.render_mode==="markers")atomMaterial.uniforms.pixelRatio.value=renderer.getPixelRatio();}}
  window.addEventListener("resize",resize);
  let diagnosticFrame=0;
  function animate(){{
    requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);
    renderer.domElement.dataset.cameraPosition=camera.position.toArray().map(value=>value.toFixed(6)).join(",");
    renderer.domElement.dataset.renderMode=config.render_mode;
    renderer.domElement.dataset.renderedInstances=String(config.instance_count);
    if(config.render_mode==="markers")renderer.domElement.dataset.renderedMarkers=String(config.instance_count);
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
