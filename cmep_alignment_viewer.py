"""Browser-based 2D calibration and translation viewer for CMEP volumes."""

from __future__ import annotations

import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np

from cmep_alignment import (
    DATASETS,
    apply_calibration_preview,
    create_alignment_state,
    load_alignment_state,
    plane_axes,
    preview_calibration,
    rotate_dataset_in_plane,
    save_alignment_state,
    slice_transformed_volume,
    transformed_geometric_center,
    translate_dataset_in_plane,
    union_depth_range,
    validate_alignment_state,
    validate_plane,
)


_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "three-r128"
_THREE_PATH = _VENDOR_DIR / "three.min.js"


class AlignmentViewer:
    """A local HTTP viewer backed by live canonical NumPy volumes."""

    def __init__(
        self,
        volumes: Mapping[str, np.ndarray],
        metadata: Mapping[str, Mapping[str, Any]],
        *,
        state: Mapping[str, Any],
        state_path: str | Path,
        colors: Mapping[str, str],
        threshold_initial: float,
        threshold_min: float,
        threshold_max: float,
        threshold_count: int,
        depth_positions: int,
        minimum_voxel_alpha: float,
        background_color: str,
        voxel_scale: float,
        rotation_min_degrees: float = -180.0,
        rotation_max_degrees: float = 180.0,
        rotation_step_degrees: float = 0.25,
    ) -> None:
        self.volumes = {name: np.asarray(volumes[name]) for name in DATASETS}
        self.metadata = {name: metadata[name] for name in DATASETS}
        self.state = validate_alignment_state(state)
        self.state_path = Path(state_path).expanduser().resolve()
        self.colors = {name: _rgb_css(_parse_rgb(colors[name])) for name in DATASETS}
        self.thresholds = _threshold_values(
            threshold_initial, threshold_min, threshold_max, threshold_count
        )
        self.threshold_initial = float(threshold_initial)
        self.depth_positions = _positive_integer(depth_positions, "depth_positions")
        self.minimum_voxel_alpha = _normalized_scalar(
            minimum_voxel_alpha, "minimum_voxel_alpha"
        )
        self.background_css = _rgb_css(_parse_rgb(background_color))
        background_rgb = _parse_rgb(background_color)
        self.foreground_css = _rgb_css(tuple(255 - value for value in background_rgb))
        self.voxel_scale = _positive_finite(voxel_scale, "voxel_scale")
        self.rotation_min_degrees = _finite_scalar(
            rotation_min_degrees, "rotation_min_degrees"
        )
        self.rotation_max_degrees = _finite_scalar(
            rotation_max_degrees, "rotation_max_degrees"
        )
        self.rotation_step_degrees = _positive_finite(
            rotation_step_degrees, "rotation_step_degrees"
        )
        if self.rotation_min_degrees >= self.rotation_max_degrees:
            raise ValueError("rotation_min_degrees must be less than rotation_max_degrees.")
        if not self.rotation_min_degrees <= 0.0 <= self.rotation_max_degrees:
            raise ValueError("The rotation range must include zero degrees.")
        self._lock = threading.RLock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url: str | None = None

    def start(self, *, open_browser: bool = True) -> "AlignmentViewer":
        """Start the localhost viewer server."""
        if self._server is not None:
            return self
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                viewer._handle_get(self)

            def do_POST(self) -> None:  # noqa: N802
                viewer._handle_post(self)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{port}/"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="cmep-alignment-viewer",
            daemon=True,
        )
        self._thread.start()
        if open_browser:
            webbrowser.open_new_tab(self.url)
        return self

    def close(self) -> None:
        """Stop the local server and release its socket."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def __repr__(self) -> str:
        status = self.url if self.url else "not started"
        return f"AlignmentViewer({status!r}, state_path={str(self.state_path)!r})"

    def _handle_get(self, request: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(request.path)
        try:
            if parsed.path == "/":
                _send_bytes(request, 200, "text/html; charset=utf-8", self._html())
                return
            if parsed.path == "/api/state":
                with self._lock:
                    payload = deepcopy(self.state)
                _send_json(request, 200, payload)
                return
            if parsed.path == "/api/slice":
                query = parse_qs(parsed.query)
                payload = self._slice_payload(query)
                _send_json(request, 200, payload)
                return
            _send_json(request, 404, {"error": "Not found."})
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return
        except Exception as exc:  # Keep browser errors informative.
            try:
                _send_json(request, 400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

    def _handle_post(self, request: BaseHTTPRequestHandler) -> None:
        try:
            length = int(request.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("Request body is too large.")
            body = json.loads(request.rfile.read(length) or b"{}")
            if not isinstance(body, Mapping):
                raise ValueError("Request body must be a JSON object.")

            if request.path == "/api/translate":
                with self._lock:
                    self.state = translate_dataset_in_plane(
                        self.state,
                        dataset=str(body.get("dataset")),
                        plane=str(body.get("plane")),
                        horizontal_delta=body.get("horizontal_delta"),
                        vertical_delta=body.get("vertical_delta"),
                        depth=body.get("depth"),
                    )
                    payload = deepcopy(self.state)
                _send_json(request, 200, payload)
                return

            if request.path == "/api/rotate":
                dataset = str(body.get("dataset"))
                with self._lock:
                    if dataset not in DATASETS:
                        raise ValueError(f"dataset must be one of {DATASETS}.")
                    self.state = rotate_dataset_in_plane(
                        self.state,
                        self.metadata[dataset],
                        dataset=dataset,
                        plane=str(body.get("plane")),
                        angle_degrees=body.get("angle_degrees"),
                        depth=body.get("depth"),
                    )
                    payload = deepcopy(self.state)
                _send_json(request, 200, payload)
                return

            if request.path == "/api/calibration-preview":
                with self._lock:
                    payload = preview_calibration(
                        self.state,
                        plane=str(body.get("plane")),
                        depth=body.get("depth"),
                        target_distance=body.get("target_distance"),
                        points=body.get("points", {}),
                    )
                _send_json(request, 200, payload)
                return

            if request.path == "/api/calibrate":
                with self._lock:
                    preview = preview_calibration(
                        self.state,
                        plane=str(body.get("plane")),
                        depth=body.get("depth"),
                        target_distance=body.get("target_distance"),
                        points=body.get("points", {}),
                    )
                    self.state = apply_calibration_preview(self.state, preview)
                    payload = {
                        "state": deepcopy(self.state),
                        "measurements": preview["measurements"],
                    }
                _send_json(request, 200, payload)
                return

            if request.path == "/api/save":
                with self._lock:
                    self.state["ui"] = {
                        "plane": validate_plane(str(body.get("plane"))),
                        "depth": float(body.get("depth")),
                    }
                    destination, snapshot = save_alignment_state(
                        self.state, self.state_path, keep_snapshot=True
                    )
                _send_json(
                    request,
                    200,
                    {
                        "state_path": str(destination),
                        "snapshot_path": str(snapshot) if snapshot else None,
                    },
                )
                return
            _send_json(request, 404, {"error": "Not found."})
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                _send_json(request, 400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

    def _slice_payload(self, query: Mapping[str, list[str]]) -> dict[str, Any]:
        plane = validate_plane(_query_value(query, "plane", self.state["ui"]["plane"]))
        with self._lock:
            state = deepcopy(self.state)
        depth_min, depth_max = union_depth_range(self.metadata, state, plane)
        depth_values = np.linspace(
            depth_min, depth_max, self.depth_positions, dtype=np.float64
        )
        if "depth_value" in query:
            requested = float(_query_value(query, "depth_value", "0"))
            depth_index = int(np.argmin(np.abs(depth_values - requested)))
        else:
            depth_index = int(_query_value(query, "depth_index", "0"))
            depth_index = min(max(depth_index, 0), self.depth_positions - 1)
        depth = float(depth_values[depth_index])
        threshold_floor = float(
            _query_value(query, "threshold_floor", str(float(self.thresholds.min())))
        )
        datasets = {}
        hull_points = []
        vertical, horizontal, depth_axis = plane_axes(plane)
        vertical_index = "xyz".index(vertical)
        horizontal_index = "xyz".index(horizontal)
        for dataset in DATASETS:
            sliced = slice_transformed_volume(
                self.volumes[dataset],
                self.metadata[dataset],
                state["transforms"][dataset],
                plane=plane,
                depth=depth,
                threshold_floor=threshold_floor,
            )
            positions = np.column_stack(
                (sliced["horizontal"], sliced["vertical"])
            ).astype(np.float32, copy=False)
            center = transformed_geometric_center(
                self.metadata[dataset], state["transforms"][dataset]
            )
            datasets[dataset] = {
                "positions_b64": _float32_base64(positions.reshape(-1)),
                "intensities_b64": _float32_base64(sliced["intensity"]),
                "displayed_point_count": sliced["displayed_point_count"],
                "sampled_point_count": sliced["sampled_point_count"],
                "voxel_size": [
                    float(sliced["voxel_size"][0]) * self.voxel_scale,
                    float(sliced["voxel_size"][1]) * self.voxel_scale,
                ],
                "hull": sliced["hull"],
                "transform": state["transforms"][dataset],
                "rotation_center": [
                    float(center[horizontal_index]),
                    float(center[vertical_index]),
                ],
                "rotation_center_xyz": center.tolist(),
            }
            hull_points.extend(sliced["hull"])
        bounds = _points_bounds(hull_points)
        horizontal_basis = np.eye(3, dtype=np.float64)[horizontal_index]
        vertical_basis = np.eye(3, dtype=np.float64)[vertical_index]
        depth_basis = np.eye(3, dtype=np.float64)["xyz".index(depth_axis)]
        rotation_screen_sign = float(
            np.dot(np.cross(horizontal_basis, vertical_basis), depth_basis)
        )
        return {
            "plane": plane,
            "vertical_axis": vertical,
            "horizontal_axis": horizontal,
            "depth_axis": depth_axis,
            "depth": depth,
            "depth_index": depth_index,
            "depth_count": self.depth_positions,
            "depth_min": float(depth_min),
            "depth_max": float(depth_max),
            "rotation_screen_sign": rotation_screen_sign,
            "bounds": bounds,
            "datasets": datasets,
        }

    def _html(self) -> bytes:
        if not _THREE_PATH.is_file():
            raise FileNotFoundError(f"Missing vendored Three.js asset: {_THREE_PATH}")
        source = _THREE_PATH.read_text(encoding="utf-8").replace("</script", "<\\/script")
        config = {
            "colors": self.colors,
            "thresholds": self.thresholds.tolist(),
            "threshold_initial": self.threshold_initial,
            "threshold_floor": float(self.thresholds.min()),
            "minimum_alpha": self.minimum_voxel_alpha,
            "background": self.background_css,
            "foreground": self.foreground_css,
            "length_unit": self.state["length_unit"],
            "initial_plane": self.state["ui"]["plane"],
            "initial_depth": self.state["ui"]["depth"],
            "rotation_min_degrees": self.rotation_min_degrees,
            "rotation_max_degrees": self.rotation_max_degrees,
            "rotation_step_degrees": self.rotation_step_degrees,
        }
        html = _HTML_TEMPLATE.replace("__THREE_SOURCE__", source).replace(
            "__CONFIG__", json.dumps(config, separators=(",", ":"))
        )
        return html.encode("utf-8")


def launch_alignment_viewer(
    volume_plan: np.ndarray,
    meta_plan: Mapping[str, Any],
    volume_cross: np.ndarray,
    meta_cross: Mapping[str, Any],
    *,
    state_path: str | Path,
    initial_plane: str = "yx",
    color_plan: str = "lightcoral",
    color_cross: str = "lightblue",
    threshold_initial: float = 0.50,
    threshold_min: float = 0.20,
    threshold_max: float = 0.80,
    threshold_count: int = 1,
    depth_positions: int = 50,
    minimum_voxel_alpha: float = 0.30,
    background_color: str = "black",
    voxel_scale: float = 1.0,
    rotation_min_degrees: float = -180.0,
    rotation_max_degrees: float = 180.0,
    rotation_step_degrees: float = 0.25,
    load_saved_state: bool = True,
    open_browser: bool = True,
) -> AlignmentViewer:
    """Launch the 2D physical-slice alignment viewer in a browser tab."""
    unit_plan = str(meta_plan.get("length_unit", "")).strip()
    unit_cross = str(meta_cross.get("length_unit", "")).strip()
    if not unit_plan or unit_plan != unit_cross:
        raise ValueError("Plan and cross metadata must use the same non-empty length_unit.")
    state_file = Path(state_path).expanduser().resolve()
    if load_saved_state and state_file.is_file():
        state = load_alignment_state(state_file, length_unit=unit_plan)
    else:
        state = create_alignment_state(unit_plan)
        state["ui"]["plane"] = validate_plane(initial_plane)
    viewer = AlignmentViewer(
        {"plan": volume_plan, "cross": volume_cross},
        {"plan": meta_plan, "cross": meta_cross},
        state=state,
        state_path=state_file,
        colors={"plan": color_plan, "cross": color_cross},
        threshold_initial=threshold_initial,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_count=threshold_count,
        depth_positions=depth_positions,
        minimum_voxel_alpha=minimum_voxel_alpha,
        background_color=background_color,
        voxel_scale=voxel_scale,
        rotation_min_degrees=rotation_min_degrees,
        rotation_max_degrees=rotation_max_degrees,
        rotation_step_degrees=rotation_step_degrees,
    )
    return viewer.start(open_browser=open_browser)


def _threshold_values(initial: float, minimum: float, maximum: float, count: int) -> np.ndarray:
    initial_value = _normalized_scalar(initial, "threshold_initial")
    minimum_value = _normalized_scalar(minimum, "threshold_min")
    maximum_value = _normalized_scalar(maximum, "threshold_max")
    if minimum_value > maximum_value:
        raise ValueError("threshold_min must not exceed threshold_max.")
    if not minimum_value <= initial_value <= maximum_value:
        raise ValueError("threshold_initial must lie within the configured threshold range.")
    sample_count = _positive_integer(count, "threshold_count")
    if sample_count == 1:
        return np.array([initial_value], dtype=np.float64)
    return np.linspace(minimum_value, maximum_value, sample_count, dtype=np.float64)


def _send_json(request: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    _send_bytes(
        request,
        status,
        "application/json; charset=utf-8",
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def _send_bytes(
    request: BaseHTTPRequestHandler, status: int, content_type: str, payload: bytes
) -> None:
    request.send_response(status)
    request.send_header("Content-Type", content_type)
    request.send_header("Content-Length", str(len(payload)))
    request.send_header("Cache-Control", "no-store")
    request.end_headers()
    request.wfile.write(payload)


def _query_value(query: Mapping[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    return values[0] if values else default


def _float32_base64(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def _points_bounds(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return [[-1.0, 1.0], [-1.0, 1.0]]
    values = np.asarray(points, dtype=np.float64)
    return [
        [float(values[:, 0].min()), float(values[:, 0].max())],
        [float(values[:, 1].min()), float(values[:, 1].max())],
    ]


def _parse_rgb(color: str) -> tuple[int, int, int]:
    if not isinstance(color, str) or not color.strip():
        raise TypeError("Viewer colors must be non-empty strings.")
    try:
        from PIL import ImageColor

        return tuple(int(value) for value in ImageColor.getrgb(color.strip())[:3])
    except (ImportError, ValueError) as exc:
        raise ValueError(f"Unsupported color value: {color!r}.") from exc


def _rgb_css(rgb: tuple[int, int, int]) -> str:
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _normalized_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMEP initial alignment</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden}
body{font-family:Arial,sans-serif;background:var(--bg);color:var(--fg);letter-spacing:0}
#viewer{position:fixed;inset:0;z-index:0}canvas{display:block;width:100%;height:100%;touch-action:none}
#toolbar{position:fixed;z-index:10;top:12px;left:12px;right:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;pointer-events:none}
#toolbar>*{pointer-events:auto}.segment{display:flex;border:1px solid color-mix(in srgb,var(--fg) 38%,transparent);border-radius:6px;overflow:hidden;background:color-mix(in srgb,var(--bg) 88%,transparent)}
button{height:34px;border:1px solid color-mix(in srgb,var(--fg) 38%,transparent);background:var(--bg);color:var(--fg);padding:0 12px;font-weight:600;cursor:pointer}
.segment button{border:0;border-right:1px solid color-mix(in srgb,var(--fg) 28%,transparent);border-radius:0}.segment button:last-child{border-right:0}
button.active{background:var(--fg);color:var(--bg)}button:disabled{opacity:.45;cursor:default}
#readout{margin-left:auto;font-size:13px;padding:7px 9px;background:color-mix(in srgb,var(--bg) 88%,transparent);border:1px solid color-mix(in srgb,var(--fg) 28%,transparent);border-radius:6px}
#sliders{position:fixed;z-index:10;left:12px;right:12px;bottom:12px;display:grid;gap:7px;pointer-events:none}
.slider-row{display:grid;grid-template-columns:88px minmax(120px,1fr) 180px;align-items:center;gap:10px;font-size:13px;pointer-events:auto;background:color-mix(in srgb,var(--bg) 88%,transparent);border:1px solid color-mix(in srgb,var(--fg) 28%,transparent);border-radius:6px;padding:7px 9px}
input[type=range]{width:100%;accent-color:var(--fg)}#status{position:fixed;z-index:10;left:12px;bottom:164px;max-width:min(720px,calc(100% - 24px));font-size:13px;line-height:1.35;padding:7px 9px;background:color-mix(in srgb,var(--bg) 88%,transparent);border-radius:6px;pointer-events:none}
#axis-h{position:fixed;z-index:5;right:14px;bottom:168px;font-size:13px;pointer-events:none}#axis-v{position:fixed;z-index:5;left:14px;top:58px;font-size:13px;pointer-events:none}
dialog{width:min(430px,calc(100% - 32px));border:1px solid var(--fg);border-radius:6px;background:var(--bg);color:var(--fg);padding:18px}dialog::backdrop{background:rgba(0,0,0,.58)}
dialog h2{font-size:17px;margin:0 0 14px}dialog label{display:grid;gap:7px;font-size:13px}dialog input{height:36px;border:1px solid color-mix(in srgb,var(--fg) 45%,transparent);background:var(--bg);color:var(--fg);padding:0 9px}.dialog-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}
@media(max-width:720px){#readout{width:100%;margin-left:0}.slider-row{grid-template-columns:70px 1fr}.slider-value{grid-column:2}.toolbar-action{padding:0 8px}#status{bottom:205px}#axis-h{bottom:209px}}
</style>
</head>
<body>
<div id="viewer"></div>
<div id="toolbar">
  <div class="segment" id="planes"><button data-plane="yx">YX</button><button data-plane="zx">ZX</button><button data-plane="zy">ZY</button></div>
  <div class="segment" id="datasets"><button data-dataset="plan" class="active">Plan</button><button data-dataset="cross">Cross</button></div>
  <button id="calibrate" class="toolbar-action">Calibrate Direction</button>
  <button id="save" class="toolbar-action">Save State</button>
  <div id="readout">Loading...</div>
</div>
<div id="axis-v"></div><div id="axis-h"></div>
<div id="status">Loading physical slices...</div>
<div id="sliders">
  <div class="slider-row"><span>Depth</span><input id="depth" type="range" min="0" value="0"><span class="slider-value" id="depth-value"></span></div>
  <div class="slider-row"><span>Rotation</span><input id="rotation" type="range" value="0"><span class="slider-value" id="rotation-value"></span></div>
  <div class="slider-row" id="threshold-row"><span>Threshold</span><input id="threshold" type="range" min="0" value="0"><span class="slider-value" id="threshold-value"></span></div>
</div>
<dialog id="distance-dialog"><form method="dialog"><h2>Calibrate Direction</h2><label>How much distance are you calibrating?<input id="distance" type="number" min="0" step="any" required></label><div class="dialog-actions"><button value="cancel">Cancel</button><button id="begin-calibration" value="default">Begin</button></div></form></dialog>
<dialog id="preview-dialog"><h2>Calibration Preview</h2><div id="preview-text"></div><div class="dialog-actions"><button id="cancel-preview">Cancel</button><button id="apply-preview">Apply</button></div></dialog>
<script>__THREE_SOURCE__</script>
<script>
"use strict";
const config=__CONFIG__;
document.documentElement.style.setProperty("--bg",config.background);
document.documentElement.style.setProperty("--fg",config.foreground);
const host=document.getElementById("viewer"),statusBox=document.getElementById("status"),readout=document.getElementById("readout");
const depthSlider=document.getElementById("depth"),rotationSlider=document.getElementById("rotation"),thresholdSlider=document.getElementById("threshold"),thresholdRow=document.getElementById("threshold-row");
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance"});renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.setClearColor(config.background,1);host.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(config.background);const camera=new THREE.OrthographicCamera(-1,1,1,-1,-100,100);camera.position.z=10;
const layers={plan:null,cross:null},markers=new THREE.Group();scene.add(markers);let axes=null,current=null,activeDataset="plan",plane=config.initial_plane,threshold=config.threshold_initial;
let calibration=null,preview=null,drag=null,loadController=null,loadTimer=null,firstLoad=true,rotationBusy=false;
const rotationSessions={};

function decodeFloat32(encoded){const binary=atob(encoded),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);return new Float32Array(bytes.buffer)}
function makeLayer(name,data,z){const positions=decodeFloat32(data.positions_b64),intensities=decodeFloat32(data.intensities_b64);const geometry=new THREE.InstancedBufferGeometry();
  geometry.setAttribute("position",new THREE.Float32BufferAttribute([-.5,-.5,0,.5,-.5,0,.5,.5,0,-.5,.5,0],3));geometry.setIndex([0,1,2,0,2,3]);geometry.setAttribute("offset",new THREE.InstancedBufferAttribute(positions,2));geometry.setAttribute("intensity",new THREE.InstancedBufferAttribute(intensities,1));geometry.instanceCount=intensities.length;
  const material=new THREE.ShaderMaterial({uniforms:{baseColor:{value:new THREE.Color(config.colors[name])},threshold:{value:threshold},minimumAlpha:{value:config.minimum_alpha},voxelSize:{value:new THREE.Vector2(data.voxel_size[0],data.voxel_size[1])}},
    vertexShader:`attribute vec2 offset;attribute float intensity;uniform vec2 voxelSize;varying float vIntensity;void main(){vIntensity=intensity;vec3 p=position;p.xy=p.xy*voxelSize+offset;gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.0);}`,
    fragmentShader:`precision highp float;uniform vec3 baseColor;uniform float threshold;uniform float minimumAlpha;varying float vIntensity;void main(){if(vIntensity<threshold)discard;float f=threshold>=1.0?1.0:clamp((vIntensity-threshold)/(1.0-threshold),0.0,1.0);gl_FragColor=vec4(baseColor,mix(minimumAlpha,1.0,f));}`,
    transparent:true,depthTest:false,depthWrite:false});
  const mesh=new THREE.Mesh(geometry,material);mesh.position.z=z;const group=new THREE.Group();group.add(mesh);
  if(data.hull.length>1){const hull=data.hull.concat([data.hull[0]]),lineGeometry=new THREE.BufferGeometry().setFromPoints(hull.map(p=>new THREE.Vector3(p[0],p[1],z+.5)));group.add(new THREE.Line(lineGeometry,new THREE.LineBasicMaterial({color:config.colors[name],transparent:true,opacity:1,depthTest:false})));}
  group.userData={name,positions,intensities,hull:data.hull,mesh};scene.add(group);return group}
function disposeLayer(layer){if(!layer)return;scene.remove(layer);layer.traverse(object=>{if(object.geometry)object.geometry.dispose();if(object.material)object.material.dispose()})}
function makeAxes(bounds){if(axes){scene.remove(axes);axes.geometry.dispose();axes.material.dispose()}const x0=bounds[0][0],x1=bounds[0][1],y0=bounds[1][0],y1=bounds[1][1];const geometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x0,y0,5),new THREE.Vector3(x1,y0,5),new THREE.Vector3(x0,y0,5),new THREE.Vector3(x0,y1,5)]);axes=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:config.foreground,depthTest:false}));scene.add(axes)}
function fitView(bounds){let width=Math.max(bounds[0][1]-bounds[0][0],1e-6),height=Math.max(bounds[1][1]-bounds[1][0],1e-6);const pad=1.10,widthFromHeight=height*innerWidth/innerHeight;if(widthFromHeight>width)width=widthFromHeight;else height=width*innerHeight/innerWidth;const cx=(bounds[0][0]+bounds[0][1])/2,cy=(bounds[1][0]+bounds[1][1])/2;camera.left=cx-width*pad/2;camera.right=cx+width*pad/2;camera.bottom=cy-height*pad/2;camera.top=cy+height*pad/2;camera.zoom=1;camera.updateProjectionMatrix()}
function rotationSession(){const key=`${activeDataset}:${plane}`;if(!rotationSessions[key])rotationSessions[key]={value:0,committed:0};return rotationSessions[key]}
function configureRotationSlider(){const session=rotationSession();rotationSlider.min=String(config.rotation_min_degrees);rotationSlider.max=String(config.rotation_max_degrees);rotationSlider.step=String(config.rotation_step_degrees);rotationSlider.value=String(session.value);rotationSlider.disabled=rotationBusy||Boolean(calibration)||Boolean(preview);updateRotationValue()}
function updateRotationValue(){const axis=current?current.depth_axis:"?",value=Number(rotationSlider.value);document.getElementById("rotation-value").textContent=`${activeDataset}: ${value.toFixed(2)} deg about ${axis}`}
function rotationDisplayMatrix(angleDegrees,center,screenSign){const angle=angleDegrees*Math.PI/180*screenSign,c=Math.cos(angle),s=Math.sin(angle),cx=center[0],cy=center[1];return new THREE.Matrix4().set(c,-s,0,cx-c*cx+s*cy,s,c,0,cy-s*cx-c*cy,0,0,1,0,0,0,0,1)}
function clearRotationPreview(){if(calibration||preview)return;for(const name of ["plan","cross"]){if(layers[name]){layers[name].matrix.identity();layers[name].matrixAutoUpdate=true}}}
function previewRotation(){if(!current||calibration||preview||rotationBusy)return;clearRotationPreview();const session=rotationSession(),value=Number(rotationSlider.value);session.value=value;const delta=value-session.committed,layer=layers[activeDataset],center=current.datasets[activeDataset].rotation_center;layer.matrix.copy(rotationDisplayMatrix(delta,center,current.rotation_screen_sign));layer.matrixAutoUpdate=false;updateRotationValue();render()}
async function api(path,options={}){const response=await fetch(path,options);const payload=await response.json();if(!response.ok)throw new Error(payload.error||`HTTP ${response.status}`);return payload}
function scheduleLoad(index){clearTimeout(loadTimer);loadTimer=setTimeout(()=>loadSlice({depthIndex:index}),100)}
async function loadSlice({depthIndex=null,depthValue=null,refit=false}={}){if(loadController)loadController.abort();loadController=new AbortController();statusBox.textContent="Loading physical slices...";const params=new URLSearchParams({plane,threshold_floor:String(config.threshold_floor)});if(depthValue!==null)params.set("depth_value",String(depthValue));else params.set("depth_index",String(depthIndex??depthSlider.value));
  try{const payload=await api(`/api/slice?${params}`,{signal:loadController.signal});current=payload;plane=payload.plane;for(const name of ["plan","cross"]){disposeLayer(layers[name]);layers[name]=makeLayer(name,payload.datasets[name],name==="plan"?1:2)}makeAxes(payload.bounds);if(firstLoad||refit){fitView(payload.bounds);firstLoad=false}depthSlider.max=String(payload.depth_count-1);depthSlider.value=String(payload.depth_index);document.getElementById("depth-value").textContent=`${payload.depth_axis} = ${payload.depth.toFixed(5)} ${config.length_unit}`;document.getElementById("axis-h").textContent=`${payload.horizontal_axis} (${config.length_unit})`;document.getElementById("axis-v").textContent=`${payload.vertical_axis} (${config.length_unit})`;document.querySelectorAll("[data-plane]").forEach(button=>button.classList.toggle("active",button.dataset.plane===plane));configureRotationSlider();updateCounts();render()}catch(error){if(error.name!=="AbortError")statusBox.textContent=error.message}}
function updateCounts(){if(!current)return;let text=[];for(const name of ["plan","cross"]){const values=layers[name].userData.intensities;let count=0;for(let i=0;i<values.length;i++)if(values[i]>=threshold)count++;text.push(`${name}: ${count.toLocaleString()}`);layers[name].userData.mesh.material.uniforms.threshold.value=threshold}readout.textContent=`${plane.toUpperCase()} | ${text.join(" | ")}`;document.getElementById("threshold-value").textContent=threshold.toFixed(2);if(!calibration&&!preview&&!rotationBusy)statusBox.textContent=`Active dataset: ${activeDataset}`}
function worldPoint(event){const rect=renderer.domElement.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width*2-1,y=-(event.clientY-rect.top)/rect.height*2+1;const point=new THREE.Vector3(x,y,0).unproject(camera);return [point.x,point.y]}
function pointInHull(point,hull){if(!hull||hull.length<3)return false;let inside=false;for(let i=0,j=hull.length-1;i<hull.length;j=i++){const xi=hull[i][0],yi=hull[i][1],xj=hull[j][0],yj=hull[j][1];if(((yi>point[1])!==(yj>point[1]))&&(point[0]<(xj-xi)*(point[1]-yi)/(yj-yi)+xi))inside=!inside}return inside}
function nearestVisible(name,point){const layer=layers[name],positions=layer.userData.positions,values=layer.userData.intensities;let best=-1,bestDistance=Infinity;for(let i=0;i<values.length;i++){if(values[i]<threshold)continue;const dx=positions[2*i]-point[0],dy=positions[2*i+1]-point[1],distance=dx*dx+dy*dy;if(distance<bestDistance){bestDistance=distance;best=i}}const worldPerPixel=(camera.top-camera.bottom)/(renderer.domElement.clientHeight*camera.zoom);return best>=0&&Math.sqrt(bestDistance)<=12*worldPerPixel?[positions[2*best],positions[2*best+1]]:point}
function addMarker(point,color){const geometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(point[0],point[1],8)]),material=new THREE.PointsMaterial({color,size:9,sizeAttenuation:false,depthTest:false});markers.add(new THREE.Points(geometry,material))}
function clearMarkers(){while(markers.children.length){const child=markers.children.pop();child.geometry.dispose();child.material.dispose()}}
function calibrationClick(point){const expected=calibration.points.plan.length<2?"plan":"cross",snapped=nearestVisible(expected,point);calibration.points[expected].push(snapped);addMarker(snapped,config.colors[expected]);const count=calibration.points[expected].length;if(expected==="plan")statusBox.textContent=count<2?"Plan: select the second point.":"Cross: select the first point.";else if(count<2)statusBox.textContent="Cross: select the second point.";else requestPreview()}
async function requestPreview(){try{const body={plane,depth:current.depth,target_distance:calibration.distance,points:calibration.points};preview=await api("/api/calibration-preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});for(const name of ["plan","cross"]){layers[name].matrix.copy(displayMatrix(preview.operations[name]));layers[name].matrixAutoUpdate=false}const p=preview.measurements,a=preview.relative_alignment,shift=a.translation_xyz.map(value=>value.toFixed(5)).join(", "),residual=Math.max(...a.endpoint_residuals);document.getElementById("preview-text").textContent=`Plan: ${p.plan.measured_distance.toFixed(5)} -> ${p.plan.target_distance.toFixed(5)} (x${p.plan.scale_factor.toFixed(6)}). Cross: ${p.cross.measured_distance.toFixed(5)} -> ${p.cross.target_distance.toFixed(5)} (x${p.cross.scale_factor.toFixed(6)}). Cross-to-plan: rotate ${a.rotation_degrees.toFixed(2)} deg about ${a.rotation_axis}, translate xyz [${shift}] ${config.length_unit}; endpoint residual ${residual.toExponential(2)} ${config.length_unit}.`;configureRotationSlider();document.getElementById("preview-dialog").showModal();statusBox.textContent="Calibration preview."}catch(error){cancelCalibration();statusBox.textContent=error.message}}
function displayMatrix(values){const m=values,h=current.horizontal_axis,v=current.vertical_axis,axis={x:0,y:1,z:2},hi=axis[h],vi=axis[v];return new THREE.Matrix4().set(m[hi][hi],m[hi][vi],0,m[hi][3],m[vi][hi],m[vi][vi],0,m[vi][3],0,0,1,0,0,0,0,1)}
function cancelCalibration(){for(const name of ["plan","cross"]){if(layers[name]){layers[name].matrix.identity();layers[name].matrixAutoUpdate=true}}calibration=null;preview=null;clearMarkers();document.getElementById("preview-dialog").close();configureRotationSlider();updateCounts()}
renderer.domElement.addEventListener("pointerdown",event=>{if(rotationBusy)return;const point=worldPoint(event);if(calibration){calibrationClick(point);return}const layer=layers[activeDataset];if(pointInHull(point,layer.userData.hull)){drag={start:point,last:point,name:activeDataset};renderer.domElement.setPointerCapture(event.pointerId)}});
renderer.domElement.addEventListener("pointermove",event=>{if(!drag)return;const point=worldPoint(event),dx=point[0]-drag.start[0],dy=point[1]-drag.start[1];drag.last=point;layers[drag.name].position.set(dx,dy,0);render()});
renderer.domElement.addEventListener("pointerup",async event=>{if(!drag)return;const finished=drag;drag=null;const dx=finished.last[0]-finished.start[0],dy=finished.last[1]-finished.start[1];layers[finished.name].position.set(0,0,0);if(dx||dy){try{await api("/api/translate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({dataset:finished.name,plane,horizontal_delta:dx,vertical_delta:dy,depth:current.depth})});await loadSlice({depthIndex:current.depth_index})}catch(error){statusBox.textContent=error.message}}});
renderer.domElement.addEventListener("wheel",event=>{event.preventDefault();camera.zoom=Math.min(100,Math.max(.05,camera.zoom*Math.exp(-event.deltaY*.001)));camera.updateProjectionMatrix();render()},{passive:false});
async function commitRotation(){if(!current||calibration||preview||rotationBusy)return;const session=rotationSession(),value=Number(rotationSlider.value),delta=value-session.committed;if(Math.abs(delta)<1e-12){clearRotationPreview();render();return}rotationBusy=true;configureRotationSlider();statusBox.textContent=`Applying ${delta.toFixed(2)} deg rotation to ${activeDataset}...`;try{await api("/api/rotate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({dataset:activeDataset,plane,angle_degrees:delta,depth:current.depth})});session.committed=value;clearRotationPreview();await loadSlice({depthValue:current.depth});statusBox.textContent=`Rotated ${activeDataset} by ${delta.toFixed(2)} deg about ${current.depth_axis}.`}catch(error){session.value=session.committed;rotationSlider.value=String(session.committed);clearRotationPreview();statusBox.textContent=error.message}finally{rotationBusy=false;configureRotationSlider();render()}}
document.querySelectorAll("[data-plane]").forEach(button=>button.addEventListener("click",()=>{clearRotationPreview();plane=button.dataset.plane;loadSlice({depthValue:0,refit:true})}));
document.querySelectorAll("[data-dataset]").forEach(button=>button.addEventListener("click",()=>{clearRotationPreview();activeDataset=button.dataset.dataset;document.querySelectorAll("[data-dataset]").forEach(item=>item.classList.toggle("active",item.dataset.dataset===activeDataset));configureRotationSlider();updateCounts()}));
depthSlider.addEventListener("input",()=>scheduleLoad(depthSlider.value));
rotationSlider.addEventListener("input",previewRotation);rotationSlider.addEventListener("change",commitRotation);
thresholdSlider.max=String(config.thresholds.length-1);thresholdSlider.value=String(config.thresholds.reduce((best,value,index)=>Math.abs(value-threshold)<Math.abs(config.thresholds[best]-threshold)?index:best,0));thresholdRow.style.display=config.thresholds.length===1?"none":"grid";thresholdSlider.addEventListener("input",()=>{threshold=config.thresholds[Number(thresholdSlider.value)];updateCounts();render()});
document.getElementById("calibrate").addEventListener("click",()=>document.getElementById("distance-dialog").showModal());
document.getElementById("begin-calibration").addEventListener("click",event=>{const value=Number(document.getElementById("distance").value);if(!(value>0)){event.preventDefault();return}clearRotationPreview();calibration={distance:value,points:{plan:[],cross:[]}};configureRotationSlider();clearMarkers();statusBox.textContent="Plan: select the first point."});
document.getElementById("cancel-preview").addEventListener("click",cancelCalibration);
document.getElementById("apply-preview").addEventListener("click",async()=>{try{await api("/api/calibrate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({plane,depth:current.depth,target_distance:calibration.distance,points:calibration.points})});document.getElementById("preview-dialog").close();calibration=null;preview=null;clearMarkers();for(const name of ["plan","cross"]){layers[name].matrix.identity();layers[name].matrixAutoUpdate=true}configureRotationSlider();await loadSlice({depthIndex:current.depth_index})}catch(error){statusBox.textContent=error.message}});
document.getElementById("save").addEventListener("click",async()=>{try{const result=await api("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({plane,depth:current.depth})});statusBox.textContent=`Saved: ${result.state_path}`}catch(error){statusBox.textContent=error.message}});
function resize(){renderer.setSize(innerWidth,innerHeight);if(current)fitView(current.bounds);render()}window.addEventListener("resize",resize);resize();
function render(){renderer.render(scene,camera);renderer.domElement.dataset.plane=plane;renderer.domElement.dataset.activeDataset=activeDataset;renderer.domElement.dataset.depth=current?String(current.depth):"";renderer.domElement.dataset.rotation=String(rotationSession().value);renderer.domElement.dataset.rotationAxis=current?current.depth_axis:"";renderer.domElement.dataset.rotationBusy=String(rotationBusy);renderer.domElement.dataset.planCount=layers.plan?String(layers.plan.userData.intensities.length):"0";renderer.domElement.dataset.crossCount=layers.cross?String(layers.cross.userData.intensities.length):"0"}
loadSlice({depthValue:config.initial_depth,refit:true});
</script>
</body>
</html>'''
