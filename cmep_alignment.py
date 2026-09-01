"""Non-destructive affine state and physical slicing for CMEP alignment."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


AXES = ("x", "y", "z")
DATASETS = ("plan", "cross")
STATE_SCHEMA = "cmep-initial-alignment-v1"
ALLOWED_PLANES = ("xy", "yx", "xz", "zx", "yz", "zy")


def create_alignment_state(length_unit: str) -> dict[str, Any]:
    """Return a fresh identity-transform alignment state."""
    if not isinstance(length_unit, str) or not length_unit.strip():
        raise ValueError("length_unit must be a non-empty string.")
    identity = np.eye(4, dtype=np.float64).tolist()
    return {
        "schema": STATE_SCHEMA,
        "length_unit": length_unit.strip(),
        "transforms": {name: deepcopy(identity) for name in DATASETS},
        "calibrations": [],
        "ui": {"plane": "yx", "depth": 0.0},
    }


def validate_alignment_state(
    state: Mapping[str, Any], *, length_unit: str | None = None
) -> dict[str, Any]:
    """Validate and return a JSON-compatible copy of an alignment state."""
    if not isinstance(state, Mapping):
        raise TypeError("Alignment state must be a mapping.")
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"Unsupported alignment state schema: {state.get('schema')!r}.")
    unit = state.get("length_unit")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Alignment state length_unit must be a non-empty string.")
    if length_unit is not None and unit.strip() != str(length_unit).strip():
        raise ValueError(
            f"Saved alignment unit {unit!r} does not match current unit {length_unit!r}."
        )

    transforms = state.get("transforms")
    if not isinstance(transforms, Mapping):
        raise ValueError("Alignment state must contain dataset transforms.")
    clean_transforms = {
        dataset: _validated_affine(transforms.get(dataset), f"{dataset} transform").tolist()
        for dataset in DATASETS
    }

    calibrations = state.get("calibrations", [])
    if not isinstance(calibrations, list):
        raise ValueError("calibrations must be a list.")
    ui = state.get("ui", {})
    if not isinstance(ui, Mapping):
        raise ValueError("ui must be a mapping.")
    plane = validate_plane(str(ui.get("plane", "yx")))
    depth = _finite_scalar(ui.get("depth", 0.0), "ui depth")

    return {
        "schema": STATE_SCHEMA,
        "length_unit": unit.strip(),
        "transforms": clean_transforms,
        "calibrations": deepcopy(calibrations),
        "ui": {"plane": plane, "depth": depth},
    }


def load_alignment_state(
    path: str | Path, *, length_unit: str | None = None
) -> dict[str, Any]:
    """Load and validate a saved alignment state."""
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read alignment state {source}: {exc}") from exc
    return validate_alignment_state(payload, length_unit=length_unit)


def save_alignment_state(
    state: Mapping[str, Any],
    path: str | Path,
    *,
    keep_snapshot: bool = True,
) -> tuple[Path, Path | None]:
    """Atomically save the current state and optionally a timestamped snapshot."""
    clean = validate_alignment_state(state)
    clean["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(destination)

    snapshot = None
    if keep_snapshot:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        snapshot = destination.with_name(f"{destination.stem}-{stamp}{destination.suffix}")
        snapshot.write_text(text, encoding="utf-8", newline="\n")
    return destination, snapshot


def validate_plane(plane: str) -> str:
    """Validate a two-letter physical display plane."""
    normalized = str(plane).strip().lower()
    if normalized not in ALLOWED_PLANES:
        raise ValueError(
            f"Unsupported plane {plane!r}; expected one of {ALLOWED_PLANES}."
        )
    return normalized


def plane_axes(plane: str) -> tuple[str, str, str]:
    """Return (vertical, horizontal, depth) physical axes for a display plane."""
    normalized = validate_plane(plane)
    depth = next(axis for axis in AXES if axis not in normalized)
    return normalized[0], normalized[1], depth


def directional_scale_affine(
    point1: Sequence[float],
    point2: Sequence[float],
    target_distance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Scale only along point1->point2, fixing point1 in world space."""
    p1 = _point3(point1, "point1")
    p2 = _point3(point2, "point2")
    distance = _positive_finite(target_distance, "target_distance")
    vector = p2 - p1
    measured = float(np.linalg.norm(vector))
    if measured <= np.finfo(np.float64).eps:
        raise ValueError("Calibration points must be distinct.")
    direction = vector / measured
    factor = distance / measured
    linear = np.eye(3, dtype=np.float64) + (factor - 1.0) * np.outer(
        direction, direction
    )
    operation = np.eye(4, dtype=np.float64)
    operation[:3, :3] = linear
    operation[:3, 3] = p1 - linear @ p1
    return operation, {
        "measured_distance": measured,
        "target_distance": distance,
        "scale_factor": factor,
    }


def translation_affine(offset: Sequence[float]) -> np.ndarray:
    """Return a world-space translation affine."""
    vector = _point3(offset, "offset")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = vector
    return result


def axis_angle_rotation_affine(
    axis: str | Sequence[float],
    angle_degrees: float,
    center: Sequence[float],
) -> np.ndarray:
    """Return a right-handed world-space rotation about a physical point."""
    if isinstance(axis, str):
        name = axis.strip().lower()
        if name not in AXES:
            raise ValueError("A named rotation axis must be 'x', 'y', or 'z'.")
        vector = np.zeros(3, dtype=np.float64)
        vector[AXES.index(name)] = 1.0
    else:
        vector = _point3(axis, "axis")
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("axis must be nonzero.")
        vector = vector / norm

    angle = np.deg2rad(_finite_scalar(angle_degrees, "angle_degrees"))
    pivot = _point3(center, "center")
    x, y, z = vector
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    complement = 1.0 - cosine
    linear = np.array(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )
    operation = np.eye(4, dtype=np.float64)
    operation[:3, :3] = linear
    operation[:3, 3] = pivot - linear @ pivot
    return operation


def compose_world_operation(
    transform: Sequence[Sequence[float]], operation: Sequence[Sequence[float]]
) -> np.ndarray:
    """Apply an operation in current world coordinates to a source transform."""
    current = _validated_affine(transform, "transform")
    world_operation = _validated_affine(operation, "operation")
    composed = world_operation @ current
    return _validated_affine(composed, "composed transform")


def transform_physical_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    transform: Sequence[Sequence[float]],
) -> np.ndarray:
    """Transform physical xyz points from source coordinates into world space."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must be a finite array with shape (point_count, 3).")
    affine = _validated_affine(transform, "transform")
    return values @ affine[:3, :3].T + affine[:3, 3]


def sample_volume_at_world_points(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    transform: Sequence[Sequence[float]],
    world_points: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Trilinearly sample a canonical volume at physical world xyz positions.

    Points outside the transformed source volume are returned as ``NaN``.
    The source volume and its coordinate vectors are never modified.
    """
    data = np.asarray(volume)
    if data.ndim != 3:
        raise ValueError(f"volume must be 3D in (x, y, z) order; got {data.shape}.")
    coordinates = _coordinate_arrays(metadata)
    if tuple(len(values) for values in coordinates) != data.shape:
        raise ValueError("Coordinate vector lengths must match the canonical volume shape.")
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("world_points must be finite with shape (point_count, 3).")
    inverse = np.linalg.inv(_validated_affine(transform, "transform"))
    source = points @ inverse[:3, :3].T + inverse[:3, 3]
    indices = np.vstack(
        [
            _physical_to_fractional_index(source[:, axis], coordinates[axis])
            for axis in range(3)
        ]
    )
    return _trilinear_sample(data, indices)


def preview_calibration(
    state: Mapping[str, Any],
    *,
    plane: str,
    depth: float,
    target_distance: float,
    points: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, Any]:
    """Independently scale each selected direction without changing placement.

    Each operation is expressed in current world coordinates and fixes that
    dataset's first selected point. Calibration therefore changes physical
    scale along the measured direction but adds no cross-to-plan rotation or
    translation.
    """
    clean = validate_alignment_state(state)
    vertical, horizontal, depth_axis = plane_axes(plane)
    depth_value = _finite_scalar(depth, "depth")
    scale_operations: dict[str, np.ndarray] = {}
    measurements: dict[str, dict[str, float]] = {}
    points3d_arrays: dict[str, np.ndarray] = {}

    for dataset in DATASETS:
        selected = points.get(dataset)
        if selected is None or len(selected) != 2:
            raise ValueError(f"Calibration requires exactly two {dataset} points.")
        converted = []
        for index, point in enumerate(selected, start=1):
            values = np.asarray(point, dtype=np.float64)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{dataset} point {index} must contain horizontal, vertical.")
            world = np.zeros(3, dtype=np.float64)
            world[AXES.index(horizontal)] = values[0]
            world[AXES.index(vertical)] = values[1]
            world[AXES.index(depth_axis)] = depth_value
            converted.append(world)
        operation, measurement = directional_scale_affine(
            converted[0], converted[1], target_distance
        )
        scale_operations[dataset] = operation
        measurements[dataset] = measurement
        points3d_arrays[dataset] = np.asarray(converted, dtype=np.float64)

    calibrated_points = {
        dataset: transform_physical_points(
            points3d_arrays[dataset], scale_operations[dataset]
        )
        for dataset in DATASETS
    }
    anchor_residuals = {
        dataset: float(
            np.linalg.norm(calibrated_points[dataset][0] - points3d_arrays[dataset][0])
        )
        for dataset in DATASETS
    }

    return {
        "plane": validate_plane(plane),
        "depth": depth_value,
        "target_distance": float(target_distance),
        "operations": {
            dataset: scale_operations[dataset].tolist() for dataset in DATASETS
        },
        "measurements": measurements,
        "points_3d": {
            dataset: points3d_arrays[dataset].tolist() for dataset in DATASETS
        },
        "calibrated_points_3d": {
            dataset: calibrated_points[dataset].tolist() for dataset in DATASETS
        },
        "placement": {
            "mode": "independent_directional_scale",
            "anchor": "first_selected_point",
            "relative_rotation_applied": False,
            "relative_translation_applied": False,
            "anchor_residuals": anchor_residuals,
        },
        "base_transforms": deepcopy(clean["transforms"]),
    }


def apply_calibration_preview(
    state: Mapping[str, Any], preview: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply a previously constructed calibration preview to both datasets."""
    clean = validate_alignment_state(state)
    base = preview.get("base_transforms")
    if base != clean["transforms"]:
        raise ValueError("Alignment state changed after calibration preview; preview again.")
    operations = preview.get("operations")
    if not isinstance(operations, Mapping):
        raise ValueError("Calibration preview is missing operations.")
    for dataset in DATASETS:
        clean["transforms"][dataset] = compose_world_operation(
            clean["transforms"][dataset], operations.get(dataset)
        ).tolist()
    calibration = {
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "plane": preview["plane"],
        "depth": float(preview["depth"]),
        "target_distance": float(preview["target_distance"]),
        "measurements": deepcopy(preview["measurements"]),
        "points_3d": deepcopy(preview["points_3d"]),
        "operations": deepcopy(preview["operations"]),
    }
    for key in ("calibrated_points_3d", "placement"):
        if key in preview:
            calibration[key] = deepcopy(preview[key])
    clean["calibrations"].append(calibration)
    clean["ui"] = {"plane": preview["plane"], "depth": float(preview["depth"])}
    return clean


def translate_dataset_in_plane(
    state: Mapping[str, Any],
    *,
    dataset: str,
    plane: str,
    horizontal_delta: float,
    vertical_delta: float,
    depth: float,
) -> dict[str, Any]:
    """Translate one dataset along the two physical axes in the active plane."""
    clean = validate_alignment_state(state)
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}.")
    vertical, horizontal, _ = plane_axes(plane)
    offset = np.zeros(3, dtype=np.float64)
    offset[AXES.index(horizontal)] = _finite_scalar(
        horizontal_delta, "horizontal_delta"
    )
    offset[AXES.index(vertical)] = _finite_scalar(vertical_delta, "vertical_delta")
    clean["transforms"][dataset] = compose_world_operation(
        clean["transforms"][dataset], translation_affine(offset)
    ).tolist()
    clean["ui"] = {
        "plane": validate_plane(plane),
        "depth": _finite_scalar(depth, "depth"),
    }
    return clean


def transformed_geometric_center(
    metadata: Mapping[str, Any], transform: Sequence[Sequence[float]]
) -> np.ndarray:
    """Return the transformed center of a dataset's physical coordinate box."""
    coordinates = _coordinate_arrays(metadata)
    source_center = np.array(
        [(float(values.min()) + float(values.max())) / 2.0 for values in coordinates],
        dtype=np.float64,
    )
    return transform_physical_points(source_center[None, :], transform)[0]


def rotate_dataset_in_plane(
    state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    dataset: str,
    plane: str,
    angle_degrees: float,
    depth: float,
) -> dict[str, Any]:
    """Rotate one complete dataset about its center along the plane normal."""
    clean = validate_alignment_state(state)
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}.")
    normalized_plane = validate_plane(plane)
    _, _, rotation_axis = plane_axes(normalized_plane)
    angle = _finite_scalar(angle_degrees, "angle_degrees")
    center = transformed_geometric_center(metadata, clean["transforms"][dataset])
    operation = axis_angle_rotation_affine(rotation_axis, angle, center)
    clean["transforms"][dataset] = compose_world_operation(
        clean["transforms"][dataset], operation
    ).tolist()
    clean["ui"] = {
        "plane": normalized_plane,
        "depth": _finite_scalar(depth, "depth"),
    }
    return clean


def transformed_box_corners(
    metadata: Mapping[str, Any], transform: Sequence[Sequence[float]]
) -> np.ndarray:
    """Return the eight transformed physical corners of a volume."""
    coordinates = _coordinate_arrays(metadata)
    limits = [(float(values.min()), float(values.max())) for values in coordinates]
    corners = np.array(
        [[x, y, z] for x in limits[0] for y in limits[1] for z in limits[2]],
        dtype=np.float64,
    )
    affine = _validated_affine(transform, "transform")
    return corners @ affine[:3, :3].T + affine[:3, 3]


def union_depth_range(
    metadata_by_dataset: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    plane: str,
) -> tuple[float, float]:
    """Return the largest transformed physical depth interval of both datasets."""
    clean = validate_alignment_state(state)
    _, _, depth_axis = plane_axes(plane)
    index = AXES.index(depth_axis)
    values = []
    for dataset in DATASETS:
        corners = transformed_box_corners(
            metadata_by_dataset[dataset], clean["transforms"][dataset]
        )
        values.extend((float(corners[:, index].min()), float(corners[:, index].max())))
    return min(values), max(values)


def suggested_alignment_depth(
    metadata_by_dataset: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    plane: str,
) -> float:
    """Return a central depth that intersects both transformed datasets.

    If their transformed depth intervals do not overlap, the plan-view
    dataset's center is used so a new projection still opens on data rather
    than in the empty gap between datasets.
    """
    clean = validate_alignment_state(state)
    _, _, depth_axis = plane_axes(plane)
    index = AXES.index(depth_axis)
    intervals: dict[str, tuple[float, float]] = {}
    for dataset in DATASETS:
        corners = transformed_box_corners(
            metadata_by_dataset[dataset], clean["transforms"][dataset]
        )
        intervals[dataset] = (
            float(corners[:, index].min()),
            float(corners[:, index].max()),
        )

    shared_lower = max(interval[0] for interval in intervals.values())
    shared_upper = min(interval[1] for interval in intervals.values())
    if shared_lower <= shared_upper:
        return (shared_lower + shared_upper) / 2.0
    plan_lower, plan_upper = intervals["plan"]
    return (plan_lower + plan_upper) / 2.0


def slice_transformed_volume(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    transform: Sequence[Sequence[float]],
    *,
    plane: str,
    depth: float,
    threshold_floor: float = 0.0,
) -> dict[str, Any]:
    """Linearly sample an affine-transformed volume on a physical world plane."""
    data = np.asarray(volume)
    if data.ndim != 3:
        raise ValueError(f"volume must be 3D in (x, y, z) order; got {data.shape}.")
    coordinates = _coordinate_arrays(metadata)
    if tuple(len(values) for values in coordinates) != data.shape:
        raise ValueError("Coordinate vector lengths must match the canonical volume shape.")
    threshold = _finite_scalar(threshold_floor, "threshold_floor")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold_floor must lie in [0, 1].")
    depth_value = _finite_scalar(depth, "depth")
    vertical, horizontal, depth_axis = plane_axes(plane)
    vertical_index = AXES.index(vertical)
    horizontal_index = AXES.index(horizontal)
    depth_index = AXES.index(depth_axis)
    affine = _validated_affine(transform, "transform")
    inverse = np.linalg.inv(affine)
    corners = transformed_box_corners(metadata, affine)

    horizontal_values, horizontal_spacing = _sampling_coordinates(
        corners[:, horizontal_index], inverse[:3, :3], horizontal_index, coordinates
    )
    vertical_values, vertical_spacing = _sampling_coordinates(
        corners[:, vertical_index], inverse[:3, :3], vertical_index, coordinates
    )
    horizontal_grid, vertical_grid = np.meshgrid(
        horizontal_values, vertical_values, indexing="xy"
    )
    count = horizontal_grid.size
    world = np.empty((4, count), dtype=np.float64)
    world[:3] = 0.0
    world[horizontal_index] = horizontal_grid.reshape(-1)
    world[vertical_index] = vertical_grid.reshape(-1)
    world[depth_index] = depth_value
    world[3] = 1.0
    source = inverse @ world
    indices = np.vstack(
        [
            _physical_to_fractional_index(source[axis_index], coordinates[axis_index])
            for axis_index in range(3)
        ]
    )
    sampled = _trilinear_sample(data, indices)
    keep = np.isfinite(sampled) & (sampled >= threshold)
    horizontal_points = horizontal_grid.reshape(-1)[keep].astype(np.float32, copy=False)
    vertical_points = vertical_grid.reshape(-1)[keep].astype(np.float32, copy=False)
    intensities = sampled[keep]
    hull = plane_box_intersection(corners, plane, depth_value)
    return {
        "horizontal": horizontal_points,
        "vertical": vertical_points,
        "intensity": intensities,
        "sampled_point_count": int(count),
        "displayed_point_count": int(keep.sum()),
        "voxel_size": (float(horizontal_spacing), float(vertical_spacing)),
        "hull": hull,
    }


def plane_box_intersection(
    transformed_corners: np.ndarray, plane: str, depth: float
) -> list[list[float]]:
    """Return the convex 2D intersection polygon of a transformed box and plane."""
    corners = np.asarray(transformed_corners, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError("transformed_corners must have shape (8, 3).")
    vertical, horizontal, depth_axis = plane_axes(plane)
    vi, hi, di = (AXES.index(vertical), AXES.index(horizontal), AXES.index(depth_axis))
    depth_value = _finite_scalar(depth, "depth")
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    tolerance = max(1.0, float(np.ptp(corners[:, di]))) * 1e-10
    points: list[tuple[float, float]] = []
    for start, stop in edges:
        first, second = corners[start], corners[stop]
        a, b = first[di] - depth_value, second[di] - depth_value
        if abs(a) <= tolerance:
            points.append((float(first[hi]), float(first[vi])))
        if abs(b) <= tolerance:
            points.append((float(second[hi]), float(second[vi])))
        if a * b < -(tolerance * tolerance):
            fraction = a / (a - b)
            point = first + fraction * (second - first)
            points.append((float(point[hi]), float(point[vi])))
    return [list(point) for point in _convex_hull(points, tolerance)]


def _sampling_coordinates(
    projected_corners: np.ndarray,
    inverse_linear: np.ndarray,
    world_axis_index: int,
    source_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, float]:
    lower = float(np.min(projected_corners))
    upper = float(np.max(projected_corners))
    source_spacing = min(_axis_spacing(values) for values in source_coordinates)
    source_distance_per_world_unit = float(
        np.linalg.norm(inverse_linear @ np.eye(3, dtype=np.float64)[:, world_axis_index])
    )
    spacing = source_spacing / max(source_distance_per_world_unit, 1e-12)
    extent = upper - lower
    sample_count = max(1, int(round(extent / spacing)) + 1)
    values = np.linspace(lower, upper, sample_count, dtype=np.float64)
    effective = extent / (sample_count - 1) if sample_count > 1 else spacing
    return values, float(effective)


def _physical_to_fractional_index(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    if len(coordinates) == 1:
        return np.zeros_like(values, dtype=np.float64)
    return (values - coordinates[0]) * (len(coordinates) - 1) / (
        coordinates[-1] - coordinates[0]
    )


def _trilinear_sample(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sample a canonical volume linearly without requiring SciPy."""
    coordinates = np.asarray(indices, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] != 3:
        raise ValueError("indices must have shape (3, sample_count).")
    sample_count = coordinates.shape[1]
    valid = np.isfinite(coordinates).all(axis=0)
    clipped = np.empty_like(coordinates)
    tolerance = 1e-8
    for axis, length in enumerate(data.shape):
        valid &= coordinates[axis] >= -tolerance
        valid &= coordinates[axis] <= (length - 1) + tolerance
        clipped[axis] = np.clip(coordinates[axis], 0.0, float(length - 1))

    lower = np.floor(clipped).astype(np.intp)
    upper = np.minimum(lower + 1, np.asarray(data.shape, dtype=np.intp)[:, None] - 1)
    fraction = (clipped - lower).astype(np.float32)
    inverse_fraction = 1.0 - fraction
    sampled = np.full(sample_count, np.nan, dtype=np.float32)
    if not np.any(valid):
        return sampled

    x0, y0, z0 = (lower[axis, valid] for axis in range(3))
    x1, y1, z1 = (upper[axis, valid] for axis in range(3))
    fx, fy, fz = (fraction[axis, valid] for axis in range(3))
    gx, gy, gz = (inverse_fraction[axis, valid] for axis in range(3))
    values = np.asarray(data, dtype=np.float32)
    interpolated = (
        values[x0, y0, z0] * gx * gy * gz
        + values[x1, y0, z0] * fx * gy * gz
        + values[x0, y1, z0] * gx * fy * gz
        + values[x1, y1, z0] * fx * fy * gz
        + values[x0, y0, z1] * gx * gy * fz
        + values[x1, y0, z1] * fx * gy * fz
        + values[x0, y1, z1] * gx * fy * fz
        + values[x1, y1, z1] * fx * fy * fz
    )
    sampled[valid] = interpolated.astype(np.float32, copy=False)
    return sampled


def _coordinate_arrays(
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = metadata.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("metadata must contain x, y, z coordinate vectors.")
    result = []
    for axis in AXES:
        values = np.asarray(coordinates.get(axis), dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"metadata coordinate {axis!r} must be a finite 1D vector.")
        if len(values) > 1:
            differences = np.diff(values)
            if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
                raise ValueError(f"metadata coordinate {axis!r} must be strictly monotonic.")
        result.append(values)
    return tuple(result)  # type: ignore[return-value]


def _convex_hull(
    points: Sequence[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for point in sorted(points):
        if not any(np.linalg.norm(np.subtract(point, other)) <= tolerance for other in unique):
            unique.append(point)
    if len(unique) <= 2:
        return unique

    def cross(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _axis_spacing(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        return 1.0
    return float(np.median(np.abs(np.diff(coordinates))))


def _validated_affine(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 affine matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError(f"{name} must have affine bottom row [0, 0, 0, 1].")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
        raise ValueError(f"{name} must have an invertible linear part.")
    return matrix


def _point3(value: Sequence[float], name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain exactly three finite coordinates.")
    return point


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return result
