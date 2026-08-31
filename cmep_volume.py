"""Physical preparation of plan-view and cross-sectional TIFF volumes.

The raw storage convention is always ``(slice, row, col)``.  The processed
volume is returned in canonical physical axis order ``(x, y, z)``.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError


PHYSICAL_AXES = ("x", "y", "z")
PLAN_ORIENTATIONS = frozenset({"xy", "yx"})
CROSS_ORIENTATIONS = frozenset({"xz", "zx", "yz", "zy"})
COORDINATE_EQUATION = (
    "point = first_slice_origin + col*col_step_vector + "
    "row*row_step_vector + slice*slice_step_vector"
)


def load_volume_input(input_data: str | PathLike[str] | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a TIFF path or accept an existing numeric 3D NumPy array.

    TIFF files are opened read-only with Pillow and stacked as
    ``(slice, row, col)``.  The returned array is not normalized.
    """
    if isinstance(input_data, np.ndarray):
        stack = np.asarray(input_data)
        input_type = "numpy_array"
        source_path = None
    elif isinstance(input_data, (str, PathLike)):
        path = Path(input_data).expanduser()
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Expected a .tif or .tiff path, got: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"TIFF input does not exist: {path}")
        try:
            with Image.open(path) as image:
                frames: list[np.ndarray] = []
                expected_shape: tuple[int, int] | None = None
                n_frames = int(getattr(image, "n_frames", 1))
                for index in range(n_frames):
                    image.seek(index)
                    frame = np.asarray(image)
                    if frame.ndim != 2:
                        raise ValueError(
                            f"TIFF frame {index} is {frame.ndim}D; grayscale 2D frames are required."
                        )
                    if expected_shape is None:
                        expected_shape = frame.shape
                    elif frame.shape != expected_shape:
                        raise ValueError(
                            f"TIFF frame {index} has shape {frame.shape}, expected {expected_shape}."
                        )
                    frames.append(np.array(frame, copy=True))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read TIFF input {path}: {exc}") from exc
        stack = np.stack(frames, axis=0)
        input_type = "tiff"
        source_path = str(path.resolve())
    else:
        raise TypeError(
            "Input must be a TIFF path or a 3D NumPy array; "
            f"received {type(input_data).__name__}."
        )

    if stack.ndim != 3:
        raise ValueError(
            f"Input must have shape (slice, row, col); received shape {stack.shape}."
        )
    if any(size < 1 for size in stack.shape):
        raise ValueError(f"Every input dimension must be non-empty; received {stack.shape}.")
    if not np.issubdtype(stack.dtype, np.number):
        raise TypeError(f"Input array must be numeric; received dtype {stack.dtype}.")

    info = {
        "input_type": input_type,
        "source_path": source_path,
        "original_input_shape": tuple(int(v) for v in stack.shape),
        "original_input_dtype": str(stack.dtype),
        "raw_array_axis_order": ("slice", "row", "col"),
    }
    return stack, info


def normalize_slices(
    stack: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize each ``(row, col)`` slice independently to float32 [0, 1]."""
    array = np.asarray(stack)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D (slice, row, col) array, got {array.shape}.")
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")

    source = array.astype(np.float32, copy=False)
    normalized = np.zeros(source.shape, dtype=np.float32)
    low_values: list[float | None] = []
    high_values: list[float | None] = []
    degenerate_slices: list[int] = []

    for index in range(source.shape[0]):
        image = source[index]
        finite = np.isfinite(image)
        if not finite.any():
            low_values.append(None)
            high_values.append(None)
            degenerate_slices.append(index)
            continue

        # Python floats reproduce the arithmetic used by the established
        # project normalizer after its initial float32 conversion. Computing
        # each percentile separately also preserves its exact NumPy behavior.
        finite_values = image[finite]
        low = float(np.percentile(finite_values, lower_percentile))
        high = float(np.percentile(finite_values, upper_percentile))
        low_values.append(low)
        high_values.append(high)
        if not high > low:
            degenerate_slices.append(index)
            continue

        safe_image = np.where(finite, image, low)
        normalized[index] = np.clip(
            (safe_image - low) / (high - low), 0.0, 1.0
        )

    details = {
        "scope": "each 2D slice independently",
        "method": "1st/99th percentile clipping and linear scaling to [0, 1]",
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "per_slice_lower": low_values,
        "per_slice_upper": high_values,
        "degenerate_or_nonfinite_only_slices": degenerate_slices,
        "output_dtype": "float32",
        "output_range": (0.0, 1.0),
    }
    return normalized, details


def validate_orientation(orientation: str, dataset_kind: str) -> dict[str, str]:
    """Validate an orientation and return the raw-dimension axis mapping."""
    if not isinstance(orientation, str):
        raise TypeError("Orientation must be a two-letter string.")
    orientation = orientation.lower()
    kind = dataset_kind.lower().replace("-", "_")

    if kind in {"plan", "plan_view", "planview"}:
        allowed = PLAN_ORIENTATIONS
        depth_axis = "z"
        canonical_kind = "plan_view"
    elif kind in {"cross", "cross_section", "crosssection"}:
        allowed = CROSS_ORIENTATIONS
        canonical_kind = "cross_section"
        if orientation in {"xz", "zx"}:
            depth_axis = "y"
        elif orientation in {"yz", "zy"}:
            depth_axis = "x"
        else:
            depth_axis = ""
    else:
        raise ValueError("dataset_kind must identify plan-view or cross-sectional data.")

    if orientation not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported {canonical_kind} orientation {orientation!r}; "
            f"allowed values are {allowed_text}."
        )

    mapping = {
        "dataset_kind": canonical_kind,
        "orientation": orientation,
        "slice": depth_axis,
        "row": orientation[0],
        "col": orientation[1],
    }
    if set((mapping["slice"], mapping["row"], mapping["col"])) != set(PHYSICAL_AXES):
        raise ValueError(f"Orientation {orientation!r} does not map uniquely onto x, y, z.")
    return mapping


def resample_stack_depth(
    normalized_stack: np.ndarray,
    physical_depth: float,
    target_spacing: float,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Linearly interpolate only the slice axis and include both endpoints.

    Raw slice zero is located at ``physical_depth`` and the final raw slice is
    at zero.  The returned stack is indexed by an ascending 0..depth vector.
    """
    stack = np.asarray(normalized_stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected a normalized 3D stack, got {stack.shape}.")
    pixel_size = _positive_finite(target_spacing, "target_spacing")
    depth = _finite_scalar(physical_depth, "physical_depth")
    n_slices = int(stack.shape[0])

    if n_slices == 1:
        if depth != 0.0:
            raise ValueError(
                "A one-slice stack has coincident first/last centers, so physical_depth must be 0."
            )
        coords = np.array([0.0], dtype=np.float64)
        return stack.copy(), coords, {
            "original_stack_spacing": None,
            "resampled_depth_sample_count": 1,
            "effective_resampled_depth_spacing": None,
        }

    if depth <= 0.0:
        raise ValueError("physical_depth must be positive for a stack with multiple slices.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    original_spacing = depth / (n_slices - 1)
    target_count = max(2, int(round(depth / pixel_size)) + 1)
    depth_coords = np.linspace(0.0, depth, target_count, dtype=np.float64)

    # Target coordinate zero maps to the last raw slice; target depth maps to
    # raw slice zero.  Interpolation positions therefore run N-1 -> 0.
    source_positions = np.linspace(n_slices - 1, 0.0, target_count, dtype=np.float64)
    lower = np.floor(source_positions).astype(np.intp)
    upper = np.ceil(source_positions).astype(np.intp)
    weights = (source_positions - lower).astype(np.float32)
    output = np.empty((target_count, stack.shape[1], stack.shape[2]), dtype=np.float32)

    for start in range(0, target_count, chunk_size):
        stop = min(start + chunk_size, target_count)
        weight = weights[start:stop, None, None]
        low_values = stack[lower[start:stop]]
        high_values = stack[upper[start:stop]]
        output[start:stop] = low_values + weight * (high_values - low_values)

    effective_spacing = depth / (target_count - 1)
    return output, depth_coords, {
        "original_stack_spacing": float(original_spacing),
        "resampled_depth_sample_count": int(target_count),
        "effective_resampled_depth_spacing": float(effective_spacing),
    }


def map_stack_to_canonical(
    depth_row_col_stack: np.ndarray,
    orientation_mapping: Mapping[str, str],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Transpose a ``(depth, row, col)`` stack into physical ``(x, y, z)``."""
    stack = np.asarray(depth_row_col_stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got {stack.shape}.")
    source_physical_axes = (
        orientation_mapping["slice"],
        orientation_mapping["row"],
        orientation_mapping["col"],
    )
    permutation = tuple(source_physical_axes.index(axis) for axis in PHYSICAL_AXES)
    return np.transpose(stack, permutation), permutation


def apply_physical_axis_flips(
    volume_xyz: np.ndarray,
    flips: Mapping[str, bool] | None = None,
) -> tuple[np.ndarray, dict[str, bool]]:
    """Reflect data along requested physical axes while retaining coordinates."""
    volume = np.asarray(volume_xyz)
    if volume.ndim != 3:
        raise ValueError(f"Expected a canonical 3D volume, got {volume.shape}.")
    provided = {} if flips is None else dict(flips)
    unknown = set(provided) - set(PHYSICAL_AXES)
    if unknown:
        raise ValueError(f"Unknown physical flip axes: {sorted(unknown)}")

    applied: dict[str, bool] = {}
    for axis_index, axis_name in enumerate(PHYSICAL_AXES):
        value = provided.get(axis_name, False)
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Flip for axis {axis_name!r} must be Boolean.")
        applied[axis_name] = bool(value)
        if value:
            volume = np.flip(volume, axis=axis_index)
    return volume, applied


def process_volume(
    input_data: str | PathLike[str] | np.ndarray,
    *,
    dataset_kind: str,
    orientation: str,
    pixel_size: float,
    physical_depth: float,
    length_unit: str,
    lat_param: float,
    flips: Mapping[str, bool] | None = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the complete raw stack -> normalized -> physical volume workflow."""
    px_size = _positive_finite(pixel_size, "pixel_size")
    lattice_parameter = _positive_finite(lat_param, "lat_param")
    if not isinstance(length_unit, str) or not length_unit.strip():
        raise ValueError("length_unit must be a non-empty string.")

    raw_stack, input_info = load_volume_input(input_data)
    mapping = validate_orientation(orientation, dataset_kind)
    normalized, normalization = normalize_slices(
        raw_stack, lower_percentile, upper_percentile
    )
    resampled, depth_coords, depth_info = resample_stack_depth(
        normalized, physical_depth, px_size
    )
    canonical, permutation = map_stack_to_canonical(resampled, mapping)
    canonical, applied_flips = apply_physical_axis_flips(canonical, flips)

    n_slices, n_rows, n_cols = normalized.shape
    row_coords = np.arange(n_rows - 1, -1, -1, dtype=np.float64) * px_size
    col_coords = np.arange(n_cols, dtype=np.float64) * px_size
    coordinates_by_axis = {
        mapping["slice"]: depth_coords,
        mapping["row"]: row_coords,
        mapping["col"]: col_coords,
    }
    coordinates = {axis: coordinates_by_axis[axis] for axis in PHYSICAL_AXES}

    original_spacing = depth_info["original_stack_spacing"]
    input_geometry = _input_geometry_metadata(
        mapping,
        rows=n_rows,
        pixel_size=px_size,
        physical_depth=float(physical_depth),
        original_stack_spacing=original_spacing,
    )

    meta: dict[str, Any] = {
        **input_info,
        "dataset_kind": mapping["dataset_kind"],
        "normalized_stack_shape": tuple(int(v) for v in normalized.shape),
        "final_resampled_canonical_shape": tuple(int(v) for v in canonical.shape),
        "original_orientation": mapping["orientation"],
        "canonical_axis_order": PHYSICAL_AXES,
        "physical_stack_axis": mapping["slice"],
        "original_slice_count": int(n_slices),
        "original_stack_spacing": original_spacing,
        "requested_physical_stack_depth": float(physical_depth),
        "resampled_depth_sample_count": depth_info["resampled_depth_sample_count"],
        "effective_resampled_depth_spacing": depth_info[
            "effective_resampled_depth_spacing"
        ],
        "in_plane_pixel_size": float(px_size),
        "length_unit": length_unit.strip(),
        "lat_param": float(lattice_parameter),
        "applied_physical_axis_flips": applied_flips,
        "normalization": normalization,
        "interpolation_method": "1D linear interpolation along physical stack/depth axis",
        "intensity_dtype": str(canonical.dtype),
        "intensity_range": (float(canonical.min()), float(canonical.max())),
        "coordinates": coordinates,
        "coordinate_ranges": {
            axis: _coordinate_range(coordinates[axis]) for axis in PHYSICAL_AXES
        },
        "coordinate_direction_with_increasing_index": {
            axis: _coordinate_direction(coordinates[axis]) for axis in PHYSICAL_AXES
        },
        "raw_axis_to_physical_axis": {
            "slice": mapping["slice"],
            "row": mapping["row"],
            "col": mapping["col"],
        },
        "canonical_transpose_permutation_from_resampled_slice_row_col": permutation,
        "input_coordinate_geometry": input_geometry,
        "origin_and_storage_note": (
            "Raw image row values retain their original top-to-bottom order. "
            "Their physical row-axis coordinates decrease from the top row to "
            "zero at the bottom row; locating the bottom-left origin does not "
            "flip image data. The resampled depth coordinate is stored 0..depth, "
            "with the last original slice at zero and the first at maximum depth."
        ),
    }
    return normalized, canonical, meta


def print_volume_summary(name: str, metadata: Mapping[str, Any]) -> None:
    """Print the key shape, mapping, spacing, and coordinate information."""
    unit = metadata["length_unit"]
    mapping = metadata["raw_axis_to_physical_axis"]
    print(f"{name}")
    print(
        f"  shapes: raw {metadata['original_input_shape']} -> normalized "
        f"{metadata['normalized_stack_shape']} -> xyz "
        f"{metadata['final_resampled_canonical_shape']}"
    )
    print(
        "  mapping: "
        f"slice->{mapping['slice']}, row->{mapping['row']}, col->{mapping['col']}"
    )
    print(
        f"  depth: {metadata['requested_physical_stack_depth']:g} {unit}; "
        f"native spacing {metadata['original_stack_spacing']} {unit}; "
        f"resampled {metadata['resampled_depth_sample_count']} points at "
        f"{metadata['effective_resampled_depth_spacing']} {unit}"
    )
    ranges = metadata["coordinate_ranges"]
    print(
        "  ranges: "
        + ", ".join(
            f"{axis}={ranges[axis][0]:g}..{ranges[axis][1]:g} {unit}"
            for axis in PHYSICAL_AXES
        )
    )
    print(f"  physical flips: {metadata['applied_physical_axis_flips']}")


def _input_geometry_metadata(
    mapping: Mapping[str, str],
    *,
    rows: int,
    pixel_size: float,
    physical_depth: float,
    original_stack_spacing: float | None,
) -> dict[str, Any]:
    axis_index = {axis: index for index, axis in enumerate(PHYSICAL_AXES)}
    origin = np.zeros(3, dtype=np.float64)
    origin[axis_index[mapping["row"]]] = (rows - 1) * pixel_size
    origin[axis_index[mapping["slice"]]] = physical_depth

    col_step = np.zeros(3, dtype=np.float64)
    col_step[axis_index[mapping["col"]]] = pixel_size
    row_step = np.zeros(3, dtype=np.float64)
    row_step[axis_index[mapping["row"]]] = -pixel_size
    slice_step = np.zeros(3, dtype=np.float64)
    if original_stack_spacing is not None:
        slice_step[axis_index[mapping["slice"]]] = -original_stack_spacing

    return {
        "coordinate_equation": COORDINATE_EQUATION,
        "vector_component_order": PHYSICAL_AXES,
        "first_slice_origin": origin,
        "col_step_vector": col_step,
        "row_step_vector": row_step,
        "slice_step_vector": slice_step,
        "coordinate_direction_convention": {
            "col": "increasing physical coordinate",
            "row": "decreasing physical coordinate (top row to bottom row)",
            "slice": "decreasing physical coordinate (first slice to last slice)",
        },
    }


def _finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _coordinate_range(values: np.ndarray) -> tuple[float, float]:
    return float(np.min(values)), float(np.max(values))


def _coordinate_direction(values: np.ndarray) -> str:
    if len(values) < 2:
        return "singleton"
    return "increasing" if values[-1] > values[0] else "decreasing"
