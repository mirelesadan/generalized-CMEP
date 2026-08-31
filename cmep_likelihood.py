"""Correlative likelihood score maps from aligned CMEP intensity volumes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


LIKELIHOOD_SCHEMA = "cmep-likelihood-map-v1"


@dataclass
class LikelihoodMapResult:
    """A complete physical correlative likelihood score volume and provenance."""

    likelihood: np.ndarray
    valid_overlap_mask: np.ndarray
    metadata: dict[str, Any]
    map_path: Path | None = None
    metadata_path: Path | None = None

    @property
    def coordinates(self) -> dict[str, np.ndarray]:
        return self.metadata["coordinates"]

    def __repr__(self) -> str:
        return (
            "LikelihoodMapResult("
            f"shape={self.likelihood.shape}, "
            f"valid_voxels={self.metadata['valid_overlap_voxels']:,}, "
            f"positive_voxels={self.metadata['positive_likelihood_voxels']:,}, "
            f"map_path={str(self.map_path)!r})"
        )


def create_likelihood_map(
    aligned_grid: str | Path | Mapping[str, Any],
    *,
    floor_plan: float = 0.10,
    floor_cross: float = 0.10,
    length_unit: str | None = None,
    chunk_voxels: int = 2_000_000,
    output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> LikelihoodMapResult:
    """Create and optionally save a full correlative likelihood score volume.

    Inside the valid overlap, the map is

    ``sqrt(plan_signal * cross_signal)``

    where each signal is independently floor-subtracted, divided by the
    remaining normalized range, and clipped to ``[0, 1]``. Values outside the
    valid overlap are stored as ``NaN``. ``chunk_voxels`` controls temporary
    batch memory only; every voxel is evaluated and no sampling occurs here.
    """
    loaded, source_path, source_hash = load_aligned_grid(aligned_grid)
    likelihood, valid = compute_likelihood(
        loaded["volume_plan"],
        loaded["volume_cross"],
        loaded["valid_overlap_mask"],
        floor_plan=floor_plan,
        floor_cross=floor_cross,
        chunk_voxels=chunk_voxels,
    )
    coordinates = _validate_coordinates(loaded["coordinates"], likelihood.shape)
    unit = str(length_unit) if length_unit is not None else str(
        loaded.get("length_unit", "")
    )
    if not unit.strip():
        raise ValueError(
            "length_unit is absent from the aligned grid; provide it explicitly."
        )
    metadata = _likelihood_metadata(
        likelihood,
        valid,
        coordinates,
        floor_plan=float(floor_plan),
        floor_cross=float(floor_cross),
        length_unit=unit,
        source_path=source_path,
        source_hash=source_hash,
    )
    result = LikelihoodMapResult(likelihood, valid, metadata)

    if output_path is not None:
        map_destination, sidecar_destination = save_likelihood_map(
            result,
            output_path,
            metadata_path=metadata_path,
        )
        result.map_path = map_destination
        result.metadata_path = sidecar_destination
    elif metadata_path is not None:
        raise ValueError("metadata_path requires output_path.")
    return result


def compute_likelihood(
    volume_plan: np.ndarray,
    volume_cross: np.ndarray,
    valid_overlap_mask: np.ndarray | None = None,
    *,
    floor_plan: float = 0.10,
    floor_cross: float = 0.10,
    chunk_voxels: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return likelihood scores and validity without changing either input."""
    plan = np.asarray(volume_plan)
    cross = np.asarray(volume_cross)
    if plan.ndim != 3 or cross.ndim != 3:
        raise ValueError(
            "volume_plan and volume_cross must both be 3D in (x, y, z) order."
        )
    if plan.shape != cross.shape:
        raise ValueError(
            f"Aligned volume shapes differ: plan={plan.shape}, cross={cross.shape}."
        )
    plan_floor = _likelihood_floor(floor_plan, "floor_plan")
    cross_floor = _likelihood_floor(floor_cross, "floor_cross")
    chunk_size = _positive_integer(chunk_voxels, "chunk_voxels")

    finite = np.isfinite(plan) & np.isfinite(cross)
    if valid_overlap_mask is None:
        valid = finite
    else:
        supplied = np.asarray(valid_overlap_mask)
        if supplied.shape != plan.shape:
            raise ValueError(
                "valid_overlap_mask must have the same shape as both volumes."
            )
        if supplied.dtype.kind != "b":
            raise TypeError("valid_overlap_mask must be Boolean.")
        valid = supplied & finite

    result = np.full(plan.shape, np.nan, dtype=np.float32)
    yz_size = max(1, plan.shape[1] * plan.shape[2])
    slab_width = max(1, chunk_size // yz_size)
    plan_scale = np.float32(1.0 / (1.0 - plan_floor))
    cross_scale = np.float32(1.0 / (1.0 - cross_floor))

    for start in range(0, plan.shape[0], slab_width):
        stop = min(start + slab_width, plan.shape[0])
        slab_valid = valid[start:stop]
        if not np.any(slab_valid):
            continue
        plan_signal = np.asarray(plan[start:stop], dtype=np.float32)
        cross_signal = np.asarray(cross[start:stop], dtype=np.float32)
        plan_signal = np.clip(
            (plan_signal - np.float32(plan_floor)) * plan_scale, 0.0, 1.0
        )
        cross_signal = np.clip(
            (cross_signal - np.float32(cross_floor)) * cross_scale, 0.0, 1.0
        )
        combined = np.sqrt(plan_signal * cross_signal).astype(np.float32, copy=False)
        destination = result[start:stop]
        destination[slab_valid] = combined[slab_valid]

    return result, valid.astype(np.bool_, copy=False)


def load_aligned_grid(
    aligned_grid: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None, str | None]:
    """Load the aligned NPZ format or normalize an in-memory aligned mapping."""
    if isinstance(aligned_grid, Mapping):
        coordinates_value = aligned_grid.get("coordinates")
        if not isinstance(coordinates_value, Mapping):
            raise ValueError("Aligned mapping must contain coordinates for x, y, and z.")
        data = {
            "coordinates": {
                axis: np.asarray(coordinates_value[axis], dtype=np.float64)
                for axis in "xyz"
            },
            "volume_plan": np.asarray(aligned_grid["volume_plan"]),
            "volume_cross": np.asarray(aligned_grid["volume_cross"]),
            "valid_overlap_mask": np.asarray(aligned_grid["valid_overlap_mask"]),
            "length_unit": _mapping_length_unit(aligned_grid),
        }
        return data, None, None

    source = Path(aligned_grid).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Aligned-grid NPZ does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "x",
                "y",
                "z",
                "volume_plan",
                "volume_cross",
                "valid_overlap_mask",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(
                    "Aligned-grid NPZ is missing arrays: " + ", ".join(missing)
                )
            data = {
                "coordinates": {
                    axis: np.asarray(archive[axis], dtype=np.float64) for axis in "xyz"
                },
                "volume_plan": np.asarray(archive["volume_plan"]),
                "volume_cross": np.asarray(archive["volume_cross"]),
                "valid_overlap_mask": np.asarray(archive["valid_overlap_mask"]),
                "length_unit": _npz_text(archive, "length_unit"),
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to read aligned-grid NPZ {source}: {exc}") from exc
    return data, source, _sha256_file(source)


def save_likelihood_map(
    result: LikelihoodMapResult,
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Atomically save the full float32 score map, coordinates, and metadata."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else destination.with_suffix(".json")
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    serializable = _serializable_metadata(result.metadata)
    metadata_json = json.dumps(serializable, sort_keys=True)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            x=np.asarray(result.coordinates["x"], dtype=np.float64),
            y=np.asarray(result.coordinates["y"], dtype=np.float64),
            z=np.asarray(result.coordinates["z"], dtype=np.float64),
            likelihood=np.asarray(result.likelihood, dtype=np.float32),
            valid_overlap_mask=np.asarray(result.valid_overlap_mask, dtype=np.bool_),
            metadata_json=np.asarray(metadata_json),
        )
    temporary.replace(destination)

    sidecar_text = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_sidecar.write_text(sidecar_text, encoding="utf-8", newline="\n")
    temporary_sidecar.replace(sidecar)
    return destination, sidecar


def load_likelihood_map(path: str | Path) -> LikelihoodMapResult:
    """Load and validate a previously saved likelihood-score-map NPZ."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Likelihood-score-map NPZ does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "x",
                "y",
                "z",
                "likelihood",
                "valid_overlap_mask",
                "metadata_json",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(
                    "Likelihood-score-map NPZ is missing arrays: "
                    + ", ".join(missing)
                )
            likelihood = np.asarray(archive["likelihood"], dtype=np.float32)
            valid = np.asarray(archive["valid_overlap_mask"], dtype=np.bool_)
            metadata = json.loads(str(archive["metadata_json"].item()))
            metadata["coordinates"] = {
                axis: np.asarray(archive[axis], dtype=np.float64) for axis in "xyz"
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read likelihood-score-map NPZ {source}: {exc}"
        ) from exc
    _validate_loaded_likelihood(likelihood, valid, metadata)
    sidecar = source.with_suffix(".json")
    return LikelihoodMapResult(
        likelihood,
        valid,
        metadata,
        map_path=source,
        metadata_path=sidecar if sidecar.is_file() else None,
    )


def print_likelihood_summary(result: LikelihoodMapResult, threshold: float | None = None) -> None:
    """Print map provenance and, optionally, an exact display selection count."""
    metadata = result.metadata
    print("Correlative likelihood score map")
    print(
        f"  shape={result.likelihood.shape}, dtype={result.likelihood.dtype}, "
        f"valid overlap={metadata['valid_overlap_voxels']:,} voxels"
    )
    print(
        "  positive likelihood score="
        f"{metadata['positive_likelihood_voxels']:,} voxels, "
        f"range={metadata['finite_likelihood_range']}"
    )
    print(
        f"  floors: plan={metadata['floor_plan']:.4g}, "
        f"cross={metadata['floor_cross']:.4g}"
    )
    if threshold is not None:
        threshold_value = _normalized_scalar(threshold, "threshold")
        count = int(np.count_nonzero(result.likelihood >= threshold_value))
        print(f"  display threshold >= {threshold_value:.4g}: {count:,} voxels")
    if result.map_path is not None:
        print(f"  likelihood score map: {result.map_path}")


def make_likelihood_histogram(
    likelihood: LikelihoodMapResult | np.ndarray,
    *,
    threshold: float = 0.50,
    bins: int = 60,
    colormap: str | Sequence[Any] = "magma",
    background_color: str = "black",
    log_y: bool = False,
):
    """Return a pre-binned Plotly histogram above a likelihood-score threshold.

    Only finite values satisfying ``likelihood >= threshold`` contribute. The
    map is not modified, and only compact bin counts are sent to the browser.
    """
    data = (
        likelihood.likelihood
        if isinstance(likelihood, LikelihoodMapResult)
        else np.asarray(likelihood)
    )
    if data.ndim != 3:
        raise ValueError(
            f"likelihood score map must be 3D in (x, y, z) order; got {data.shape}."
        )
    threshold_value = _normalized_scalar(threshold, "threshold")
    bin_count = _positive_integer(bins, "bins")
    if not isinstance(log_y, (bool, np.bool_)):
        raise TypeError("log_y must be Boolean.")

    finite = data[np.isfinite(data)]
    if finite.size and (
        float(np.min(finite)) < -1e-6 or float(np.max(finite)) > 1.0 + 1e-6
    ):
        raise ValueError("Finite likelihood scores must lie in [0, 1].")
    selected = finite[finite >= threshold_value]

    if threshold_value < 1.0:
        edges = np.linspace(threshold_value, 1.0, bin_count + 1)
        counts, edges = np.histogram(selected, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        x_range = [threshold_value, 1.0]
    else:
        counts = np.array([selected.size], dtype=np.int64)
        centers = np.array([1.0])
        widths = np.array([0.01])
        x_range = [0.99, 1.0]

    foreground = _inverted_color(background_color)
    go = _plotly_graph_objects()
    figure = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            marker={
                "color": centers,
                "colorscale": colormap,
                "cmin": 0.0,
                "cmax": 1.0,
                "showscale": True,
                "colorbar": {
                    "title": {
                        "text": "Likelihood score",
                        "font": {"color": foreground},
                    },
                    "tickfont": {"color": foreground},
                },
                "line": {"width": 0},
            },
            hovertemplate=(
                "Likelihood score: %{x:.4f}<br>Voxels: %{y:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=(
            f"Correlative likelihood score distribution: {selected.size:,} voxels "
            f">= {threshold_value:.4g}"
        ),
        xaxis_title="Likelihood score",
        yaxis_title="Voxel count",
        bargap=0,
        showlegend=False,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font={"color": foreground},
        margin={"l": 70, "r": 90, "t": 70, "b": 65},
    )
    figure.update_xaxes(range=x_range, gridcolor=foreground, gridwidth=0.35)
    figure.update_yaxes(
        type="log" if bool(log_y) else "linear",
        gridcolor=foreground,
        gridwidth=0.35,
        tickformat=",",
    )
    if not bool(log_y):
        figure.update_yaxes(rangemode="tozero")
    return figure


def _likelihood_metadata(
    likelihood: np.ndarray,
    valid: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    *,
    floor_plan: float,
    floor_cross: float,
    length_unit: str,
    source_path: Path | None,
    source_hash: str | None,
) -> dict[str, Any]:
    finite_values = likelihood[valid]
    finite_range = (
        [float(np.min(finite_values)), float(np.max(finite_values))]
        if finite_values.size
        else [None, None]
    )
    return {
        "schema_version": LIKELIHOOD_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_kind": "correlative_likelihood_score",
        "quantity_name": "correlative_likelihood_score",
        "axis_order": ["x", "y", "z"],
        "shape_xyz": list(likelihood.shape),
        "voxel_count": int(likelihood.size),
        "intensity_dtype": "float32",
        "likelihood_range": [0.0, 1.0],
        "finite_likelihood_range": finite_range,
        "outside_overlap_value": "NaN",
        "valid_overlap_voxels": int(np.count_nonzero(valid)),
        "positive_likelihood_voxels": int(np.count_nonzero(likelihood > 0.0)),
        "floor_plan": float(floor_plan),
        "floor_cross": float(floor_cross),
        "formula": (
            "sqrt(clip((plan-floor_plan)/(1-floor_plan),0,1) * "
            "clip((cross-floor_cross)/(1-floor_cross),0,1))"
        ),
        "length_unit": length_unit,
        "coordinate_ranges": {
            axis: [float(values[0]), float(values[-1])]
            for axis, values in coordinates.items()
        },
        "coordinates": dict(coordinates),
        "source_aligned_grid_path": str(source_path) if source_path else None,
        "source_aligned_grid_sha256": source_hash,
    }


def _validate_loaded_likelihood(
    likelihood: np.ndarray,
    valid: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    if likelihood.ndim != 3:
        raise ValueError(
            "Saved likelihood score map must be 3D in (x, y, z) order."
        )
    if valid.shape != likelihood.shape:
        raise ValueError(
            "Saved likelihood score and valid-overlap mask shapes differ."
        )
    if metadata.get("schema_version") != LIKELIHOOD_SCHEMA:
        raise ValueError("Unsupported likelihood-score-map schema version.")
    _validate_coordinates(metadata.get("coordinates", {}), likelihood.shape)
    if np.any(np.isfinite(likelihood[~valid])):
        raise ValueError(
            "Likelihood scores outside the valid overlap must be NaN."
        )
    finite = likelihood[valid]
    if finite.size and (
        not np.isfinite(finite).all()
        or float(np.min(finite)) < -1e-6
        or float(np.max(finite)) > 1.0 + 1e-6
    ):
        raise ValueError(
            "Valid likelihood scores must be finite and lie in [0, 1]."
        )


def _validate_coordinates(
    coordinates: Mapping[str, Any], shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    if not isinstance(coordinates, Mapping):
        raise ValueError("Coordinates must map x, y, and z to vectors.")
    result: dict[str, np.ndarray] = {}
    for axis_index, axis in enumerate("xyz"):
        if axis not in coordinates:
            raise ValueError(f"Coordinates are missing axis {axis!r}.")
        values = np.asarray(coordinates[axis], dtype=np.float64)
        if values.ndim != 1 or len(values) != shape[axis_index]:
            raise ValueError(
                f"Coordinate {axis!r} must be 1D with length {shape[axis_index]}."
            )
        if not np.isfinite(values).all() or np.any(np.diff(values) <= 0.0):
            raise ValueError(f"Coordinate {axis!r} must be finite and increasing.")
        result[axis] = values
    return result


def _serializable_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "coordinates"}


def _mapping_length_unit(aligned_grid: Mapping[str, Any]) -> str:
    direct = aligned_grid.get("length_unit")
    if direct is not None:
        return str(direct)
    metadata = aligned_grid.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("length_unit") is not None:
        return str(metadata["length_unit"])
    return ""


def _npz_text(archive: Any, name: str) -> str:
    if name not in archive.files:
        return ""
    value = archive[name]
    return str(value.item()) if np.asarray(value).ndim == 0 else str(value)


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def _likelihood_floor(value: float, name: str) -> float:
    result = _normalized_scalar(value, name)
    if result >= 1.0:
        raise ValueError(f"{name} must be less than 1 so the signal can be rescaled.")
    return result


def _normalized_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not Boolean.")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _inverted_color(background_color: str) -> str:
    if not isinstance(background_color, str) or not background_color.strip():
        raise TypeError("background_color must be a non-empty color string.")
    try:
        from PIL import ImageColor

        red, green, blue = ImageColor.getrgb(background_color.strip())[:3]
    except (ImportError, ValueError) as exc:
        raise ValueError(f"Unsupported background color: {background_color!r}.") from exc
    return f"rgb({255 - red},{255 - green},{255 - blue})"


def _plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Likelihood-score histograms require plotly from requirements-cmep.txt."
        ) from exc
    return go
