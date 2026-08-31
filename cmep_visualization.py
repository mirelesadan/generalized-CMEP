"""Interactive Plotly visualization for prepared CMEP intensity volumes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def select_threshold_points(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    threshold: float,
    max_points: int = 150_000,
    chunk_voxels: int = 2_000_000,
) -> dict[str, Any]:
    """Select evenly distributed thresholded points without changing the volume.

    Selection is a two-pass, chunked operation.  If more than ``max_points``
    voxels pass the threshold, evenly spaced ranks from the full thresholded
    set are retained for visualization only.
    """
    data, coordinate_arrays = _validated_volume_and_coordinates(volume, metadata)
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")
    if max_points < 1:
        raise ValueError("max_points must be a positive integer.")
    if chunk_voxels < 1:
        raise ValueError("chunk_voxels must be a positive integer.")

    slab_size = max(1, chunk_voxels // max(1, data.shape[1] * data.shape[2]))
    threshold_count = 0
    for start in range(0, data.shape[0], slab_size):
        stop = min(start + slab_size, data.shape[0])
        threshold_count += int(np.count_nonzero(data[start:stop] >= threshold_value))

    sample_count = min(threshold_count, int(max_points))
    if sample_count == 0:
        return _empty_point_set()

    target_ranks = (
        np.arange(sample_count, dtype=np.int64) * threshold_count // sample_count
    )
    selected_linear = np.empty(sample_count, dtype=np.int64)
    cumulative = 0
    selected_position = 0
    yz_size = data.shape[1] * data.shape[2]

    for start in range(0, data.shape[0], slab_size):
        stop = min(start + slab_size, data.shape[0])
        local_hits = np.flatnonzero(data[start:stop].reshape(-1) >= threshold_value)
        local_count = int(local_hits.size)
        if local_count:
            left = int(np.searchsorted(target_ranks, cumulative, side="left"))
            right = int(
                np.searchsorted(target_ranks, cumulative + local_count, side="left")
            )
            if right > left:
                requested_local_ranks = target_ranks[left:right] - cumulative
                chosen = local_hits[requested_local_ranks]
                selected_linear[selected_position : selected_position + len(chosen)] = (
                    chosen + start * yz_size
                )
                selected_position += len(chosen)
        cumulative += local_count

    if selected_position != sample_count:
        raise RuntimeError("Internal threshold sampling count mismatch.")

    return _point_set_from_linear_indices(
        data,
        coordinate_arrays,
        selected_linear,
        threshold_count=threshold_count,
    )


def make_volume_figure(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    threshold: float = 0.70,
    color: str = "red",
    max_voxels: int = 40_000,
    voxel_scale: float = 0.92,
    minimum_voxel_alpha: float = 0.30,
    background_color: str = "black",
    figure_size: int = 800,
):
    """Build an interactive Plotly 3D thresholded voxel rendering."""
    go = _plotly()
    points = select_threshold_points(
        volume, metadata, threshold=threshold, max_points=max_voxels
    )
    figure = go.Figure(
        data=[
            _make_voxel_mesh(
                go,
                points,
                metadata,
                color=color,
                voxel_scale=voxel_scale,
                minimum_voxel_alpha=minimum_voxel_alpha,
                threshold=threshold,
            )
        ]
    )
    _configure_figure(
        figure,
        metadata,
        threshold,
        points,
        background_color=background_color,
        figure_size=figure_size,
    )
    return figure


def visualize_volume(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    color: str = "red",
    threshold: float = 0.70,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_step: float = 0.10,
    threshold_count: int | None = None,
    threshold_values: Sequence[float] | None = None,
    max_voxels: int = 40_000,
    voxel_scale: float = 0.92,
    minimum_voxel_alpha: float = 0.30,
    background_color: str = "black",
    figure_size: int = 800,
):
    """Return a standard Plotly figure with an optional native threshold slider.

    Plotly supplies its normal 3D rotate, zoom, pan, and hover interaction.  The
    slider switches physical voxel meshes by intensity band; setting
    ``threshold_count=1`` uses only ``threshold`` and omits the slider.  The
    numerical volume is never downsampled or modified.  A normal ``go.Figure``
    avoids the Jupyter widget-manager dependency required by ``FigureWidget``.
    """
    go = _plotly()
    thresholds = _threshold_grid(
        threshold,
        threshold_values,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        threshold_count=threshold_count,
    )
    initial_index = int(np.argmin(np.abs(thresholds - float(threshold))))
    band_sets, threshold_summaries = _select_threshold_bands(
        volume,
        metadata,
        thresholds,
        max_voxels=max_voxels,
    )
    traces = []
    for index, points in enumerate(band_sets):
        trace = _make_voxel_mesh(
            go,
            points,
            metadata,
            color=color,
            voxel_scale=voxel_scale,
            minimum_voxel_alpha=minimum_voxel_alpha,
            threshold=float(thresholds[initial_index]),
        )
        trace.visible = index >= initial_index
        traces.append(trace)

    figure = go.Figure(data=traces)
    _configure_figure(
        figure,
        metadata,
        float(thresholds[initial_index]),
        threshold_summaries[initial_index],
        background_color=background_color,
        figure_size=figure_size,
    )
    if len(thresholds) == 1:
        return figure

    text_color = _inverted_color(background_color)
    steps = []
    for index, value in enumerate(thresholds):
        visible = [trace_index >= index for trace_index in range(len(traces))]
        steps.append(
            {
                "method": "update",
                "label": f"{value:.2f}",
                "args": [
                    {
                        "visible": visible,
                        "cmin": [_color_minimum(float(value))] * len(traces),
                    },
                    {
                        "title": {
                            "text": _title_text(
                                metadata,
                                float(value),
                                threshold_summaries[index],
                            ),
                            "x": 0.02,
                        }
                    },
                ],
            }
        )
    figure.update_layout(
        sliders=[
            {
                "active": initial_index,
                "font": {"color": text_color},
                "currentvalue": {
                    "prefix": "Threshold: ",
                    "font": {"color": text_color},
                },
                "pad": {"t": 18},
                "steps": steps,
            }
        ],
        margin={"l": 0, "r": 0, "t": 52, "b": 70},
    )
    return figure


def _make_voxel_mesh(
    go,
    points: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    color: str,
    voxel_scale: float,
    minimum_voxel_alpha: float,
    threshold: float,
):
    scale = float(voxel_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("voxel_scale must be a positive finite number.")

    coordinates = metadata["coordinates"]
    half_sizes = np.array(
        [_axis_spacing(np.asarray(coordinates[axis])) for axis in "xyz"],
        dtype=np.float64,
    ) * (0.5 * scale)
    centers = np.column_stack((points["x"], points["y"], points["z"]))
    corners = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices = (
        centers[:, None, :] + corners[None, :, :] * half_sizes
    ).reshape(-1, 3).astype(np.float32, copy=False)
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    offsets = np.arange(len(centers), dtype=np.uint32)[:, None, None] * 8
    triangles = (offsets + faces[None, :, :]).reshape(-1, 3)
    vertex_intensities = np.repeat(
        np.asarray(points["intensity"], dtype=np.float32), 8
    )

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        intensity=vertex_intensities,
        intensitymode="vertex",
        colorscale=_alpha_colorscale(color, minimum_voxel_alpha),
        cmin=_color_minimum(threshold),
        cmax=1.0,
        showscale=False,
        opacity=1.0,
        flatshading=True,
        lighting={
            "ambient": 0.55,
            "diffuse": 0.8,
            "specular": 0.08,
            "roughness": 0.9,
        },
        hoverinfo="skip",
        showlegend=False,
    )


def _configure_figure(
    figure,
    metadata: Mapping[str, Any],
    threshold: float,
    points: Mapping[str, Any],
    *,
    background_color: str,
    figure_size: int,
) -> None:
    size = int(figure_size)
    if size < 300:
        raise ValueError("figure_size must be at least 300 pixels.")
    unit = metadata.get("length_unit", "")
    foreground_color = _inverted_color(background_color)

    def axis_style(title: str) -> dict[str, Any]:
        return {
            "title": {"text": title, "font": {"color": foreground_color}},
            "color": foreground_color,
            "showbackground": False,
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": foreground_color,
            "linewidth": 2,
            "ticks": "outside",
            "tickcolor": foreground_color,
            "tickfont": {"color": foreground_color},
        }

    figure.update_layout(
        title={"text": _title_text(metadata, threshold, points), "x": 0.02},
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        width=size,
        height=size,
        autosize=False,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font={"color": foreground_color},
        scene={
            "bgcolor": background_color,
            "xaxis": axis_style(f"x ({unit})"),
            "yaxis": axis_style(f"y ({unit})"),
            "zaxis": axis_style(f"z ({unit})"),
            "aspectmode": "data",
            "dragmode": "orbit",
        },
        uirevision="keep-camera",
    )


def _alpha_colorscale(color: str, minimum_alpha: float) -> list[list[Any]]:
    alpha = float(minimum_alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("minimum_voxel_alpha must lie in [0, 1].")
    red, green, blue = _parse_rgb(color)
    return [
        [0.0, f"rgba({red},{green},{blue},{alpha:g})"],
        [1.0, f"rgba({red},{green},{blue},1)"],
    ]


def _color_minimum(threshold: float) -> float:
    """Keep Plotly's color interval nonzero when the threshold equals one."""
    return min(float(threshold), float(np.nextafter(1.0, 0.0)))


def _inverted_color(background_color: str) -> str:
    red, green, blue = _parse_rgb(background_color)
    return f"rgb({255 - red},{255 - green},{255 - blue})"


def _parse_rgb(color: str) -> tuple[int, int, int]:
    if not isinstance(color, str) or not color.strip():
        raise TypeError("Plot colors must be non-empty strings.")
    try:
        from PIL import ImageColor

        parsed = ImageColor.getrgb(color.strip())
    except (ImportError, ValueError) as exc:
        raise ValueError(f"Unsupported color value: {color!r}.") from exc
    return tuple(int(component) for component in parsed[:3])


def _title_text(
    metadata: Mapping[str, Any],
    threshold: float,
    points: Mapping[str, Any],
) -> str:
    label = str(metadata.get("dataset_kind", "volume")).replace("_", " ").title()
    if points["visualization_downsampled"]:
        count_text = (
            f"{points['displayed_count']:,} of "
            f"{points['threshold_count']:,} displayed"
        )
    else:
        count_text = f"{points['displayed_count']:,} displayed"
    return (
        f"{label}: intensity >= {threshold:.2f} "
        f"({count_text})"
    )


def _threshold_grid(
    threshold: float,
    threshold_values: Sequence[float] | None,
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    threshold_count: int | None,
) -> np.ndarray:
    initial = float(threshold)
    if not np.isfinite(initial) or not 0.0 <= initial <= 1.0:
        raise ValueError("threshold must be finite and lie in [0, 1].")

    count: int | None = None
    if threshold_count is not None:
        if isinstance(threshold_count, (bool, np.bool_)):
            raise TypeError("threshold_count must be a positive integer.")
        try:
            count = int(threshold_count)
        except (TypeError, ValueError) as exc:
            raise TypeError("threshold_count must be a positive integer.") from exc
        if count < 1 or count != threshold_count:
            raise ValueError("threshold_count must be a positive integer.")
        if threshold_values is not None:
            raise ValueError(
                "Use either threshold_count or threshold_values, not both."
            )
        if count == 1:
            return np.array([initial], dtype=np.float64)

    minimum = float(threshold_min)
    maximum = float(threshold_max)
    step = float(threshold_step)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("threshold_min and threshold_max must be finite.")
    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError("Slider bounds must satisfy 0 <= min < max <= 1.")
    if not minimum <= initial <= maximum:
        raise ValueError("threshold must lie within the configured slider bounds.")
    if count is not None:
        return np.linspace(minimum, maximum, count, dtype=np.float64)

    if threshold_values is None:
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("threshold_step must be a positive finite number.")
        interval_count = int(np.floor((maximum - minimum) / step + 1e-12))
        values = minimum + np.arange(interval_count + 1, dtype=np.float64) * step
        if values[-1] < maximum - 1e-12:
            values = np.append(values, maximum)
        else:
            values[-1] = maximum
    else:
        values = np.asarray(list(threshold_values), dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("threshold_values must be a non-empty 1D sequence.")
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Every threshold value must be finite and lie in [0, 1].")
        if np.any((values < minimum) | (values > maximum)):
            raise ValueError("threshold_values must lie within the configured slider bounds.")

    return np.unique(np.round(np.append(values, [minimum, maximum, initial]), 12))


def _select_threshold_bands(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    thresholds: np.ndarray,
    *,
    max_voxels: int,
    chunk_voxels: int = 2_000_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select deterministic intensity-band samples under one global cap.

    The slider thresholds define disjoint intensity bands.  The display budget
    is shared approximately equally among nonempty bands, with unused capacity
    redistributed.  Within each band, evenly spaced ranks from flattened
    C-order voxel indices are retained in a chunked second pass; no random
    sampling or modification of the numerical volume is involved.
    """
    data, coordinate_arrays = _validated_volume_and_coordinates(volume, metadata)
    if max_voxels < 1:
        raise ValueError("max_voxels must be a positive integer.")
    if chunk_voxels < 1:
        raise ValueError("chunk_voxels must be a positive integer.")

    thresholds = np.asarray(thresholds, dtype=np.float64)
    bin_counts = np.zeros(len(thresholds), dtype=np.int64)
    slab_size = max(1, chunk_voxels // max(1, data.shape[1] * data.shape[2]))

    for start in range(0, data.shape[0], slab_size):
        stop = min(start + slab_size, data.shape[0])
        values = data[start:stop].reshape(-1)
        eligible = values >= thresholds[0]
        if eligible.any():
            bin_indices = np.searchsorted(
                thresholds, values[eligible], side="right"
            ) - 1
            bin_counts += np.bincount(
                bin_indices, minlength=len(thresholds)
            ).astype(np.int64)

    sample_counts = _allocate_band_samples(bin_counts, int(max_voxels))
    selected_by_bin: list[np.ndarray] = []
    target_ranks: list[np.ndarray] = []
    for count, sample_count_value in zip(bin_counts, sample_counts):
        sample_count = int(sample_count_value)
        selected_by_bin.append(np.empty(sample_count, dtype=np.int64))
        if sample_count:
            target_ranks.append(
                np.arange(sample_count, dtype=np.int64) * int(count) // sample_count
            )
        else:
            target_ranks.append(np.empty(0, dtype=np.int64))

    cumulative = np.zeros(len(thresholds), dtype=np.int64)
    selected_positions = np.zeros(len(thresholds), dtype=np.int64)
    yz_size = data.shape[1] * data.shape[2]

    for start in range(0, data.shape[0], slab_size):
        stop = min(start + slab_size, data.shape[0])
        values = data[start:stop].reshape(-1)
        eligible = values >= thresholds[0]
        if not eligible.any():
            continue
        eligible_linear = np.flatnonzero(eligible)
        bin_indices = np.searchsorted(
            thresholds, values[eligible], side="right"
        ) - 1

        for bin_index in range(len(thresholds)):
            local_hits = eligible_linear[bin_indices == bin_index]
            local_count = int(local_hits.size)
            if not local_count:
                continue
            ranks = target_ranks[bin_index]
            left = int(
                np.searchsorted(ranks, cumulative[bin_index], side="left")
            )
            right = int(
                np.searchsorted(
                    ranks, cumulative[bin_index] + local_count, side="left"
                )
            )
            if right > left:
                requested = ranks[left:right] - cumulative[bin_index]
                chosen = local_hits[requested] + start * yz_size
                position = int(selected_positions[bin_index])
                selected_by_bin[bin_index][position : position + len(chosen)] = chosen
                selected_positions[bin_index] += len(chosen)
            cumulative[bin_index] += local_count

    expected_positions = np.array([len(values) for values in selected_by_bin])
    if not np.array_equal(selected_positions, expected_positions):
        raise RuntimeError("Internal multi-threshold sampling count mismatch.")

    band_sets: list[dict[str, Any]] = []
    for bin_index, selected in enumerate(selected_by_bin):
        if bin_counts[bin_index] == 0:
            band_sets.append(_empty_point_set())
        else:
            band_sets.append(
                _point_set_from_linear_indices(
                    data,
                    coordinate_arrays,
                    selected,
                    threshold_count=int(bin_counts[bin_index]),
                )
            )

    threshold_counts = np.cumsum(bin_counts[::-1])[::-1]
    displayed_counts = np.cumsum(sample_counts[::-1])[::-1]
    summaries = [
        {
            "threshold_count": int(threshold_counts[index]),
            "displayed_count": int(displayed_counts[index]),
            "visualization_downsampled": (
                threshold_counts[index] > displayed_counts[index]
            ),
        }
        for index in range(len(thresholds))
    ]
    return band_sets, summaries


def _allocate_band_samples(bin_counts: np.ndarray, max_voxels: int) -> np.ndarray:
    allocations = np.zeros_like(bin_counts, dtype=np.int64)
    remaining = min(int(max_voxels), int(np.sum(bin_counts)))
    active = [index for index, count in enumerate(bin_counts) if count > 0]

    while remaining and active:
        share = max(1, remaining // len(active))
        progressed = 0
        for index in reversed(active):
            available = int(bin_counts[index] - allocations[index])
            take = min(available, share, remaining)
            allocations[index] += take
            remaining -= take
            progressed += take
            if remaining == 0:
                break
        if progressed == 0:
            break
        active = [
            index for index in active if allocations[index] < bin_counts[index]
        ]
    return allocations


def _axis_spacing(coordinates: np.ndarray) -> float:
    if coordinates.ndim != 1 or len(coordinates) < 2:
        return 1.0
    differences = np.abs(np.diff(coordinates.astype(np.float64)))
    positive = differences[differences > 0.0]
    return float(np.median(positive)) if len(positive) else 1.0


def _validated_volume_and_coordinates(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.asarray(volume)
    if data.ndim != 3:
        raise ValueError(f"volume must be 3D in (x, y, z) order; got {data.shape}.")
    coordinates = metadata.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("metadata must contain x, y, z coordinate vectors.")
    coordinate_arrays = {axis: np.asarray(coordinates[axis]) for axis in "xyz"}
    for axis_index, axis_name in enumerate("xyz"):
        if coordinate_arrays[axis_name].ndim != 1:
            raise ValueError(f"Coordinate vector {axis_name!r} must be one-dimensional.")
        if len(coordinate_arrays[axis_name]) != data.shape[axis_index]:
            raise ValueError(
                f"Coordinate vector {axis_name!r} has length "
                f"{len(coordinate_arrays[axis_name])}, expected {data.shape[axis_index]}."
            )
    return data, coordinate_arrays


def _point_set_from_linear_indices(
    data: np.ndarray,
    coordinate_arrays: Mapping[str, np.ndarray],
    linear_indices: np.ndarray,
    *,
    threshold_count: int,
) -> dict[str, Any]:
    ix, iy, iz = np.unravel_index(linear_indices, data.shape)
    displayed_count = int(len(linear_indices))
    return {
        "x": coordinate_arrays["x"][ix],
        "y": coordinate_arrays["y"][iy],
        "z": coordinate_arrays["z"][iz],
        "intensity": np.asarray(data[ix, iy, iz], dtype=np.float32),
        "threshold_count": int(threshold_count),
        "displayed_count": displayed_count,
        "visualization_downsampled": threshold_count > displayed_count,
    }


def _empty_point_set() -> dict[str, Any]:
    empty_float = np.empty(0, dtype=np.float64)
    return {
        "x": empty_float,
        "y": empty_float.copy(),
        "z": empty_float.copy(),
        "intensity": np.empty(0, dtype=np.float32),
        "threshold_count": 0,
        "displayed_count": 0,
        "visualization_downsampled": False,
    }


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Interactive visualization requires plotly. Install the packages "
            "listed in requirements-cmep.txt."
        ) from exc
    return go
