"""Multiresolution correlative registration for prepared CMEP volumes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cmep_alignment import (
    AXES,
    DATASETS,
    load_alignment_state,
    sample_volume_at_world_points,
    transform_physical_points,
    transformed_box_corners,
    validate_alignment_state,
)


CORRELATIVE_SCHEMA = "cmep-correlative-alignment-v2"


@dataclass(frozen=True)
class WorldGrid:
    """Regular physical xyz grid used only for scoring or aligned output."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    requested_spacing: float

    @property
    def coordinates(self) -> dict[str, np.ndarray]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.x), len(self.y), len(self.z)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def bounds(self) -> np.ndarray:
        return np.array(
            [[values[0], values[-1]] for values in (self.x, self.y, self.z)],
            dtype=np.float64,
        )

    @property
    def effective_spacing(self) -> tuple[float, float, float]:
        return tuple(_axis_spacing(values) for values in (self.x, self.y, self.z))

    def metadata(self) -> dict[str, Any]:
        return {
            "shape_xyz": list(self.shape),
            "voxel_count": self.size,
            "requested_spacing": float(self.requested_spacing),
            "effective_spacing_xyz": list(self.effective_spacing),
            "bounds_xyz": self.bounds.tolist(),
        }


@dataclass
class CorrelativeRegistrationResult:
    """Optimized transforms plus optional aligned physical-grid intensities."""

    summary: dict[str, Any]
    optimized_alignment_state: dict[str, Any]
    aligned_grid: dict[str, Any] | None
    state_path: Path | None
    aligned_grid_path: Path | None

    @property
    def optimized_transforms(self) -> dict[str, np.ndarray]:
        return {
            dataset: np.asarray(matrix, dtype=np.float64)
            for dataset, matrix in self.optimized_alignment_state["transforms"].items()
        }

    def __repr__(self) -> str:
        optimization = self.summary["optimization"]
        return (
            "CorrelativeRegistrationResult("
            f"moving={optimization['moving_dataset']!r}, "
            f"initial_score={optimization['initial_score']:.6f}, "
            f"optimized_score={optimization['optimized_score']:.6f}, "
            f"state_path={str(self.state_path)!r})"
        )


class NormalizedProductScorer:
    """Score a moving transform at fixed-grid locations that can contribute.

    The product numerator is exactly zero wherever the fixed signal is below
    ``intensity_floor``, so candidate interpolation is restricted to active
    fixed locations.  A support-coverage factor prevents a small, bright subset
    from receiving a high cosine score when most signal does not overlap.
    """

    def __init__(
        self,
        fixed_volume: np.ndarray,
        fixed_metadata: Mapping[str, Any],
        fixed_transform: Sequence[Sequence[float]],
        moving_volume: np.ndarray,
        moving_metadata: Mapping[str, Any],
        moving_normalization_transform: Sequence[Sequence[float]],
        grid: WorldGrid,
        *,
        intensity_floor: float,
        chunk_voxels: int,
    ) -> None:
        self.fixed_volume = np.asarray(fixed_volume)
        self.fixed_metadata = fixed_metadata
        self.fixed_transform = np.asarray(fixed_transform, dtype=np.float64)
        self.moving_volume = np.asarray(moving_volume)
        self.moving_metadata = moving_metadata
        self.grid = grid
        self.intensity_floor = _normalized_scalar(intensity_floor, "intensity_floor")
        self.chunk_voxels = _positive_integer(chunk_voxels, "chunk_voxels")

        fixed = sample_volume_on_grid(
            self.fixed_volume,
            self.fixed_metadata,
            self.fixed_transform,
            grid,
            chunk_voxels=self.chunk_voxels,
        )
        fixed_signal = _registration_signal(fixed, self.intensity_floor)
        self.fixed_norm_squared = float(
            np.sum(fixed_signal * fixed_signal, dtype=np.float64)
        )
        active_indices = np.nonzero(fixed_signal > 0.0)
        self.fixed_signal_values = fixed_signal[active_indices]
        self.fixed_world_points = np.column_stack(
            (
                grid.x[active_indices[0]],
                grid.y[active_indices[1]],
                grid.z[active_indices[2]],
            )
        ).astype(np.float64, copy=False)
        self.fixed_signal_count = int(len(self.fixed_signal_values))
        if self.fixed_norm_squared <= 0.0:
            raise ValueError(
                "The fixed volume has no signal above intensity_floor on the scoring grid."
            )
        moving_baseline = sample_volume_on_grid(
            self.moving_volume,
            self.moving_metadata,
            moving_normalization_transform,
            grid,
            chunk_voxels=self.chunk_voxels,
        )
        moving_baseline_signal = _registration_signal(
            moving_baseline, self.intensity_floor
        )
        self.moving_norm_squared = float(
            np.sum(
                moving_baseline_signal * moving_baseline_signal,
                dtype=np.float64,
            )
        )
        self.moving_signal_count = int(np.count_nonzero(moving_baseline_signal))
        if self.moving_norm_squared <= 0.0:
            raise ValueError(
                "The moving volume has no signal above intensity_floor on the scoring grid."
            )
        del fixed, fixed_signal, moving_baseline, moving_baseline_signal

    def evaluate(self, moving_transform: Sequence[Sequence[float]]) -> dict[str, Any]:
        """Return normalized-product score and overlap diagnostics."""
        product = 0.0
        moving_overlap_norm_squared = 0.0
        overlap_count = 0
        for start in range(0, self.fixed_signal_count, self.chunk_voxels):
            stop = min(start + self.chunk_voxels, self.fixed_signal_count)
            moving = sample_volume_at_world_points(
                self.moving_volume,
                self.moving_metadata,
                moving_transform,
                self.fixed_world_points[start:stop],
            )
            moving_signal = _registration_signal(moving, self.intensity_floor)
            fixed_signal = self.fixed_signal_values[start:stop]
            product += float(
                np.sum(fixed_signal * moving_signal, dtype=np.float64)
            )
            moving_overlap_norm_squared += float(
                np.sum(moving_signal * moving_signal, dtype=np.float64)
            )
            moving_active = moving_signal > 0.0
            overlap_count += int(np.count_nonzero(moving_active))

        denominator = np.sqrt(
            self.fixed_norm_squared * moving_overlap_norm_squared
        )
        normalized_product = product / denominator if denominator > 0.0 else 0.0
        smaller_signal_count = min(
            self.fixed_signal_count, self.moving_signal_count
        )
        coverage = overlap_count / smaller_signal_count if smaller_signal_count else 0.0
        coverage = min(float(coverage), 1.0)
        score = normalized_product * coverage
        return {
            "score": float(score),
            "normalized_product": float(normalized_product),
            "product_sum": float(product),
            "fixed_signal_norm": float(np.sqrt(self.fixed_norm_squared)),
            "moving_signal_norm": float(np.sqrt(self.moving_norm_squared)),
            "moving_signal_norm_on_fixed_support": float(
                np.sqrt(moving_overlap_norm_squared)
            ),
            "fixed_signal_voxels": self.fixed_signal_count,
            "moving_signal_voxels": self.moving_signal_count,
            "overlap_signal_voxels": overlap_count,
            "signal_coverage": float(coverage),
        }


def optimize_correlative_alignment(
    volume_plan: np.ndarray,
    meta_plan: Mapping[str, Any],
    volume_cross: np.ndarray,
    meta_cross: Mapping[str, Any],
    *,
    alignment_state: str | Path | Mapping[str, Any],
    moving_dataset: str = "cross",
    coarse_search_axes: str = "xyz",
    coarse_search_radius: int = 1,
    coarse_translation_step: float | None = None,
    coarse_optimization_spacing: float | None = None,
    refinement_optimization_spacing: float | None = None,
    fine_translation_limit: float = 0.10,
    fine_translation_initial_step: float = 0.05,
    fine_translation_tolerance: float = 0.00625,
    fine_rotation_limit_degrees_xyz: float | Sequence[float] = (1.0, 1.0, 1.0),
    fine_rotation_initial_step_degrees_xyz: float | Sequence[float] = (
        0.5,
        0.5,
        0.5,
    ),
    fine_rotation_tolerance_degrees_xyz: float | Sequence[float] = (
        0.0625,
        0.0625,
        0.0625,
    ),
    intensity_floor: float = 0.10,
    chunk_voxels: int = 500_000,
    native_validation: bool = True,
    materialize_aligned_grid: bool = True,
    result_state_path: str | Path | None = None,
    aligned_grid_path: str | Path | None = None,
    progress: bool | Callable[[str], None] = True,
) -> CorrelativeRegistrationResult:
    """Optimize one volume's six-degree-of-freedom pose against the other.

    The input volumes remain unchanged.  Every candidate is represented by a
    physical source-to-world affine matrix, and the final aligned arrays are
    sampled only after the best transform has been selected. Fine rotations
    use one world-frame xyz rotation vector, avoiding Euler-angle ordering.
    """
    volumes = {"plan": np.asarray(volume_plan), "cross": np.asarray(volume_cross)}
    metadata = {"plan": meta_plan, "cross": meta_cross}
    _validate_volume_pair(volumes, metadata)
    moving = _dataset_name(moving_dataset)
    fixed = next(dataset for dataset in DATASETS if dataset != moving)
    unit = str(metadata[fixed]["length_unit"]).strip()
    state, source_path, source_hash = _load_source_state(alignment_state, unit)

    lattice_parameter = _common_lattice_parameter(metadata)
    coarse_step = _positive_finite(
        lattice_parameter if coarse_translation_step is None else coarse_translation_step,
        "coarse_translation_step",
    )
    coarse_spacing = _positive_finite(
        lattice_parameter / 4.0
        if coarse_optimization_spacing is None
        else coarse_optimization_spacing,
        "coarse_optimization_spacing",
    )
    refinement_spacing = _positive_finite(
        lattice_parameter / 8.0
        if refinement_optimization_spacing is None
        else refinement_optimization_spacing,
        "refinement_optimization_spacing",
    )
    if refinement_spacing > coarse_spacing:
        raise ValueError(
            "refinement_optimization_spacing must not exceed coarse_optimization_spacing."
        )
    axes = _coarse_axes(coarse_search_axes)
    radius = _nonnegative_integer(coarse_search_radius, "coarse_search_radius")
    fine_limit = _nonnegative_finite(fine_translation_limit, "fine_translation_limit")
    fine_step = _positive_finite(
        fine_translation_initial_step, "fine_translation_initial_step"
    )
    fine_tolerance = _positive_finite(
        fine_translation_tolerance, "fine_translation_tolerance"
    )
    if fine_step > fine_limit and fine_limit > 0.0:
        raise ValueError("fine_translation_initial_step must not exceed its limit.")
    if fine_tolerance > fine_step:
        raise ValueError("fine_translation_tolerance must not exceed its initial step.")
    rotation_limits = _xyz_parameter_vector(
        fine_rotation_limit_degrees_xyz,
        "fine_rotation_limit_degrees_xyz",
        allow_zero=True,
    )
    rotation_steps = _xyz_parameter_vector(
        fine_rotation_initial_step_degrees_xyz,
        "fine_rotation_initial_step_degrees_xyz",
        allow_zero=False,
    )
    rotation_tolerances = _xyz_parameter_vector(
        fine_rotation_tolerance_degrees_xyz,
        "fine_rotation_tolerance_degrees_xyz",
        allow_zero=False,
    )
    active_rotation_axes = rotation_limits > 0.0
    if np.any(rotation_steps[active_rotation_axes] > rotation_limits[active_rotation_axes]):
        raise ValueError(
            "Active fine rotation initial steps must not exceed their xyz limits."
        )
    if np.any(rotation_tolerances > rotation_steps):
        raise ValueError(
            "Fine rotation tolerances must not exceed their xyz initial steps."
        )
    floor = _normalized_scalar(intensity_floor, "intensity_floor")
    chunk_size = _positive_integer(chunk_voxels, "chunk_voxels")
    reporter = _progress_reporter(progress)

    initial_transforms = {
        dataset: np.asarray(state["transforms"][dataset], dtype=np.float64)
        for dataset in DATASETS
    }
    source_center = intensity_weighted_center_of_mass(
        volumes[moving], metadata[moving], intensity_floor=floor
    )
    world_center = transform_physical_points(
        source_center.reshape(1, 3), initial_transforms[moving]
    )[0]
    search_bounds = registration_search_bounds(
        metadata,
        initial_transforms,
        moving_dataset=moving,
        moving_center=world_center,
        coarse_search_axes=axes,
        coarse_search_radius=radius,
        coarse_translation_step=coarse_step,
        fine_translation_limit=fine_limit,
        rotation_limit_degrees_xyz=rotation_limits,
    )
    coarse_grid = create_world_grid(search_bounds, coarse_spacing)
    reporter(
        f"Coarse grid {coarse_grid.shape} ({coarse_grid.size:,} voxels) at "
        f"{coarse_spacing:g} {unit}."
    )
    coarse_scorer = NormalizedProductScorer(
        volumes[fixed],
        metadata[fixed],
        initial_transforms[fixed],
        volumes[moving],
        metadata[moving],
        initial_transforms[moving],
        coarse_grid,
        intensity_floor=floor,
        chunk_voxels=chunk_size,
    )
    initial_metrics = coarse_scorer.evaluate(initial_transforms[moving])

    coarse_offsets = coarse_translation_offsets(axes, radius, coarse_step)
    candidate_records: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    translation_bounds = np.array([[-fine_limit, fine_limit]] * 3, dtype=np.float64)
    rotation_bounds = np.column_stack((-rotation_limits, rotation_limits))
    parameter_bounds = np.vstack((translation_bounds, rotation_bounds))
    parameter_steps = np.concatenate(
        (np.full(3, fine_step, dtype=np.float64), rotation_steps)
    )
    parameter_tolerances = np.concatenate(
        (np.full(3, fine_tolerance, dtype=np.float64), rotation_tolerances)
    )

    for index, coarse_offset in enumerate(coarse_offsets, start=1):
        evaluator = _CandidateEvaluator(
            coarse_scorer,
            initial_transforms[moving],
            coarse_offset,
            world_center,
        )
        search = hierarchical_coordinate_search(
            evaluator,
            initial_parameters=np.zeros(6, dtype=np.float64),
            parameter_bounds=parameter_bounds,
            initial_steps=parameter_steps,
            tolerances=parameter_tolerances,
        )
        record = {
            "coarse_index": index - 1,
            "coarse_offset_xyz": coarse_offset.tolist(),
            "fine_translation_xyz": search["parameters"][:3].tolist(),
            "fine_rotation_vector_degrees_xyz": search["parameters"][3:6].tolist(),
            "fine_rotation_angle_degrees": float(
                np.linalg.norm(search["parameters"][3:6])
            ),
            "score": float(search["metrics"]["score"]),
            "signal_coverage": float(search["metrics"]["signal_coverage"]),
            "evaluation_count": int(search["evaluation_count"]),
        }
        candidate_records.append(record)
        candidate = {**record, "parameters": search["parameters"], "metrics": search["metrics"]}
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate
        reporter(
            f"[{index:>{len(str(len(coarse_offsets)))}}/{len(coarse_offsets)}] "
            f"coarse {tuple(round(float(value), 6) for value in coarse_offset)} -> "
            f"score {record['score']:.6f}."
        )

    assert best_candidate is not None
    best_coarse_offset = np.asarray(best_candidate["coarse_offset_xyz"], dtype=np.float64)
    refinement_grid = create_world_grid(search_bounds, refinement_spacing)
    reporter(
        f"Refinement grid {refinement_grid.shape} ({refinement_grid.size:,} voxels) "
        f"at {refinement_spacing:g} {unit}."
    )
    refinement_scorer = NormalizedProductScorer(
        volumes[fixed],
        metadata[fixed],
        initial_transforms[fixed],
        volumes[moving],
        metadata[moving],
        initial_transforms[moving],
        refinement_grid,
        intensity_floor=floor,
        chunk_voxels=chunk_size,
    )
    refinement_evaluator = _CandidateEvaluator(
        refinement_scorer,
        initial_transforms[moving],
        best_coarse_offset,
        world_center,
    )
    refinement_manual_metrics = refinement_scorer.evaluate(
        initial_transforms[moving]
    )
    refinement_start_metrics, _ = refinement_evaluator(
        np.asarray(best_candidate["parameters"], dtype=np.float64)
    )
    refinement_steps = np.concatenate(
        (
            np.full(3, fine_step / 2.0, dtype=np.float64),
            rotation_steps / 2.0,
        )
    )
    refinement_steps = np.maximum(refinement_steps, parameter_tolerances)
    refined = hierarchical_coordinate_search(
        refinement_evaluator,
        initial_parameters=np.asarray(best_candidate["parameters"], dtype=np.float64),
        parameter_bounds=parameter_bounds,
        initial_steps=refinement_steps,
        tolerances=parameter_tolerances,
    )
    optimized_transform = refinement_evaluator.transform(refined["parameters"])
    optimized_metrics = refined["metrics"]
    optimized_rotation_vector = np.asarray(refined["parameters"][3:6], dtype=np.float64)
    optimized_rotation_angle, optimized_rotation_axis = rotation_vector_axis_angle(
        optimized_rotation_vector
    )
    reporter(
        f"Refined score {optimized_metrics['score']:.6f}; translation "
        f"{tuple(round(float(value), 6) for value in (best_coarse_offset + refined['parameters'][:3]))} "
        f"{unit}; rotation vector xyz "
        f"{tuple(round(float(value), 6) for value in optimized_rotation_vector)} "
        f"degrees (angle {optimized_rotation_angle:.6f} degrees)."
    )
    exact_refinement_initial = score_volume_pair_on_grid(
        volumes[fixed],
        metadata[fixed],
        initial_transforms[fixed],
        volumes[moving],
        metadata[moving],
        initial_transforms[moving],
        refinement_grid,
        intensity_floor=floor,
        chunk_voxels=chunk_size,
    )
    exact_refinement_optimized = score_volume_pair_on_grid(
        volumes[fixed],
        metadata[fixed],
        initial_transforms[fixed],
        volumes[moving],
        metadata[moving],
        optimized_transform,
        refinement_grid,
        intensity_floor=floor,
        chunk_voxels=chunk_size,
    )
    reporter(
        "Exact refinement-grid normalized product "
        f"{exact_refinement_initial['score']:.6f} -> "
        f"{exact_refinement_optimized['score']:.6f}."
    )

    native_metrics = None
    native_grid_metadata = None
    if native_validation:
        native_spacing = _native_world_spacing(metadata, initial_transforms)
        final_bounds = transformed_union_bounds(
            metadata,
            {
                fixed: initial_transforms[fixed],
                moving: optimized_transform,
            },
            padding=native_spacing,
        )
        native_grid = create_world_grid(final_bounds, native_spacing)
        reporter(
            f"Native diagnostic grid {native_grid.shape} ({native_grid.size:,} voxels)."
        )
        native_metrics = score_volume_pair_on_grid(
            volumes[fixed],
            metadata[fixed],
            initial_transforms[fixed],
            volumes[moving],
            metadata[moving],
            optimized_transform,
            native_grid,
            intensity_floor=floor,
            chunk_voxels=chunk_size,
        )
        native_grid_metadata = native_grid.metadata()
        reporter(f"Native diagnostic score {native_metrics['score']:.6f}.")

    optimized_state = deepcopy(state)
    optimized_state["transforms"][moving] = optimized_transform.tolist()
    aligned_data = None
    grid_destination = None
    if materialize_aligned_grid:
        overlap_bounds = transformed_overlap_bounds(
            metadata,
            {
                fixed: initial_transforms[fixed],
                moving: optimized_transform,
            },
        )
        output_grid = create_world_grid(overlap_bounds, refinement_spacing)
        reporter(
            f"Aligned overlap grid {output_grid.shape} ({output_grid.size:,} voxels)."
        )
        aligned_data = materialize_aligned_volume_pair(
            volumes,
            metadata,
            {
                fixed: initial_transforms[fixed],
                moving: optimized_transform,
            },
            output_grid,
            chunk_voxels=chunk_size,
        )
        if aligned_grid_path is not None:
            grid_destination = save_aligned_grid(aligned_data, aligned_grid_path)

    now = datetime.now(timezone.utc).isoformat()
    total_translation = best_coarse_offset + refined["parameters"][:3]
    parameter_values = np.asarray(refined["parameters"], dtype=np.float64)
    at_lower_bound = np.isclose(parameter_values, parameter_bounds[:, 0], atol=1e-12)
    at_upper_bound = np.isclose(parameter_values, parameter_bounds[:, 1], atol=1e-12)
    fixed_parameters = np.isclose(
        parameter_bounds[:, 0], parameter_bounds[:, 1], atol=1e-12
    )
    at_lower_bound &= ~fixed_parameters
    at_upper_bound &= ~fixed_parameters
    bound_labels = (
        "fine_x",
        "fine_y",
        "fine_z",
        "rotation_x",
        "rotation_y",
        "rotation_z",
    )
    parameters_at_bounds = {
        label: (
            "lower" if lower else "upper" if upper else None
        )
        for label, lower, upper in zip(
            bound_labels, at_lower_bound.tolist(), at_upper_bound.tolist()
        )
    }
    active_bound_labels = [
        label for label, side in parameters_at_bounds.items() if side is not None
    ]
    if active_bound_labels:
        reporter(
            "Warning: optimized parameters reached configured bounds: "
            + ", ".join(
                f"{label}={parameters_at_bounds[label]}" for label in active_bound_labels
            )
            + "."
        )
    configuration = {
        "moving_dataset": moving,
        "fixed_dataset": fixed,
        "coarse_search_axes": axes,
        "coarse_search_radius": radius,
        "coarse_translation_step": coarse_step,
        "coarse_position_count": len(coarse_offsets),
        "coarse_additional_position_count": len(coarse_offsets) - 1,
        "coarse_optimization_spacing": coarse_spacing,
        "refinement_optimization_spacing": refinement_spacing,
        "fine_translation_limit": fine_limit,
        "fine_translation_initial_step": fine_step,
        "fine_translation_tolerance": fine_tolerance,
        "rotation_center_world_xyz": world_center.tolist(),
        "rotation_center_method": (
            "moving intensity-weighted center of mass above intensity_floor"
        ),
        "fine_rotation_parameterization": (
            "world-frame xyz rotation vector (exponential coordinates)"
        ),
        "fine_rotation_limit_degrees_xyz": rotation_limits.tolist(),
        "fine_rotation_initial_step_degrees_xyz": rotation_steps.tolist(),
        "fine_rotation_tolerance_degrees_xyz": rotation_tolerances.tolist(),
        "intensity_floor": floor,
        "score_method": (
            "normalized product of floor-subtracted intensities multiplied by "
            "signal-support coverage"
        ),
        "validation_score_method": (
            "full-grid normalized product of floor-subtracted intensities"
        ),
        "interpolation_method": "trilinear",
        "chunk_voxels": chunk_size,
    }
    optimization_summary = {
        **configuration,
        "initial_score": float(refinement_manual_metrics["score"]),
        "coarse_initial_score": float(initial_metrics["score"]),
        "coarse_best_score": float(best_candidate["score"]),
        "refinement_start_score": float(refinement_start_metrics["score"]),
        "optimized_score": float(optimized_metrics["score"]),
        "exact_refinement_initial_score": float(
            exact_refinement_initial["score"]
        ),
        "exact_refinement_optimized_score": float(
            exact_refinement_optimized["score"]
        ),
        "exact_refinement_score_improved": bool(
            exact_refinement_optimized["score"]
            > exact_refinement_initial["score"]
        ),
        "exact_refinement_initial_coverage": float(
            exact_refinement_initial["signal_coverage"]
        ),
        "exact_refinement_optimized_coverage": float(
            exact_refinement_optimized["signal_coverage"]
        ),
        "native_validation_score": (
            float(native_metrics["score"]) if native_metrics is not None else None
        ),
        "best_coarse_offset_xyz": best_coarse_offset.tolist(),
        "best_fine_translation_xyz": refined["parameters"][:3].tolist(),
        "total_optimized_translation_xyz": total_translation.tolist(),
        "optimized_rotation_vector_degrees_xyz": optimized_rotation_vector.tolist(),
        "optimized_rotation_angle_degrees": optimized_rotation_angle,
        "optimized_rotation_axis_vector_xyz": optimized_rotation_axis,
        "optimized_signal_coverage": float(optimized_metrics["signal_coverage"]),
        "parameters_at_bounds": parameters_at_bounds,
        "bound_warning": bool(active_bound_labels),
        "refinement_evaluation_count": int(refined["evaluation_count"]),
    }
    summary = {
        "schema": CORRELATIVE_SCHEMA,
        "created_at_utc": now,
        "length_unit": unit,
        "source_alignment_path": str(source_path) if source_path else None,
        "source_alignment_sha256": source_hash,
        "source_alignment_schema": state["schema"],
        "optimization": optimization_summary,
        "coarse_grid": coarse_grid.metadata(),
        "refinement_grid": refinement_grid.metadata(),
        "native_validation_grid": native_grid_metadata,
        "coarse_candidates": candidate_records,
        "initial_transforms": {
            dataset: matrix.tolist() for dataset, matrix in initial_transforms.items()
        },
        "optimized_transforms": deepcopy(optimized_state["transforms"]),
        "aligned_grid_path": str(grid_destination) if grid_destination else None,
        "aligned_grid": aligned_data["metadata"] if aligned_data is not None else None,
    }
    state_destination = None
    if result_state_path is not None:
        state_destination, snapshot = save_correlative_result(
            summary,
            optimized_state,
            result_state_path,
        )
        summary["result_state_path"] = str(state_destination)
        summary["result_state_snapshot_path"] = str(snapshot)

    return CorrelativeRegistrationResult(
        summary=summary,
        optimized_alignment_state=optimized_state,
        aligned_grid=aligned_data,
        state_path=state_destination,
        aligned_grid_path=grid_destination,
    )


def hierarchical_coordinate_search(
    evaluator: Callable[[np.ndarray], tuple[dict[str, Any], np.ndarray]],
    *,
    initial_parameters: Sequence[float],
    parameter_bounds: np.ndarray,
    initial_steps: Sequence[float],
    tolerances: Sequence[float],
    maximum_sweeps_per_level: int = 8,
) -> dict[str, Any]:
    """Deterministic bounded coordinate search with successively halved steps."""
    parameters = np.asarray(initial_parameters, dtype=np.float64).copy()
    bounds = np.asarray(parameter_bounds, dtype=np.float64)
    steps = np.asarray(initial_steps, dtype=np.float64).copy()
    minimum_steps = np.asarray(tolerances, dtype=np.float64)
    if parameters.ndim != 1 or bounds.shape != (len(parameters), 2):
        raise ValueError("parameter_bounds must have shape (parameter_count, 2).")
    if steps.shape != parameters.shape or minimum_steps.shape != parameters.shape:
        raise ValueError("initial_steps and tolerances must match initial_parameters.")
    if np.any(bounds[:, 0] > bounds[:, 1]) or np.any(parameters < bounds[:, 0]) or np.any(
        parameters > bounds[:, 1]
    ):
        raise ValueError("Initial parameters must lie inside valid bounds.")
    if np.any(steps <= 0.0) or np.any(minimum_steps <= 0.0):
        raise ValueError("Search steps and tolerances must be positive.")
    sweep_limit = _positive_integer(maximum_sweeps_per_level, "maximum_sweeps_per_level")
    metrics, transform = evaluator(parameters)
    evaluation_count = 1
    levels = 0

    while np.any(steps >= minimum_steps - 1e-15):
        levels += 1
        for _ in range(sweep_limit):
            improved = False
            for dimension in range(len(parameters)):
                best_parameters = parameters
                best_metrics = metrics
                best_transform = transform
                for direction in (-1.0, 1.0):
                    candidate = parameters.copy()
                    candidate[dimension] = np.clip(
                        candidate[dimension] + direction * steps[dimension],
                        bounds[dimension, 0],
                        bounds[dimension, 1],
                    )
                    if candidate[dimension] == parameters[dimension]:
                        continue
                    candidate_metrics, candidate_transform = evaluator(candidate)
                    evaluation_count += 1
                    if candidate_metrics["score"] > best_metrics["score"] + 1e-12:
                        best_parameters = candidate
                        best_metrics = candidate_metrics
                        best_transform = candidate_transform
                if best_parameters is not parameters:
                    parameters = best_parameters
                    metrics = best_metrics
                    transform = best_transform
                    improved = True
            if not improved:
                break
        steps *= 0.5

    return {
        "parameters": parameters,
        "metrics": metrics,
        "transform": transform,
        "evaluation_count": evaluation_count,
        "level_count": levels,
    }


class _CandidateEvaluator:
    def __init__(
        self,
        scorer: NormalizedProductScorer,
        initial_transform: np.ndarray,
        coarse_offset: np.ndarray,
        rotation_center: np.ndarray,
    ) -> None:
        self.scorer = scorer
        self.initial_transform = np.asarray(initial_transform, dtype=np.float64)
        self.coarse_offset = np.asarray(coarse_offset, dtype=np.float64)
        self.rotation_center = np.asarray(rotation_center, dtype=np.float64)
        self.cache: dict[tuple[float, ...], tuple[dict[str, Any], np.ndarray]] = {}

    def transform(self, parameters: Sequence[float]) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(
                "Registration parameters must be finite [tx, ty, tz, rx, ry, rz]."
            )
        rotation = rotation_vector_affine(
            values[3:6], self.rotation_center
        )
        translation = translation_affine(self.coarse_offset + values[:3])
        return translation @ rotation @ self.initial_transform

    def __call__(self, parameters: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        key = tuple(float(round(value, 12)) for value in parameters)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        transform = self.transform(parameters)
        result = self.scorer.evaluate(transform), transform
        self.cache[key] = result
        return result


def create_world_grid(bounds: Sequence[Sequence[float]], spacing: float) -> WorldGrid:
    """Create an endpoint-inclusive regular physical grid."""
    limits = np.asarray(bounds, dtype=np.float64)
    if limits.shape != (3, 2) or not np.isfinite(limits).all():
        raise ValueError("bounds must be finite with shape (3, 2).")
    if np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError("Every world-grid upper bound must exceed its lower bound.")
    requested = _positive_finite(spacing, "spacing")
    vectors = []
    for lower, upper in limits:
        count = max(2, int(np.ceil((upper - lower) / requested)) + 1)
        vectors.append(np.linspace(lower, upper, count, dtype=np.float64))
    return WorldGrid(*vectors, requested_spacing=requested)


def iter_world_grid_chunks(
    grid: WorldGrid, chunk_voxels: int
):
    """Yield x-slabs as Nx3 physical world positions."""
    maximum = _positive_integer(chunk_voxels, "chunk_voxels")
    yz_count = len(grid.y) * len(grid.z)
    slab_width = max(1, maximum // yz_count)
    y_flat = np.repeat(grid.y, len(grid.z))
    z_flat = np.tile(grid.z, len(grid.y))
    for start in range(0, len(grid.x), slab_width):
        stop = min(start + slab_width, len(grid.x))
        x_values = grid.x[start:stop]
        points = np.empty((len(x_values) * yz_count, 3), dtype=np.float64)
        points[:, 0] = np.repeat(x_values, yz_count)
        points[:, 1] = np.tile(y_flat, len(x_values))
        points[:, 2] = np.tile(z_flat, len(x_values))
        yield slice(start, stop), points


def sample_volume_on_grid(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    transform: Sequence[Sequence[float]],
    grid: WorldGrid,
    *,
    chunk_voxels: int = 500_000,
) -> np.ndarray:
    """Materialize one transformed volume on a world grid with NaN outside."""
    output = np.empty(grid.shape, dtype=np.float32)
    for x_slice, points in iter_world_grid_chunks(grid, chunk_voxels):
        output[x_slice] = sample_volume_at_world_points(
            volume, metadata, transform, points
        ).reshape(x_slice.stop - x_slice.start, len(grid.y), len(grid.z))
    return output


def score_volume_pair_on_grid(
    fixed_volume: np.ndarray,
    fixed_metadata: Mapping[str, Any],
    fixed_transform: Sequence[Sequence[float]],
    moving_volume: np.ndarray,
    moving_metadata: Mapping[str, Any],
    moving_transform: Sequence[Sequence[float]],
    grid: WorldGrid,
    *,
    intensity_floor: float,
    chunk_voxels: int = 500_000,
) -> dict[str, Any]:
    """Calculate a chunked normalized-product score without storing either grid."""
    floor = _normalized_scalar(intensity_floor, "intensity_floor")
    fixed_norm_squared = 0.0
    moving_norm_squared = 0.0
    product = 0.0
    fixed_count = 0
    moving_count = 0
    overlap_count = 0
    for _, points in iter_world_grid_chunks(grid, chunk_voxels):
        fixed = _registration_signal(
            sample_volume_at_world_points(
                fixed_volume, fixed_metadata, fixed_transform, points
            ),
            floor,
        )
        moving = _registration_signal(
            sample_volume_at_world_points(
                moving_volume, moving_metadata, moving_transform, points
            ),
            floor,
        )
        fixed_norm_squared += float(np.sum(fixed * fixed, dtype=np.float64))
        moving_norm_squared += float(np.sum(moving * moving, dtype=np.float64))
        product += float(np.sum(fixed * moving, dtype=np.float64))
        fixed_active = fixed > 0.0
        moving_active = moving > 0.0
        fixed_count += int(np.count_nonzero(fixed_active))
        moving_count += int(np.count_nonzero(moving_active))
        overlap_count += int(np.count_nonzero(fixed_active & moving_active))
    denominator = np.sqrt(fixed_norm_squared * moving_norm_squared)
    score = product / denominator if denominator > 0.0 else 0.0
    smaller = min(fixed_count, moving_count)
    return {
        "score": float(score),
        "normalized_product": float(score),
        "product_sum": float(product),
        "fixed_signal_norm": float(np.sqrt(fixed_norm_squared)),
        "moving_signal_norm": float(np.sqrt(moving_norm_squared)),
        "fixed_signal_voxels": fixed_count,
        "moving_signal_voxels": moving_count,
        "overlap_signal_voxels": overlap_count,
        "signal_coverage": float(overlap_count / smaller) if smaller else 0.0,
    }


def intensity_weighted_center_of_mass(
    volume: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    intensity_floor: float,
    slab_voxels: int = 5_000_000,
) -> np.ndarray:
    """Return source-space xyz center of mass above the registration floor."""
    data = np.asarray(volume)
    if data.ndim != 3:
        raise ValueError("volume must be canonical 3D data.")
    coordinates = _metadata_coordinates(metadata, data.shape)
    floor = _normalized_scalar(intensity_floor, "intensity_floor")
    yz_size = data.shape[1] * data.shape[2]
    slab_width = max(1, _positive_integer(slab_voxels, "slab_voxels") // yz_size)
    total = 0.0
    moments = np.zeros(3, dtype=np.float64)
    for start in range(0, data.shape[0], slab_width):
        stop = min(start + slab_width, data.shape[0])
        signal = _registration_signal(data[start:stop], floor)
        weights_x = np.sum(signal, axis=(1, 2), dtype=np.float64)
        weights_y = np.sum(signal, axis=(0, 2), dtype=np.float64)
        weights_z = np.sum(signal, axis=(0, 1), dtype=np.float64)
        total += float(weights_x.sum())
        moments[0] += float(np.dot(weights_x, coordinates[0][start:stop]))
        moments[1] += float(np.dot(weights_y, coordinates[1]))
        moments[2] += float(np.dot(weights_z, coordinates[2]))
    if total <= 0.0:
        return np.array(
            [(values[0] + values[-1]) / 2.0 for values in coordinates],
            dtype=np.float64,
        )
    return moments / total


def validate_rotation_axis(
    rotation_axis: str | Sequence[float],
) -> tuple[np.ndarray, str]:
    """Accept one named physical axis or an arbitrary xyz direction vector."""
    if isinstance(rotation_axis, str):
        name = rotation_axis.strip().lower()
        if name not in AXES:
            raise ValueError("A named rotation_axis must be 'x', 'y', or 'z'.")
        vector = np.zeros(3, dtype=np.float64)
        vector[AXES.index(name)] = 1.0
        return vector, name
    vector = np.asarray(rotation_axis, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("rotation_axis must be a named axis or finite xyz vector.")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("rotation_axis vector must be nonzero.")
    normalized = vector / norm
    return normalized, [float(value) for value in normalized]  # type: ignore[return-value]


def rotation_affine(
    axis: Sequence[float], angle_degrees: float, center: Sequence[float]
) -> np.ndarray:
    """Return a world-space axis-angle rotation about a physical center."""
    vector, _ = validate_rotation_axis(axis)
    point = np.asarray(center, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("rotation center must be a finite xyz point.")
    angle = np.deg2rad(float(angle_degrees))
    if not np.isfinite(angle):
        raise ValueError("rotation angle must be finite.")
    x, y, z = vector
    cosine, sine = np.cos(angle), np.sin(angle)
    cross = 1.0 - cosine
    matrix = np.array(
        [
            [cosine + x * x * cross, x * y * cross - z * sine, x * z * cross + y * sine],
            [y * x * cross + z * sine, cosine + y * y * cross, y * z * cross - x * sine],
            [z * x * cross - y * sine, z * y * cross + x * sine, cosine + z * z * cross],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix
    result[:3, 3] = point - matrix @ point
    return result


def rotation_vector_axis_angle(
    rotation_vector_degrees_xyz: Sequence[float],
) -> tuple[float, list[float] | None]:
    """Return total angle and axis for a world-frame xyz rotation vector."""
    vector = np.asarray(rotation_vector_degrees_xyz, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("rotation_vector_degrees_xyz must be a finite xyz vector.")
    angle = float(np.linalg.norm(vector))
    if angle <= np.finfo(np.float64).eps:
        return 0.0, None
    return angle, (vector / angle).tolist()


def rotation_vector_affine(
    rotation_vector_degrees_xyz: Sequence[float], center: Sequence[float]
) -> np.ndarray:
    """Convert exponential-coordinate rotation components into one affine."""
    angle, axis = rotation_vector_axis_angle(rotation_vector_degrees_xyz)
    point = np.asarray(center, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("rotation center must be a finite xyz point.")
    if axis is None:
        return np.eye(4, dtype=np.float64)
    return rotation_affine(axis, angle, point)


def translation_affine(offset: Sequence[float]) -> np.ndarray:
    vector = np.asarray(offset, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("translation offset must be a finite xyz vector.")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = vector
    return result


def coarse_translation_offsets(
    axes: str, radius: int, step: float
) -> list[np.ndarray]:
    """Return the initial position first, followed by all Cartesian coarse hops."""
    selected = _coarse_axes(axes)
    search_radius = _nonnegative_integer(radius, "radius")
    jump = _positive_finite(step, "step")
    integer_offsets = []
    for values in itertools.product(
        range(-search_radius, search_radius + 1), repeat=len(selected)
    ):
        offset = np.zeros(3, dtype=np.int64)
        for axis, value in zip(selected, values):
            offset[AXES.index(axis)] = value
        integer_offsets.append(offset)
    integer_offsets.sort(
        key=lambda value: (
            int(np.sum(value * value)),
            int(np.sum(np.abs(value))),
            tuple(int(component) for component in value),
        )
    )
    return [value.astype(np.float64) * jump for value in integer_offsets]


def registration_search_bounds(
    metadata: Mapping[str, Mapping[str, Any]],
    transforms: Mapping[str, Sequence[Sequence[float]]],
    *,
    moving_dataset: str,
    moving_center: Sequence[float],
    coarse_search_axes: str,
    coarse_search_radius: int,
    coarse_translation_step: float,
    fine_translation_limit: float,
    rotation_limit_degrees_xyz: Sequence[float],
) -> np.ndarray:
    """Return fixed bounds containing every requested moving candidate."""
    moving = _dataset_name(moving_dataset)
    corners = {
        dataset: transformed_box_corners(metadata[dataset], transforms[dataset])
        for dataset in DATASETS
    }
    all_corners = np.vstack(tuple(corners.values()))
    bounds = np.column_stack((all_corners.min(axis=0), all_corners.max(axis=0)))
    translation_extent = np.full(3, float(fine_translation_limit), dtype=np.float64)
    for axis in _coarse_axes(coarse_search_axes):
        translation_extent[AXES.index(axis)] += (
            coarse_search_radius * coarse_translation_step
        )
    center = np.asarray(moving_center, dtype=np.float64)
    radius = float(np.max(np.linalg.norm(corners[moving] - center, axis=1)))
    maximum_rotation_angle = float(
        np.linalg.norm(
            _xyz_parameter_vector(
                rotation_limit_degrees_xyz,
                "rotation_limit_degrees_xyz",
                allow_zero=True,
            )
        )
    )
    conservative_angle = min(maximum_rotation_angle, 180.0)
    rotation_displacement = 2.0 * radius * np.sin(
        np.deg2rad(conservative_angle) / 2.0
    )
    expansion = translation_extent + rotation_displacement
    bounds[:, 0] -= expansion
    bounds[:, 1] += expansion
    return bounds


def transformed_union_bounds(
    metadata: Mapping[str, Mapping[str, Any]],
    transforms: Mapping[str, Sequence[Sequence[float]]],
    *,
    padding: float = 0.0,
) -> np.ndarray:
    corners = np.vstack(
        [
            transformed_box_corners(metadata[dataset], transforms[dataset])
            for dataset in DATASETS
        ]
    )
    pad = _nonnegative_finite(padding, "padding")
    bounds = np.column_stack((corners.min(axis=0), corners.max(axis=0)))
    bounds[:, 0] -= pad
    bounds[:, 1] += pad
    return bounds


def transformed_overlap_bounds(
    metadata: Mapping[str, Mapping[str, Any]],
    transforms: Mapping[str, Sequence[Sequence[float]]],
) -> np.ndarray:
    per_dataset = []
    for dataset in DATASETS:
        corners = transformed_box_corners(metadata[dataset], transforms[dataset])
        per_dataset.append(
            np.column_stack((corners.min(axis=0), corners.max(axis=0)))
        )
    lower = np.maximum(per_dataset[0][:, 0], per_dataset[1][:, 0])
    upper = np.minimum(per_dataset[0][:, 1], per_dataset[1][:, 1])
    if np.any(upper <= lower):
        raise ValueError("Optimized volume bounding boxes do not overlap in 3D.")
    return np.column_stack((lower, upper))


def materialize_aligned_volume_pair(
    volumes: Mapping[str, np.ndarray],
    metadata: Mapping[str, Mapping[str, Any]],
    transforms: Mapping[str, Sequence[Sequence[float]]],
    grid: WorldGrid,
    *,
    chunk_voxels: int = 500_000,
) -> dict[str, Any]:
    """Return both transformed intensities on one common physical xyz grid."""
    aligned = {
        dataset: sample_volume_on_grid(
            volumes[dataset],
            metadata[dataset],
            transforms[dataset],
            grid,
            chunk_voxels=chunk_voxels,
        )
        for dataset in DATASETS
    }
    valid = np.isfinite(aligned["plan"]) & np.isfinite(aligned["cross"])
    return {
        "coordinates": grid.coordinates,
        "volume_plan": aligned["plan"],
        "volume_cross": aligned["cross"],
        "valid_overlap_mask": valid,
        "transforms": {
            dataset: np.asarray(transforms[dataset], dtype=np.float64)
            for dataset in DATASETS
        },
        "metadata": {
            **grid.metadata(),
            "axis_order": ["x", "y", "z"],
            "intensity_dtype": "float32",
            "outside_source_value": "NaN",
            "valid_overlap_voxels": int(np.count_nonzero(valid)),
        },
    }


def save_aligned_grid(aligned_grid: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically save transformed intensities and physical coordinate vectors."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    coordinates = aligned_grid["coordinates"]
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            x=np.asarray(coordinates["x"], dtype=np.float64),
            y=np.asarray(coordinates["y"], dtype=np.float64),
            z=np.asarray(coordinates["z"], dtype=np.float64),
            volume_plan=np.asarray(aligned_grid["volume_plan"], dtype=np.float32),
            volume_cross=np.asarray(aligned_grid["volume_cross"], dtype=np.float32),
            valid_overlap_mask=np.asarray(
                aligned_grid["valid_overlap_mask"], dtype=np.bool_
            ),
            transform_plan=np.asarray(aligned_grid["transforms"]["plan"]),
            transform_cross=np.asarray(aligned_grid["transforms"]["cross"]),
        )
    temporary.replace(destination)
    return destination


def save_correlative_result(
    summary: Mapping[str, Any],
    optimized_alignment_state: Mapping[str, Any],
    path: str | Path,
) -> tuple[Path, Path]:
    """Save the optimized state without overwriting the manual alignment state."""
    state = validate_alignment_state(optimized_alignment_state)
    payload = {
        **deepcopy(dict(summary)),
        "optimized_alignment_state": state,
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    snapshot = destination.with_name(
        f"{destination.stem}-{stamp}{destination.suffix}"
    )
    snapshot.write_text(text, encoding="utf-8", newline="\n")
    return destination, snapshot


def print_registration_summary(result: CorrelativeRegistrationResult) -> None:
    """Print the compact scientific summary needed after optimization."""
    values = result.summary["optimization"]
    unit = result.summary["length_unit"]
    print("Correlative registration")
    print(
        f"  fixed={values['fixed_dataset']}, moving={values['moving_dataset']}, "
        f"coarse positions={values['coarse_position_count']}"
    )
    print(
        f"  score: {values['initial_score']:.6f} -> "
        f"{values['optimized_score']:.6f}"
    )
    print(
        "  exact normalized product: "
        f"{values['exact_refinement_initial_score']:.6f} -> "
        f"{values['exact_refinement_optimized_score']:.6f}"
    )
    print(
        "  translation xyz: "
        + ", ".join(
            f"{value:.6g}" for value in values["total_optimized_translation_xyz"]
        )
        + f" {unit}"
    )
    rotation_vector = values["optimized_rotation_vector_degrees_xyz"]
    rotation_axis = values["optimized_rotation_axis_vector_xyz"]
    print(
        "  rotation vector xyz: "
        + ", ".join(f"{value:.6g}" for value in rotation_vector)
        + " deg"
    )
    if rotation_axis is None:
        print("  rotation angle: 0 deg (axis undefined)")
    else:
        print(
            f"  rotation angle: {values['optimized_rotation_angle_degrees']:.6g} "
            "deg about axis ["
            + ", ".join(f"{value:.6g}" for value in rotation_axis)
            + "]"
        )
    print(f"  optimized state: {result.state_path}")
    print(f"  aligned grid: {result.aligned_grid_path}")


def _registration_signal(values: np.ndarray, intensity_floor: float) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    signal = signal - np.float32(intensity_floor)
    np.maximum(signal, 0.0, out=signal)
    return signal


def _load_source_state(
    value: str | Path | Mapping[str, Any], length_unit: str
) -> tuple[dict[str, Any], Path | None, str | None]:
    if isinstance(value, Mapping):
        return validate_alignment_state(value, length_unit=length_unit), None, None
    source = Path(value).expanduser().resolve()
    state = load_alignment_state(source, length_unit=length_unit)
    digest = hashlib.sha256(source.read_bytes()).hexdigest().upper()
    return state, source, digest


def _validate_volume_pair(
    volumes: Mapping[str, np.ndarray], metadata: Mapping[str, Mapping[str, Any]]
) -> None:
    units = []
    for dataset in DATASETS:
        volume = volumes[dataset]
        if volume.ndim != 3:
            raise ValueError(f"{dataset} volume must be canonical 3D data.")
        _metadata_coordinates(metadata[dataset], volume.shape)
        units.append(str(metadata[dataset].get("length_unit", "")).strip())
    if not units[0] or units[0] != units[1]:
        raise ValueError("Both datasets must use the same non-empty length_unit.")


def _metadata_coordinates(
    metadata: Mapping[str, Any], shape: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = metadata.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("metadata must contain physical x, y, z coordinates.")
    result = tuple(np.asarray(coordinates[axis], dtype=np.float64) for axis in AXES)
    if tuple(len(values) for values in result) != tuple(shape):
        raise ValueError("Coordinate lengths must match canonical volume shape.")
    return result  # type: ignore[return-value]


def _common_lattice_parameter(
    metadata: Mapping[str, Mapping[str, Any]]
) -> float:
    values = [
        _positive_finite(metadata[dataset].get("lat_param"), "lat_param")
        for dataset in DATASETS
    ]
    if not np.isclose(values[0], values[1], rtol=1e-9, atol=0.0):
        raise ValueError("Plan and cross metadata use different lattice parameters.")
    return values[0]


def _native_world_spacing(
    metadata: Mapping[str, Mapping[str, Any]],
    transforms: Mapping[str, np.ndarray],
) -> float:
    spacings = []
    for dataset in DATASETS:
        coordinates = metadata[dataset]["coordinates"]
        linear = np.asarray(transforms[dataset], dtype=np.float64)[:3, :3]
        for axis_index, axis in enumerate(AXES):
            source_step = _axis_spacing(np.asarray(coordinates[axis], dtype=np.float64))
            world_step = np.linalg.norm(linear[:, axis_index] * source_step)
            spacings.append(float(world_step))
    return min(spacings)


def _axis_spacing(values: np.ndarray) -> float:
    if len(values) < 2:
        return 1.0
    return float(np.median(np.abs(np.diff(values.astype(np.float64)))))


def _coarse_axes(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("coarse_search_axes must be a string containing x, y, and/or z.")
    axes = value.strip().lower()
    if any(axis not in AXES for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError(
            "coarse_search_axes must contain unique physical axes, for example 'xyz' or 'yx'."
        )
    return axes


def _dataset_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}.")
    return normalized


def _progress_reporter(
    progress: bool | Callable[[str], None]
) -> Callable[[str], None]:
    if callable(progress):
        return progress
    if progress:
        return lambda message: print(message, flush=True)
    return lambda message: None


def _normalized_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return result


def _xyz_parameter_vector(
    value: float | Sequence[float], name: str, *, allow_zero: bool
) -> np.ndarray:
    """Validate a scalar broadcast or explicit xyz optimization setting."""
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite scalar or xyz sequence.") from exc
    if values.ndim == 0:
        values = np.full(3, float(values), dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite scalar or length-three xyz sequence.")
    if allow_zero:
        if np.any(values < 0.0):
            raise ValueError(f"{name} components must be nonnegative.")
    elif np.any(values <= 0.0):
        raise ValueError(f"{name} components must be positive.")
    return values


def _positive_finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number.")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a nonnegative integer.")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return result
