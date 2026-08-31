"""Physical peak detection and subvoxel atom localization for CMEP."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cmep_likelihood import (
    LikelihoodMapResult,
    load_aligned_grid,
    load_likelihood_map,
)


ATOM_LOCALIZATION_SCHEMA = "cmep-atom-localization-v1"


@dataclass
class AtomLocalizationResult:
    """Localized physical atom centers plus candidate-level diagnostics."""

    atoms: dict[str, np.ndarray]
    candidates: dict[str, np.ndarray]
    metadata: dict[str, Any]
    npz_path: Path | None = None
    csv_path: Path | None = None
    metadata_path: Path | None = None

    @property
    def positions_xyz(self) -> np.ndarray:
        return self.atoms["position_xyz"]

    @property
    def scores(self) -> np.ndarray:
        return self.atoms["likelihood_score"]

    def __repr__(self) -> str:
        return (
            "AtomLocalizationResult("
            f"atom_count={len(self.positions_xyz):,}, "
            f"candidate_count={len(self.candidates['seed_score']):,}, "
            f"npz_path={str(self.npz_path)!r})"
        )


def localize_atoms(
    likelihood_score_map: str | Path | LikelihoodMapResult,
    aligned_grid: str | Path | Mapping[str, Any],
    *,
    lat_param: float,
    score_threshold: float,
    minimum_atom_separation: float | None = None,
    detection_smoothing_sigma: float = 0.0,
    fit_radius: float | None = None,
    maximum_subvoxel_shift: float | None = None,
    initial_step: float | None = None,
    position_tolerance: float | None = None,
    minimum_valid_fraction: float = 0.70,
    output_prefix: str | Path | None = None,
    progress: bool = True,
) -> AtomLocalizationResult:
    """Detect score maxima and refine one shared center against both datasets.

    Candidate detection uses the correlative likelihood score map. Refinement
    fits the aligned plan-view and cross-sectional intensities directly with a
    shared center and dataset-specific Gaussian widths, amplitudes, and
    backgrounds. All distances use the map's physical length unit.
    """
    score_result, score_source, score_hash = _load_score_result(likelihood_score_map)
    score = np.asarray(score_result.likelihood, dtype=np.float32)
    coordinates = {
        axis: np.asarray(score_result.coordinates[axis], dtype=np.float64)
        for axis in "xyz"
    }
    aligned, aligned_source, aligned_hash = load_aligned_grid(aligned_grid)
    plan = np.asarray(aligned["volume_plan"], dtype=np.float32)
    cross = np.asarray(aligned["volume_cross"], dtype=np.float32)
    valid = np.asarray(aligned["valid_overlap_mask"], dtype=np.bool_)
    _validate_common_grid(score, coordinates, plan, cross, valid, aligned["coordinates"])

    lattice = _positive_finite(lat_param, "lat_param")
    threshold = _normalized_scalar(score_threshold, "score_threshold")
    smoothing = _nonnegative_finite(
        detection_smoothing_sigma, "detection_smoothing_sigma"
    )
    valid_fraction = float(minimum_valid_fraction)
    if not np.isfinite(valid_fraction) or not 0.0 < valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must lie in (0, 1].")

    spacing = np.array([_axis_spacing(coordinates[axis]) for axis in "xyz"])
    separation = (
        lattice * np.sqrt(3.0) / 2.0 * 0.7
        if minimum_atom_separation is None
        else _positive_finite(minimum_atom_separation, "minimum_atom_separation")
    )
    radius = (
        separation * 0.45
        if fit_radius is None
        else _positive_finite(fit_radius, "fit_radius")
    )
    max_shift = (
        float(np.max(spacing))
        if maximum_subvoxel_shift is None
        else _positive_finite(maximum_subvoxel_shift, "maximum_subvoxel_shift")
    )
    step = (
        float(np.min(spacing)) / 4.0
        if initial_step is None
        else _positive_finite(initial_step, "initial_step")
    )
    tolerance = (
        float(np.min(spacing)) / 32.0
        if position_tolerance is None
        else _positive_finite(position_tolerance, "position_tolerance")
    )
    if tolerance > step:
        raise ValueError("position_tolerance must not exceed initial_step.")
    if radius <= max_shift:
        raise ValueError("fit_radius must be greater than maximum_subvoxel_shift.")

    floor_plan = _metadata_floor(score_result.metadata, "floor_plan")
    floor_cross = _metadata_floor(score_result.metadata, "floor_cross")
    detection = _smooth_score(score, coordinates, smoothing)
    raw_indices, raw_scores = _local_maxima(score, detection, threshold)
    raw_positions = _indices_to_positions(raw_indices, coordinates)
    keep = _physical_nms(raw_positions, raw_scores, separation)
    seed_indices = raw_indices[keep]
    seed_positions = raw_positions[keep]
    seed_scores = raw_scores[keep]

    if progress:
        print(
            f"Detected {len(raw_indices):,} local maxima; "
            f"{len(seed_indices):,} remain after physical separation."
        )

    template = _fit_template(spacing, radius)
    candidate_count = len(seed_indices)
    refined_positions = seed_positions.copy()
    refined_scores = np.full(candidate_count, np.nan, dtype=np.float32)
    plan_amplitudes = np.full(candidate_count, np.nan, dtype=np.float32)
    cross_amplitudes = np.full(candidate_count, np.nan, dtype=np.float32)
    plan_backgrounds = np.full(candidate_count, np.nan, dtype=np.float32)
    cross_backgrounds = np.full(candidate_count, np.nan, dtype=np.float32)
    plan_sigmas = np.full((candidate_count, 3), np.nan, dtype=np.float32)
    cross_sigmas = np.full((candidate_count, 3), np.nan, dtype=np.float32)
    plan_fit_quality = np.full(candidate_count, np.nan, dtype=np.float32)
    cross_fit_quality = np.full(candidate_count, np.nan, dtype=np.float32)
    joint_fit_quality = np.full(candidate_count, np.nan, dtype=np.float32)
    shifts = np.zeros(candidate_count, dtype=np.float32)
    valid_fractions = np.zeros(candidate_count, dtype=np.float32)
    evaluation_counts = np.zeros(candidate_count, dtype=np.int32)
    statuses = np.full(candidate_count, "unprocessed", dtype="<U32")

    for candidate_index, (seed_index, seed_position) in enumerate(
        zip(seed_indices, seed_positions)
    ):
        fit = _refine_candidate(
            seed_index,
            seed_position,
            plan,
            cross,
            valid,
            score,
            coordinates,
            template,
            floor_plan=floor_plan,
            floor_cross=floor_cross,
            maximum_shift=max_shift,
            initial_step=step,
            tolerance=tolerance,
            minimum_valid_fraction=valid_fraction,
        )
        refined_positions[candidate_index] = fit["position"]
        refined_scores[candidate_index] = fit["score"]
        plan_amplitudes[candidate_index] = fit["plan_amplitude"]
        cross_amplitudes[candidate_index] = fit["cross_amplitude"]
        plan_backgrounds[candidate_index] = fit["plan_background"]
        cross_backgrounds[candidate_index] = fit["cross_background"]
        plan_sigmas[candidate_index] = fit["plan_sigma_xyz"]
        cross_sigmas[candidate_index] = fit["cross_sigma_xyz"]
        plan_fit_quality[candidate_index] = fit["plan_fit_quality"]
        cross_fit_quality[candidate_index] = fit["cross_fit_quality"]
        joint_fit_quality[candidate_index] = fit["joint_fit_quality"]
        shifts[candidate_index] = fit["shift"]
        valid_fractions[candidate_index] = fit["valid_fraction"]
        evaluation_counts[candidate_index] = fit["evaluation_count"]
        statuses[candidate_index] = fit["status"]
        if progress and (candidate_index + 1) % 2_000 == 0:
            print(f"  refined {candidate_index + 1:,}/{candidate_count:,} candidates")

    provisionally_valid = (
        (statuses == "refined")
        & np.isfinite(refined_scores)
        & (refined_scores >= threshold)
        & np.all(np.isfinite(refined_positions), axis=1)
    )
    provisional_indices = np.flatnonzero(provisionally_valid)
    final_local = _physical_nms(
        refined_positions[provisional_indices],
        refined_scores[provisional_indices],
        separation,
    )
    accepted_indices = provisional_indices[final_local]
    accepted = np.zeros(candidate_count, dtype=np.bool_)
    accepted[accepted_indices] = True
    statuses[provisionally_valid & ~accepted] = "post_refinement_duplicate"
    statuses[~provisionally_valid & (statuses == "refined")] = "below_score_threshold"

    candidates = {
        "seed_index_xyz": seed_indices.astype(np.int32, copy=False),
        "seed_position_xyz": seed_positions.astype(np.float64, copy=False),
        "seed_score": seed_scores.astype(np.float32, copy=False),
        "position_xyz": refined_positions.astype(np.float64, copy=False),
        "likelihood_score": refined_scores,
        "accepted": accepted,
        "status": statuses,
        "subvoxel_shift": shifts,
        "valid_fraction": valid_fractions,
        "plan_amplitude": plan_amplitudes,
        "cross_amplitude": cross_amplitudes,
        "plan_background": plan_backgrounds,
        "cross_background": cross_backgrounds,
        "plan_sigma_xyz": plan_sigmas,
        "cross_sigma_xyz": cross_sigmas,
        "plan_fit_quality": plan_fit_quality,
        "cross_fit_quality": cross_fit_quality,
        "joint_fit_quality": joint_fit_quality,
        "evaluation_count": evaluation_counts,
    }
    atoms = _accepted_atom_table(candidates, accepted_indices)
    metadata = _localization_metadata(
        score_result,
        coordinates,
        atoms,
        candidates,
        lattice=lattice,
        score_threshold=threshold,
        minimum_atom_separation=separation,
        detection_smoothing_sigma=smoothing,
        fit_radius=radius,
        maximum_subvoxel_shift=max_shift,
        initial_step=step,
        position_tolerance=tolerance,
        minimum_valid_fraction=valid_fraction,
        score_source=score_source,
        score_hash=score_hash,
        aligned_source=aligned_source,
        aligned_hash=aligned_hash,
        raw_local_maximum_count=len(raw_indices),
    )
    result = AtomLocalizationResult(atoms, candidates, metadata)
    if output_prefix is not None:
        result.npz_path, result.csv_path, result.metadata_path = save_atom_localization(
            result, output_prefix
        )
    if progress:
        print(f"Accepted {len(atoms['position_xyz']):,} localized atomic centers.")
    return result


def save_atom_localization(
    result: AtomLocalizationResult, output_prefix: str | Path
) -> tuple[Path, Path, Path]:
    """Atomically save coordinates, diagnostics, CSV, and JSON metadata."""
    prefix = Path(output_prefix).expanduser().resolve()
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    csv_path = prefix.with_suffix(".csv")
    metadata_path = prefix.with_suffix(".json")

    temporary_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    payload: dict[str, np.ndarray] = {}
    for group_name, group in (("atom", result.atoms), ("candidate", result.candidates)):
        for name, values in group.items():
            payload[f"{group_name}_{name}"] = np.asarray(values)
    payload["metadata_json"] = np.asarray(
        json.dumps(result.metadata, sort_keys=True)
    )
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary_npz.replace(npz_path)

    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    fieldnames = [
        "atom_id",
        "x",
        "y",
        "z",
        "likelihood_score",
        "seed_score",
        "subvoxel_shift",
        "joint_fit_quality",
        "plan_fit_quality",
        "cross_fit_quality",
        "plan_amplitude",
        "cross_amplitude",
        "plan_sigma_x",
        "plan_sigma_y",
        "plan_sigma_z",
        "cross_sigma_x",
        "cross_sigma_y",
        "cross_sigma_z",
        "status",
    ]
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(result.atoms["atom_id"])):
            position = result.atoms["position_xyz"][index]
            plan_sigma = result.atoms["plan_sigma_xyz"][index]
            cross_sigma = result.atoms["cross_sigma_xyz"][index]
            writer.writerow(
                {
                    "atom_id": int(result.atoms["atom_id"][index]),
                    "x": f"{position[0]:.12g}",
                    "y": f"{position[1]:.12g}",
                    "z": f"{position[2]:.12g}",
                    "likelihood_score": f"{result.atoms['likelihood_score'][index]:.8g}",
                    "seed_score": f"{result.atoms['seed_score'][index]:.8g}",
                    "subvoxel_shift": f"{result.atoms['subvoxel_shift'][index]:.8g}",
                    "joint_fit_quality": f"{result.atoms['joint_fit_quality'][index]:.8g}",
                    "plan_fit_quality": f"{result.atoms['plan_fit_quality'][index]:.8g}",
                    "cross_fit_quality": f"{result.atoms['cross_fit_quality'][index]:.8g}",
                    "plan_amplitude": f"{result.atoms['plan_amplitude'][index]:.8g}",
                    "cross_amplitude": f"{result.atoms['cross_amplitude'][index]:.8g}",
                    "plan_sigma_x": f"{plan_sigma[0]:.8g}",
                    "plan_sigma_y": f"{plan_sigma[1]:.8g}",
                    "plan_sigma_z": f"{plan_sigma[2]:.8g}",
                    "cross_sigma_x": f"{cross_sigma[0]:.8g}",
                    "cross_sigma_y": f"{cross_sigma[1]:.8g}",
                    "cross_sigma_z": f"{cross_sigma[2]:.8g}",
                    "status": str(result.atoms["status"][index]),
                }
            )
    temporary_csv.replace(csv_path)

    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_metadata.replace(metadata_path)
    return npz_path, csv_path, metadata_path


def load_atom_localization(path: str | Path) -> AtomLocalizationResult:
    """Load a saved localization NPZ without requiring pickle."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Atom-localization NPZ does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        atoms = {
            name.removeprefix("atom_"): np.asarray(archive[name])
            for name in archive.files
            if name.startswith("atom_")
        }
        candidates = {
            name.removeprefix("candidate_"): np.asarray(archive[name])
            for name in archive.files
            if name.startswith("candidate_")
        }
    if metadata.get("schema_version") != ATOM_LOCALIZATION_SCHEMA:
        raise ValueError("Unsupported atom-localization schema version.")
    positions = np.asarray(atoms.get("position_xyz"))
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Saved atom positions must have shape (N, 3).")
    prefix = source.with_suffix("")
    csv_path = prefix.with_suffix(".csv")
    metadata_path = prefix.with_suffix(".json")
    return AtomLocalizationResult(
        atoms,
        candidates,
        metadata,
        npz_path=source,
        csv_path=csv_path if csv_path.is_file() else None,
        metadata_path=metadata_path if metadata_path.is_file() else None,
    )


def print_atom_localization_summary(result: AtomLocalizationResult) -> None:
    """Print the key physical and numerical localization results."""
    metadata = result.metadata
    unit = metadata["length_unit"]
    print("Subvoxel atomic-center localization")
    print(
        f"  candidates={metadata['candidate_count']:,}, "
        f"accepted atoms={metadata['accepted_atom_count']:,}"
    )
    print(
        f"  score threshold={metadata['score_threshold']:.6g}, "
        f"minimum separation={metadata['minimum_atom_separation']:.6g} {unit}"
    )
    print(
        f"  fit radius={metadata['fit_radius']:.6g} {unit}, "
        f"position tolerance={metadata['position_tolerance']:.6g} {unit}"
    )
    if len(result.positions_xyz):
        print(
            "  score range="
            f"{float(np.min(result.scores)):.6g} .. {float(np.max(result.scores)):.6g}"
        )
        print(
            "  median subvoxel shift="
            f"{float(np.median(result.atoms['subvoxel_shift'])):.6g} {unit}"
        )
    if result.npz_path is not None:
        print(f"  localization NPZ: {result.npz_path}")
        print(f"  atom table CSV: {result.csv_path}")


def _load_score_result(
    value: str | Path | LikelihoodMapResult,
) -> tuple[LikelihoodMapResult, Path | None, str | None]:
    if isinstance(value, LikelihoodMapResult) or _is_likelihood_result(value):
        map_path = getattr(value, "map_path", None)
        source = None if map_path is None else Path(map_path).expanduser().resolve()
        return value, source, _hash_if_file(source)
    try:
        source = Path(value).expanduser().resolve()
    except TypeError as exc:
        raise TypeError(
            "likelihood_score_map must be a path or a "
            "LikelihoodMapResult-compatible object"
        ) from exc
    return load_likelihood_map(source), source, _sha256_file(source)


def _is_likelihood_result(value: Any) -> bool:
    """Recognize result instances retained across notebook module reloads."""
    return (
        hasattr(value, "likelihood")
        and hasattr(value, "valid_overlap_mask")
        and isinstance(getattr(value, "metadata", None), Mapping)
        and isinstance(getattr(value, "coordinates", None), Mapping)
    )


def _validate_common_grid(
    score: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    plan: np.ndarray,
    cross: np.ndarray,
    valid: np.ndarray,
    aligned_coordinates: Mapping[str, Any],
) -> None:
    if score.ndim != 3 or plan.shape != score.shape or cross.shape != score.shape:
        raise ValueError("Score, plan, and cross volumes must share one 3D shape.")
    if valid.shape != score.shape or valid.dtype.kind != "b":
        raise ValueError("The aligned valid-overlap mask must match the volume shape.")
    for axis_index, axis in enumerate("xyz"):
        score_axis = np.asarray(coordinates[axis], dtype=np.float64)
        aligned_axis = np.asarray(aligned_coordinates[axis], dtype=np.float64)
        if len(score_axis) != score.shape[axis_index] or not np.allclose(
            score_axis, aligned_axis, rtol=0.0, atol=1e-10
        ):
            raise ValueError(f"Score and aligned coordinate vectors differ on {axis}.")


def _smooth_score(
    score: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    physical_sigma: float,
) -> np.ndarray:
    if physical_sigma == 0.0:
        return np.asarray(score, dtype=np.float32)
    valid = np.isfinite(score)
    numerator = np.where(valid, score, 0.0).astype(np.float32)
    denominator = valid.astype(np.float32)
    for axis_index, axis in enumerate("xyz"):
        sigma_voxels = physical_sigma / _axis_spacing(coordinates[axis])
        kernel = _gaussian_kernel(sigma_voxels)
        numerator = _convolve_axis(numerator, kernel, axis_index)
        denominator = _convolve_axis(denominator, kernel, axis_index)
    output = np.full(score.shape, np.nan, dtype=np.float32)
    supported = denominator > 1e-6
    output[supported] = numerator[supported] / denominator[supported]
    return output


def _gaussian_kernel(sigma_voxels: float) -> np.ndarray:
    if sigma_voxels <= 1e-12:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * sigma_voxels)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_voxels) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _convolve_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    if radius == 0:
        return values.copy()
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(values, pad_width, mode="edge")
    output = np.zeros_like(values, dtype=np.float32)
    for kernel_index, weight in enumerate(kernel):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(kernel_index, kernel_index + values.shape[axis])
        output += np.float32(weight) * padded[tuple(slices)]
    return output


def _local_maxima(
    score: np.ndarray, detection: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    if min(score.shape) < 3:
        raise ValueError("Every score-map axis needs at least three samples.")
    filled = np.where(np.isfinite(detection), detection, -np.inf)
    center = filled[1:-1, 1:-1, 1:-1]
    original_center = score[1:-1, 1:-1, 1:-1]
    local = np.isfinite(original_center) & (original_center >= threshold)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                neighbor = filled[
                    1 + dx : score.shape[0] - 1 + dx,
                    1 + dy : score.shape[1] - 1 + dy,
                    1 + dz : score.shape[2] - 1 + dz,
                ]
                local &= center >= neighbor
    indices = np.argwhere(local).astype(np.int32) + 1
    scores = score[tuple(indices.T)].astype(np.float32, copy=False)
    return indices, scores


def _physical_nms(
    positions: np.ndarray, scores: np.ndarray, minimum_separation: float
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(positions) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(scores, kind="stable")[::-1]
    bins: dict[tuple[int, int, int], list[int]] = {}
    kept: list[int] = []
    for index in order:
        point = positions[index]
        key_array = np.floor(point / minimum_separation).astype(np.int64)
        key = tuple(int(value) for value in key_array)
        rejected = False
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for offset_z in (-1, 0, 1):
                    neighbor_key = (
                        key[0] + offset_x,
                        key[1] + offset_y,
                        key[2] + offset_z,
                    )
                    for other in bins.get(neighbor_key, ()):
                        if np.linalg.norm(point - positions[other]) < minimum_separation:
                            rejected = True
                            break
                    if rejected:
                        break
                if rejected:
                    break
            if rejected:
                break
        if rejected:
            continue
        kept.append(int(index))
        bins.setdefault(key, []).append(int(index))
    return np.asarray(kept, dtype=np.int64)


def _indices_to_positions(
    indices: np.ndarray, coordinates: Mapping[str, np.ndarray]
) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack(
        [coordinates[axis][indices[:, axis_index]] for axis_index, axis in enumerate("xyz")]
    ).astype(np.float64, copy=False)


def _fit_template(spacing: np.ndarray, radius: float) -> dict[str, np.ndarray]:
    index_radii = np.maximum(1, np.ceil(radius / spacing).astype(np.int64))
    index_axes = [
        np.arange(-axis_radius, axis_radius + 1, dtype=np.int32)
        for axis_radius in index_radii
    ]
    index_grid = np.meshgrid(*index_axes, indexing="ij")
    index_offsets = np.column_stack([values.ravel() for values in index_grid])
    physical_offsets = index_offsets * spacing[None, :]
    distances = np.linalg.norm(physical_offsets, axis=1)
    inside = distances <= radius + 1e-12
    index_offsets = index_offsets[inside]
    physical_offsets = physical_offsets[inside]
    distances = distances[inside]
    return {
        "index_offsets": index_offsets,
        "physical_offsets": physical_offsets,
        "radial_weight": np.exp(-0.5 * (distances / (radius * 0.65)) ** 2),
        "spacing": spacing,
    }


def _refine_candidate(
    seed_index: np.ndarray,
    seed_position: np.ndarray,
    plan: np.ndarray,
    cross: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    template: Mapping[str, np.ndarray],
    *,
    floor_plan: float,
    floor_cross: float,
    maximum_shift: float,
    initial_step: float,
    tolerance: float,
    minimum_valid_fraction: float,
) -> dict[str, Any]:
    offsets = np.asarray(template["index_offsets"], dtype=np.int32)
    indices = seed_index[None, :] + offsets
    inside = np.all((indices >= 0) & (indices < np.array(score.shape)), axis=1)
    indices = indices[inside]
    radial_weight = np.asarray(template["radial_weight"])[inside]
    expected_count = len(offsets)
    if len(indices) < 10:
        return _fallback_fit(seed_position, score, coordinates, "insufficient_patch")
    index_tuple = tuple(indices[:, axis] for axis in range(3))
    patch_valid = (
        valid[index_tuple]
        & np.isfinite(plan[index_tuple])
        & np.isfinite(cross[index_tuple])
        & np.isfinite(score[index_tuple])
    )
    valid_fraction = float(np.count_nonzero(patch_valid) / expected_count)
    if valid_fraction < minimum_valid_fraction or np.count_nonzero(patch_valid) < 10:
        result = _fallback_fit(seed_position, score, coordinates, "incomplete_patch")
        result["valid_fraction"] = valid_fraction
        return result
    indices = indices[patch_valid]
    radial_weight = radial_weight[patch_valid]
    points = np.column_stack(
        [coordinates[axis][indices[:, axis_index]] for axis_index, axis in enumerate("xyz")]
    )
    plan_values = _floor_signal(plan[index_tuple][patch_valid], floor_plan)
    cross_values = _floor_signal(cross[index_tuple][patch_valid], floor_cross)
    plan_sigma = _estimate_sigma(points, plan_values, seed_position, template["spacing"])
    cross_sigma = _estimate_sigma(points, cross_values, seed_position, template["spacing"])

    bounds = np.column_stack(
        (seed_position - maximum_shift, seed_position + maximum_shift)
    )
    lower_map = np.array([coordinates[axis][0] for axis in "xyz"])
    upper_map = np.array([coordinates[axis][-1] for axis in "xyz"])
    bounds[:, 0] = np.maximum(bounds[:, 0], lower_map)
    bounds[:, 1] = np.minimum(bounds[:, 1], upper_map)
    quadratic_seed = _quadratic_seed(
        points,
        0.5 * _normalize_patch(plan_values) + 0.5 * _normalize_patch(cross_values),
        radial_weight,
        seed_position,
        maximum_shift,
    )
    initial = np.clip(quadratic_seed, bounds[:, 0], bounds[:, 1])
    fit = _coordinate_refine_center(
        points,
        plan_values,
        cross_values,
        radial_weight,
        plan_sigma,
        cross_sigma,
        initial,
        bounds,
        initial_step,
        tolerance,
    )
    position = fit["position"]
    sampled_score = _trilinear_sample(score, coordinates, position)
    status = "refined" if np.isfinite(sampled_score) else "score_sample_failed"
    return {
        "position": position,
        "score": np.float32(sampled_score),
        "plan_amplitude": np.float32(fit["plan_amplitude"]),
        "cross_amplitude": np.float32(fit["cross_amplitude"]),
        "plan_background": np.float32(fit["plan_background"]),
        "cross_background": np.float32(fit["cross_background"]),
        "plan_sigma_xyz": plan_sigma.astype(np.float32),
        "cross_sigma_xyz": cross_sigma.astype(np.float32),
        "plan_fit_quality": np.float32(fit["plan_fit_quality"]),
        "cross_fit_quality": np.float32(fit["cross_fit_quality"]),
        "joint_fit_quality": np.float32(fit["joint_fit_quality"]),
        "shift": np.float32(np.linalg.norm(position - seed_position)),
        "valid_fraction": np.float32(valid_fraction),
        "evaluation_count": np.int32(fit["evaluation_count"]),
        "status": status,
    }


def _floor_signal(values: np.ndarray, floor: float) -> np.ndarray:
    return np.clip((np.asarray(values, dtype=np.float64) - floor) / (1.0 - floor), 0.0, 1.0)


def _normalize_patch(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(values))
    scale = float(np.max(values) - minimum)
    return (values - minimum) / scale if scale > 1e-12 else np.zeros_like(values)


def _quadratic_seed(
    points: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    origin: np.ndarray,
    maximum_shift: float,
) -> np.ndarray:
    offsets = points - origin
    design = np.column_stack(
        (
            np.ones(len(points)),
            offsets,
            offsets[:, 0] ** 2,
            offsets[:, 1] ** 2,
            offsets[:, 2] ** 2,
            offsets[:, 0] * offsets[:, 1],
            offsets[:, 0] * offsets[:, 2],
            offsets[:, 1] * offsets[:, 2],
        )
    )
    root_weight = np.sqrt(np.asarray(weights, dtype=np.float64))
    try:
        coefficients = np.linalg.lstsq(
            design * root_weight[:, None], values * root_weight, rcond=None
        )[0]
        gradient = coefficients[1:4]
        hessian = np.array(
            [
                [2 * coefficients[4], coefficients[7], coefficients[8]],
                [coefficients[7], 2 * coefficients[5], coefficients[9]],
                [coefficients[8], coefficients[9], 2 * coefficients[6]],
            ]
        )
        if np.max(np.linalg.eigvalsh(hessian)) >= -1e-10:
            return origin.copy()
        displacement = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return origin.copy()
    if not np.isfinite(displacement).all() or np.linalg.norm(displacement) > maximum_shift:
        return origin.copy()
    return origin + displacement


def _estimate_sigma(
    points: np.ndarray,
    values: np.ndarray,
    center: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    background = float(np.percentile(values, 20.0))
    weights = np.maximum(values - background, 0.0)
    if float(np.sum(weights)) <= 1e-12:
        return np.maximum(spacing, np.full(3, float(np.max(spacing))))
    variance = np.sum(weights[:, None] * (points - center) ** 2, axis=0) / np.sum(weights)
    lower = spacing * 0.60
    upper = np.maximum(lower, np.ptp(points, axis=0) * 0.75)
    return np.clip(np.sqrt(np.maximum(variance, 0.0)), lower, upper)


def _coordinate_refine_center(
    points: np.ndarray,
    plan_values: np.ndarray,
    cross_values: np.ndarray,
    radial_weight: np.ndarray,
    plan_sigma: np.ndarray,
    cross_sigma: np.ndarray,
    initial: np.ndarray,
    bounds: np.ndarray,
    initial_step: float,
    tolerance: float,
) -> dict[str, Any]:
    center = np.asarray(initial, dtype=np.float64).copy()
    metrics = _joint_gaussian_metrics(
        center,
        points,
        plan_values,
        cross_values,
        radial_weight,
        plan_sigma,
        cross_sigma,
    )
    evaluations = 1
    step = float(initial_step)
    while step >= tolerance - 1e-15:
        for _ in range(3):
            improved = False
            for axis in range(3):
                best_center = center
                best_metrics = metrics
                for direction in (-1.0, 1.0):
                    candidate = center.copy()
                    candidate[axis] = np.clip(
                        candidate[axis] + direction * step,
                        bounds[axis, 0],
                        bounds[axis, 1],
                    )
                    if candidate[axis] == center[axis]:
                        continue
                    candidate_metrics = _joint_gaussian_metrics(
                        candidate,
                        points,
                        plan_values,
                        cross_values,
                        radial_weight,
                        plan_sigma,
                        cross_sigma,
                    )
                    evaluations += 1
                    if candidate_metrics["joint_fit_quality"] > best_metrics[
                        "joint_fit_quality"
                    ] + 1e-12:
                        best_center = candidate
                        best_metrics = candidate_metrics
                if best_center is not center:
                    center = best_center
                    metrics = best_metrics
                    improved = True
            if not improved:
                break
        step *= 0.5
    return {"position": center, "evaluation_count": evaluations, **metrics}


def _joint_gaussian_metrics(
    center: np.ndarray,
    points: np.ndarray,
    plan_values: np.ndarray,
    cross_values: np.ndarray,
    radial_weight: np.ndarray,
    plan_sigma: np.ndarray,
    cross_sigma: np.ndarray,
) -> dict[str, float]:
    plan_gaussian = np.exp(
        -0.5 * np.sum(((points - center) / plan_sigma) ** 2, axis=1)
    )
    cross_gaussian = np.exp(
        -0.5 * np.sum(((points - center) / cross_sigma) ** 2, axis=1)
    )
    plan_fit = _linear_profile_fit(plan_values, plan_gaussian, radial_weight)
    cross_fit = _linear_profile_fit(cross_values, cross_gaussian, radial_weight)
    return {
        "plan_amplitude": plan_fit["amplitude"],
        "cross_amplitude": cross_fit["amplitude"],
        "plan_background": plan_fit["background"],
        "cross_background": cross_fit["background"],
        "plan_fit_quality": plan_fit["quality"],
        "cross_fit_quality": cross_fit["quality"],
        "joint_fit_quality": 0.5 * (plan_fit["quality"] + cross_fit["quality"]),
    }


def _linear_profile_fit(
    values: np.ndarray, gaussian: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    gaussian = np.asarray(gaussian, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    design = np.column_stack((np.ones(len(values)), gaussian))
    root_weight = np.sqrt(weights)
    try:
        background, amplitude = np.linalg.lstsq(
            design * root_weight[:, None], values * root_weight, rcond=None
        )[0]
    except np.linalg.LinAlgError:
        background, amplitude = float(np.mean(values)), 0.0
    if amplitude < 0.0:
        background, amplitude = float(np.average(values, weights=weights)), 0.0
    prediction = background + amplitude * gaussian
    residual = values - prediction
    weighted_mean = float(np.average(values, weights=weights))
    total = float(np.sum(weights * (values - weighted_mean) ** 2))
    error = float(np.sum(weights * residual**2))
    quality = 1.0 - error / (total + 1e-12)
    return {
        "background": float(background),
        "amplitude": float(amplitude),
        "quality": float(quality),
    }


def _fallback_fit(
    seed_position: np.ndarray,
    score: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    status: str,
) -> dict[str, Any]:
    sampled = _trilinear_sample(score, coordinates, seed_position)
    return {
        "position": np.asarray(seed_position, dtype=np.float64),
        "score": np.float32(sampled),
        "plan_amplitude": np.float32(np.nan),
        "cross_amplitude": np.float32(np.nan),
        "plan_background": np.float32(np.nan),
        "cross_background": np.float32(np.nan),
        "plan_sigma_xyz": np.full(3, np.nan, dtype=np.float32),
        "cross_sigma_xyz": np.full(3, np.nan, dtype=np.float32),
        "plan_fit_quality": np.float32(np.nan),
        "cross_fit_quality": np.float32(np.nan),
        "joint_fit_quality": np.float32(np.nan),
        "shift": np.float32(0.0),
        "valid_fraction": np.float32(0.0),
        "evaluation_count": np.int32(0),
        "status": status,
    }


def _trilinear_sample(
    volume: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    point: np.ndarray,
) -> float:
    lower_indices = []
    fractions = []
    for axis_index, axis in enumerate("xyz"):
        values = coordinates[axis]
        coordinate = float(point[axis_index])
        if coordinate < values[0] or coordinate > values[-1]:
            return float("nan")
        upper = int(np.searchsorted(values, coordinate, side="right"))
        if upper == 0:
            lower, upper = 0, 1
        elif upper >= len(values):
            lower, upper = len(values) - 2, len(values) - 1
        else:
            lower = upper - 1
        delta = values[upper] - values[lower]
        fraction = 0.0 if delta == 0.0 else (coordinate - values[lower]) / delta
        lower_indices.append(lower)
        fractions.append(float(fraction))
    x0, y0, z0 = lower_indices
    fx, fy, fz = fractions
    block = np.asarray(volume[x0 : x0 + 2, y0 : y0 + 2, z0 : z0 + 2])
    if block.shape != (2, 2, 2) or not np.isfinite(block).all():
        return float("nan")
    x_values = block[0] * (1.0 - fx) + block[1] * fx
    y_values = x_values[0] * (1.0 - fy) + x_values[1] * fy
    return float(y_values[0] * (1.0 - fz) + y_values[1] * fz)


def _accepted_atom_table(
    candidates: Mapping[str, np.ndarray], accepted_indices: np.ndarray
) -> dict[str, np.ndarray]:
    if len(accepted_indices):
        positions = candidates["position_xyz"][accepted_indices]
        order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        indices = accepted_indices[order]
    else:
        indices = accepted_indices
    atoms = {
        name: np.asarray(values)[indices]
        for name, values in candidates.items()
        if name not in {"accepted"}
    }
    atoms["atom_id"] = np.arange(1, len(indices) + 1, dtype=np.int64)
    atoms["candidate_index"] = indices.astype(np.int64, copy=False)
    return atoms


def _localization_metadata(
    score_result: LikelihoodMapResult,
    coordinates: Mapping[str, np.ndarray],
    atoms: Mapping[str, np.ndarray],
    candidates: Mapping[str, np.ndarray],
    **parameters: Any,
) -> dict[str, Any]:
    score_source = parameters.pop("score_source")
    score_hash = parameters.pop("score_hash")
    aligned_source = parameters.pop("aligned_source")
    aligned_hash = parameters.pop("aligned_hash")
    return {
        "schema_version": ATOM_LOCALIZATION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quantity_name": "localized_atomic_centers",
        "axis_order": ["x", "y", "z"],
        "length_unit": str(score_result.metadata.get("length_unit", "")),
        "coordinate_ranges": {
            axis: [float(values[0]), float(values[-1])]
            for axis, values in coordinates.items()
        },
        "candidate_count": int(len(candidates["seed_score"])),
        "accepted_atom_count": int(len(atoms["position_xyz"])),
        "candidate_status_counts": {
            str(status): int(count)
            for status, count in zip(*np.unique(candidates["status"], return_counts=True))
        },
        "method": {
            "candidate_detection": "3D local maxima plus Euclidean physical NMS",
            "subvoxel_initialization": "weighted local 3D quadratic",
            "subvoxel_refinement": (
                "bounded shared-center coordinate search against aligned plan and "
                "cross intensities with dataset-specific anisotropic Gaussian profiles"
            ),
            "interpolation": "trilinear score sampling at refined centers",
        },
        "source_likelihood_score_map_path": (
            str(score_source) if score_source is not None else None
        ),
        "source_likelihood_score_map_sha256": score_hash,
        "source_aligned_grid_path": (
            str(aligned_source) if aligned_source is not None else None
        ),
        "source_aligned_grid_sha256": aligned_hash,
        **{
            key: _json_value(value)
            for key, value in parameters.items()
        },
    }


def _metadata_floor(metadata: Mapping[str, Any], name: str) -> float:
    if name not in metadata:
        raise ValueError(f"Likelihood score metadata is missing {name!r}.")
    result = _normalized_scalar(metadata[name], name)
    if result >= 1.0:
        raise ValueError(f"{name} must be below 1.")
    return result


def _axis_spacing(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    differences = np.diff(values)
    if values.ndim != 1 or len(values) < 2 or np.any(differences <= 0.0):
        raise ValueError("Coordinate vectors must be increasing with at least two values.")
    return float(np.median(differences))


def _normalized_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite.")
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _hash_if_file(path: Path | None) -> str | None:
    return _sha256_file(path) if path is not None and path.is_file() else None


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()
