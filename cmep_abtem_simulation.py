"""abTEM simulation and multislice ptychography helpers for CMEP validation.

The numerical defaults are a reproducible starting point, not converged
publication settings. Use the resource report first, then perform convergence
tests for sampling, slice thickness, potential parametrization, vacuum, scan
step, detector angle, phonons, dose, and reconstruction iterations.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cmep_au_model import OrientedAtomsResult, atomic_model_fingerprint


SIMULATION_SCHEMA = "cmep.abtem-4dstem.v3"
ORACLE_SCHEMA = "cmep.abtem-oracle-potential.v2"
RECONSTRUCTION_SCHEMA = "cmep.abtem-multislice-reconstruction.v4"
QC_SCHEMA = "cmep.abtem-4dstem-qc.v2"
COMPLEX_OBJECT_INITIALIZATION_SCHEMA = "cmep.abtem-complex-object-initialization.v1"
RECONSTRUCTION_HISTORY_SCHEMA = "cmep.abtem-reconstruction-history.v1"
DEPTH_IDENTIFIABILITY_SCHEMA = "cmep.abtem-depth-identifiability.v1"

_PROBE_INITIALIZATION_MODES = {"abtem_default", "simulation_exact"}
_OBJECT_INITIALIZATION_MODES = {"uniform", "split_projection", "provided_complex"}
_OBJECT_CONSTRAINT_MODES = {"none", "pure_phase", "pure_phase_positive"}
_RECONSTRUCTION_GRID_MODES = {"detector", "simulation_native"}
_NON_SIMULATION_CONFIG_FIELDS = {
    "device",
    "max_batch",
    "cpu_chunk_size",
    "gpu_chunk_size",
    "reconstruction_slice_thickness_angstrom",
    "reconstruction_iterations",
}

_WINDOWS_LEGACY_PATH_LIMIT = 260
# Zarr 3 appends nested array/chunk keys and a 32-character atomic-write token.
_ZARR_INTERNAL_PATH_RESERVE = 64

VALIDATION_CONDITION_DESCRIPTIONS = {
    "ideal_static": (
        "Same post-vacancy structure, one reference configuration, no counting noise."
    ),
    "thermal_only": (
        "Incoherently averaged frozen phonons, with no counting or detector noise."
    ),
    "thermal_dose_limited": (
        "Frozen phonons followed by Poisson counting noise at the configured dose."
    ),
    "instrument_model": (
        "Thermal and dose-limited data with all enabled calibrated instrument effects."
    ),
}

_KNOWN_ABTEM_ABERRATIONS = {
    "C10", "C12", "phi12", "C21", "phi21", "C23", "phi23",
    "C30", "C32", "phi32", "C34", "phi34", "C41", "phi41",
    "C43", "phi43", "C45", "phi45", "C50", "C52", "phi52",
    "C54", "phi54", "C56", "phi56", "defocus", "astigmatism",
    "astigmatism_angle", "coma", "coma_angle", "Cs", "C5",
}


@dataclass(frozen=True)
class PtychographyConfig:
    """Physical and numerical settings shared by all particle diameters."""

    energy_ev: float = 200_000.0
    semiangle_mrad: float = 25.0
    potential_sampling_angstrom: float = 0.20
    potential_slice_thickness_angstrom: float = 1.0
    potential_parametrization: str = "lobato"
    potential_projection: str = "finite"
    scan_step_angstrom: float = 0.50
    scan_margin_angstrom: float = 3.0
    detector_max_angle_mrad: float = 50.0
    frozen_phonon_configs: int = 4
    thermal_sigma_angstrom: float | dict[str, float] = 0.08
    dose_electrons_per_angstrom2: float | None = 100_000.0
    random_seed: int = 17
    device: str = "gpu"
    max_batch: int | str = "auto"
    cpu_chunk_size: str = "128 MB"
    gpu_chunk_size: str = "512 MB"
    reconstruction_slice_thickness_angstrom: float = 2.0
    reconstruction_iterations: int = 20
    condition_name: str = "custom"
    probe_aberrations: dict[str, float] | None = None
    beam_tilt_mrad: tuple[float, float] = (0.0, 0.0)
    partial_coherence_source_sigma_angstrom: float = 0.0
    scan_position_error_std_angstrom: float = 0.0
    detector_background_mean_counts: float = 0.0
    detector_read_noise_std_counts: float = 0.0
    detector_gain_std_fraction: float = 0.0
    detector_dead_pixel_fraction: float = 0.0
    detector_saturation_counts: float | None = None
    probe_aperture_soft: bool = True

    def validated(self) -> "PtychographyConfig":
        """Return this immutable configuration after informative validation."""
        for name in (
            "energy_ev",
            "semiangle_mrad",
            "potential_sampling_angstrom",
            "potential_slice_thickness_angstrom",
            "scan_step_angstrom",
            "detector_max_angle_mrad",
            "reconstruction_slice_thickness_angstrom",
        ):
            _positive_finite(getattr(self, name), name)
        _normalize_thermal_sigma(self.thermal_sigma_angstrom)
        _positive_finite(
            self.scan_margin_angstrom, "scan_margin_angstrom", allow_zero=True
        )
        for name in (
            "partial_coherence_source_sigma_angstrom",
            "scan_position_error_std_angstrom",
            "detector_background_mean_counts",
            "detector_read_noise_std_counts",
            "detector_gain_std_fraction",
        ):
            _positive_finite(getattr(self, name), name, allow_zero=True)
        if self.dose_electrons_per_angstrom2 is not None:
            _positive_finite(
                self.dose_electrons_per_angstrom2,
                "dose_electrons_per_angstrom2",
            )
        if int(self.frozen_phonon_configs) != self.frozen_phonon_configs:
            raise ValueError("frozen_phonon_configs must be an integer.")
        if self.frozen_phonon_configs < 1:
            raise ValueError("frozen_phonon_configs must be at least 1.")
        if int(self.reconstruction_iterations) != self.reconstruction_iterations:
            raise ValueError("reconstruction_iterations must be an integer.")
        if self.reconstruction_iterations < 1:
            raise ValueError("reconstruction_iterations must be at least 1.")
        if int(self.random_seed) != self.random_seed or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer.")
        condition_name = str(self.condition_name).strip().lower()
        if not condition_name:
            raise ValueError("condition_name must not be empty.")
        beam_tilt = np.asarray(self.beam_tilt_mrad, dtype=np.float64)
        if beam_tilt.shape != (2,) or not np.isfinite(beam_tilt).all():
            raise ValueError("beam_tilt_mrad must contain two finite values.")
        dead_fraction = float(self.detector_dead_pixel_fraction)
        if not np.isfinite(dead_fraction) or not 0.0 <= dead_fraction < 1.0:
            raise ValueError("detector_dead_pixel_fraction must be finite and in [0, 1).")
        if self.detector_saturation_counts is not None:
            _positive_finite(
                self.detector_saturation_counts, "detector_saturation_counts"
            )
        if self.probe_aberrations is not None:
            if not isinstance(self.probe_aberrations, Mapping):
                raise TypeError("probe_aberrations must be a mapping or None.")
            unsupported = sorted(set(self.probe_aberrations) - _KNOWN_ABTEM_ABERRATIONS)
            if unsupported:
                raise ValueError(
                    "Unsupported abTEM aberration symbols: " + ", ".join(unsupported)
                )
            if any(not np.isfinite(float(value)) for value in self.probe_aberrations.values()):
                raise ValueError("All probe_aberrations values must be finite.")
        if _detector_count_effects_enabled(self) and self.dose_electrons_per_angstrom2 is None:
            raise ValueError(
                "Detector background/read/gain/dead-pixel/saturation effects require "
                "dose_electrons_per_angstrom2 so their count units are meaningful."
            )
        _validate_named_condition(self, condition_name)
        device = str(self.device).strip().lower()
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'.")
        if self.potential_projection not in {"finite", "infinite"}:
            raise ValueError("potential_projection must be 'finite' or 'infinite'.")
        if str(self.potential_parametrization).strip().lower() not in {
            "lobato",
            "kirkland",
            "peng",
        }:
            raise ValueError(
                "potential_parametrization must be 'lobato', 'kirkland', or 'peng'."
            )
        if not isinstance(self.probe_aperture_soft, bool):
            raise TypeError("probe_aperture_soft must be Boolean.")
        if not (
            self.max_batch == "auto"
            or isinstance(self.max_batch, int)
            and self.max_batch >= 1
        ):
            raise ValueError("max_batch must be 'auto' or a positive integer.")
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["device"] = str(self.device).strip().lower()
        data["condition_name"] = str(self.condition_name).strip().lower()
        data["potential_parametrization"] = (
            str(self.potential_parametrization).strip().lower()
        )
        data["beam_tilt_mrad"] = [float(value) for value in self.beam_tilt_mrad]
        data["thermal_sigma_angstrom"] = _normalize_thermal_sigma(
            self.thermal_sigma_angstrom
        )
        if self.probe_aberrations is not None:
            data["probe_aberrations"] = {
                str(key): float(value)
                for key, value in sorted(self.probe_aberrations.items())
            }
        return data


def simulation_configuration(
    value: PtychographyConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Return only settings that can change simulated diffraction data."""
    if isinstance(value, PtychographyConfig):
        data = value.validated().to_dict()
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        raise TypeError(
            "value must be a PtychographyConfig or configuration mapping."
        )
    return {
        key: data[key]
        for key in sorted(data)
        if key not in _NON_SIMULATION_CONFIG_FIELDS
    }


def simulation_configuration_differences(
    cached: PtychographyConfig | Mapping[str, Any],
    requested: PtychographyConfig | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Report physical settings that prevent reuse of a 4D-STEM cache."""
    cached_data = simulation_configuration(cached)
    requested_data = simulation_configuration(requested)
    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(cached_data) | set(requested_data)):
        cached_value = cached_data.get(key, "<missing>")
        requested_value = requested_data.get(key, "<missing>")
        if cached_value != requested_value:
            differences[key] = {
                "cached": cached_value,
                "requested": requested_value,
            }
    return differences


def make_validation_conditions(
    base_config: PtychographyConfig,
) -> dict[str, PtychographyConfig]:
    """Create the explicit four-rung validation ladder from one shared recipe."""
    base = base_config.validated()
    if base.dose_electrons_per_angstrom2 is None:
        raise ValueError(
            "base_config.dose_electrons_per_angstrom2 is required to construct "
            "the dose-limited validation conditions."
        )
    ideal_instrument = {
        "probe_aberrations": None,
        "beam_tilt_mrad": (0.0, 0.0),
        "partial_coherence_source_sigma_angstrom": 0.0,
        "scan_position_error_std_angstrom": 0.0,
        "detector_background_mean_counts": 0.0,
        "detector_read_noise_std_counts": 0.0,
        "detector_gain_std_fraction": 0.0,
        "detector_dead_pixel_fraction": 0.0,
        "detector_saturation_counts": None,
    }
    conditions = {
        "ideal_static": replace(
            base,
            condition_name="ideal_static",
            frozen_phonon_configs=1,
            dose_electrons_per_angstrom2=None,
            **ideal_instrument,
        ),
        "thermal_only": replace(
            base,
            condition_name="thermal_only",
            dose_electrons_per_angstrom2=None,
            **ideal_instrument,
        ),
        "thermal_dose_limited": replace(
            base,
            condition_name="thermal_dose_limited",
            **ideal_instrument,
        ),
        "instrument_model": replace(base, condition_name="instrument_model"),
    }
    return {name: config.validated() for name, config in conditions.items()}


def print_validation_conditions(
    conditions: Mapping[str, PtychographyConfig],
) -> None:
    """Print which physical effects are active in each validation condition."""
    print("Validation condition ladder")
    for name, config in conditions.items():
        effects = _condition_effects(config.validated())
        enabled = [key for key, value in effects.items() if value is True]
        text = ", ".join(enabled) if enabled else "static ideal data"
        instrument_keys = {
            "probe_aberrations",
            "beam_tilt",
            "partial_coherence",
            "scan_position_errors",
            "detector_response",
        }
        if name == "instrument_model" and not instrument_keys.intersection(enabled):
            text += "; no additional instrument effects configured"
        print(f"  {name}: {text}")


@dataclass(frozen=True)
class ResourceEstimate:
    """Conservative preflight estimates, not hard execution limits."""

    details: dict[str, Any]


@dataclass(frozen=True)
class OraclePotentialResult:
    """A static truth potential exported in CMEP raw-stack convention."""

    stack_slice_row_col: np.ndarray
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    z_angstrom: np.ndarray
    metadata: dict[str, Any]
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SimulationResult:
    """Cached or newly computed 4D-STEM diffraction patterns."""

    diffraction_patterns: Any
    metadata: dict[str, Any]
    output_path: Path
    manifest_path: Path
    scan_positions_path: Path | None = None
    simulation_state_path: Path | None = None


@dataclass(frozen=True)
class SimulationQCResult:
    """Compact quality-control products derived without materializing all 4D data."""

    mean_diffraction_pattern: np.ndarray
    scan_integrated_image: np.ndarray
    histogram_counts: np.ndarray
    histogram_edges: np.ndarray
    metadata: dict[str, Any]
    output_path: Path | None
    manifest_path: Path | None


@dataclass(frozen=True)
class ReconstructionResult:
    """Multislice ptychographic object slices and physical coordinates."""

    objects_complex_slice_xy: np.ndarray
    phase_stack_slice_row_col: np.ndarray
    probes_complex_slice_xy: np.ndarray
    positions_angstrom: np.ndarray
    error: float
    metadata: dict[str, Any]
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ComplexObjectInitializationResult:
    """Complex transmission slices on a physically described source grid."""

    objects_complex_slice_xy: np.ndarray
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    z_edges_angstrom: np.ndarray
    metadata: dict[str, Any]
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ReconstructionHistoryResult:
    """Selected object states captured from one reconstruction run."""

    iterations: np.ndarray
    errors: np.ndarray
    objects_complex_iteration_slice_xy: np.ndarray
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    z_angstrom: np.ndarray
    metadata: dict[str, Any]
    output_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class DepthIdentifiabilityResult:
    """Oracle-referenced metrics for depth migration during reconstruction."""

    iterations: np.ndarray
    errors: np.ndarray
    phase_fractions: np.ndarray
    near_wrap_counts: np.ndarray
    oracle_correlations: np.ndarray
    reversed_oracle_correlations: np.ndarray
    per_slice_oracle_correlations: np.ndarray
    global_phase_scales: np.ndarray
    scale_adjusted_nrmse: np.ndarray
    transmission_departure: np.ndarray
    metadata: dict[str, Any]
    output_path: Path
    manifest_path: Path


def environment_report() -> dict[str, Any]:
    """Report the runtime and whether CuPy can see a CUDA device."""
    report: dict[str, Any] = {
        "python": _python_version(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "ase", "abtem", "cupy", "zarr", "dask")
        },
        "gpu": {
            "available": False,
            "count": 0,
            "name": None,
            "free_memory_gib": None,
            "total_memory_gib": None,
            "error": None,
        },
    }
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        report["gpu"]["count"] = count
        report["gpu"]["available"] = count > 0
        if count:
            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            report["gpu"]["name"] = (
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
            )
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            report["gpu"]["free_memory_gib"] = _gib(int(free_bytes))
            report["gpu"]["total_memory_gib"] = _gib(int(total_bytes))
    except Exception as exc:
        report["gpu"]["error"] = f"{type(exc).__name__}: {exc}"
    return report


def configure_abtem_runtime(
    config: PtychographyConfig,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Apply abTEM device/chunk settings and use project-local GPU caches."""
    cfg = config.validated()
    if cache_dir is not None:
        cache_root = Path(cache_dir).expanduser()
        cupy_cache = cache_root / "cupy"
        temp_dir = cache_root / "tmp"
        mpl_cache = cache_root / "matplotlib"
        for directory in (cupy_cache, temp_dir, mpl_cache):
            directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CUPY_CACHE_DIR", str(cupy_cache.resolve()))
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))
        os.environ.setdefault("TEMP", str(temp_dir.resolve()))
        os.environ.setdefault("TMP", str(temp_dir.resolve()))

    import abtem

    abtem.config.set(
        {
            "device": str(cfg.device).strip().lower(),
            "dask.chunk-size": cfg.cpu_chunk_size,
            "dask.chunk-size-gpu": cfg.gpu_chunk_size,
        }
    )
    report = environment_report()
    if cfg.device == "gpu" and not report["gpu"]["available"]:
        raise RuntimeError(
            "device='gpu' was requested, but CuPy did not detect a CUDA device: "
            f"{report['gpu']['error']}"
        )
    return report


def estimate_simulation_resources(
    view: OrientedAtomsResult,
    config: PtychographyConfig,
) -> ResourceEstimate:
    """Estimate grids and bytes before constructing an expensive Dask graph."""
    cfg = config.validated()
    cell = np.asarray(view.atoms.cell.lengths(), dtype=np.float64)
    if np.any(cell <= 2.0 * cfg.scan_margin_angstrom):
        raise ValueError("scan_margin_angstrom leaves no scan area in the view cell.")
    potential_gpts = np.ceil(cell[:2] / cfg.potential_sampling_angstrom).astype(int)
    potential_slices = int(
        math.ceil(cell[2] / cfg.potential_slice_thickness_angstrom)
    )
    scan_gpts = np.ceil(
        (cell[:2] - 2.0 * cfg.scan_margin_angstrom) / cfg.scan_step_angstrom
    ).astype(int)

    wavelength = electron_wavelength_angstrom(cfg.energy_ev)
    qmax = cfg.detector_max_angle_mrad * 1e-3 / wavelength
    detector_gpts = np.minimum(
        potential_gpts,
        2 * np.ceil(qmax * cell[:2]).astype(int),
    )
    detector_gpts = np.maximum(detector_gpts, 2)
    scan_positions = int(np.prod(scan_gpts, dtype=np.int64))
    detector_pixels = int(np.prod(detector_gpts, dtype=np.int64))
    stored_bytes = scan_positions * detector_pixels * np.dtype(np.float32).itemsize
    potential_bytes = (
        potential_slices
        * int(np.prod(potential_gpts, dtype=np.int64))
        * np.dtype(np.float32).itemsize
    )
    unreduced_frozen_bytes = stored_bytes * cfg.frozen_phonon_configs
    details = {
        "condition_name": cfg.condition_name,
        "atom_count": int(len(view.atoms)),
        "cell_angstrom": cell.tolist(),
        "potential_gpts_xy": potential_gpts.tolist(),
        "potential_slices": potential_slices,
        "potential_array_gib": _gib(potential_bytes),
        "scan_gpts_xy": scan_gpts.tolist(),
        "scan_positions": scan_positions,
        "estimated_detector_gpts_xy": detector_gpts.tolist(),
        "estimated_detector_pixels": detector_pixels,
        "stored_4dstem_float32_gib": _gib(stored_bytes),
        "unreduced_frozen_phonon_float32_gib": _gib(unreduced_frozen_bytes),
        "frozen_phonon_configs": int(cfg.frozen_phonon_configs),
        "electron_wavelength_angstrom": wavelength,
        "device": cfg.device,
        "notes": [
            "No atom, scan-position, detector-pixel, or output-size cap is applied.",
            "Frozen-phonon intensities are averaged lazily before the stored dataset.",
            "Instrument effects are included only when enabled in this condition.",
            "Peak memory depends on Dask chunks and abTEM work arrays, not only output size.",
            "The current abTEM ptychography operator loads its reconstruction data in memory.",
        ],
    }
    return ResourceEstimate(details=details)


def print_resource_estimate(estimate: ResourceEstimate, *, label: str = "view") -> None:
    """Print the high-signal portion of a preflight estimate."""
    info = estimate.details
    print(f"abTEM resource estimate: {label}")
    print(
        f"  atoms={info['atom_count']:,}, cell={_format_triplet(info['cell_angstrom'])} A"
    )
    print(
        f"  potential={tuple(info['potential_gpts_xy'])} x {info['potential_slices']} slices "
        f"(~{info['potential_array_gib']:.3f} GiB raw)"
    )
    print(
        f"  scan={tuple(info['scan_gpts_xy'])} = {info['scan_positions']:,} positions"
    )
    print(
        f"  detector~{tuple(info['estimated_detector_gpts_xy'])}; "
        f"stored 4D-STEM~{info['stored_4dstem_float32_gib']:.2f} GiB"
    )
    print(
        f"  unreduced {info['frozen_phonon_configs']}-phonon equivalent~"
        f"{info['unreduced_frozen_phonon_float32_gib']:.2f} GiB (processed lazily)"
    )


def export_oracle_potential(
    view: OrientedAtomsResult,
    config: PtychographyConfig,
    output_path: str | Path,
    *,
    model_fingerprint: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> OraclePotentialResult:
    """Build the static abTEM potential and export `(slice,row=y,col=x)` data."""
    cfg = config.validated()
    path = _require_suffix(output_path, ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".json")
    model_hash = model_fingerprint or atomic_model_fingerprint(view.atoms)
    fingerprint = _fingerprint(
        {
            "schema": ORACLE_SCHEMA,
            "model_fingerprint": model_hash,
            "view": view.metadata,
            "potential": _potential_config(cfg),
        }
    )
    if _cache_matches(
        path,
        manifest_path,
        fingerprint,
        overwrite=overwrite,
        schema=ORACLE_SCHEMA,
    ):
        return load_oracle_potential(path)

    potential = _make_potential(view, cfg, frozen_phonons=False)
    potential_array = potential.build(lazy=True).compute(progress_bar=progress)
    from abtem.core.backend import asnumpy

    array_slice_xy = np.asarray(asnumpy(potential_array.array), dtype=np.float32)
    if array_slice_xy.ndim != 3:
        raise RuntimeError(
            "Static abTEM potential was expected to have shape (slice,x,y), "
            f"received {array_slice_xy.shape}."
        )
    stack = np.transpose(array_slice_xy, (0, 2, 1)).copy()
    sampling = tuple(float(value) for value in potential_array.sampling)
    x = np.arange(array_slice_xy.shape[1], dtype=np.float64) * sampling[0]
    y = np.arange(array_slice_xy.shape[2], dtype=np.float64) * sampling[1]
    endpoints = np.asarray(potential_array.axes_metadata[0].values, dtype=np.float64)
    starts = np.concatenate(([0.0], endpoints[:-1]))
    z = (starts + endpoints) / 2.0
    slice_thicknesses = endpoints - starts
    metadata = {
        "schema": ORACLE_SCHEMA,
        "fingerprint_sha256": fingerprint,
        "model_fingerprint_sha256": model_hash,
        "view": view.metadata,
        "configuration": _potential_config(cfg),
        "abtem_native_shape_slice_xy": list(array_slice_xy.shape),
        "export_shape_slice_row_col": list(stack.shape),
        "export_axis_order": ["slice_z_view", "row_y_view", "col_x_view"],
        "orientation_string_within_view_frame": "yx",
        "coordinate_unit": "angstrom",
        "potential_value_description": "abTEM projected electrostatic potential per slice",
        "sampling_xy_angstrom": list(sampling),
        "slice_thicknesses_angstrom": slice_thicknesses.tolist(),
        "coordinate_ranges_angstrom": {
            "x": _range(x),
            "y": _range(y),
            "z_slice_centers": _range(z),
        },
        "source_tiffs_modified": False,
        "software_versions": _software_versions(),
    }
    np.savez_compressed(
        path,
        potential_stack_slice_row_col=stack,
        x_angstrom=x,
        y_angstrom=y,
        z_angstrom=z,
    )
    _write_json(manifest_path, metadata)
    return OraclePotentialResult(
        stack_slice_row_col=stack,
        x_angstrom=x,
        y_angstrom=y,
        z_angstrom=z,
        metadata=metadata,
        output_path=path.resolve(),
        manifest_path=manifest_path.resolve(),
    )


def load_oracle_potential(path: str | Path) -> OraclePotentialResult:
    """Load an oracle-potential cache and its required metadata."""
    source = _require_suffix(path, ".npz")
    manifest = source.with_suffix(".json")
    if not source.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Oracle potential cache is incomplete: {source}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as data:
        stack = np.asarray(data["potential_stack_slice_row_col"], dtype=np.float32)
        x = np.asarray(data["x_angstrom"], dtype=np.float64)
        y = np.asarray(data["y_angstrom"], dtype=np.float64)
        z = np.asarray(data["z_angstrom"], dtype=np.float64)
    return OraclePotentialResult(
        stack_slice_row_col=stack,
        x_angstrom=x,
        y_angstrom=y,
        z_angstrom=z,
        metadata=metadata,
        output_path=source.resolve(),
        manifest_path=manifest.resolve(),
    )


def oracle_potential_to_complex_object(
    oracle: OraclePotentialResult | str | Path,
    output_path: str | Path,
    *,
    energy_ev: float,
    target_slice_count: int,
    overwrite: bool = False,
) -> ComplexObjectInitializationResult:
    """Convert oracle potential into thin-phase transmission slabs.

    The abTEM potential values are already projected through each native slice,
    so depth rebinning conserves projected potential before applying
    ``exp(1j * sigma * V_projected)``. No extra thickness factor is applied.
    The returned in-plane grid remains the oracle grid; reconstruction performs
    coordinate-aware interpolation after its padded object grid is known.
    """
    source = load_oracle_potential(oracle) if isinstance(oracle, (str, Path)) else oracle
    if not isinstance(source, OraclePotentialResult):
        raise TypeError("oracle must be an OraclePotentialResult or oracle .npz path.")
    energy = _positive_finite(energy_ev, "energy_ev")
    if isinstance(target_slice_count, (bool, np.bool_)):
        raise TypeError("target_slice_count must be a positive integer.")
    if int(target_slice_count) != target_slice_count or target_slice_count < 1:
        raise ValueError("target_slice_count must be a positive integer.")
    num_target_slices = int(target_slice_count)

    stack = np.asarray(source.stack_slice_row_col, dtype=np.float64)
    x = _strict_coordinate_vector(source.x_angstrom, "oracle x_angstrom")
    y = _strict_coordinate_vector(source.y_angstrom, "oracle y_angstrom")
    z = np.asarray(source.z_angstrom, dtype=np.float64)
    if stack.ndim != 3 or stack.shape != (len(z), len(y), len(x)):
        raise ValueError(
            "Oracle potential shape must be (slice,row=y,col=x) and match its "
            f"coordinates; received {stack.shape}."
        )
    if not np.isfinite(stack).all():
        raise ValueError("Oracle potential contains non-finite values.")

    thicknesses = np.asarray(
        source.metadata.get("slice_thicknesses_angstrom"), dtype=np.float64
    )
    if thicknesses.shape != (stack.shape[0],) or not np.isfinite(thicknesses).all():
        raise ValueError(
            "Oracle metadata must contain one finite slice thickness per slice."
        )
    if np.any(thicknesses <= 0.0):
        raise ValueError("Oracle slice thicknesses must be positive.")
    source_edges = np.concatenate(([0.0], np.cumsum(thicknesses)))
    expected_centers = (source_edges[:-1] + source_edges[1:]) / 2.0
    if z.shape != expected_centers.shape or not np.allclose(
        z, expected_centers, rtol=0.0, atol=1e-7
    ):
        raise ValueError(
            "Oracle z coordinates are inconsistent with its slice thicknesses."
        )

    target_edges = np.linspace(
        0.0, float(source_edges[-1]), num_target_slices + 1, dtype=np.float64
    )
    overlap = np.maximum(
        0.0,
        np.minimum(target_edges[1:, None], source_edges[None, 1:])
        - np.maximum(target_edges[:-1, None], source_edges[None, :-1]),
    )
    weights = overlap / thicknesses[None, :]
    if not np.allclose(weights.sum(axis=0), 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError("Depth rebinning did not conserve every oracle slice.")
    target_potential = np.tensordot(weights, stack, axes=(1, 0))

    from abtem.core.energy import energy2sigma

    sigma = float(energy2sigma(energy))
    transmission_slice_row_col = np.exp(1j * sigma * target_potential)
    objects = np.transpose(transmission_slice_row_col, (0, 2, 1)).astype(
        np.complex64
    )
    target_centers = (target_edges[:-1] + target_edges[1:]) / 2.0
    path = _require_suffix(output_path, ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".json")
    conversion = {
        "schema": COMPLEX_OBJECT_INITIALIZATION_SCHEMA,
        "source_oracle_fingerprint_sha256": source.metadata.get(
            "fingerprint_sha256"
        ),
        "model_fingerprint_sha256": source.metadata.get(
            "model_fingerprint_sha256"
        ),
        "view": source.metadata.get("view"),
        "source_oracle_configuration": source.metadata.get("configuration"),
        "energy_ev": energy,
        "interaction_parameter_sigma": sigma,
        "target_slice_count": num_target_slices,
        "target_z_edges_angstrom": target_edges.tolist(),
        "conversion_rule": (
            "Conserve projected potential by physical slab overlap, then apply "
            "exp(1j * sigma * V_projected); no additional thickness factor."
        ),
    }
    fingerprint = _fingerprint(conversion)
    if _cache_matches(
        path,
        manifest_path,
        fingerprint,
        overwrite=overwrite,
        schema=COMPLEX_OBJECT_INITIALIZATION_SCHEMA,
    ):
        return load_complex_object_initialization(path)

    metadata = {
        **conversion,
        "fingerprint_sha256": fingerprint,
        "source_oracle_path": str(source.output_path.resolve()),
        "source_oracle_shape_slice_row_col": list(stack.shape),
        "source_oracle_slice_thicknesses_angstrom": thicknesses.tolist(),
        "source_oracle_z_edges_angstrom": source_edges.tolist(),
        "objects_shape_slice_xy": list(objects.shape),
        "axis_order": ["slice_z_view", "x_view", "y_view"],
        "coordinate_unit": "angstrom",
        "x_range_angstrom": _range(x),
        "y_range_angstrom": _range(y),
        "z_center_range_angstrom": _range(target_centers),
        "complex_array_dtype": str(objects.dtype),
        "complex_array_sha256": _sha256_array(objects),
        "simulation_only_truth_initialization": True,
        "software_versions": _software_versions(),
    }
    np.savez_compressed(
        path,
        objects_complex_slice_xy=objects,
        x_angstrom=x,
        y_angstrom=y,
        z_edges_angstrom=target_edges,
    )
    _write_json(manifest_path, metadata)
    return ComplexObjectInitializationResult(
        objects_complex_slice_xy=objects,
        x_angstrom=x,
        y_angstrom=y,
        z_edges_angstrom=target_edges,
        metadata=metadata,
        output_path=path.resolve(),
        manifest_path=manifest_path.resolve(),
    )


def load_complex_object_initialization(
    path: str | Path,
) -> ComplexObjectInitializationResult:
    """Load a physically described complex-object initializer."""
    source = _require_suffix(path, ".npz")
    manifest = source.with_suffix(".json")
    if not source.is_file() or not manifest.is_file():
        raise FileNotFoundError(
            f"Complex-object initialization cache is incomplete: {source}"
        )
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("schema") != COMPLEX_OBJECT_INITIALIZATION_SCHEMA:
        raise ValueError(
            "Complex-object initialization manifest has an unsupported schema."
        )
    with np.load(source, allow_pickle=False) as data:
        objects = np.asarray(data["objects_complex_slice_xy"], dtype=np.complex64)
        x = np.asarray(data["x_angstrom"], dtype=np.float64)
        y = np.asarray(data["y_angstrom"], dtype=np.float64)
        z_edges = np.asarray(data["z_edges_angstrom"], dtype=np.float64)
    if _sha256_array(objects) != metadata.get("complex_array_sha256"):
        raise ValueError("Complex-object initialization array hash does not match.")
    return ComplexObjectInitializationResult(
        objects_complex_slice_xy=objects,
        x_angstrom=x,
        y_angstrom=y,
        z_edges_angstrom=z_edges,
        metadata=metadata,
        output_path=source.resolve(),
        manifest_path=manifest.resolve(),
    )


def _validate_zarr_output_path(path: Path) -> None:
    """Reject Windows paths that leave too little room for Zarr internals."""
    if os.name != "nt":
        return
    resolved = str(path.resolve())
    estimated_internal_length = len(resolved) + _ZARR_INTERNAL_PATH_RESERVE
    if estimated_internal_length >= _WINDOWS_LEGACY_PATH_LIMIT:
        maximum_root_length = (
            _WINDOWS_LEGACY_PATH_LIMIT - _ZARR_INTERNAL_PATH_RESERVE - 1
        )
        raise ValueError(
            "Zarr output path is too long for reliable Windows atomic writes: "
            f"the resolved .zarr path is {len(resolved)} characters and Zarr may "
            f"need about {_ZARR_INTERNAL_PATH_RESERVE} more. Keep the .zarr path "
            f"at or below {maximum_root_length} characters by shortening the "
            "output hierarchy or cache filename."
        )


def simulate_4dstem(
    view: OrientedAtomsResult,
    config: PtychographyConfig,
    output_path: str | Path,
    *,
    model_fingerprint: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> SimulationResult:
    """Simulate one explicit validation condition and cache it independently."""
    cfg = config.validated()
    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".zarr":
        raise ValueError("output_path must end in .zarr.")
    _validate_zarr_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".json")
    model_hash = model_fingerprint or atomic_model_fingerprint(view.atoms)
    simulation_config = simulation_configuration(cfg)
    if path.exists() and manifest_path.is_file() and not overwrite:
        cached = load_4dstem(path)
        cached_config = cached.metadata.get(
            "simulation_configuration",
            cached.metadata.get("configuration", {}),
        )
        cache_is_compatible = (
            not simulation_configuration_differences(cached_config, cfg)
            and cached.metadata.get("view") == view.metadata
            and cached.metadata.get("model_fingerprint_sha256") == model_hash
        )
        if cache_is_compatible:
            return cached
    fingerprint = _fingerprint(
        {
            "schema": SIMULATION_SCHEMA,
            "model_fingerprint": model_hash,
            "view": view.metadata,
            "simulation_configuration": simulation_config,
        }
    )
    if _cache_matches(
        path,
        manifest_path,
        fingerprint,
        overwrite=overwrite,
        schema=SIMULATION_SCHEMA,
    ):
        return load_4dstem(path)

    seed_namespace = _fingerprint(
        {"model_fingerprint": model_hash, "view": view.metadata}
    )
    seeds = _derived_seeds(cfg.random_seed, namespace=seed_namespace)
    potential = _make_potential(
        view,
        cfg,
        frozen_phonons=True,
        frozen_seed=seeds["frozen_phonons"],
    )
    (
        probe,
        scan,
        detector,
        nominal_scan,
        nominal_positions,
        actual_positions,
        scan_details,
    ) = _make_probe_scan_detector(potential, cfg, scan_seed=seeds["scan_positions"])
    patterns = probe.scan(
        potential,
        scan=scan,
        detectors=detector,
        max_batch=cfg.max_batch,
        lazy=True,
    )
    patterns = patterns.reduce_ensemble()
    if cfg.scan_position_error_std_angstrom > 0.0:
        patterns = _restore_nominal_scan_grid(
            patterns,
            nominal_scan,
            nominal_positions.shape[:2],
        )
    if cfg.partial_coherence_source_sigma_angstrom > 0.0:
        patterns = patterns.gaussian_source_size(
            cfg.partial_coherence_source_sigma_angstrom
        )
    if cfg.dose_electrons_per_angstrom2 is not None:
        patterns = patterns.poisson_noise(
            dose_per_area=cfg.dose_electrons_per_angstrom2,
            samples=1,
            seed=seeds["poisson_counts"],
        )
    patterns, detector_details, detector_state = _apply_detector_response(
        patterns,
        cfg,
        detector_seed=seeds["detector_response"],
    )

    simulation_state_path = path.parent / f"{path.stem}_simulation_state.npz"
    metadata = {
        "schema": SIMULATION_SCHEMA,
        "fingerprint_sha256": fingerprint,
        "model_fingerprint_sha256": model_hash,
        "view": view.metadata,
        "configuration": cfg.to_dict(),
        "simulation_configuration": simulation_config,
        "condition": {
            "name": cfg.condition_name,
            "description": VALIDATION_CONDITION_DESCRIPTIONS.get(
                cfg.condition_name, "User-defined simulation condition."
            ),
            "effects": _condition_effects(cfg),
        },
        "random_seed_master": cfg.random_seed,
        "random_seed_namespace_sha256": seed_namespace,
        "random_seeds": seeds,
        "diffraction_shape": list(patterns.shape),
        "diffraction_dtype": str(patterns.dtype),
        "diffraction_fftshift": bool(patterns.fftshift),
        "axis_order": [axis.label for axis in patterns.axes_metadata],
        "angular_sampling_mrad": [float(value) for value in patterns.angular_sampling],
        "scan_positions_shape": list(actual_positions.shape),
        "nominal_scan_shape": list(nominal_positions.shape[:2]),
        "scan_start_angstrom": nominal_positions.reshape(-1, 2).min(axis=0).tolist(),
        "scan_end_angstrom": nominal_positions.reshape(-1, 2).max(axis=0).tolist(),
        "scan_position_model": scan_details,
        "detector_response_model": detector_details,
        "stored_frozen_phonon_axis": False,
        "frozen_phonon_reduction": "incoherent intensity mean",
        "poisson_noise_applied": cfg.dose_electrons_per_angstrom2 is not None,
        "partial_coherence_model": (
            "abTEM Gaussian source-size mixing across scan axes"
            if cfg.partial_coherence_source_sigma_angstrom > 0.0
            else None
        ),
        "ground_truth_positions": (
            "Reference post-vacancy atom coordinates are the validation truth; "
            "frozen-phonon displacements are nuisance realizations."
        ),
        "files": {
            "diffraction_zarr": str(path.resolve()),
            "simulation_state_npz": str(simulation_state_path.resolve()),
            "scan_positions_npz": str(simulation_state_path.resolve()),
        },
        "instrument_parameter_units": {
            "probe_aberration_lengths": "angstrom",
            "probe_aberration_angles": "radian",
            "beam_tilt": "mrad",
            "source_size_sigma": "angstrom in scan plane",
            "scan_position_error": "angstrom per Cartesian scan component",
            "detector_response": "electron counts after Poisson dose sampling",
        },
        "software_versions": _software_versions(),
    }
    patterns.to_zarr(
        str(path.resolve()),
        compute=True,
        overwrite=overwrite,
        progress_bar=progress,
    )
    state_arrays: dict[str, np.ndarray] = {
        "nominal_positions_angstrom": nominal_positions,
        "actual_positions_angstrom": actual_positions,
        **detector_state,
    }
    np.savez_compressed(simulation_state_path, **state_arrays)
    _write_json(manifest_path, metadata)
    return load_4dstem(path)


def load_4dstem(path: str | Path) -> SimulationResult:
    """Open a cached abTEM DiffractionPatterns Zarr dataset lazily."""
    source = Path(path).expanduser()
    manifest = source.with_suffix(".json")
    if not source.exists() or not manifest.is_file():
        raise FileNotFoundError(f"4D-STEM cache is incomplete: {source}")
    from abtem.measurements import DiffractionPatterns

    patterns = DiffractionPatterns.from_zarr(str(source.resolve()))
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    files = metadata.get("files", {})
    scan_path_value = files.get("simulation_state_npz") or files.get(
        "scan_positions_npz"
    )
    scan_positions_path = Path(scan_path_value) if scan_path_value else None
    if scan_positions_path is not None and not scan_positions_path.is_file():
        raise FileNotFoundError(
            f"4D-STEM simulation-state sidecar is missing: {scan_positions_path}"
        )
    return SimulationResult(
        diffraction_patterns=patterns,
        metadata=metadata,
        output_path=source.resolve(),
        manifest_path=manifest.resolve(),
        scan_positions_path=(
            scan_positions_path.resolve() if scan_positions_path is not None else None
        ),
        simulation_state_path=(
            scan_positions_path.resolve() if scan_positions_path is not None else None
        ),
    )


def analyze_4dstem_quality(
    simulation: SimulationResult | str | Path,
    output_path: str | Path | None = None,
    *,
    histogram_bins: int = 80,
    overwrite: bool = False,
    progress: bool = True,
) -> SimulationQCResult:
    """Calculate diffraction, scan, count, finiteness, and storage QC products.

    The full 4D array remains lazy. Only the mean diffraction pattern, one
    scan-integrated image, scalar checks, and a histogram are materialized.
    No scan positions or detector pixels are capped or subsampled.
    """
    sim = load_4dstem(simulation) if isinstance(simulation, (str, Path)) else simulation
    if not isinstance(histogram_bins, int) or histogram_bins < 2:
        raise ValueError("histogram_bins must be an integer of at least 2.")

    path: Path | None = None
    manifest_path: Path | None = None
    fingerprint = _fingerprint(
        {
            "schema": QC_SCHEMA,
            "simulation_fingerprint": sim.metadata["fingerprint_sha256"],
            "histogram_bins": histogram_bins,
        }
    )
    if output_path is not None:
        path = _require_suffix(output_path, ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = path.with_suffix(".json")
        if _cache_matches(
            path,
            manifest_path,
            fingerprint,
            overwrite=overwrite,
            schema=QC_SCHEMA,
        ):
            return load_4dstem_quality(path)

    array = sim.diffraction_patterns.array
    if array.ndim < 4:
        raise ValueError(
            "QC expects two scan axes followed by two detector axes; received "
            f"shape {array.shape}."
        )
    scan_axes = tuple(range(array.ndim - 2))
    detector_axes = (array.ndim - 2, array.ndim - 1)

    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar

    lazy_array = array if isinstance(array, da.Array) else da.from_array(array)
    calculations = (
        lazy_array.mean(axis=scan_axes),
        lazy_array.sum(axis=detector_axes),
        da.isfinite(lazy_array).sum(dtype=np.int64),
        da.isnan(lazy_array).sum(dtype=np.int64),
        da.isinf(lazy_array).sum(dtype=np.int64),
        (lazy_array == 0).sum(dtype=np.int64),
        da.nanmin(lazy_array),
        da.nanmax(lazy_array),
        da.nanmean(lazy_array),
    )
    context = ProgressBar() if progress else nullcontext()
    with context:
        (
            mean_pattern,
            scan_integrated,
            finite_count,
            nan_count,
            inf_count,
            zero_count,
            minimum,
            maximum,
            mean,
        ) = dask.compute(*calculations)

    mean_pattern = np.asarray(mean_pattern, dtype=np.float32)
    scan_integrated = np.asarray(scan_integrated, dtype=np.float64)
    finite_scan_values = scan_integrated[np.isfinite(scan_integrated)]
    if finite_scan_values.size:
        histogram_counts, histogram_edges = np.histogram(
            finite_scan_values,
            bins=histogram_bins,
        )
    else:
        histogram_counts = np.zeros(histogram_bins, dtype=np.int64)
        histogram_edges = np.linspace(0.0, 1.0, histogram_bins + 1)

    total_values = int(np.prod(array.shape, dtype=np.int64))
    stored_bytes = _path_size_bytes(sim.output_path)
    units = str(sim.diffraction_patterns.metadata.get("units", "arbitrary intensity"))
    metadata = {
        "schema": QC_SCHEMA,
        "fingerprint_sha256": fingerprint,
        "simulation_fingerprint_sha256": sim.metadata["fingerprint_sha256"],
        "condition_name": sim.metadata.get("condition", {}).get("name"),
        "diffraction_shape": list(array.shape),
        "scan_shape": list(array.shape[:-2]),
        "detector_shape": list(array.shape[-2:]),
        "diffraction_dtype": str(array.dtype),
        "measurement_units": units,
        "mean_diffraction_pattern_shape": list(mean_pattern.shape),
        "scan_integrated_image_shape": list(scan_integrated.shape),
        "histogram_bins": histogram_bins,
        "histogram_quantity": (
            "scan-integrated detector counts per probe position"
            if sim.metadata.get("poisson_noise_applied")
            else "scan-integrated detector intensity per probe position"
        ),
        "scan_start_angstrom": sim.metadata.get("scan_start_angstrom"),
        "scan_end_angstrom": sim.metadata.get("scan_end_angstrom"),
        "scan_step_angstrom": sim.metadata.get("configuration", {}).get(
            "scan_step_angstrom"
        ),
        "angular_sampling_mrad": sim.metadata.get("angular_sampling_mrad"),
        "diffraction_fftshift": sim.metadata.get("diffraction_fftshift", True),
        "total_value_count": total_values,
        "finite_value_count": int(finite_count),
        "nan_value_count": int(nan_count),
        "infinite_value_count": int(inf_count),
        "zero_value_count": int(zero_count),
        "zero_value_fraction": float(int(zero_count) / total_values),
        "zero_diffraction_pattern_count": int(np.count_nonzero(scan_integrated == 0)),
        "minimum": float(minimum),
        "maximum": float(maximum),
        "mean": float(mean),
        "stored_zarr_size_bytes": stored_bytes,
        "stored_zarr_size_gib": _gib(stored_bytes),
        "no_visualization_or_data_cap_applied": True,
        "software_versions": _software_versions(),
    }

    if path is not None and manifest_path is not None:
        np.savez_compressed(
            path,
            mean_diffraction_pattern=mean_pattern,
            scan_integrated_image=scan_integrated,
            histogram_counts=np.asarray(histogram_counts, dtype=np.int64),
            histogram_edges=np.asarray(histogram_edges, dtype=np.float64),
        )
        _write_json(manifest_path, metadata)
        resolved_path: Path | None = path.resolve()
        resolved_manifest: Path | None = manifest_path.resolve()
    else:
        resolved_path = None
        resolved_manifest = None
    return SimulationQCResult(
        mean_diffraction_pattern=mean_pattern,
        scan_integrated_image=scan_integrated,
        histogram_counts=np.asarray(histogram_counts, dtype=np.int64),
        histogram_edges=np.asarray(histogram_edges, dtype=np.float64),
        metadata=metadata,
        output_path=resolved_path,
        manifest_path=resolved_manifest,
    )


def load_4dstem_quality(path: str | Path) -> SimulationQCResult:
    """Load a cached 4D-STEM quality-control product."""
    source = _require_suffix(path, ".npz")
    manifest = source.with_suffix(".json")
    if not source.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"4D-STEM QC cache is incomplete: {source}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as data:
        mean_pattern = np.asarray(data["mean_diffraction_pattern"], dtype=np.float32)
        scan_integrated = np.asarray(data["scan_integrated_image"], dtype=np.float64)
        counts = np.asarray(data["histogram_counts"], dtype=np.int64)
        edges = np.asarray(data["histogram_edges"], dtype=np.float64)
    return SimulationQCResult(
        mean_diffraction_pattern=mean_pattern,
        scan_integrated_image=scan_integrated,
        histogram_counts=counts,
        histogram_edges=edges,
        metadata=metadata,
        output_path=source.resolve(),
        manifest_path=manifest.resolve(),
    )


def print_4dstem_quality_summary(
    result: SimulationQCResult,
    *,
    label: str = "dataset",
) -> None:
    """Print the main QC checks and actual stored size."""
    info = result.metadata
    print(f"4D-STEM QC: {label}")
    print(
        f"  shape={tuple(info['diffraction_shape'])}, detector={tuple(info['detector_shape'])}, "
        f"dtype={info['diffraction_dtype']}"
    )
    print(
        f"  finite={info['finite_value_count']:,}/{info['total_value_count']:,}, "
        f"NaN={info['nan_value_count']:,}, inf={info['infinite_value_count']:,}, "
        f"zeros={info['zero_value_count']:,}"
    )
    print(
        f"  range=[{info['minimum']:.6g}, {info['maximum']:.6g}], "
        f"mean={info['mean']:.6g}, stored={info['stored_zarr_size_gib']:.3f} GiB"
    )


def plot_4dstem_quality(
    result: SimulationQCResult,
    *,
    cmap: str = "magma",
    figsize: tuple[float, float] = (15.0, 4.5),
) -> tuple[Any, Any]:
    """Plot scan-integrated signal, mean diffraction, and its count distribution."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    scan_start = result.metadata.get("scan_start_angstrom")
    scan_end = result.metadata.get("scan_end_angstrom")
    scan_step = result.metadata.get("scan_step_angstrom")
    scan_extent = None
    if scan_start is not None and scan_end is not None and scan_step is not None:
        scan_extent = [
            float(scan_start[0]) - float(scan_step) / 2.0,
            float(scan_end[0]) + float(scan_step) / 2.0,
            float(scan_start[1]) - float(scan_step) / 2.0,
            float(scan_end[1]) + float(scan_step) / 2.0,
        ]
    scan_image = axes[0].imshow(
        result.scan_integrated_image.T,
        origin="lower",
        cmap=cmap,
        extent=scan_extent,
    )
    axes[0].set(title="Scan-integrated image", xlabel="scan x (A)", ylabel="scan y (A)")
    fig.colorbar(scan_image, ax=axes[0], shrink=0.78)

    angular_sampling = result.metadata.get("angular_sampling_mrad")
    diffraction_extent = None
    if angular_sampling is not None:
        nx, ny = result.mean_diffraction_pattern.shape
        sx, sy = (float(value) for value in angular_sampling)
        if result.metadata.get("diffraction_fftshift", True):
            x_centers = (np.arange(nx) - nx // 2) * sx
            y_centers = (np.arange(ny) - ny // 2) * sy
        else:
            x_centers = np.arange(nx) * sx
            y_centers = np.arange(ny) * sy
        diffraction_extent = [
            float(x_centers[0] - sx / 2.0),
            float(x_centers[-1] + sx / 2.0),
            float(y_centers[0] - sy / 2.0),
            float(y_centers[-1] + sy / 2.0),
        ]
    diffraction_image = axes[1].imshow(
        np.log1p(np.maximum(result.mean_diffraction_pattern.T, 0.0)),
        origin="lower",
        cmap=cmap,
        extent=diffraction_extent,
    )
    axes[1].set(
        title="Mean diffraction pattern (log1p)",
        xlabel="detector qx (mrad)",
        ylabel="detector qy (mrad)",
    )
    fig.colorbar(diffraction_image, ax=axes[1], shrink=0.78)

    axes[2].stairs(result.histogram_counts, result.histogram_edges, fill=True)
    axes[2].set(
        title="Scan-position signal distribution",
        xlabel=result.metadata["histogram_quantity"],
        ylabel="positions",
    )
    return fig, axes


def reconstruct_multislice_ptychography(
    simulation: SimulationResult | str | Path,
    view: OrientedAtomsResult,
    config: PtychographyConfig,
    output_path: str | Path,
    *,
    probe_initialization: str = "abtem_default",
    object_initialization: str = "uniform",
    object_initialization_source: (
        ReconstructionResult | ComplexObjectInitializationResult | str | Path | None
    ) = None,
    object_step_size: float = 1.0,
    probe_step_size: float = 1.0,
    step_size_damping_rate: float = 0.995,
    probe_correction_start_iteration: int | None = 0,
    position_correction: bool = False,
    object_constraint: str = "none",
    phase_sign: int = 1,
    vacuum_guard_slices: tuple[int, int] = (0, 0),
    reconstruction_grid: str = "simulation_native",
    object_antialiasing: bool = True,
    final_slice_propagation: bool = True,
    capture_iterations: tuple[int, ...] | list[int] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> ReconstructionResult:
    """Run abTEM MS-PIE and export complex objects plus CMEP-ready phase slices.

    abTEM's current operator is in-memory. Loading a large 8 nm 4D dataset may
    therefore require substantially more RAM/VRAM than its compressed Zarr size.
    ``probe_initialization='simulation_exact'`` rebuilds the coherent incident
    probe encoded by the simulation manifest on the reconstruction grid before
    the first PIE update. ``object_initialization='split_projection'`` requires
    a compatible one-slice reconstruction and distributes its complex
    transmission evenly across the requested target slices.
    ``object_initialization='provided_complex'`` accepts physically described
    complex slices and linearly resamples only their in-plane coordinates onto
    the reconstruction grid. ``capture_iterations`` stores selected states from
    one run; iteration 0 is the initialized object before any PIE update. Step
    sizes and damping are forwarded to abTEM's iterative operator. Optional
    object constraints are applied after initialization and after every complete
    MS-PIE iteration. ``reconstruction_grid='simulation_native'`` center-pads
    cropped diffraction patterns so the object retains the simulation potential's
    real-space sampling. ``object_antialiasing=True`` applies abTEM's transmission
    antialias aperture after initialization and each iteration. ``pure_phase``
    forces unit amplitude before that bandlimit, while
    ``pure_phase_positive`` additionally retains only the phase polarity selected
    by ``phase_sign``. ``vacuum_guard_slices=(entrance, exit)`` freezes terminal
    depth slices to vacuum transmission. ``final_slice_propagation=True`` matches
    conventional abTEM multislice at the detector plane and applies its adjoint
    before PIE updates. Probe correction can start after a whole number of scan
    iterations; ``None`` disables it. Position correction is disabled by default
    for exact simulated scan coordinates and is not currently combined with
    iteration-level object projections.
    """
    cfg = config.validated()
    sim = (
        load_4dstem(simulation)
        if isinstance(simulation, (str, Path))
        else simulation
    )
    probe_mode = _normalize_probe_initialization(probe_initialization)
    probe_descriptor = _reconstruction_probe_descriptor(sim, probe_mode)
    if probe_mode == "simulation_exact" and not np.isclose(
        cfg.energy_ev,
        probe_descriptor["energy_ev"],
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "simulation_exact probe initialization requires the reconstruction "
            "energy_ev to match the simulated energy_ev."
        )
    probe_descriptor_fingerprint = _fingerprint(probe_descriptor)
    object_mode = _normalize_object_initialization(object_initialization)
    object_source, object_descriptor = _reconstruction_object_descriptor(
        sim,
        cfg,
        object_mode,
        object_initialization_source,
    )
    object_descriptor_fingerprint = _fingerprint(object_descriptor)
    object_step = _positive_finite(object_step_size, "object_step_size")
    probe_step = _positive_finite(probe_step_size, "probe_step_size")
    damping_rate = _positive_finite(
        step_size_damping_rate, "step_size_damping_rate"
    )
    if damping_rate > 1.0:
        raise ValueError("step_size_damping_rate must be less than or equal to 1.")
    if probe_correction_start_iteration is not None:
        if isinstance(probe_correction_start_iteration, (bool, np.bool_)):
            raise TypeError(
                "probe_correction_start_iteration must be an integer or None."
            )
        if int(probe_correction_start_iteration) != probe_correction_start_iteration:
            raise ValueError(
                "probe_correction_start_iteration must be an integer or None."
            )
        if probe_correction_start_iteration < 0:
            raise ValueError(
                "probe_correction_start_iteration must be non-negative or None."
            )
        probe_start_iteration: int | None = int(
            probe_correction_start_iteration
        )
    else:
        probe_start_iteration = None
    if not isinstance(position_correction, (bool, np.bool_)):
        raise TypeError("position_correction must be Boolean.")
    grid_mode = _normalize_reconstruction_grid(reconstruction_grid)
    if not isinstance(object_antialiasing, (bool, np.bool_)):
        raise TypeError("object_antialiasing must be Boolean.")
    if not isinstance(final_slice_propagation, (bool, np.bool_)):
        raise TypeError("final_slice_propagation must be Boolean.")
    use_object_antialiasing = bool(object_antialiasing)
    use_final_slice_propagation = bool(final_slice_propagation)
    constraint_mode = _normalize_object_constraint(object_constraint)
    normalized_phase_sign = _normalize_phase_sign(phase_sign)
    normalized_vacuum_guards = _normalize_vacuum_guard_slices(
        vacuum_guard_slices
    )
    projections_active = (
        constraint_mode != "none"
        or normalized_vacuum_guards != (0, 0)
        or use_object_antialiasing
    )
    if projections_active and position_correction:
        raise ValueError(
            "position_correction is not supported with iteration-level object "
            "projections. Disable position correction for projected runs."
        )
    checkpoint_iterations = _normalize_capture_iterations(
        capture_iterations, cfg.reconstruction_iterations
    )

    detector_shape = tuple(
        int(value) for value in sim.diffraction_patterns.shape[-2:]
    )
    grid_metadata = _reconstruction_grid_metadata(
        view,
        cfg,
        detector_shape=detector_shape,
        mode=grid_mode,
        simulation_metadata=sim.metadata,
    )
    region_of_interest_shape = tuple(
        int(value) for value in grid_metadata["region_of_interest_gpts_xy"]
    )

    base_reconstruction_controls = {
        "probe_initialization": probe_mode,
        "probe_descriptor_fingerprint_sha256": probe_descriptor_fingerprint,
        "object_step_size": object_step,
        "probe_step_size": probe_step,
        "step_size_damping_rate": damping_rate,
        "probe_correction_start_iteration": probe_start_iteration,
        "position_correction": bool(position_correction),
        "reconstruction_grid": grid_mode,
        "region_of_interest_gpts_xy": list(region_of_interest_shape),
        "object_antialiasing": use_object_antialiasing,
        "final_slice_propagation": use_final_slice_propagation,
    }
    reconstruction_controls = {
        **base_reconstruction_controls,
        "object_initialization": object_mode,
        "object_descriptor_fingerprint_sha256": object_descriptor_fingerprint,
    }
    if projections_active:
        reconstruction_controls["object_constraint"] = constraint_mode
        reconstruction_controls["phase_sign"] = normalized_phase_sign
        reconstruction_controls["vacuum_guard_slices"] = list(
            normalized_vacuum_guards
        )
        reconstruction_controls["object_projection_application"] = (
            "after initialization and after every completed MS-PIE iteration"
        )
    if checkpoint_iterations:
        reconstruction_controls["capture_iterations"] = list(
            checkpoint_iterations
        )
    fingerprint_controls = reconstruction_controls
    path = _require_suffix(output_path, ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".json")
    fingerprint = _fingerprint(
        {
            "schema": RECONSTRUCTION_SCHEMA,
            "simulation_fingerprint": sim.metadata["fingerprint_sha256"],
            "view": view.metadata,
            "configuration": cfg.to_dict(),
            "reconstruction_controls": fingerprint_controls,
        }
    )
    if _cache_matches(
        path,
        manifest_path,
        fingerprint,
        overwrite=overwrite,
        schema=RECONSTRUCTION_SCHEMA,
    ):
        cached = load_reconstruction(path)
        if checkpoint_iterations:
            load_reconstruction_history(_reconstruction_history_path(path))
        return cached

    patterns = sim.diffraction_patterns.compute(progress_bar=verbose)
    scan_position_count = int(np.prod(patterns.shape[:-2], dtype=np.int64))
    total_update_steps = int(cfg.reconstruction_iterations * scan_position_count)
    if probe_start_iteration is None:
        pre_probe_correction_update_steps = total_update_steps + 1
    elif probe_start_iteration == 0:
        pre_probe_correction_update_steps = None
    else:
        pre_probe_correction_update_steps = int(
            probe_start_iteration * scan_position_count
        )
    pre_position_correction_update_steps = (
        0 if bool(position_correction) else None
    )
    cell_z = float(view.atoms.cell.lengths()[2])
    num_slices = max(
        1, int(round(cell_z / cfg.reconstruction_slice_thickness_angstrom))
    )
    slice_thickness = cell_z / num_slices
    if sum(normalized_vacuum_guards) >= num_slices:
        raise ValueError(
            "vacuum_guard_slices must leave at least one reconstructable depth slice; "
            f"received {normalized_vacuum_guards} for {num_slices} slices."
        )

    compatibility_shim_applied = _ensure_abtem_reconstruction_compatibility()
    from abtem.reconstruct import MultislicePtychographicOperator

    operator = MultislicePtychographicOperator(
        patterns,
        energy=cfg.energy_ev,
        num_slices=num_slices,
        slice_thicknesses=slice_thickness,
        region_of_interest_shape=region_of_interest_shape,
        semiangle_cutoff=cfg.semiangle_mrad,
        preprocess=True,
        device=str(cfg.device).strip().lower(),
    )
    if grid_mode == "simulation_native" and not np.allclose(
        np.asarray(operator.sampling, dtype=np.float64),
        np.asarray(
            grid_metadata["requested_effective_sampling_xy_angstrom"],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "The padded MS-PIE grid did not recover the requested "
            "reconstruction "
            f"sampling: {operator.sampling} versus "
            f"{grid_metadata['requested_effective_sampling_xy_angstrom']}."
        )
    if verbose:
        print(
            "MS-PIE grid: "
            f"detector={detector_shape}, ROI={region_of_interest_shape}, "
            f"sampling={tuple(float(value) for value in operator.sampling)} A"
        )
    probe_initialization_metadata = _initialize_reconstruction_probe(
        operator,
        probe_descriptor,
        descriptor_fingerprint=probe_descriptor_fingerprint,
    )
    object_initialization_metadata = _initialize_reconstruction_object(
        operator,
        sim,
        object_descriptor,
        object_source,
        target_z_edges_angstrom=np.linspace(
            0.0, cell_z, num_slices + 1, dtype=np.float64
        ),
        descriptor_fingerprint=object_descriptor_fingerprint,
        antialias=use_object_antialiasing,
    )
    constraint_metadata = {
        "active": projections_active,
        "mode": constraint_mode,
        "phase_sign": normalized_phase_sign,
        "object_antialiasing": use_object_antialiasing,
        "vacuum_guard_slices": list(normalized_vacuum_guards),
        "application": (
            "after initialization and after every completed MS-PIE iteration"
            if projections_active
            else None
        ),
        "guard_value": "1+0j vacuum transmission",
        "phase_representation": "wrapped angle in [-pi, pi]",
    }
    if projections_active:
        operator._objects = _apply_object_constraint_array(
            operator._objects,
            mode=constraint_mode,
            phase_sign=normalized_phase_sign,
            vacuum_guard_slices=normalized_vacuum_guards,
            antialias_sampling=(
                tuple(float(value) for value in operator.sampling)
                if use_object_antialiasing
                else None
            ),
        )

    initial_object_array: np.ndarray | None = None
    if checkpoint_iterations:
        from abtem.core.backend import asnumpy

        initial_object_array = np.asarray(
            asnumpy(operator._objects), dtype=np.complex64
        ).copy()
    constrained_checkpoints: dict[int, np.ndarray] = {}
    constrained_checkpoint_errors: dict[int, float] = {}
    if projections_active:
        from abtem.core.backend import asnumpy

        if initial_object_array is not None:
            constrained_checkpoints[0] = initial_object_array
            constrained_checkpoint_errors[0] = np.nan
        if verbose:
            print(
                "Projected ptychographic reconstruction will perform "
                f"{cfg.reconstruction_iterations} MS-PIE iterations."
            )
        iteration_width = len(str(cfg.reconstruction_iterations))
        error = np.nan
        for iteration_index in range(cfg.reconstruction_iterations):
            if (
                probe_start_iteration is None
                or iteration_index < probe_start_iteration
            ):
                epoch_pre_probe_correction_steps = scan_position_count + 1
            else:
                epoch_pre_probe_correction_steps = None
            epoch_functions_queue = _multislice_functions_queue(
                operator,
                max_iterations=1,
                pre_probe_correction_update_steps=(
                    epoch_pre_probe_correction_steps
                ),
                pre_position_correction_update_steps=None,
                final_slice_propagation=use_final_slice_propagation,
            )
            _, _, _, epoch_error = operator.reconstruct(
                max_iterations=1,
                return_iterations=False,
                random_seed=(cfg.random_seed if iteration_index == 0 else None),
                verbose=False,
                functions_queue=epoch_functions_queue,
                parameters={
                    "object_step_size": (
                        object_step * damping_rate**iteration_index
                    ),
                    "probe_step_size": probe_step * damping_rate**iteration_index,
                    "step_size_damping_rate": damping_rate,
                    "pre_probe_correction_update_steps": (
                        epoch_pre_probe_correction_steps
                    ),
                    "pre_position_correction_update_steps": None,
                },
            )
            error = float(epoch_error)
            operator._objects = _apply_object_constraint_array(
                operator._objects,
                mode=constraint_mode,
                phase_sign=normalized_phase_sign,
                vacuum_guard_slices=normalized_vacuum_guards,
                antialias_sampling=(
                    tuple(float(value) for value in operator.sampling)
                    if use_object_antialiasing
                    else None
                ),
            )
            completed_iteration = iteration_index + 1
            if completed_iteration in checkpoint_iterations:
                constrained_checkpoints[completed_iteration] = np.asarray(
                    asnumpy(operator._objects), dtype=np.complex64
                ).copy()
                constrained_checkpoint_errors[completed_iteration] = error
            if verbose:
                print(
                    f"----Projected iteration {iteration_index:<{iteration_width}}, "
                    f"SSE = {error:.3e}"
                )

        object_array = np.asarray(
            asnumpy(operator._objects), dtype=np.complex64
        )
        probe_array = np.asarray(asnumpy(operator._probes), dtype=np.complex64)
        positions_array = np.asarray(
            asnumpy(operator._positions_px), dtype=np.float64
        ) * np.asarray(operator.sampling, dtype=np.float64)
        sampling = tuple(float(value) for value in operator.sampling)
    else:
        functions_queue = _multislice_functions_queue(
            operator,
            max_iterations=cfg.reconstruction_iterations,
            pre_probe_correction_update_steps=pre_probe_correction_update_steps,
            pre_position_correction_update_steps=pre_position_correction_update_steps,
            final_slice_propagation=use_final_slice_propagation,
        )
        reconstruction_output = operator.reconstruct(
            max_iterations=cfg.reconstruction_iterations,
            return_iterations=bool(checkpoint_iterations),
            random_seed=cfg.random_seed,
            verbose=verbose,
            functions_queue=functions_queue,
            parameters={
                "object_step_size": object_step,
                "probe_step_size": probe_step,
                "step_size_damping_rate": damping_rate,
                "pre_probe_correction_update_steps": (
                    pre_probe_correction_update_steps
                ),
                "pre_position_correction_update_steps": (
                    pre_position_correction_update_steps
                ),
            },
        )
        if checkpoint_iterations:
            (
                object_iterations,
                probe_iterations,
                position_iterations,
                error_iterations,
            ) = reconstruction_output
            objects = object_iterations[-1]
            probes = probe_iterations[-1]
            positions = position_iterations[-1]
            error = error_iterations[-1]
        else:
            objects, probes, positions, error = reconstruction_output
        object_array = np.asarray(objects.array, dtype=np.complex64)
        probe_array = np.asarray(probes.array, dtype=np.complex64)
        positions_array = np.asarray(positions, dtype=np.float64)
        sampling = tuple(float(value) for value in objects.sampling)

    phase_stack = np.transpose(np.angle(object_array), (0, 2, 1)).astype(np.float32)
    padding_px = np.asarray(
        operator._experimental_parameters["object_px_padding"], dtype=np.float64
    )
    scan_start = np.asarray(sim.metadata["scan_start_angstrom"], dtype=np.float64)
    object_origin_xy = scan_start - padding_px * np.asarray(sampling)
    x = object_origin_xy[0] + np.arange(object_array.shape[1]) * sampling[0]
    y = object_origin_xy[1] + np.arange(object_array.shape[2]) * sampling[1]
    z = (np.arange(num_slices, dtype=np.float64) + 0.5) * slice_thickness
    history_metadata: dict[str, Any] | None = None
    if checkpoint_iterations:
        if initial_object_array is None:
            raise RuntimeError("The requested iteration-0 object was not captured.")
        selected_objects = []
        selected_errors = []
        for iteration in checkpoint_iterations:
            if projections_active:
                selected_objects.append(constrained_checkpoints[iteration])
                selected_errors.append(constrained_checkpoint_errors[iteration])
            elif iteration == 0:
                selected_objects.append(initial_object_array)
                selected_errors.append(np.nan)
            else:
                selected_objects.append(
                    np.asarray(
                        object_iterations[iteration - 1].array,
                        dtype=np.complex64,
                    )
                )
                selected_errors.append(float(error_iterations[iteration - 1]))
        history_path = _reconstruction_history_path(path)
        history_manifest_path = history_path.with_suffix(".json")
        history_arrays = np.stack(selected_objects, axis=0).astype(np.complex64)
        history_fingerprint = _fingerprint(
            {
                "schema": RECONSTRUCTION_HISTORY_SCHEMA,
                "reconstruction_fingerprint_sha256": fingerprint,
                "iterations": list(checkpoint_iterations),
                "objects_sha256": _sha256_array(history_arrays),
            }
        )
        history_metadata = {
            "schema": RECONSTRUCTION_HISTORY_SCHEMA,
            "fingerprint_sha256": history_fingerprint,
            "reconstruction_fingerprint_sha256": fingerprint,
            "simulation_fingerprint_sha256": sim.metadata["fingerprint_sha256"],
            "view": view.metadata,
            "configuration": cfg.to_dict(),
            "object_initialization": object_initialization_metadata,
            "object_constraint": constraint_metadata,
            "iterations": list(checkpoint_iterations),
            "iteration_zero_definition": (
                "Initialized object after coordinate resampling and optional "
                "constraint projection, before any PIE update."
            ),
            "checkpoint_state_definition": (
                "Object after the completed iteration and constraint projection."
                if projections_active
                else "Object after the completed unconstrained iteration."
            ),
            "checkpoint_error_definition": (
                "SSE measured immediately before the post-iteration constraint."
                if projections_active
                else "SSE from the completed unconstrained iteration."
            ),
            "error_at_iteration_zero": None,
            "objects_shape_iteration_slice_xy": list(history_arrays.shape),
            "object_sampling_xy_angstrom": list(sampling),
            "object_origin_xy_angstrom": object_origin_xy.tolist(),
            "slice_thickness_angstrom": slice_thickness,
            "coordinate_unit": "angstrom",
            "complex_array_dtype": str(history_arrays.dtype),
            "complex_array_sha256": _sha256_array(history_arrays),
            "software_versions": _software_versions(),
        }
        np.savez_compressed(
            history_path,
            iterations=np.asarray(checkpoint_iterations, dtype=np.int64),
            errors=np.asarray(selected_errors, dtype=np.float64),
            objects_complex_iteration_slice_xy=history_arrays,
            x_angstrom=x,
            y_angstrom=y,
            z_angstrom=z,
        )
        _write_json(history_manifest_path, history_metadata)
    metadata = {
        "schema": RECONSTRUCTION_SCHEMA,
        "fingerprint_sha256": fingerprint,
        "simulation_fingerprint_sha256": sim.metadata["fingerprint_sha256"],
        "view": view.metadata,
        "configuration": cfg.to_dict(),
        "reconstruction_grid": {
            **grid_metadata,
            "effective_sampling_xy_angstrom": list(sampling),
            "padded_detector_pixels_xy": [
                int(region_of_interest_shape[index] - detector_shape[index])
                for index in range(2)
            ],
        },
        "probe_initialization": probe_initialization_metadata,
        "object_initialization": object_initialization_metadata,
        "object_constraint": constraint_metadata,
        "reconstruction_controls": {
            **reconstruction_controls,
            "scan_position_count": scan_position_count,
            "total_update_steps": total_update_steps,
            "abtem_pre_probe_correction_update_steps": (
                pre_probe_correction_update_steps
            ),
            "abtem_pre_position_correction_update_steps": (
                pre_position_correction_update_steps
            ),
        },
        "objects_shape_slice_xy": list(object_array.shape),
        "phase_export_shape_slice_row_col": list(phase_stack.shape),
        "phase_export_axis_order": ["slice_z_view", "row_y_view", "col_x_view"],
        "phase_orientation_string_within_view_frame": "yx",
        "phase_value": "wrapped phase angle of reconstructed complex transmission",
        "coordinate_unit": "angstrom",
        "object_sampling_xy_angstrom": list(sampling),
        "object_origin_xy_angstrom": object_origin_xy.tolist(),
        "slice_thickness_angstrom": slice_thickness,
        "coordinate_ranges_angstrom": {"x": _range(x), "y": _range(y), "z": _range(z)},
        "reconstruction_error": float(error),
        "abtem_fresnel_compatibility_shim_applied": compatibility_shim_applied,
        "iteration_history": (
            None
            if history_metadata is None
            else {
                "output_path": str(_reconstruction_history_path(path).resolve()),
                "manifest_path": str(
                    _reconstruction_history_path(path).with_suffix(".json").resolve()
                ),
                "fingerprint_sha256": history_metadata["fingerprint_sha256"],
                "iterations": history_metadata["iterations"],
            }
        ),
        "software_versions": _software_versions(),
    }
    np.savez_compressed(
        path,
        objects_complex_slice_xy=object_array,
        phase_stack_slice_row_col=phase_stack,
        probes_complex_slice_xy=probe_array,
        positions_angstrom=positions_array,
        x_angstrom=x,
        y_angstrom=y,
        z_angstrom=z,
    )
    _write_json(manifest_path, metadata)
    return ReconstructionResult(
        objects_complex_slice_xy=object_array,
        phase_stack_slice_row_col=phase_stack,
        probes_complex_slice_xy=probe_array,
        positions_angstrom=positions_array,
        error=float(error),
        metadata=metadata,
        output_path=path.resolve(),
        manifest_path=manifest_path.resolve(),
    )


def load_reconstruction(path: str | Path) -> ReconstructionResult:
    """Load a cached multislice reconstruction."""
    source = _require_suffix(path, ".npz")
    manifest = source.with_suffix(".json")
    if not source.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Reconstruction cache is incomplete: {source}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as data:
        objects = np.asarray(data["objects_complex_slice_xy"], dtype=np.complex64)
        phase = np.asarray(data["phase_stack_slice_row_col"], dtype=np.float32)
        probes = np.asarray(data["probes_complex_slice_xy"], dtype=np.complex64)
        positions = np.asarray(data["positions_angstrom"], dtype=np.float64)
    return ReconstructionResult(
        objects_complex_slice_xy=objects,
        phase_stack_slice_row_col=phase,
        probes_complex_slice_xy=probes,
        positions_angstrom=positions,
        error=float(metadata["reconstruction_error"]),
        metadata=metadata,
        output_path=source.resolve(),
        manifest_path=manifest.resolve(),
    )


def load_reconstruction_history(path: str | Path) -> ReconstructionHistoryResult:
    """Load selected iteration states captured during one reconstruction."""
    source = _require_suffix(path, ".npz")
    manifest = source.with_suffix(".json")
    if not source.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Reconstruction history cache is incomplete: {source}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("schema") != RECONSTRUCTION_HISTORY_SCHEMA:
        raise ValueError("Reconstruction history manifest has an unsupported schema.")
    with np.load(source, allow_pickle=False) as data:
        iterations = np.asarray(data["iterations"], dtype=np.int64)
        errors = np.asarray(data["errors"], dtype=np.float64)
        objects = np.asarray(
            data["objects_complex_iteration_slice_xy"], dtype=np.complex64
        )
        x = np.asarray(data["x_angstrom"], dtype=np.float64)
        y = np.asarray(data["y_angstrom"], dtype=np.float64)
        z = np.asarray(data["z_angstrom"], dtype=np.float64)
    if objects.ndim != 4 or objects.shape[0] != len(iterations):
        raise ValueError(
            "Reconstruction history objects must have shape "
            "(iteration,slice,x,y)."
        )
    if errors.shape != iterations.shape:
        raise ValueError("Reconstruction history errors and iterations do not match.")
    if _sha256_array(objects) != metadata.get("complex_array_sha256"):
        raise ValueError("Reconstruction history complex-array hash does not match.")
    return ReconstructionHistoryResult(
        iterations=iterations,
        errors=errors,
        objects_complex_iteration_slice_xy=objects,
        x_angstrom=x,
        y_angstrom=y,
        z_angstrom=z,
        metadata=metadata,
        output_path=source.resolve(),
        manifest_path=manifest.resolve(),
    )


def analyze_oracle_depth_identifiability(
    history: ReconstructionHistoryResult | str | Path,
    output_path: str | Path,
    *,
    support_fraction_of_peak: float = 1e-4,
    near_wrap_fraction_of_pi: float = 0.9,
    overwrite: bool = False,
) -> DepthIdentifiabilityResult:
    """Measure whether oracle-initialized transmission migrates across depth.

    Iteration 0 is treated as the oracle reference after physical resampling onto
    the reconstruction grid. Correlation and scale-adjusted NRMSE use wrapped
    phase inside an oracle-defined support. Complex-transmission departure uses
    the full grid and is normalized by the initial signal relative to vacuum.
    """
    source = (
        load_reconstruction_history(history)
        if isinstance(history, (str, Path))
        else history
    )
    if not isinstance(source, ReconstructionHistoryResult):
        raise TypeError(
            "history must be a ReconstructionHistoryResult or history .npz path."
        )
    if 0 not in source.iterations:
        raise ValueError("Oracle depth diagnostics require a captured iteration 0.")
    initialization = source.metadata.get("object_initialization", {})
    if initialization.get("mode") != "provided_complex" or not initialization.get(
        "simulation_only_truth_initialization", False
    ):
        raise ValueError(
            "Oracle depth diagnostics require a simulation-only provided_complex "
            "initializer."
        )
    support_fraction = _fraction_in_closed_interval(
        support_fraction_of_peak,
        "support_fraction_of_peak",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    near_wrap_fraction = _fraction_in_closed_interval(
        near_wrap_fraction_of_pi,
        "near_wrap_fraction_of_pi",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )

    objects = np.asarray(
        source.objects_complex_iteration_slice_xy, dtype=np.complex128
    )
    truth_index = int(np.flatnonzero(source.iterations == 0)[0])
    truth = objects[truth_index]
    truth_phase = np.angle(truth)
    truth_signal = np.abs(truth - 1.0)
    support_threshold = max(
        float(np.max(truth_signal)) * support_fraction,
        np.finfo(np.float64).eps,
    )
    support = truth_signal >= support_threshold
    if not np.any(support):
        raise ValueError("Oracle initialization has no non-vacuum support.")

    checkpoint_count, slice_count = objects.shape[:2]
    phase_fractions = np.empty((checkpoint_count, slice_count), dtype=np.float64)
    near_wrap_counts = np.empty((checkpoint_count, slice_count), dtype=np.int64)
    oracle_correlations = np.empty(checkpoint_count, dtype=np.float64)
    reversed_correlations = np.empty(checkpoint_count, dtype=np.float64)
    per_slice_correlations = np.empty(
        (checkpoint_count, slice_count), dtype=np.float64
    )
    global_scales = np.empty(checkpoint_count, dtype=np.float64)
    nrmse = np.empty(checkpoint_count, dtype=np.float64)
    departure = np.empty(checkpoint_count, dtype=np.float64)

    reversed_truth_phase = truth_phase[::-1]
    reversed_support = support[::-1]
    truth_norm = max(
        float(np.linalg.norm(truth_phase[support])), np.finfo(np.float64).eps
    )
    transmission_norm = max(
        float(np.linalg.norm(truth - 1.0)), np.finfo(np.float64).eps
    )
    for checkpoint_index, candidate in enumerate(objects):
        phase = np.angle(candidate)
        per_slice_signal = np.sum(np.abs(phase), axis=(1, 2), dtype=np.float64)
        signal_total = max(float(per_slice_signal.sum()), np.finfo(np.float64).eps)
        phase_fractions[checkpoint_index] = per_slice_signal / signal_total
        near_wrap_counts[checkpoint_index] = np.count_nonzero(
            np.abs(phase) >= near_wrap_fraction * np.pi, axis=(1, 2)
        )
        oracle_correlations[checkpoint_index] = _masked_correlation(
            phase, truth_phase, support
        )
        reversed_correlations[checkpoint_index] = _masked_correlation(
            phase, reversed_truth_phase, reversed_support
        )
        for slice_index in range(slice_count):
            per_slice_correlations[checkpoint_index, slice_index] = (
                _masked_correlation(
                    phase[slice_index],
                    truth_phase[slice_index],
                    support[slice_index],
                )
            )
        candidate_values = phase[support]
        truth_values = truth_phase[support]
        denominator = float(np.dot(candidate_values, candidate_values))
        scale = (
            float(np.dot(candidate_values, truth_values) / denominator)
            if denominator > np.finfo(np.float64).eps
            else 0.0
        )
        global_scales[checkpoint_index] = scale
        nrmse[checkpoint_index] = float(
            np.linalg.norm(scale * candidate_values - truth_values) / truth_norm
        )
        departure[checkpoint_index] = float(
            np.linalg.norm(candidate - truth) / transmission_norm
        )

    path = _require_suffix(output_path, ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".json")
    settings = {
        "schema": DEPTH_IDENTIFIABILITY_SCHEMA,
        "history_fingerprint_sha256": source.metadata.get("fingerprint_sha256"),
        "support_fraction_of_peak": support_fraction,
        "near_wrap_fraction_of_pi": near_wrap_fraction,
    }
    fingerprint = _fingerprint(settings)
    if _cache_matches(
        path,
        manifest_path,
        fingerprint,
        overwrite=overwrite,
        schema=DEPTH_IDENTIFIABILITY_SCHEMA,
    ):
        return load_oracle_depth_identifiability(path)
    metadata = {
        **settings,
        "fingerprint_sha256": fingerprint,
        "history_path": str(source.output_path.resolve()),
        "simulation_fingerprint_sha256": source.metadata.get(
            "simulation_fingerprint_sha256"
        ),
        "view": source.metadata.get("view"),
        "iterations": source.iterations.tolist(),
        "slice_count": slice_count,
        "support_threshold_complex_distance_from_vacuum": support_threshold,
        "support_voxel_count": int(np.count_nonzero(support)),
        "metric_definitions": {
            "phase_fraction": "sum(abs(wrapped phase)) per slice / all slices",
            "oracle_correlation": (
                "Pearson correlation of wrapped phase against iteration 0 "
                "inside oracle support"
            ),
            "reversed_oracle_correlation": (
                "Same metric after reversing the oracle slice order"
            ),
            "scale_adjusted_nrmse": (
                "NRMSE after one least-squares scalar rescales all candidate "
                "wrapped-phase voxels inside oracle support"
            ),
            "transmission_departure": (
                "L2(candidate - iteration0) / L2(iteration0 - vacuum) on full grid"
            ),
        },
        "interpretation": (
            "Simulation-only depth-identifiability diagnostic; oracle truth is "
            "used for initialization and evaluation, never for experimental data."
        ),
        "software_versions": _software_versions(),
    }
    np.savez_compressed(
        path,
        iterations=source.iterations,
        errors=source.errors,
        phase_fractions=phase_fractions,
        near_wrap_counts=near_wrap_counts,
        oracle_correlations=oracle_correlations,
        reversed_oracle_correlations=reversed_correlations,
        per_slice_oracle_correlations=per_slice_correlations,
        global_phase_scales=global_scales,
        scale_adjusted_nrmse=nrmse,
        transmission_departure=departure,
    )
    _write_json(manifest_path, metadata)
    return DepthIdentifiabilityResult(
        iterations=source.iterations,
        errors=source.errors,
        phase_fractions=phase_fractions,
        near_wrap_counts=near_wrap_counts,
        oracle_correlations=oracle_correlations,
        reversed_oracle_correlations=reversed_correlations,
        per_slice_oracle_correlations=per_slice_correlations,
        global_phase_scales=global_scales,
        scale_adjusted_nrmse=nrmse,
        transmission_departure=departure,
        metadata=metadata,
        output_path=path.resolve(),
        manifest_path=manifest_path.resolve(),
    )


def load_oracle_depth_identifiability(
    path: str | Path,
) -> DepthIdentifiabilityResult:
    """Load a cached oracle depth-identifiability report."""
    source = _require_suffix(path, ".npz")
    manifest = source.with_suffix(".json")
    if not source.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Depth-identifiability cache is incomplete: {source}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("schema") != DEPTH_IDENTIFIABILITY_SCHEMA:
        raise ValueError("Depth-identifiability manifest has an unsupported schema.")
    with np.load(source, allow_pickle=False) as data:
        values = {key: np.asarray(data[key]) for key in data.files}
    return DepthIdentifiabilityResult(
        iterations=values["iterations"].astype(np.int64),
        errors=values["errors"].astype(np.float64),
        phase_fractions=values["phase_fractions"].astype(np.float64),
        near_wrap_counts=values["near_wrap_counts"].astype(np.int64),
        oracle_correlations=values["oracle_correlations"].astype(np.float64),
        reversed_oracle_correlations=values[
            "reversed_oracle_correlations"
        ].astype(np.float64),
        per_slice_oracle_correlations=values[
            "per_slice_oracle_correlations"
        ].astype(np.float64),
        global_phase_scales=values["global_phase_scales"].astype(np.float64),
        scale_adjusted_nrmse=values["scale_adjusted_nrmse"].astype(np.float64),
        transmission_departure=values["transmission_departure"].astype(np.float64),
        metadata=metadata,
        output_path=source.resolve(),
        manifest_path=manifest.resolve(),
    )


def print_oracle_depth_identifiability_summary(
    result: DepthIdentifiabilityResult,
) -> None:
    """Print checkpoint metrics in a compact table."""
    print("Oracle-initialized depth-identifiability diagnostic")
    print("  iter       SSE   phase fraction by slice   corr  reverse   NRMSE  departure")
    for index, iteration in enumerate(result.iterations):
        error = result.errors[index]
        error_text = "       -" if not np.isfinite(error) else f"{error:8.2e}"
        fractions = ",".join(f"{value:.3f}" for value in result.phase_fractions[index])
        per_slice = ",".join(
            f"{value:.3f}"
            for value in result.per_slice_oracle_correlations[index]
        )
        near_wrap = ",".join(
            str(int(value)) for value in result.near_wrap_counts[index]
        )
        print(
            f"  {int(iteration):4d}  {error_text}   [{fractions}]   "
            f"{result.oracle_correlations[index]:.3f}   "
            f"{result.reversed_oracle_correlations[index]:.3f}   "
            f"{result.scale_adjusted_nrmse[index]:.3f}   "
            f"{result.transmission_departure[index]:.3f}"
        )
        print(
            f"          slice corr=[{per_slice}], near-wrap=[{near_wrap}], "
            f"global phase scale={result.global_phase_scales[index]:.3g}"
        )
    print("  report:", result.output_path)


def plot_oracle_depth_identifiability(
    result: DepthIdentifiabilityResult,
) -> tuple[Any, Any]:
    """Plot slice-signal migration and oracle agreement across checkpoints."""
    import matplotlib.pyplot as plt

    iterations = result.iterations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for slice_index in range(result.phase_fractions.shape[1]):
        axes[0].plot(
            iterations,
            result.phase_fractions[:, slice_index],
            marker="o",
            label=f"slice {slice_index}",
        )
    axes[0].set(
        title="Depth-signal migration",
        xlabel="completed iteration",
        ylabel="fraction of |phase|",
        ylim=(0.0, 1.0),
    )
    axes[0].legend()
    axes[1].plot(
        iterations, result.oracle_correlations, marker="o", label="oracle order"
    )
    axes[1].plot(
        iterations,
        result.reversed_oracle_correlations,
        marker="o",
        label="reversed oracle",
    )
    axes[1].set(
        title="Depth-order agreement",
        xlabel="completed iteration",
        ylabel="phase correlation",
        ylim=(-1.0, 1.0),
    )
    axes[1].legend()
    axes[2].plot(
        iterations,
        result.scale_adjusted_nrmse,
        marker="o",
        label="scale-adjusted phase NRMSE",
    )
    axes[2].plot(
        iterations,
        result.transmission_departure,
        marker="o",
        label="complex transmission departure",
    )
    axes[2].set(
        title="Departure from oracle initialization",
        xlabel="completed iteration",
        ylabel="normalized error",
    )
    axes[2].legend()
    return fig, axes


def plot_reconstruction_checkpoint_phases(
    history: ReconstructionHistoryResult,
    *,
    cmap: str = "RdBu_r",
) -> tuple[Any, Any]:
    """Show every captured slice with one symmetric scale per checkpoint."""
    import matplotlib.pyplot as plt

    phase = np.angle(history.objects_complex_iteration_slice_xy)
    rows, columns = phase.shape[:2]
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.1 * columns, 2.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for row in range(rows):
        limit = max(float(np.percentile(np.abs(phase[row]), 99.9)), 1e-12)
        fractions = np.sum(np.abs(phase[row]), axis=(1, 2), dtype=np.float64)
        fractions /= max(float(fractions.sum()), np.finfo(np.float64).eps)
        for column in range(columns):
            axes[row, column].imshow(
                phase[row, column].T,
                origin="lower",
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
            )
            axes[row, column].set_title(
                f"iter {int(history.iterations[row])}, slice {column}\n"
                f"{fractions[column]:.1%} of |phase|"
            )
            axes[row, column].set_axis_off()
    return fig, axes


def print_environment_report(report: dict[str, Any]) -> None:
    """Print versions and device status in one compact block."""
    package_text = ", ".join(
        f"{name}={version}" for name, version in report["packages"].items()
    )
    print(f"Python {report['python']}; {package_text}")
    if report["gpu"]["available"]:
        if report["gpu"]["free_memory_gib"] is not None:
            print(
                f"CUDA device: {report['gpu']['name']} "
                f"({report['gpu']['free_memory_gib']:.2f}/"
                f"{report['gpu']['total_memory_gib']:.2f} GiB free)"
            )
        else:
            print(f"CUDA device: {report['gpu']['name']}")
    else:
        print(f"CUDA device unavailable: {report['gpu']['error']}")


def electron_wavelength_angstrom(energy_ev: float) -> float:
    """Relativistic electron wavelength in angstrom for accelerating voltage eV."""
    voltage = _positive_finite(energy_ev, "energy_ev")
    return float(12.2639 / math.sqrt(voltage * (1.0 + 0.97845e-6 * voltage)))


def _ensure_abtem_reconstruction_compatibility() -> bool:
    """Restore the Fresnel-array hook expected by abTEM 1.0.10 MS-PIE.

    The packaged reconstruction operator calls ``_evaluate_propagator_array``,
    while the same release's ``FresnelPropagator`` exposes the equivalent newer
    implementation through ``get_array``. The reconstruction path operates on
    bare arrays, so this local shim evaluates the same first-order Fresnel phase
    and antialias aperture without modifying the installed package.
    """
    from abtem.antialias import antialias_aperture
    from abtem.core.complex import complex_exponential
    from abtem.core.grid import spatial_frequencies
    from abtem.multislice import FresnelPropagator

    if hasattr(FresnelPropagator, "_evaluate_propagator_array"):
        return False

    def _evaluate_propagator_array(
        self: Any,
        gpts: tuple[int, ...],
        sampling: tuple[float, float],
        wavelength: float,
        thickness: float,
        tilt: Any,
        xp: Any,
    ) -> Any:
        del self, tilt
        valid_gpts = tuple(int(value) for value in gpts[-2:])
        valid_sampling = tuple(float(value) for value in sampling)
        kx, ky = spatial_frequencies(valid_gpts, valid_sampling, xp=xp)
        k_squared = kx[:, None] ** 2 + ky[None, :] ** 2
        array = complex_exponential(
            -k_squared * np.pi * float(thickness) * float(wavelength)
        )
        return array * antialias_aperture(valid_gpts, valid_sampling, xp)

    setattr(FresnelPropagator, "_evaluate_propagator_array", _evaluate_propagator_array)
    return True


def _multislice_overlap_projection_with_final_propagation(
    objects: Any,
    probes: Any,
    position: Any,
    old_position: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Run MS-PIE overlap and propagate the final slice to the detector plane."""
    from abtem.reconstruct import (
        MultislicePtychographicOperator,
        _propagate_array,
    )

    probes, exit_waves = MultislicePtychographicOperator._overlap_projection(
        objects,
        probes,
        position,
        old_position,
        **kwargs,
    )
    slice_thicknesses = np.asarray(kwargs["slice_thicknesses"], dtype=np.float64)
    exit_waves[-1] = _propagate_array(
        kwargs["propagator"],
        exit_waves[-1],
        sampling=kwargs["sampling"],
        wavelength=kwargs["wavelength"],
        thickness=float(slice_thicknesses[-1]),
        overwrite=False,
        xp=kwargs.get("xp", np),
    )
    return probes, exit_waves


def _multislice_update_with_final_propagation(
    objects: Any,
    probes: Any,
    position: Any,
    exit_waves: Any,
    modified_exit_waves: Any,
    diffraction_patterns: Any,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Apply the final propagator adjoint before the standard MS-PIE update."""
    from abtem.reconstruct import (
        MultislicePtychographicOperator,
        _propagate_array,
    )

    slice_thicknesses = np.asarray(kwargs["slice_thicknesses"], dtype=np.float64)
    object_plane_exit_waves = exit_waves.copy()
    object_plane_modified_waves = modified_exit_waves.copy()
    propagation_kwargs = {
        "sampling": kwargs["sampling"],
        "wavelength": kwargs["wavelength"],
        "thickness": -float(slice_thicknesses[-1]),
        "overwrite": False,
        "xp": kwargs.get("xp", np),
    }
    object_plane_exit_waves[-1] = _propagate_array(
        kwargs["propagator"], exit_waves[-1], **propagation_kwargs
    )
    object_plane_modified_waves[-1] = _propagate_array(
        kwargs["propagator"], modified_exit_waves[-1], **propagation_kwargs
    )
    return MultislicePtychographicOperator._update_function(
        objects,
        probes,
        position,
        object_plane_exit_waves,
        object_plane_modified_waves,
        diffraction_patterns,
        **kwargs,
    )


def _multislice_functions_queue(
    operator: Any,
    *,
    max_iterations: int,
    pre_probe_correction_update_steps: int | None,
    pre_position_correction_update_steps: int | None,
    final_slice_propagation: bool,
) -> Any:
    """Prepare abTEM's queue, optionally replacing the final-plane operators."""
    if not final_slice_propagation:
        return None
    queue, _ = operator._prepare_functions_queue(
        max_iterations,
        pre_probe_correction_update_steps=pre_probe_correction_update_steps,
        pre_position_correction_update_steps=pre_position_correction_update_steps,
    )
    return [
        [
            (
                _multislice_overlap_projection_with_final_propagation,
                fourier_projection,
                _multislice_update_with_final_propagation,
                position_correction,
            )
            for (
                _overlap_projection,
                fourier_projection,
                _update_function,
                position_correction,
            ) in iteration
        ]
        for iteration in queue
    ]


def _make_potential(
    view: OrientedAtomsResult,
    config: PtychographyConfig,
    *,
    frozen_phonons: bool,
    frozen_seed: int | None = None,
) -> Any:
    from abtem import FrozenPhonons, Potential

    atoms_or_ensemble: Any = view.atoms
    if frozen_phonons and config.frozen_phonon_configs > 1:
        atoms_or_ensemble = FrozenPhonons(
            view.atoms,
            num_configs=config.frozen_phonon_configs,
            sigmas=_thermal_sigmas_for_atoms(
                config.thermal_sigma_angstrom, view.atoms
            ),
            ensemble_mean=True,
            seed=config.random_seed if frozen_seed is None else frozen_seed,
        )
    return Potential(
        atoms_or_ensemble,
        sampling=config.potential_sampling_angstrom,
        slice_thickness=config.potential_slice_thickness_angstrom,
        parametrization=str(config.potential_parametrization).strip().lower(),
        projection=config.potential_projection,
        periodic=False,
        device=str(config.device).strip().lower(),
    )


def _normalize_probe_initialization(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("probe_initialization must be a string.")
    mode = value.strip().lower()
    if mode not in _PROBE_INITIALIZATION_MODES:
        allowed = ", ".join(sorted(_PROBE_INITIALIZATION_MODES))
        raise ValueError(f"probe_initialization must be one of: {allowed}.")
    return mode


def _normalize_object_initialization(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("object_initialization must be a string.")
    mode = value.strip().lower()
    if mode not in _OBJECT_INITIALIZATION_MODES:
        allowed = ", ".join(sorted(_OBJECT_INITIALIZATION_MODES))
        raise ValueError(f"object_initialization must be one of: {allowed}.")
    return mode


def _normalize_object_constraint(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("object_constraint must be a string.")
    mode = value.strip().lower()
    if mode not in _OBJECT_CONSTRAINT_MODES:
        allowed = ", ".join(sorted(_OBJECT_CONSTRAINT_MODES))
        raise ValueError(f"object_constraint must be one of: {allowed}.")
    return mode


def _normalize_reconstruction_grid(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("reconstruction_grid must be a string.")
    mode = value.strip().lower()
    if mode not in _RECONSTRUCTION_GRID_MODES:
        allowed = ", ".join(sorted(_RECONSTRUCTION_GRID_MODES))
        raise ValueError(f"reconstruction_grid must be one of: {allowed}.")
    return mode


def _reconstruction_grid_metadata(
    view: OrientedAtomsResult,
    config: PtychographyConfig,
    *,
    detector_shape: tuple[int, int],
    mode: str,
    simulation_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the detector padding needed to retain simulation sampling."""
    cell_xy = np.asarray(view.atoms.cell.lengths()[:2], dtype=np.float64)
    requested_sampling = float(config.potential_sampling_angstrom)
    simulation_config = simulation_metadata.get("configuration")
    if isinstance(simulation_config, Mapping):
        simulated_sampling = simulation_config.get("potential_sampling_angstrom")
        if simulated_sampling is not None and not np.isclose(
            float(simulated_sampling), requested_sampling, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "Reconstruction potential_sampling_angstrom must match the cached "
                f"simulation: {requested_sampling} != {simulated_sampling}."
            )

    native_gpts = tuple(
        int(value) for value in np.ceil(cell_xy / requested_sampling).astype(int)
    )
    detector_gpts = tuple(int(value) for value in detector_shape)
    if mode == "simulation_native":
        # Center-pad cropped detectors to the native potential grid, but retain
        # a detector that already samples more finely than that grid.
        roi_gpts = tuple(
            max(detector, native)
            for detector, native in zip(detector_gpts, native_gpts)
        )
    else:
        roi_gpts = detector_gpts

    effective_sampling = cell_xy / np.asarray(roi_gpts, dtype=np.float64)

    return {
        "mode": mode,
        "detector_gpts_xy": list(detector_gpts),
        "simulation_native_gpts_xy": list(native_gpts),
        "region_of_interest_gpts_xy": list(roi_gpts),
        "simulation_cell_xy_angstrom": cell_xy.tolist(),
        "requested_potential_sampling_angstrom": requested_sampling,
        "simulation_native_sampling_xy_angstrom": (
            cell_xy / np.asarray(native_gpts, dtype=np.float64)
        ).tolist(),
        "requested_effective_sampling_xy_angstrom": effective_sampling.tolist(),
        "native_sampling_recovered_exactly": roi_gpts == native_gpts,
        "detector_patterns_center_padded": roi_gpts != detector_gpts,
    }


def _normalize_phase_sign(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise ValueError("phase_sign must be either +1 or -1.")
    sign = int(value)
    if sign not in {-1, 1}:
        raise ValueError("phase_sign must be either +1 or -1.")
    return sign


def _normalize_vacuum_guard_slices(value: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise TypeError(
            "vacuum_guard_slices must contain entrance and exit slice counts."
        )
    try:
        counts = tuple(value)
    except TypeError as exc:
        raise TypeError(
            "vacuum_guard_slices must contain entrance and exit slice counts."
        ) from exc
    if len(counts) != 2:
        raise ValueError(
            "vacuum_guard_slices must contain exactly two slice counts."
        )
    normalized = []
    for count in counts:
        if isinstance(count, (bool, np.bool_)) or int(count) != count:
            raise ValueError("vacuum guard counts must be non-negative integers.")
        integer = int(count)
        if integer < 0:
            raise ValueError("vacuum guard counts must be non-negative integers.")
        normalized.append(integer)
    return normalized[0], normalized[1]


def _apply_object_constraint_array(
    objects: Any,
    *,
    mode: str,
    phase_sign: int,
    vacuum_guard_slices: tuple[int, int],
    antialias_sampling: tuple[float, float] | None = None,
) -> Any:
    """Project complex object slices onto physical phase and bandwidth constraints."""
    from abtem.core.backend import get_array_module

    xp = get_array_module(objects)
    constrained = objects.copy()
    if mode in {"pure_phase", "pure_phase_positive"}:
        phase = xp.angle(constrained)
        if mode == "pure_phase_positive":
            phase = phase_sign * xp.maximum(phase_sign * phase, 0.0)
        constrained = xp.exp(1j * phase).astype(objects.dtype, copy=False)

    entrance, exit_ = vacuum_guard_slices
    if entrance:
        constrained[:entrance] = 1.0 + 0.0j
    if exit_:
        constrained[-exit_:] = 1.0 + 0.0j
    if antialias_sampling is not None:
        constrained = _bandlimit_object_array(
            constrained, sampling=antialias_sampling
        )
        if entrance:
            constrained[:entrance] = 1.0 + 0.0j
        if exit_:
            constrained[-exit_:] = 1.0 + 0.0j
    return constrained


def _bandlimit_object_array(
    objects: Any,
    *,
    sampling: tuple[float, float],
) -> Any:
    """Apply the same lateral antialias aperture used for abTEM transmission."""
    from abtem.antialias import antialias_aperture
    from abtem.core.backend import get_array_module
    from abtem.core.fft import fft2_convolve

    if len(objects.shape) < 2:
        raise ValueError("objects must have at least two lateral dimensions.")
    normalized_sampling = tuple(float(value) for value in sampling)
    if len(normalized_sampling) != 2 or any(
        not np.isfinite(value) or value <= 0.0
        for value in normalized_sampling
    ):
        raise ValueError("sampling must contain two positive finite values.")
    xp = get_array_module(objects)
    kernel = antialias_aperture(
        tuple(int(value) for value in objects.shape[-2:]),
        normalized_sampling,
        xp,
    )
    bandlimited = fft2_convolve(objects, kernel, overwrite_x=False)
    return bandlimited.astype(objects.dtype, copy=False)


def _normalize_capture_iterations(
    value: tuple[int, ...] | list[int] | None,
    maximum_iteration: int,
) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("capture_iterations must be a sequence of integers or None.")
    normalized = []
    for iteration in value:
        if isinstance(iteration, (bool, np.bool_)) or int(iteration) != iteration:
            raise ValueError("capture_iterations must contain only integers.")
        integer = int(iteration)
        if integer < 0 or integer > maximum_iteration:
            raise ValueError(
                "capture_iterations values must be between 0 and "
                f"reconstruction_iterations ({maximum_iteration})."
            )
        normalized.append(integer)
    return tuple(sorted(set(normalized)))


def _reconstruction_history_path(reconstruction_path: Path) -> Path:
    return reconstruction_path.with_name(
        f"{reconstruction_path.stem}_iteration_history.npz"
    )


def _strict_coordinate_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain at least two finite coordinates.")
    if np.any(np.diff(vector) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return vector


def _resample_complex_object_xy(
    objects: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Linearly interpolate complex slices in physical x/y with vacuum fill."""
    source_x = _strict_coordinate_vector(source_x, "source x_angstrom")
    source_y = _strict_coordinate_vector(source_y, "source y_angstrom")
    target_x = _strict_coordinate_vector(target_x, "target x_angstrom")
    target_y = _strict_coordinate_vector(target_y, "target y_angstrom")
    source_array = np.asarray(objects, dtype=np.complex64)
    if source_array.shape[1:] != (len(source_x), len(source_y)):
        raise ValueError("Complex-object source shape does not match its x/y grid.")

    same_grid = (
        len(source_x) == len(target_x)
        and len(source_y) == len(target_y)
        and np.allclose(source_x, target_x, rtol=0.0, atol=1e-10)
        and np.allclose(source_y, target_y, rtol=0.0, atol=1e-10)
    )
    if same_grid:
        output = source_array.copy()
    else:
        from scipy.interpolate import RegularGridInterpolator

        target_x_grid, target_y_grid = np.meshgrid(
            target_x, target_y, indexing="ij"
        )
        query = np.column_stack((target_x_grid.ravel(), target_y_grid.ravel()))
        output = np.empty(
            (source_array.shape[0], len(target_x), len(target_y)),
            dtype=np.complex64,
        )
        for slice_index, source_slice in enumerate(source_array):
            real = RegularGridInterpolator(
                (source_x, source_y),
                source_slice.real,
                method="linear",
                bounds_error=False,
                fill_value=1.0,
            )(query)
            imag = RegularGridInterpolator(
                (source_x, source_y),
                source_slice.imag,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )(query)
            output[slice_index] = (real + 1j * imag).reshape(
                len(target_x), len(target_y)
            )
    inside_x = (target_x >= source_x[0]) & (target_x <= source_x[-1])
    inside_y = (target_y >= source_y[0]) & (target_y <= source_y[-1])
    inside_fraction = float(np.mean(inside_x[:, None] & inside_y[None, :]))
    return output, {
        "method": "none (identical grids)" if same_grid else "linear",
        "outside_source_fill_value": "1+0j (vacuum transmission)",
        "source_shape_xy": [len(source_x), len(source_y)],
        "target_shape_xy": [len(target_x), len(target_y)],
        "target_grid_fraction_inside_source_extent": inside_fraction,
    }


def _fraction_in_closed_interval(
    value: float,
    name: str,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool,
) -> float:
    number = float(value)
    valid_lower = number >= lower if lower_inclusive else number > lower
    if not np.isfinite(number) or not valid_lower or number > upper:
        left = "[" if lower_inclusive else "("
        raise ValueError(f"{name} must be finite and in {left}{lower}, {upper}].")
    return number


def _masked_correlation(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> float:
    first_values = np.asarray(first, dtype=np.float64)[mask]
    second_values = np.asarray(second, dtype=np.float64)[mask]
    if len(first_values) < 2:
        return float("nan")
    first_values = first_values - np.mean(first_values)
    second_values = second_values - np.mean(second_values)
    denominator = float(
        np.linalg.norm(first_values) * np.linalg.norm(second_values)
    )
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.dot(first_values, second_values) / denominator)


def _reconstruction_object_descriptor(
    simulation: SimulationResult,
    config: PtychographyConfig,
    mode: str,
    source: (
        ReconstructionResult | ComplexObjectInitializationResult | str | Path | None
    ),
) -> tuple[
    ReconstructionResult | ComplexObjectInitializationResult | None,
    dict[str, Any],
]:
    """Validate and fingerprint an object initializer before expensive work."""
    if mode == "uniform":
        if source is not None:
            raise ValueError(
                "object_initialization_source is only valid when "
                "object_initialization is 'split_projection' or 'provided_complex'."
            )
        return None, {
            "mode": mode,
            "source": "abTEM unit-transmission initializer",
        }

    if source is None:
        requirement = (
            "a one-slice reconstruction result or .npz path"
            if mode == "split_projection"
            else "a ComplexObjectInitializationResult or initialization .npz path"
        )
        raise ValueError(
            f"object_initialization='{mode}' requires "
            f"object_initialization_source to be {requirement}."
        )
    if mode == "split_projection":
        if isinstance(source, (str, Path)):
            result: ReconstructionResult | ComplexObjectInitializationResult = (
                load_reconstruction(source)
            )
        elif isinstance(source, ReconstructionResult):
            result = source
        else:
            raise TypeError(
                "split_projection source must be a ReconstructionResult or a "
                "path to a reconstruction .npz file."
            )

        objects = np.asarray(result.objects_complex_slice_xy, dtype=np.complex64)
        if objects.ndim != 3 or objects.shape[0] != 1:
            raise ValueError(
                "split_projection requires a source reconstruction containing "
                f"exactly one object slice; received shape {objects.shape}."
            )
        if not np.isfinite(objects.real).all() or not np.isfinite(objects.imag).all():
            raise ValueError(
                "split_projection source objects contain non-finite complex values."
            )

        expected_simulation = simulation.metadata.get("fingerprint_sha256")
        source_simulation = result.metadata.get("simulation_fingerprint_sha256")
        if not expected_simulation or source_simulation != expected_simulation:
            raise ValueError(
                "split_projection source and target must use the same 4D-STEM "
                "simulation fingerprint."
            )

        source_sampling = _object_grid_vector(
            result.metadata,
            "object_sampling_xy_angstrom",
        )
        source_origin = _object_grid_vector(
            result.metadata,
            "object_origin_xy_angstrom",
        )
        descriptor = {
            "mode": mode,
            "source": "complex transmission from a one-slice reconstruction",
            "source_reconstruction_fingerprint_sha256": result.metadata.get(
                "fingerprint_sha256"
            ),
            "source_simulation_fingerprint_sha256": source_simulation,
            "source_object_shape_slice_xy": list(objects.shape),
            "source_object_sampling_xy_angstrom": source_sampling.tolist(),
            "source_object_origin_xy_angstrom": source_origin.tolist(),
            "source_complex_array_sha256": _sha256_array(objects),
            "split_rule": (
                "principal complex Nth root: abs(object)**(1/N) * "
                "exp(1j*angle(object)/N)"
            ),
        }
        return result, descriptor

    if isinstance(source, (str, Path)):
        result = load_complex_object_initialization(source)
    elif isinstance(source, ComplexObjectInitializationResult):
        result = source
    else:
        raise TypeError(
            "provided_complex source must be a ComplexObjectInitializationResult "
            "or a path to its .npz file."
        )
    objects = np.asarray(result.objects_complex_slice_xy, dtype=np.complex64)
    if objects.ndim != 3 or not objects.shape[0]:
        raise ValueError("provided_complex source must have shape (slice,x,y).")
    if not np.isfinite(objects.real).all() or not np.isfinite(objects.imag).all():
        raise ValueError("provided_complex source contains non-finite values.")
    x = _strict_coordinate_vector(result.x_angstrom, "source x_angstrom")
    y = _strict_coordinate_vector(result.y_angstrom, "source y_angstrom")
    z_edges = _strict_coordinate_vector(
        result.z_edges_angstrom, "source z_edges_angstrom"
    )
    if objects.shape != (len(z_edges) - 1, len(x), len(y)):
        raise ValueError(
            "provided_complex shape must match its z edges, x coordinates, and "
            f"y coordinates; received {objects.shape}."
        )
    expected_model = simulation.metadata.get("model_fingerprint_sha256")
    if not expected_model or result.metadata.get("model_fingerprint_sha256") != expected_model:
        raise ValueError(
            "provided_complex source and 4D-STEM data must use the same model "
            "fingerprint."
        )
    if result.metadata.get("view") != simulation.metadata.get("view"):
        raise ValueError(
            "provided_complex source and 4D-STEM data must use the same view metadata."
        )
    source_energy = float(result.metadata.get("energy_ev", np.nan))
    simulated_energy = float(
        simulation.metadata.get("configuration", {}).get("energy_ev", np.nan)
    )
    if not np.isclose(source_energy, config.energy_ev, rtol=0.0, atol=1e-9):
        raise ValueError(
            "provided_complex source energy does not match reconstruction energy_ev."
        )
    if not np.isclose(source_energy, simulated_energy, rtol=0.0, atol=1e-9):
        raise ValueError(
            "provided_complex source energy does not match simulated energy_ev."
        )
    source_potential_config = result.metadata.get("source_oracle_configuration")
    simulation_config = simulation.metadata.get("configuration")
    if not isinstance(source_potential_config, Mapping) or not isinstance(
        simulation_config, Mapping
    ):
        raise ValueError(
            "provided_complex source and 4D-STEM data require potential settings "
            "in their manifests."
        )
    potential_keys = (
        "potential_sampling_angstrom",
        "potential_slice_thickness_angstrom",
        "potential_parametrization",
        "potential_projection",
    )
    mismatched_potential_keys = [
        key
        for key in potential_keys
        if source_potential_config.get(key) != simulation_config.get(key)
    ]
    if mismatched_potential_keys:
        raise ValueError(
            "provided_complex oracle and 4D-STEM potential settings differ: "
            + ", ".join(mismatched_potential_keys)
        )
    descriptor = {
        **dict(result.metadata),
        "mode": mode,
        "source": "physically described supplied complex transmission slices",
        "source_output_path": str(result.output_path.resolve()),
        "source_object_shape_slice_xy": list(objects.shape),
        "source_x_angstrom": x.tolist(),
        "source_y_angstrom": y.tolist(),
        "source_z_edges_angstrom": z_edges.tolist(),
        "source_complex_array_sha256": _sha256_array(objects),
        "interpolation_method": "linear in physical x/y; vacuum fill outside source",
    }
    return result, descriptor


def _object_grid_vector(metadata: Mapping[str, Any], key: str) -> np.ndarray:
    raw = metadata.get(key)
    vector = np.asarray(raw, dtype=np.float64) if raw is not None else np.empty(0)
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError(
            f"Source reconstruction metadata must contain two finite {key} values."
        )
    return vector


def _reconstruction_probe_descriptor(
    simulation: SimulationResult,
    mode: str,
) -> dict[str, Any]:
    """Describe the incident probe source before loading diffraction data."""
    if not isinstance(simulation, SimulationResult):
        raise TypeError("simulation must be a SimulationResult or a 4D-STEM path.")
    if mode == "abtem_default":
        return {
            "mode": mode,
            "source": "abTEM MultislicePtychographicOperator default initializer",
        }

    raw_config = simulation.metadata.get("configuration")
    if not isinstance(raw_config, Mapping):
        raise ValueError(
            "simulation_exact probe initialization requires configuration "
            "metadata in the 4D-STEM manifest."
        )
    required = {
        "energy_ev",
        "semiangle_mrad",
        "probe_aperture_soft",
        "beam_tilt_mrad",
        "probe_aberrations",
    }
    missing = sorted(required - set(raw_config))
    if missing:
        raise ValueError(
            "The 4D-STEM manifest lacks probe settings required for "
            "simulation_exact initialization: " + ", ".join(missing)
        )

    energy = _positive_finite(raw_config["energy_ev"], "simulated energy_ev")
    semiangle = _positive_finite(
        raw_config["semiangle_mrad"], "simulated semiangle_mrad"
    )
    soft = raw_config["probe_aperture_soft"]
    if not isinstance(soft, (bool, np.bool_)):
        raise TypeError("Simulated probe_aperture_soft metadata must be Boolean.")
    tilt = np.asarray(raw_config["beam_tilt_mrad"], dtype=np.float64)
    if tilt.shape != (2,) or not np.isfinite(tilt).all():
        raise ValueError(
            "Simulated beam_tilt_mrad metadata must contain two finite values."
        )
    raw_aberrations = raw_config["probe_aberrations"]
    if raw_aberrations is None:
        aberrations = None
    elif isinstance(raw_aberrations, Mapping):
        aberrations = {
            str(key): float(value)
            for key, value in sorted(raw_aberrations.items())
        }
        if any(not np.isfinite(value) for value in aberrations.values()):
            raise ValueError("Simulated probe aberrations must be finite.")
    else:
        raise TypeError(
            "Simulated probe_aberrations metadata must be a mapping or None."
        )

    source_sigma = float(
        raw_config.get("partial_coherence_source_sigma_angstrom", 0.0)
    )
    if not np.isfinite(source_sigma) or source_sigma < 0.0:
        raise ValueError(
            "Simulated partial_coherence_source_sigma_angstrom must be finite "
            "and non-negative."
        )
    return {
        "mode": mode,
        "source": "coherent incident-probe model from the 4D-STEM manifest",
        "simulation_fingerprint_sha256": simulation.metadata.get(
            "fingerprint_sha256"
        ),
        "energy_ev": energy,
        "semiangle_mrad": semiangle,
        "probe_aperture_soft": bool(soft),
        "beam_tilt_mrad": tilt.tolist(),
        "probe_aberrations": aberrations,
        "partial_coherence_source_sigma_angstrom": source_sigma,
        "partial_coherence_note": (
            "This is the nominal coherent probe before Gaussian source-size mixing."
            if source_sigma > 0.0
            else "No source-size mixing was applied."
        ),
    }


def _initialize_reconstruction_probe(
    operator: Any,
    descriptor: Mapping[str, Any],
    *,
    descriptor_fingerprint: str,
) -> dict[str, Any]:
    """Install and fingerprint the requested entrance probe before PIE updates."""
    mode = str(descriptor["mode"])
    if mode == "simulation_exact":
        from abtem import Probe

        # abTEM exposes the reconstruction sampling only after preprocessing.
        # Rebuilding on that reciprocal-space grid retains the simulated CTF
        # while matching the detector-limited reconstruction dimensions.
        rebuilt = Probe(
            energy=float(descriptor["energy_ev"]),
            semiangle_cutoff=float(descriptor["semiangle_mrad"]),
            soft=bool(descriptor["probe_aperture_soft"]),
            gpts=tuple(
                int(value) for value in operator._region_of_interest_shape
            ),
            sampling=tuple(float(value) for value in operator.sampling),
            tilt=tuple(float(value) for value in descriptor["beam_tilt_mrad"]),
            aberrations=descriptor["probe_aberrations"],
            device=str(operator._device),
        ).build(lazy=False).array
        if tuple(rebuilt.shape) != tuple(operator._probes[0].shape):
            raise RuntimeError(
                "Rebuilt simulation probe shape does not match the reconstruction "
                f"grid: {rebuilt.shape} != {operator._probes[0].shape}."
            )
        operator._probes[0] = rebuilt

    from abtem.core.backend import asnumpy

    initial_probe = np.asarray(asnumpy(operator._probes[0]), dtype=np.complex64)
    return {
        **dict(descriptor),
        "descriptor_fingerprint_sha256": descriptor_fingerprint,
        "reconstruction_grid_gpts": [int(value) for value in initial_probe.shape],
        "reconstruction_sampling_angstrom": [
            float(value) for value in operator.sampling
        ],
        "complex_array_dtype": str(initial_probe.dtype),
        "complex_array_sha256": _sha256_array(initial_probe),
    }


def _initialize_reconstruction_object(
    operator: Any,
    simulation: SimulationResult,
    descriptor: Mapping[str, Any],
    source: ReconstructionResult | ComplexObjectInitializationResult | None,
    *,
    target_z_edges_angstrom: np.ndarray,
    descriptor_fingerprint: str,
    antialias: bool,
) -> dict[str, Any]:
    """Install a validated object guess after abTEM establishes its grid."""
    from abtem.core.backend import asnumpy, copy_to_device

    target_shape = tuple(int(value) for value in operator._objects.shape)
    target_sampling = np.asarray(operator.sampling, dtype=np.float64)
    padding_px = np.asarray(
        operator._experimental_parameters["object_px_padding"], dtype=np.float64
    )
    scan_start = np.asarray(
        simulation.metadata["scan_start_angstrom"], dtype=np.float64
    )
    target_origin = scan_start - padding_px * target_sampling
    target_x = target_origin[0] + np.arange(target_shape[1]) * target_sampling[0]
    target_y = target_origin[1] + np.arange(target_shape[2]) * target_sampling[1]
    target_z_edges = np.asarray(target_z_edges_angstrom, dtype=np.float64)
    mode = str(descriptor["mode"])
    interpolation_metadata: dict[str, Any] | None = None

    if mode == "split_projection":
        if source is None:
            raise RuntimeError("Validated split_projection source is unavailable.")
        source_objects = np.asarray(
            source.objects_complex_slice_xy, dtype=np.complex64
        )
        source_projection = source_objects[0]
        if tuple(source_projection.shape) != target_shape[1:]:
            raise ValueError(
                "split_projection source and target object grids have different "
                f"shapes: {source_projection.shape} != {target_shape[1:]}."
            )
        source_sampling = np.asarray(
            descriptor["source_object_sampling_xy_angstrom"], dtype=np.float64
        )
        source_origin = np.asarray(
            descriptor["source_object_origin_xy_angstrom"], dtype=np.float64
        )
        if not np.allclose(source_sampling, target_sampling, rtol=0.0, atol=1e-9):
            raise ValueError(
                "split_projection source and target object sampling do not match: "
                f"{source_sampling.tolist()} != {target_sampling.tolist()}."
            )
        if not np.allclose(source_origin, target_origin, rtol=0.0, atol=1e-8):
            raise ValueError(
                "split_projection source and target object origins do not match: "
                f"{source_origin.tolist()} != {target_origin.tolist()}."
            )

        num_slices = target_shape[0]
        amplitude_root = np.power(
            np.abs(source_projection).astype(np.float64), 1.0 / num_slices
        )
        phase_root = np.angle(source_projection).astype(np.float64) / num_slices
        initial_slice = amplitude_root * np.exp(1j * phase_root)
        initial_objects = np.repeat(
            initial_slice[np.newaxis, :, :], num_slices, axis=0
        ).astype(np.complex64)
        operator._objects = copy_to_device(initial_objects, operator._device)

        recombined = np.prod(initial_objects.astype(np.complex128), axis=0)
        denominator = max(
            float(np.max(np.abs(source_projection))), np.finfo(np.float64).eps
        )
        product_relative_max_error = float(
            np.max(np.abs(recombined - source_projection)) / denominator
        )
        source_path = str(source.output_path.resolve())
    elif mode == "provided_complex":
        if not isinstance(source, ComplexObjectInitializationResult):
            raise RuntimeError("Validated provided_complex source is unavailable.")
        source_objects = np.asarray(
            source.objects_complex_slice_xy, dtype=np.complex64
        )
        if source_objects.shape[0] != target_shape[0]:
            raise ValueError(
                "provided_complex source and reconstruction require the same "
                f"slice count: {source_objects.shape[0]} != {target_shape[0]}."
            )
        source_z_edges = np.asarray(source.z_edges_angstrom, dtype=np.float64)
        if source_z_edges.shape != target_z_edges.shape or not np.allclose(
            source_z_edges, target_z_edges, rtol=0.0, atol=1e-7
        ):
            raise ValueError(
                "provided_complex source z edges do not match the reconstruction "
                f"slabs: {source_z_edges.tolist()} != {target_z_edges.tolist()}."
            )
        initial_objects, interpolation_metadata = _resample_complex_object_xy(
            source_objects,
            source.x_angstrom,
            source.y_angstrom,
            target_x,
            target_y,
        )
        operator._objects = copy_to_device(initial_objects, operator._device)
        product_relative_max_error = None
        source_path = str(source.output_path.resolve())
    else:
        initial_objects = np.asarray(
            asnumpy(operator._objects), dtype=np.complex64
        )
        product_relative_max_error = None
        source_path = None

    pre_antialias_objects = initial_objects
    if antialias:
        operator._objects = _bandlimit_object_array(
            operator._objects,
            sampling=tuple(float(value) for value in target_sampling),
        )
        initial_objects = np.asarray(
            asnumpy(operator._objects), dtype=np.complex64
        )
    amplitude_deviation = np.abs(initial_objects) - 1.0
    antialias_metadata = {
        "applied": bool(antialias),
        "method": (
            "abTEM two-thirds lateral antialias aperture"
            if antialias
            else None
        ),
        "relative_complex_change": (
            float(
                np.linalg.norm(initial_objects - pre_antialias_objects)
                / max(
                    float(np.linalg.norm(pre_antialias_objects)),
                    np.finfo(np.float64).eps,
                )
            )
            if antialias
            else 0.0
        ),
        "rms_amplitude_deviation_from_one": float(
            np.sqrt(np.mean(amplitude_deviation.astype(np.float64) ** 2))
        ),
    }

    return {
        **dict(descriptor),
        "descriptor_fingerprint_sha256": descriptor_fingerprint,
        "source_output_path": source_path,
        "target_object_shape_slice_xy": list(target_shape),
        "target_object_sampling_xy_angstrom": target_sampling.tolist(),
        "target_object_origin_xy_angstrom": target_origin.tolist(),
        "target_z_edges_angstrom": target_z_edges.tolist(),
        "initial_complex_array_dtype": str(initial_objects.dtype),
        "initial_complex_array_sha256": _sha256_array(initial_objects),
        "recombined_source_relative_max_error": product_relative_max_error,
        "coordinate_resampling": interpolation_metadata,
        "transmission_antialiasing": antialias_metadata,
    }


def _make_probe_scan_detector(
    potential: Any,
    config: PtychographyConfig,
    *,
    scan_seed: int,
) -> tuple[Any, Any, Any, Any, np.ndarray, np.ndarray, dict[str, Any]]:
    from abtem import CustomScan, GridScan, PixelatedDetector, Probe

    extent = np.asarray(potential.extent, dtype=np.float64)
    margin = config.scan_margin_angstrom
    if np.any(extent <= 2.0 * margin):
        raise ValueError("scan_margin_angstrom leaves no scan area.")
    probe = Probe(
        energy=config.energy_ev,
        semiangle_cutoff=config.semiangle_mrad,
        soft=config.probe_aperture_soft,
        extent=tuple(float(value) for value in extent),
        sampling=potential.sampling,
        tilt=tuple(float(value) for value in config.beam_tilt_mrad),
        aberrations=(
            None
            if config.probe_aberrations is None
            else {key: float(value) for key, value in config.probe_aberrations.items()}
        ),
        device=str(config.device).strip().lower(),
    )
    nominal_scan = GridScan(
        start=(margin, margin),
        end=tuple(float(value - margin) for value in extent),
        sampling=config.scan_step_angstrom,
        endpoint=False,
    )
    nominal_positions = np.asarray(nominal_scan.get_positions(), dtype=np.float64)
    actual_positions = nominal_positions.copy()
    if config.scan_position_error_std_angstrom > 0.0:
        rng = np.random.default_rng(scan_seed)
        actual_positions += rng.normal(
            0.0,
            config.scan_position_error_std_angstrom,
            size=actual_positions.shape,
        )
        outside = np.any(
            (actual_positions < 0.0) | (actual_positions >= extent), axis=-1
        )
        if np.any(outside):
            raise ValueError(
                f"scan-position errors moved {int(np.count_nonzero(outside))} probe "
                "positions outside the simulation cell. Reduce "
                "scan_position_error_std_angstrom or increase vacuum/scan margin."
            )
        scan = CustomScan(actual_positions.reshape(-1, 2), squeeze=False)
    else:
        scan = nominal_scan
    detector = PixelatedDetector(
        max_angle=config.detector_max_angle_mrad,
        to_cpu=True,
    )
    displacement = actual_positions - nominal_positions
    scan_details = {
        "model": (
            "independent Gaussian xy perturbations with no clipping"
            if config.scan_position_error_std_angstrom > 0.0
            else "nominal regular GridScan"
        ),
        "requested_std_angstrom": config.scan_position_error_std_angstrom,
        "realized_rms_displacement_angstrom": float(
            np.sqrt(np.mean(np.sum(displacement**2, axis=-1)))
        ),
        "clipped_position_count": 0,
        "actual_positions_physically_used": True,
        "stored_diffraction_scan_axes": "nominal regular grid",
    }
    return (
        probe,
        scan,
        detector,
        nominal_scan,
        nominal_positions,
        actual_positions,
        scan_details,
    )


def _restore_nominal_scan_grid(
    patterns: Any,
    nominal_scan: Any,
    nominal_shape: tuple[int, int],
) -> Any:
    from abtem.measurements import DiffractionPatterns

    expected = int(np.prod(nominal_shape, dtype=np.int64))
    if patterns.shape[0] != expected:
        raise RuntimeError(
            "Perturbed scan returned an unexpected number of diffraction patterns: "
            f"{patterns.shape[0]} != {expected}."
        )
    array = patterns.array.reshape(tuple(nominal_shape) + tuple(patterns.shape[-2:]))
    return DiffractionPatterns(
        array,
        sampling=patterns.sampling,
        fftshift=patterns.fftshift,
        ensemble_axes_metadata=nominal_scan.ensemble_axes_metadata,
        metadata=patterns.metadata,
    )


def _apply_detector_response(
    patterns: Any,
    config: PtychographyConfig,
    *,
    detector_seed: int,
) -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    """Apply one fixed detector realization and per-count read/background noise."""
    details = {
        "background_mean_counts": config.detector_background_mean_counts,
        "read_noise_std_counts": config.detector_read_noise_std_counts,
        "gain_std_fraction": config.detector_gain_std_fraction,
        "dead_pixel_fraction": config.detector_dead_pixel_fraction,
        "saturation_counts": config.detector_saturation_counts,
        "enabled": _detector_count_effects_enabled(config),
        "order": [
            "Poisson background",
            "fixed detector gain and dead-pixel maps",
            "Gaussian read noise",
            "non-negative clipping",
            "saturation clipping",
        ],
    }
    if not details["enabled"]:
        return patterns, details, {}

    import dask.array as da

    array = patterns.array.astype(np.float32)
    is_lazy = isinstance(array, da.Array)
    detector_shape = tuple(int(value) for value in array.shape[-2:])
    map_rng = np.random.default_rng(detector_seed)
    gain = np.ones(detector_shape, dtype=np.float32)
    if config.detector_gain_std_fraction > 0.0:
        gain = map_rng.normal(
            1.0,
            config.detector_gain_std_fraction,
            size=detector_shape,
        ).astype(np.float32)
        gain = np.maximum(gain, 0.0)
    live_pixels = np.ones(detector_shape, dtype=np.float32)
    if config.detector_dead_pixel_fraction > 0.0:
        live_pixels = (
            map_rng.random(detector_shape) >= config.detector_dead_pixel_fraction
        ).astype(np.float32)
    response_map = gain * live_pixels
    state = {
        "detector_gain_map": gain,
        "detector_live_pixel_mask": live_pixels.astype(bool),
    }
    details["realized_dead_pixel_count"] = int(np.count_nonzero(live_pixels == 0.0))
    details["realized_gain_mean"] = float(gain.mean())
    details["realized_gain_std"] = float(gain.std())

    if is_lazy:
        noise_rng = da.random.default_rng(detector_seed + 1)
        if config.detector_background_mean_counts > 0.0:
            array = array + noise_rng.poisson(
                config.detector_background_mean_counts,
                size=array.shape,
                chunks=array.chunks,
            ).astype(np.float32)
        array = array * response_map
        if config.detector_read_noise_std_counts > 0.0:
            array = array + noise_rng.normal(
                0.0,
                config.detector_read_noise_std_counts,
                size=array.shape,
                chunks=array.chunks,
            ).astype(np.float32)
        array = da.maximum(array, 0.0)
        if config.detector_saturation_counts is not None:
            array = da.minimum(array, config.detector_saturation_counts)
    else:
        noise_rng = np.random.default_rng(detector_seed + 1)
        if config.detector_background_mean_counts > 0.0:
            array = array + noise_rng.poisson(
                config.detector_background_mean_counts,
                size=array.shape,
            ).astype(np.float32)
        array = array * response_map
        if config.detector_read_noise_std_counts > 0.0:
            array = array + noise_rng.normal(
                0.0,
                config.detector_read_noise_std_counts,
                size=array.shape,
            ).astype(np.float32)
        array = np.maximum(array, 0.0)
        if config.detector_saturation_counts is not None:
            array = np.minimum(array, config.detector_saturation_counts)
    return (
        _replace_diffraction_array(patterns, array.astype(np.float32)),
        details,
        state,
    )


def _replace_diffraction_array(patterns: Any, array: Any) -> Any:
    from abtem.measurements import DiffractionPatterns

    return DiffractionPatterns(
        array,
        sampling=patterns.sampling,
        fftshift=patterns.fftshift,
        ensemble_axes_metadata=patterns.ensemble_axes_metadata,
        metadata=patterns.metadata,
    )


def _condition_effects(config: PtychographyConfig) -> dict[str, bool]:
    return {
        "frozen_phonons": config.frozen_phonon_configs > 1,
        "poisson_counting_noise": config.dose_electrons_per_angstrom2 is not None,
        "probe_aberrations": bool(config.probe_aberrations),
        "beam_tilt": any(float(value) != 0.0 for value in config.beam_tilt_mrad),
        "partial_coherence": config.partial_coherence_source_sigma_angstrom > 0.0,
        "scan_position_errors": config.scan_position_error_std_angstrom > 0.0,
        "detector_response": _detector_count_effects_enabled(config),
    }


def _validate_named_condition(
    config: PtychographyConfig,
    condition_name: str,
) -> None:
    if condition_name not in VALIDATION_CONDITION_DESCRIPTIONS:
        return
    instrument_enabled = bool(
        config.probe_aberrations
        or any(float(value) != 0.0 for value in config.beam_tilt_mrad)
        or config.partial_coherence_source_sigma_angstrom > 0.0
        or config.scan_position_error_std_angstrom > 0.0
        or _detector_count_effects_enabled(config)
    )
    if condition_name == "ideal_static":
        if (
            config.frozen_phonon_configs != 1
            or config.dose_electrons_per_angstrom2 is not None
            or instrument_enabled
        ):
            raise ValueError(
                "ideal_static requires one configuration, no dose, and no "
                "instrument effects."
            )
    elif condition_name == "thermal_only":
        if (
            config.frozen_phonon_configs < 2
            or config.dose_electrons_per_angstrom2 is not None
            or instrument_enabled
        ):
            raise ValueError(
                "thermal_only requires at least two frozen-phonon configurations, "
                "no dose, and no instrument effects."
            )
    elif condition_name == "thermal_dose_limited":
        if (
            config.frozen_phonon_configs < 2
            or config.dose_electrons_per_angstrom2 is None
            or instrument_enabled
        ):
            raise ValueError(
                "thermal_dose_limited requires at least two frozen-phonon "
                "configurations, a dose, and no additional instrument effects."
            )
    elif condition_name == "instrument_model" and (
        config.frozen_phonon_configs < 2
        or config.dose_electrons_per_angstrom2 is None
    ):
        raise ValueError(
            "instrument_model requires at least two frozen-phonon configurations "
            "and a finite dose. Individual instrument effects may remain disabled "
            "until calibrated."
        )


def _detector_count_effects_enabled(config: PtychographyConfig) -> bool:
    return bool(
        config.detector_background_mean_counts > 0.0
        or config.detector_read_noise_std_counts > 0.0
        or config.detector_gain_std_fraction > 0.0
        or config.detector_dead_pixel_fraction > 0.0
        or config.detector_saturation_counts is not None
    )


def _derived_seeds(random_seed: int, *, namespace: str) -> dict[str, int]:
    namespace_value = int(namespace[:16], 16)
    generated = np.random.SeedSequence(
        [int(random_seed), namespace_value]
    ).generate_state(4, dtype=np.uint32)
    return {
        "frozen_phonons": int(generated[0]),
        "scan_positions": int(generated[1]),
        "poisson_counts": int(generated[2]),
        "detector_response": int(generated[3]),
    }


def _potential_config(config: PtychographyConfig) -> dict[str, Any]:
    return {
        "potential_sampling_angstrom": config.potential_sampling_angstrom,
        "potential_slice_thickness_angstrom": (
            config.potential_slice_thickness_angstrom
        ),
        "potential_parametrization": (
            str(config.potential_parametrization).strip().lower()
        ),
        "potential_projection": config.potential_projection,
        "device": config.device,
    }


def _cache_matches(
    output_path: Path,
    manifest_path: Path,
    fingerprint: str,
    *,
    overwrite: bool,
    schema: str,
) -> bool:
    output_exists = output_path.exists()
    manifest_exists = manifest_path.is_file()
    if not output_exists and not manifest_exists:
        return False
    if overwrite:
        return False
    if not output_exists or not manifest_exists:
        raise FileExistsError(
            "A partial cached output exists. Set overwrite=True or choose a new path."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not validate cache manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema") != schema or manifest.get("fingerprint_sha256") != fingerprint:
        raise FileExistsError(
            "Cached output settings do not match this run. Set overwrite=True "
            "or choose a new output path."
        )
    return True


def _software_versions() -> dict[str, str | None]:
    return {
        name: _package_version(name)
        for name in ("numpy", "ase", "abtem", "cupy", "zarr", "dask")
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _python_version() -> str:
    import platform

    return platform.python_version()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_thermal_sigma(
    value: float | Mapping[str, float],
) -> float | dict[str, float]:
    """Validate a shared or element-specific frozen-phonon displacement sigma."""
    if not isinstance(value, Mapping):
        return _positive_finite(value, "thermal_sigma_angstrom")
    if not value:
        raise ValueError("thermal_sigma_angstrom mapping must not be empty.")
    try:
        from ase.data import atomic_numbers
    except ImportError as exc:
        raise ImportError(
            "ASE is required. Run this notebook with the 'Python (cmep-abtem)' kernel."
        ) from exc
    normalized: dict[str, float] = {}
    for raw_symbol, raw_sigma in value.items():
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise TypeError(
                "thermal_sigma_angstrom mapping keys must be element symbols."
            )
        symbol = raw_symbol.strip().capitalize()
        if symbol not in atomic_numbers or atomic_numbers[symbol] <= 0:
            raise ValueError(f"Unknown element in thermal_sigma_angstrom: {raw_symbol!r}")
        if symbol in normalized:
            raise ValueError(
                f"Duplicate element in thermal_sigma_angstrom after normalization: {symbol}"
            )
        normalized[symbol] = _positive_finite(
            raw_sigma, f"thermal_sigma_angstrom[{symbol!r}]"
        )
    return {symbol: normalized[symbol] for symbol in sorted(normalized)}


def _thermal_sigmas_for_atoms(
    value: float | Mapping[str, float], atoms: Any
) -> float | dict[str, float]:
    """Require an explicit displacement sigma for every simulated element."""
    normalized = _normalize_thermal_sigma(value)
    if isinstance(normalized, dict):
        present = set(atoms.get_chemical_symbols())
        missing = sorted(present - set(normalized))
        if missing:
            raise ValueError(
                "thermal_sigma_angstrom is missing simulated elements: "
                + ", ".join(missing)
            )
    return normalized


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    valid = number >= 0.0 if allow_zero else number > 0.0
    if not np.isfinite(number) or not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}.")
    return number


def _require_suffix(path: str | Path, suffix: str) -> Path:
    result = Path(path).expanduser()
    if result.suffix.lower() != suffix:
        raise ValueError(f"Path must end in {suffix}: {result}")
    return result


def _range(values: np.ndarray) -> list[float]:
    return [float(values[0]), float(values[-1])]


def _gib(number_of_bytes: int) -> float:
    return float(number_of_bytes / 1024**3)


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))
    return 0


def _format_triplet(values: list[float]) -> str:
    return " x ".join(f"{float(value):.2f}" for value in values)
