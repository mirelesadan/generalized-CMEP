"""Known atomic models and physical view bookkeeping for validation.

ASE and abTEM use angstrom internally. Ground-truth coordinates are also
exported in nanometres for direct comparison with the CMEP workflow. The
default generator remains a defective Au nanoparticle, while external inputs
may contain any elements supported by ASE and the selected abTEM potential.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ATOMIC_MODEL_SCHEMA = "cmep.atomic-model.v3"
AU_MODEL_SCHEMA = ATOMIC_MODEL_SCHEMA  # Backward-compatible public name.
VIEW_SCHEMA = "cmep.au-view.v2"
DEFAULT_AU_LATTICE_CONSTANT_ANGSTROM = 4.078


@dataclass(frozen=True)
class AtomicModelResult:
    """A prepared model, its pre-modification parent, and exact provenance."""

    atoms: Any
    parent_atoms: Any
    removed_atoms: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @property
    def pristine_atoms(self) -> Any:
        """Compatibility alias for results created before the parent rename."""
        return self.parent_atoms


AuModelResult = AtomicModelResult


@dataclass(frozen=True)
class ViewFrame:
    """A right-handed image/beam frame embedded in common world coordinates."""

    name: str
    requested_zone_axis: tuple[float, float, float]
    requested_image_up: tuple[float, float, float]
    zone_axis: tuple[float, float, float]
    reference_image_up: tuple[float, float, float]
    image_up: tuple[float, float, float]
    image_right: tuple[float, float, float]
    in_plane_rotation_deg: float
    projection_label: str
    world_to_view_matrix: np.ndarray


@dataclass(frozen=True)
class OrientedAtomsResult:
    """ASE atoms in a positive orthogonal simulation cell plus its transform."""

    atoms: Any
    frame: ViewFrame
    metadata: dict[str, Any]


def prepare_atomic_model(
    particle_diameter_nm: float | None = 4.0,
    *,
    input_model: str | Path | Any | None = None,
    particle_shape: str = "marks_decahedron",
    lattice_constant_angstrom: float = DEFAULT_AU_LATTICE_CONSTANT_ANGSTROM,
    decahedron_q_fraction: float = 0.35,
    decahedron_r_fraction: float = 0.12,
    recenter_model: bool = True,
    pbc_policy: str = "disable",
    defect_mode: str = "vacancy_cluster",
    vacancy_fraction: float = 0.002,
    vacancy_center_fraction: Sequence[float] = (0.18, -0.12, 0.08),
    vacancy_neighbor_cutoff_angstrom: float | None = None,
) -> AtomicModelResult:
    """Build the default Au particle or prepare a known external atomic model.

    ``input_model`` accepts an ASE-readable file path or an in-memory
    :class:`ase.Atoms` object. Inputs are copied before preparation. The default
    ``None`` path builds the established defective Au Marks decahedron.

    ``pbc_policy`` is ``'disable'``, ``'preserve'``, or
    ``'require_nonperiodic'``. Simulation views are still constructed as finite,
    nonperiodic cells by :func:`orient_atoms_for_abtem`; preserving PBC here is
    useful for provenance and separate downstream handling.

    ``defect_mode`` is ``'none'`` or ``'vacancy_cluster'``. For an external
    model, a positive ``vacancy_neighbor_cutoff_angstrom`` is required when a
    new vacancy cluster is requested. This avoids inferring bonding from an
    arbitrary composition. Existing defects are preserved when the mode is
    ``'none'``.
    """
    if not isinstance(recenter_model, (bool, np.bool_)):
        raise TypeError("recenter_model must be Boolean.")
    recenter = bool(recenter_model)
    pbc_mode = str(pbc_policy).strip().lower()
    if pbc_mode not in {"disable", "preserve", "require_nonperiodic"}:
        raise ValueError(
            "pbc_policy must be 'disable', 'preserve', or 'require_nonperiodic'."
        )
    selected_defect_mode = str(defect_mode).strip().lower()
    if selected_defect_mode not in {"none", "vacancy_cluster"}:
        raise ValueError("defect_mode must be 'none' or 'vacancy_cluster'.")

    lattice = _positive_finite(
        lattice_constant_angstrom, "lattice_constant_angstrom"
    )
    diameter_parameter = (
        None
        if particle_diameter_nm is None
        else _positive_finite(particle_diameter_nm, "particle_diameter_nm")
    )
    if input_model is None:
        if diameter_parameter is None:
            raise ValueError(
                "particle_diameter_nm is required when input_model is None."
            )
        parent, builder = _build_ase_cluster(
            diameter_parameter,
            particle_shape=particle_shape,
            lattice_constant_angstrom=lattice,
            decahedron_q_fraction=decahedron_q_fraction,
            decahedron_r_fraction=decahedron_r_fraction,
        )
        source = {
            "input_type": "generated_au_nanoparticle",
            "input_path": None,
        }
        requested_diameter = diameter_parameter
        effective_shape = str(particle_shape).strip().lower()
    else:
        parent, source = _coerce_atomic_model(input_model)
        builder = {
            "particle_shape": "external_model",
            "builder": source["input_loader"],
            "particle_diameter_parameter_applied": False,
        }
        requested_diameter = None
        effective_shape = "external_model"

    _validate_atomic_model(parent)
    parent = parent.copy()
    input_pbc = np.asarray(parent.pbc, dtype=bool)
    input_cell = np.asarray(parent.cell.array, dtype=np.float64)
    if pbc_mode == "require_nonperiodic" and input_pbc.any():
        raise ValueError(
            "The input model is periodic but pbc_policy='require_nonperiodic'."
        )
    if pbc_mode == "disable":
        parent.set_pbc(False)

    center_before = np.asarray(parent.positions, dtype=np.float64).mean(axis=0)
    center_translation = -center_before if recenter else np.zeros(3, dtype=np.float64)
    if recenter:
        parent.positions += center_translation
    _ensure_truth_atom_ids(parent)

    requested_vacancy_fraction = float(vacancy_fraction)
    if (
        not np.isfinite(requested_vacancy_fraction)
        or not 0.0 <= requested_vacancy_fraction < 1.0
    ):
        raise ValueError("vacancy_fraction must be finite and in [0, 1).")
    vacancy_center = np.asarray(vacancy_center_fraction, dtype=np.float64)
    if vacancy_center.shape != (3,) or not np.isfinite(vacancy_center).all():
        raise ValueError("vacancy_center_fraction must contain three finite values.")
    if np.linalg.norm(vacancy_center) >= 0.8:
        raise ValueError(
            "vacancy_center_fraction must lie safely inside the particle "
            "(vector norm < 0.8)."
        )

    neighbor_cutoff: float | None = None
    cutoff_source: str | None = None
    if vacancy_neighbor_cutoff_angstrom is not None:
        neighbor_cutoff = _positive_finite(
            vacancy_neighbor_cutoff_angstrom,
            "vacancy_neighbor_cutoff_angstrom",
        )
        cutoff_source = "user"
    if (
        selected_defect_mode == "vacancy_cluster"
        and requested_vacancy_fraction > 0.0
        and neighbor_cutoff is None
    ):
        if input_model is not None:
            raise ValueError(
                "vacancy_neighbor_cutoff_angstrom is required when applying a "
                "vacancy cluster to an external atomic model. Set defect_mode='none' "
                "to preserve the supplied structure without new vacancies."
            )
        neighbor_cutoff = 1.25 * lattice / np.sqrt(2.0)
        cutoff_source = "1.25_times_Au_fcc_nearest_neighbor"

    if selected_defect_mode == "vacancy_cluster":
        defective, removed_atoms = _apply_vacancy_cluster(
            parent,
            vacancy_fraction=requested_vacancy_fraction,
            center_fraction=vacancy_center,
            neighbor_cutoff_angstrom=neighbor_cutoff,
        )
        effective_vacancy_fraction = requested_vacancy_fraction
    else:
        defective = parent.copy()
        removed_atoms = []
        _ensure_near_vacancy_array(defective)
        effective_vacancy_fraction = 0.0

    _validate_atomic_model(defective)
    realized_diameter = radial_diameter_nm(defective)
    spans_nm = np.ptp(defective.positions, axis=0) / 10.0
    symbols = defective.get_chemical_symbols()
    element_counts = {
        symbol: int(symbols.count(symbol)) for symbol in sorted(set(symbols))
    }

    config = {
        "schema": ATOMIC_MODEL_SCHEMA,
        "requested_diameter_nm": requested_diameter,
        "particle_diameter_parameter_nm": diameter_parameter,
        "particle_shape": effective_shape,
        "lattice_constant_angstrom": lattice,
        "decahedron_q_fraction": float(decahedron_q_fraction),
        "decahedron_r_fraction": float(decahedron_r_fraction),
        "recenter_model": recenter,
        "pbc_policy": pbc_mode,
        "defect_mode": selected_defect_mode,
        "requested_vacancy_fraction": requested_vacancy_fraction,
        "vacancy_fraction": effective_vacancy_fraction,
        "vacancy_center_fraction": vacancy_center.tolist(),
        "vacancy_neighbor_cutoff_angstrom": neighbor_cutoff,
        "vacancy_neighbor_cutoff_source": cutoff_source,
    }
    fingerprint = atomic_model_fingerprint(defective, config)
    metadata = {
        **config,
        **source,
        "builder_details": builder,
        "chemical_formula": defective.get_chemical_formula(mode="hill"),
        "element_counts": element_counts,
        "elements": list(element_counts),
        "diameter_definition": (
            "2 * maximum distance from an atom center to the arithmetic "
            "mean atomic position"
        ),
        "realized_diameter_nm": realized_diameter,
        "axis_spans_nm": {axis: float(value) for axis, value in zip("xyz", spans_nm)},
        "parent_atom_count": int(len(parent)),
        "pristine_atom_count": int(len(parent)),
        "final_atom_count": int(len(defective)),
        "removed_atom_count": int(len(removed_atoms)),
        "defect_modification_applied": bool(removed_atoms),
        "intrinsic_twin_boundaries": (
            effective_shape == "marks_decahedron" and input_model is None
        ),
        "geometry_relaxation_applied_by_workflow": False,
        "geometry_relaxation_note": (
            "This workflow does not relax coordinates. Generated particles retain "
            "the ASE cluster geometry after defect modification; an external input "
            "may already have been relaxed before loading."
        ),
        "coordinate_frame": (
            "common_world_xyz_centered_at_arithmetic_mean"
            if recenter
            else "common_world_xyz_as_supplied"
        ),
        "recenter_translation_angstrom": center_translation.tolist(),
        "input_pbc_xyz": input_pbc.tolist(),
        "prepared_pbc_xyz": np.asarray(parent.pbc, dtype=bool).tolist(),
        "input_cell_angstrom": input_cell.tolist(),
        "prepared_cell_angstrom": np.asarray(parent.cell.array).tolist(),
        "simulation_boundary_note": (
            "orient_atoms_for_abtem creates a finite orthogonal nonperiodic cell "
            "for each selected beam view."
        ),
        "internal_length_unit": "angstrom",
        "ground_truth_export_length_unit": "nm",
        "model_fingerprint_sha256": fingerprint,
    }
    return AtomicModelResult(
        atoms=defective,
        parent_atoms=parent,
        removed_atoms=tuple(removed_atoms),
        metadata=metadata,
    )


def prepare_au_model(
    particle_diameter_nm: float,
    *,
    input_path: str | Path | None = None,
    particle_shape: str = "marks_decahedron",
    lattice_constant_angstrom: float = DEFAULT_AU_LATTICE_CONSTANT_ANGSTROM,
    decahedron_q_fraction: float = 0.35,
    decahedron_r_fraction: float = 0.12,
    recenter_model: bool = True,
    pbc_policy: str = "disable",
    defect_mode: str = "vacancy_cluster",
    vacancy_fraction: float = 0.002,
    vacancy_center_fraction: Sequence[float] = (0.18, -0.12, 0.08),
    vacancy_neighbor_cutoff_angstrom: float | None = None,
) -> AtomicModelResult:
    """Compatibility wrapper around :func:`prepare_atomic_model`."""
    return prepare_atomic_model(
        particle_diameter_nm,
        input_model=input_path,
        particle_shape=particle_shape,
        lattice_constant_angstrom=lattice_constant_angstrom,
        decahedron_q_fraction=decahedron_q_fraction,
        decahedron_r_fraction=decahedron_r_fraction,
        recenter_model=recenter_model,
        pbc_policy=pbc_policy,
        defect_mode=defect_mode,
        vacancy_fraction=vacancy_fraction,
        vacancy_center_fraction=vacancy_center_fraction,
        vacancy_neighbor_cutoff_angstrom=vacancy_neighbor_cutoff_angstrom,
    )


def load_atomic_model(path: str | Path) -> Any:
    """Read a structure supported by ASE without changing its source file."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Atomic model does not exist: {source}")
    try:
        from ase.io import read

        atoms = read(source, index=-1)
    except Exception as exc:
        raise ValueError(f"Could not read atomic model {source}: {exc}") from exc
    _validate_atomic_model(atoms, source_label=str(source))
    return atoms


def _coerce_atomic_model(value: str | Path | Any) -> tuple[Any, dict[str, Any]]:
    """Copy an ASE Atoms object or load an ASE-readable file with provenance."""
    if isinstance(value, (str, Path)):
        source = Path(value).expanduser().resolve()
        return load_atomic_model(source), {
            "input_type": "atomic_model_file",
            "input_path": str(source),
            "input_loader": "ase.io.read(index=-1)",
        }
    try:
        from ase import Atoms
    except ImportError as exc:
        raise ImportError(
            "ASE is required. Run this notebook with the 'Python (cmep-abtem)' kernel."
        ) from exc
    if not isinstance(value, Atoms):
        raise TypeError(
            "input_model must be None, an ASE-readable file path, or ase.Atoms."
        )
    _validate_atomic_model(value, source_label="input ase.Atoms")
    return value.copy(), {
        "input_type": "ase_atoms",
        "input_path": None,
        "input_loader": "in_memory_ase.Atoms.copy",
    }


def _validate_atomic_model(atoms: Any, *, source_label: str = "atomic model") -> None:
    """Validate the minimum geometry and chemistry required by this workflow."""
    try:
        positions = np.asarray(atoms.positions, dtype=np.float64)
        numbers = np.asarray(atoms.numbers, dtype=np.int64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
    except Exception as exc:
        raise TypeError(f"{source_label} must be an ase.Atoms object.") from exc
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 1:
        raise ValueError(f"{source_label} must contain at least one 3D atom position.")
    if not np.isfinite(positions).all():
        raise ValueError(f"{source_label} contains non-finite atom positions.")
    if numbers.shape != (len(positions),) or np.any(numbers <= 0):
        raise ValueError(
            f"{source_label} must assign a real chemical element to every atom."
        )
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError(f"{source_label} contains an invalid simulation cell.")


def _ensure_truth_atom_ids(atoms: Any) -> None:
    """Preserve valid supplied truth IDs or create deterministic integer IDs."""
    if "truth_atom_id" not in atoms.arrays:
        atoms.set_array("truth_atom_id", np.arange(len(atoms), dtype=np.int64))
        return
    values = np.asarray(atoms.arrays["truth_atom_id"])
    if values.shape != (len(atoms),):
        raise ValueError("Existing truth_atom_id must have one value per atom.")
    integer_values = values.astype(np.int64)
    if not np.all(values == integer_values) or len(np.unique(integer_values)) != len(atoms):
        raise ValueError("Existing truth_atom_id values must be unique integers.")
    atoms.set_array("truth_atom_id", integer_values)


def _ensure_near_vacancy_array(atoms: Any) -> None:
    """Provide the stable ground-truth export label without inventing defects."""
    if "near_vacancy" not in atoms.arrays:
        atoms.set_array("near_vacancy", np.zeros(len(atoms), dtype=bool))
        return
    values = np.asarray(atoms.arrays["near_vacancy"])
    if values.shape != (len(atoms),):
        raise ValueError("Existing near_vacancy must have one value per atom.")
    atoms.set_array("near_vacancy", values.astype(bool))


def radial_diameter_nm(atoms: Any) -> float:
    """Return twice the largest atom-center radius about the arithmetic mean."""
    positions = np.asarray(atoms.positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("atoms must contain at least one three-dimensional position.")
    centered = positions - positions.mean(axis=0)
    return float(2.0 * np.linalg.norm(centered, axis=1).max() / 10.0)


def make_view_frame(
    name: str,
    zone_axis: Sequence[float],
    image_up: Sequence[float],
    *,
    in_plane_rotation_deg: float = 0.0,
    projection_label: str | None = None,
) -> ViewFrame:
    """Create an arbitrary right-handed projection frame in common world space.

    The returned matrix follows the column-vector convention
    ``point_view = world_to_view_matrix @ point_world``. Its rows are the
    view x, view y (image-up), and view z (beam) unit vectors in world space.
    """
    requested_zone = _vector3(zone_axis, "zone_axis")
    requested_up = _vector3(image_up, "image_up")
    z_axis = _unit_vector(requested_zone, "zone_axis")
    up = _unit_vector(requested_up, "image_up")
    reference_y = up - np.dot(up, z_axis) * z_axis
    y_norm = np.linalg.norm(reference_y)
    if y_norm <= 1e-10:
        raise ValueError("image_up must not be parallel to zone_axis.")
    reference_y /= y_norm
    reference_x = np.cross(reference_y, z_axis)
    reference_x /= np.linalg.norm(reference_x)

    angle = float(in_plane_rotation_deg)
    if not np.isfinite(angle):
        raise ValueError("in_plane_rotation_deg must be finite.")
    theta = np.deg2rad(angle)
    x_axis = np.cos(theta) * reference_x + np.sin(theta) * reference_y
    y_axis = -np.sin(theta) * reference_x + np.cos(theta) * reference_y

    label = "custom" if projection_label is None else str(projection_label).strip().lower()
    if not label:
        raise ValueError("projection_label must not be empty.")
    if label != "custom" and (
        len(label) != 2 or any(axis not in "xyz" for axis in label) or label[0] == label[1]
    ):
        raise ValueError(
            "projection_label must be 'custom' or two different letters from 'xyz'."
        )
    matrix = np.stack((x_axis, y_axis, z_axis), axis=0)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10):
        raise RuntimeError("Internal error: view frame is not orthonormal.")
    if np.linalg.det(matrix) <= 0.0:
        raise RuntimeError("Internal error: view frame is not right-handed.")
    return ViewFrame(
        name=str(name),
        requested_zone_axis=tuple(float(v) for v in requested_zone),
        requested_image_up=tuple(float(v) for v in requested_up),
        zone_axis=tuple(float(v) for v in z_axis),
        reference_image_up=tuple(float(v) for v in reference_y),
        image_up=tuple(float(v) for v in y_axis),
        image_right=tuple(float(v) for v in x_axis),
        in_plane_rotation_deg=angle,
        projection_label=label,
        world_to_view_matrix=matrix,
    )


def orient_atoms_for_abtem(
    atoms: Any,
    frame: ViewFrame,
    *,
    vacuum_angstrom: float = 6.0,
) -> OrientedAtomsResult:
    """Rotate a common-world model into a positive orthogonal abTEM cell."""
    vacuum = _positive_finite(vacuum_angstrom, "vacuum_angstrom", allow_zero=True)
    source = atoms.copy()
    source_model_pbc = np.asarray(source.pbc, dtype=bool)
    source_model_cell = np.asarray(source.cell.array, dtype=np.float64)
    world_center = np.asarray(source.positions, dtype=np.float64).mean(axis=0)
    centered_world = np.asarray(source.positions, dtype=np.float64) - world_center
    view_centered = centered_world @ frame.world_to_view_matrix.T
    view_min = view_centered.min(axis=0)
    view_shift = -view_min + vacuum
    positive_view = view_centered + view_shift
    cell_lengths = np.ptp(view_centered, axis=0) + 2.0 * vacuum

    source.positions = positive_view
    source.set_cell(np.diag(cell_lengths), scale_atoms=False)
    source.set_pbc(False)
    metadata = {
        "schema": VIEW_SCHEMA,
        "view_name": frame.name,
        "projection_label": frame.projection_label,
        "projection_label_interpretation": (
            "Reference row/column labels before in-plane rotation. Exact stored "
            "row, column, and beam vectors are authoritative."
        ),
        "requested_zone_axis_world": list(frame.requested_zone_axis),
        "requested_image_up_world": list(frame.requested_image_up),
        "zone_axis_world": list(frame.zone_axis),
        "beam_unit_vector_world": list(frame.zone_axis),
        "image_up_world_before_in_plane_rotation": list(frame.reference_image_up),
        "image_up_world_orthogonalized": list(frame.image_up),
        "image_row_positive_unit_vector_world": list(frame.image_up),
        "image_column_positive_unit_vector_world": list(frame.image_right),
        "in_plane_rotation_deg": frame.in_plane_rotation_deg,
        "world_to_view_matrix_column_convention": frame.world_to_view_matrix.tolist(),
        "view_to_world_matrix_column_convention": frame.world_to_view_matrix.T.tolist(),
        "world_center_subtracted_angstrom": world_center.tolist(),
        "view_shift_to_positive_cell_angstrom": view_shift.tolist(),
        "simulation_cell_angstrom": cell_lengths.tolist(),
        "vacuum_each_side_angstrom": vacuum,
        "source_model_pbc_xyz": source_model_pbc.tolist(),
        "source_model_cell_angstrom": source_model_cell.tolist(),
        "simulation_pbc_xyz": [False, False, False],
        "simulation_boundary_mode": "finite_nonperiodic_bounding_cell",
        "view_axes": {
            "x": "image_column_positive_or_image_right",
            "y": "image_row_positive_or_image_up",
            "z": "incident_beam",
        },
        "matrix_row_meanings": [
            "image_column_positive_unit_vector_world",
            "image_row_positive_unit_vector_world",
            "beam_unit_vector_world",
        ],
        "in_plane_rotation_convention": (
            "Positive rotation actively rotates the view x/y basis about +beam; "
            "fixed world content therefore rotates oppositely in displayed coordinates."
        ),
        "transform_equation": (
            "view = R @ (world - world_center) + view_shift; "
            "world = R.T @ (view - view_shift) + world_center"
        ),
    }
    return OrientedAtomsResult(atoms=source, frame=frame, metadata=metadata)


def make_atomic_model_figure(
    model: AtomicModelResult | Any,
    *,
    background_color: str = "black",
    marker_size: float = 4.0,
    near_vacancy_marker_size: float = 7.0,
    atom_opacity: float = 0.85,
    near_vacancy_color: str = "#ff4d4d",
    figure_size: int = 850,
):
    """Build an interactive Plotly view of every atom in a known structure.

    Ordinary atoms use ASE's element colors and vacancy-neighbour atoms use a
    separate high-contrast color. Plotly supplies orbit rotation, pan, zoom,
    modebar controls, and per-atom hover inspection. There is no atom-count cap
    or visualization subsampling.
    """
    atoms = model.atoms if isinstance(model, AtomicModelResult) else model
    _validate_atomic_model(atoms)
    size = int(figure_size)
    if size < 300:
        raise ValueError("figure_size must be at least 300 pixels.")
    normal_size = _positive_finite(marker_size, "marker_size")
    defect_size = _positive_finite(
        near_vacancy_marker_size, "near_vacancy_marker_size"
    )
    opacity = float(atom_opacity)
    if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("atom_opacity must be finite and lie in [0, 1].")
    background_css, foreground_css = _background_and_foreground(background_color)
    defect_color = _validated_css_color(
        near_vacancy_color, "near_vacancy_color"
    )

    try:
        import plotly.graph_objects as go
        from ase.data import atomic_numbers
        from ase.data.colors import jmol_colors
    except ImportError as exc:
        raise ImportError(
            "Plotly and ASE are required for the interactive atomic-model viewer."
        ) from exc

    positions_nm = np.asarray(atoms.positions, dtype=np.float64) / 10.0
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    atom_ids = np.asarray(
        atoms.arrays.get("truth_atom_id", np.arange(len(atoms))), dtype=np.int64
    )
    near_vacancy = np.asarray(
        atoms.arrays.get("near_vacancy", np.zeros(len(atoms))), dtype=bool
    )
    if atom_ids.shape != (len(atoms),) or near_vacancy.shape != (len(atoms),):
        raise ValueError("Atomic truth labels must contain one value per atom.")

    traces = []
    ordered_elements = sorted(set(symbols.tolist()), key=atomic_numbers.__getitem__)
    for symbol in ordered_elements:
        mask = (symbols == symbol) & ~near_vacancy
        if not np.any(mask):
            continue
        atomic_number = atomic_numbers[symbol]
        element_color = _rgb_fraction_to_css(jmol_colors[atomic_number])
        traces.append(
            go.Scatter3d(
                x=positions_nm[mask, 0],
                y=positions_nm[mask, 1],
                z=positions_nm[mask, 2],
                mode="markers",
                name=symbol,
                customdata=atom_ids[mask],
                marker={
                    "size": normal_size,
                    "color": element_color,
                    "opacity": opacity,
                    "line": {"width": 0},
                },
                hovertemplate=(
                    f"{symbol} atom %{{customdata}}<br>"
                    "x=%{x:.4f} nm<br>y=%{y:.4f} nm<br>z=%{z:.4f} nm"
                    f"<extra>{symbol}</extra>"
                ),
            )
        )

    if np.any(near_vacancy):
        defect_custom = np.empty((int(np.count_nonzero(near_vacancy)), 2), dtype=object)
        defect_custom[:, 0] = atom_ids[near_vacancy]
        defect_custom[:, 1] = symbols[near_vacancy]
        traces.append(
            go.Scatter3d(
                x=positions_nm[near_vacancy, 0],
                y=positions_nm[near_vacancy, 1],
                z=positions_nm[near_vacancy, 2],
                mode="markers",
                name="Vacancy neighbours",
                customdata=defect_custom,
                marker={
                    "size": defect_size,
                    "color": defect_color,
                    "opacity": 1.0,
                    "line": {"color": foreground_css, "width": 0.7},
                },
                hovertemplate=(
                    "%{customdata[1]} atom %{customdata[0]}<br>"
                    "vacancy neighbour<br>x=%{x:.4f} nm<br>"
                    "y=%{y:.4f} nm<br>z=%{z:.4f} nm"
                    "<extra>Vacancy neighbour</extra>"
                ),
            )
        )

    ranges = _padded_coordinate_ranges(positions_nm)

    def axis_style(title: str, limits: list[float]) -> dict[str, Any]:
        return {
            "title": {"text": title, "font": {"color": foreground_css}},
            "range": limits,
            "color": foreground_css,
            "showbackground": False,
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": foreground_css,
            "linewidth": 2,
            "ticks": "outside",
            "tickcolor": foreground_css,
            "tickfont": {"color": foreground_css},
            "showspikes": False,
        }

    if isinstance(model, AtomicModelResult):
        metadata = model.metadata
        title = (
            f"Known atomic model: {metadata['chemical_formula']} "
            f"({metadata['final_atom_count']:,} atoms; "
            f"{metadata['removed_atom_count']} removed)"
        )
    else:
        title = f"Known atomic model ({len(atoms):,} atoms)"

    figure = go.Figure(data=traces)
    figure.update_layout(
        title={"text": title, "x": 0.02},
        width=size,
        height=size,
        autosize=False,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        paper_bgcolor=background_css,
        plot_bgcolor=background_css,
        font={"color": foreground_css},
        hoverlabel={"bgcolor": background_css, "font": {"color": foreground_css}},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": foreground_css},
            "x": 0.01,
            "y": 0.99,
        },
        scene={
            "bgcolor": background_css,
            "xaxis": axis_style("x (nm)", ranges[0]),
            "yaxis": axis_style("y (nm)", ranges[1]),
            "zaxis": axis_style("z (nm)", ranges[2]),
            "aspectmode": "data",
            "dragmode": "orbit",
            "camera": {"eye": {"x": 1.35, "y": -1.45, "z": 0.90}},
        },
        uirevision="keep-atomic-model-camera",
    )
    return figure


def view_frame_fingerprint(frame: ViewFrame) -> str:
    """Return a stable hash for cache-safe naming of a physical view frame."""
    return _json_fingerprint(
        {
            "schema": VIEW_SCHEMA,
            "name": frame.name,
            "projection_label": frame.projection_label,
            "requested_zone_axis": frame.requested_zone_axis,
            "requested_image_up": frame.requested_image_up,
            "in_plane_rotation_deg": frame.in_plane_rotation_deg,
            "world_to_view_matrix": frame.world_to_view_matrix.tolist(),
        }
    )


def view_points_to_world(points_view_angstrom: np.ndarray, view: OrientedAtomsResult) -> np.ndarray:
    """Map one or more simulation-view points back to common world coordinates."""
    points = np.asarray(points_view_angstrom, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError("points_view_angstrom must end with a length-three axis.")
    shift = np.asarray(view.metadata["view_shift_to_positive_cell_angstrom"])
    center = np.asarray(view.metadata["world_center_subtracted_angstrom"])
    return (points - shift) @ view.frame.world_to_view_matrix + center


def world_points_to_view(points_world_angstrom: np.ndarray, view: OrientedAtomsResult) -> np.ndarray:
    """Map one or more common-world points into positive simulation coordinates."""
    points = np.asarray(points_world_angstrom, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError("points_world_angstrom must end with a length-three axis.")
    shift = np.asarray(view.metadata["view_shift_to_positive_cell_angstrom"])
    center = np.asarray(view.metadata["world_center_subtracted_angstrom"])
    return (points - center) @ view.frame.world_to_view_matrix.T + shift


def export_ground_truth(
    result: AtomicModelResult,
    output_dir: str | Path,
    *,
    run_tag: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export parent/defective structures, truth CSV, and a manifest."""
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "parent_extxyz": destination / f"{run_tag}_parent.extxyz",
        "defective_extxyz": destination / f"{run_tag}_defective.extxyz",
        "ground_truth_csv": destination / f"{run_tag}_ground_truth.csv",
        "manifest_json": destination / f"{run_tag}_model_manifest.json",
    }
    if _matching_cached_manifest(
        paths,
        paths["manifest_json"],
        result.metadata["model_fingerprint_sha256"],
        overwrite=overwrite,
    ):
        return paths

    from ase.io import write

    write(paths["parent_extxyz"], _atoms_for_extxyz(result.parent_atoms))
    write(paths["defective_extxyz"], _atoms_for_extxyz(result.atoms))
    near_vacancy = np.asarray(result.atoms.arrays["near_vacancy"], dtype=bool)
    atom_ids = np.asarray(result.atoms.arrays["truth_atom_id"], dtype=np.int64)
    with paths["ground_truth_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["atom_id", "element", "x_nm", "y_nm", "z_nm", "near_vacancy"]
        )
        for atom_id, symbol, position, is_near in zip(
            atom_ids,
            result.atoms.get_chemical_symbols(),
            np.asarray(result.atoms.positions) / 10.0,
            near_vacancy,
        ):
            writer.writerow(
                [
                    int(atom_id),
                    symbol,
                    *(f"{float(value):.10g}" for value in position),
                    int(is_near),
                ]
            )
    manifest = {
        **result.metadata,
        "removed_atoms": list(result.removed_atoms),
        "files": {name: str(path.resolve()) for name, path in paths.items()},
    }
    _write_json(paths["manifest_json"], manifest)
    return paths


def export_oriented_view(
    view: OrientedAtomsResult,
    output_dir: str | Path,
    *,
    run_tag: str,
    model_fingerprint: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export one simulation-oriented model and its invertible frame metadata."""
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{run_tag}_{view.frame.name}"
    paths = {
        "oriented_extxyz": destination / f"{stem}_oriented.extxyz",
        "view_manifest_json": destination / f"{stem}_view_manifest.json",
    }
    view_fingerprint = _json_fingerprint(
        {"model_fingerprint": model_fingerprint, "view": view.metadata}
    )
    if _matching_cached_manifest(
        paths,
        paths["view_manifest_json"],
        view_fingerprint,
        overwrite=overwrite,
        fingerprint_key="view_fingerprint_sha256",
    ):
        return paths

    from ase.io import write

    write(paths["oriented_extxyz"], _atoms_for_extxyz(view.atoms))
    manifest = {
        **view.metadata,
        "model_fingerprint_sha256": model_fingerprint,
        "view_fingerprint_sha256": view_fingerprint,
        "files": {name: str(path.resolve()) for name, path in paths.items()},
    }
    _write_json(paths["view_manifest_json"], manifest)
    return paths


def print_model_summary(result: AtomicModelResult) -> None:
    """Print a compact, notebook-friendly description of the truth model."""
    meta = result.metadata
    print("Known atomic model")
    requested = meta["requested_diameter_nm"]
    size_text = f", requested={requested:.3f} nm" if requested is not None else ""
    print(
        f"  source={meta['input_type']}, formula={meta['chemical_formula']}, "
        f"shape={meta['particle_shape']}{size_text}, "
        f"measured={meta['realized_diameter_nm']:.3f} nm"
    )
    print(
        f"  atoms: {meta['parent_atom_count']} parent -> "
        f"{meta['final_atom_count']} final ({meta['removed_atom_count']} removed)"
    )
    print(
        f"  preparation: recenter={meta['recenter_model']}, "
        f"pbc_policy={meta['pbc_policy']}, defect_mode={meta['defect_mode']}"
    )
    print(f"  intrinsic twin boundaries: {meta['intrinsic_twin_boundaries']}")
    print(f"  truth frame: {meta['coordinate_frame']}")
    print(f"  fingerprint: {meta['model_fingerprint_sha256'][:16]}...")


def atomic_model_fingerprint(atoms: Any, config: dict[str, Any] | None = None) -> str:
    """Hash chemistry, geometry, boundaries, truth labels, and configuration."""
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype="<i4").tobytes())
    digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())
    digest.update(np.asarray(atoms.cell.array, dtype="<f8").tobytes())
    digest.update(np.asarray(atoms.pbc, dtype=np.uint8).tobytes())
    if "truth_atom_id" in atoms.arrays:
        digest.update(np.asarray(atoms.arrays["truth_atom_id"], dtype="<i8").tobytes())
    if "near_vacancy" in atoms.arrays:
        digest.update(np.asarray(atoms.arrays["near_vacancy"], dtype=np.uint8).tobytes())
    if config is not None:
        digest.update(_canonical_json(config).encode("utf-8"))
    return digest.hexdigest()


def _build_ase_cluster(
    target_diameter_nm: float,
    *,
    particle_shape: str,
    lattice_constant_angstrom: float,
    decahedron_q_fraction: float,
    decahedron_r_fraction: float,
) -> tuple[Any, dict[str, Any]]:
    try:
        from ase.cluster import Decahedron, Octahedron
    except ImportError as exc:
        raise ImportError(
            "ASE is required. Run this notebook with the 'Python (cmep-abtem)' kernel."
        ) from exc

    shape = str(particle_shape).strip().lower()
    candidates: list[tuple[float, Any, dict[str, Any]]] = []
    if shape == "marks_decahedron":
        q_fraction = _positive_finite(
            decahedron_q_fraction, "decahedron_q_fraction"
        )
        r_fraction = _positive_finite(
            decahedron_r_fraction, "decahedron_r_fraction"
        )
        for p in range(2, 101):
            q = max(2, int(round(p * q_fraction)))
            r = max(1, int(round(p * r_fraction)))
            atoms = Decahedron(
                "Au", p=p, q=q, r=r, latticeconstant=lattice_constant_angstrom
            )
            actual = radial_diameter_nm(atoms)
            candidates.append(
                (
                    abs(actual - target_diameter_nm),
                    atoms,
                    {"builder": "ase.cluster.Decahedron", "p": p, "q": q, "r": r},
                )
            )
            if actual >= target_diameter_nm and p >= 4:
                break
    elif shape == "truncated_octahedron":
        for cutoff in range(1, 101):
            length = 3 * cutoff + 1
            atoms = Octahedron(
                "Au",
                length=length,
                cutoff=cutoff,
                latticeconstant=lattice_constant_angstrom,
            )
            actual = radial_diameter_nm(atoms)
            candidates.append(
                (
                    abs(actual - target_diameter_nm),
                    atoms,
                    {
                        "builder": "ase.cluster.Octahedron",
                        "length": length,
                        "cutoff": cutoff,
                    },
                )
            )
            if actual >= target_diameter_nm:
                break
    else:
        raise ValueError(
            "particle_shape must be 'marks_decahedron' or 'truncated_octahedron'."
        )

    if not candidates:
        raise RuntimeError("Could not construct an ASE nanoparticle candidate.")
    _, selected_atoms, details = min(candidates, key=lambda item: item[0])
    return selected_atoms, {
        "particle_shape": shape,
        "requested_diameter_nm": target_diameter_nm,
        **details,
    }


def _apply_vacancy_cluster(
    parent: Any,
    *,
    vacancy_fraction: float,
    center_fraction: np.ndarray,
    neighbor_cutoff_angstrom: float | None,
) -> tuple[Any, list[dict[str, Any]]]:
    defective = parent.copy()
    defective.set_array("near_vacancy", np.zeros(len(defective), dtype=bool))
    if vacancy_fraction == 0.0:
        return defective, []
    if neighbor_cutoff_angstrom is None:
        raise RuntimeError("A vacancy-neighbour cutoff is required for atom removal.")

    number_to_remove = max(1, int(round(len(parent) * vacancy_fraction)))
    geometric_center = np.asarray(parent.positions, dtype=np.float64).mean(axis=0)
    centered_positions = np.asarray(parent.positions, dtype=np.float64) - geometric_center
    radial_extent = np.linalg.norm(centered_positions, axis=1).max()
    target = geometric_center + center_fraction * radial_extent
    distances = np.linalg.norm(parent.positions - target, axis=1)
    atom_ids = np.asarray(parent.arrays["truth_atom_id"], dtype=np.int64)
    order = np.lexsort((atom_ids, distances))
    remove_indices = np.sort(order[:number_to_remove])
    removed: list[dict[str, Any]] = []
    removed_positions = np.asarray(parent.positions[remove_indices], dtype=np.float64)
    for index in remove_indices:
        removed.append(
            {
                "truth_atom_id": int(atom_ids[index]),
                "element": parent[index].symbol,
                "position_world_angstrom": parent.positions[index].tolist(),
                "position_world_nm": (parent.positions[index] / 10.0).tolist(),
            }
        )
    del defective[remove_indices.tolist()]

    if len(removed_positions):
        separation = np.linalg.norm(
            defective.positions[:, None, :] - removed_positions[None, :, :], axis=2
        )
        defective.arrays["near_vacancy"] = np.any(
            separation <= neighbor_cutoff_angstrom, axis=1
        )
    return defective, removed


def _atoms_for_extxyz(atoms: Any) -> Any:
    copy = atoms.copy()
    copy.info.clear()
    copy.info["coordinate_unit"] = "angstrom"
    copy.info["coordinate_frame"] = "cmep_validation"
    return copy


def _matching_cached_manifest(
    paths: dict[str, Path],
    manifest_path: Path,
    fingerprint: str,
    *,
    overwrite: bool,
    fingerprint_key: str = "model_fingerprint_sha256",
) -> bool:
    existing = [path for path in paths.values() if path.exists()]
    if not existing:
        return False
    if overwrite:
        return False
    if len(existing) != len(paths) or not manifest_path.is_file():
        raise FileExistsError(
            "A partial cached output exists. Set overwrite=True or choose a new run_tag."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not validate cached manifest {manifest_path}: {exc}") from exc
    if manifest.get(fingerprint_key) != fingerprint:
        raise FileExistsError(
            "Cached outputs were created from different settings. Set overwrite=True "
            "or choose a new run_tag."
        )
    return True


def _validated_css_color(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty color string.")
    try:
        from PIL import ImageColor

        red, green, blue = ImageColor.getrgb(value.strip())[:3]
    except (ImportError, ValueError) as exc:
        raise ValueError(f"Unsupported {name}: {value!r}.") from exc
    return f"rgb({red},{green},{blue})"


def _background_and_foreground(background_color: str) -> tuple[str, str]:
    background = _validated_css_color(background_color, "background_color")
    components = tuple(int(value) for value in background[4:-1].split(","))
    foreground = f"rgb({255 - components[0]},{255 - components[1]},{255 - components[2]})"
    return background, foreground


def _rgb_fraction_to_css(value: Sequence[float]) -> str:
    components = np.asarray(value, dtype=np.float64)
    if components.shape != (3,) or not np.isfinite(components).all():
        raise ValueError("Element color must contain three finite RGB components.")
    components = np.clip(np.rint(255.0 * components), 0.0, 255.0).astype(int)
    return f"rgb({components[0]},{components[1]},{components[2]})"


def _padded_coordinate_ranges(positions: np.ndarray) -> list[list[float]]:
    minimum = np.min(positions, axis=0)
    maximum = np.max(positions, axis=0)
    spans = maximum - minimum
    overall_span = max(float(np.max(spans)), 0.1)
    padding = np.maximum(0.06 * spans, 0.02 * overall_span)
    return [
        [float(low - pad), float(high + pad)]
        for low, high, pad in zip(minimum, maximum, padding)
    ]


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    valid = number >= 0.0 if allow_zero else number > 0.0
    if not np.isfinite(number) or not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}.")
    return number


def _unit_vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = _vector3(value, name)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values.")
    return vector


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
