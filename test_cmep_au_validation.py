"""Focused tests for the known-structure Au ptychography validation workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cmep_abtem_simulation import (
    PtychographyConfig,
    _derived_seeds,
    _thermal_sigmas_for_atoms,
    _validate_zarr_output_path,
    analyze_4dstem_quality,
    estimate_simulation_resources,
    export_oracle_potential,
    load_4dstem,
    make_validation_conditions,
    plot_4dstem_quality,
    reconstruct_multislice_ptychography,
    simulate_4dstem,
)
from cmep_au_model import (
    export_ground_truth,
    export_oriented_view,
    make_atomic_model_figure,
    make_view_frame,
    orient_atoms_for_abtem,
    prepare_atomic_model,
    prepare_au_model,
    view_frame_fingerprint,
    view_points_to_world,
    world_points_to_view,
)


class AuValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_4nm = prepare_au_model(4.0)
        cls.model_8nm = prepare_au_model(8.0)

    def test_diameter_parameter_changes_size_not_model_recipe(self) -> None:
        small = self.model_4nm.metadata
        large = self.model_8nm.metadata
        self.assertEqual(small["particle_shape"], "marks_decahedron")
        self.assertEqual(large["particle_shape"], "marks_decahedron")
        self.assertEqual(small["input_type"], "generated_au_nanoparticle")
        self.assertEqual(small["elements"], ["Au"])
        self.assertEqual(small["defect_mode"], "vacancy_cluster")
        self.assertTrue(small["intrinsic_twin_boundaries"])
        self.assertTrue(large["intrinsic_twin_boundaries"])
        self.assertFalse(small["geometry_relaxation_applied_by_workflow"])
        self.assertLess(abs(small["realized_diameter_nm"] - 4.0), 0.35)
        self.assertLess(abs(large["realized_diameter_nm"] - 8.0), 0.35)
        self.assertGreater(large["final_atom_count"], small["final_atom_count"])
        self.assertEqual(small["parent_atom_count"], len(self.model_4nm.parent_atoms))
        self.assertIs(self.model_4nm.pristine_atoms, self.model_4nm.parent_atoms)
        self.assertGreater(small["removed_atom_count"], 0)
        self.assertGreater(large["removed_atom_count"], 0)

    def test_model_and_vacancies_are_deterministic(self) -> None:
        repeat = prepare_au_model(4.0)
        np.testing.assert_allclose(repeat.atoms.positions, self.model_4nm.atoms.positions)
        np.testing.assert_array_equal(
            repeat.atoms.arrays["truth_atom_id"],
            self.model_4nm.atoms.arrays["truth_atom_id"],
        )
        self.assertEqual(repeat.removed_atoms, self.model_4nm.removed_atoms)
        self.assertEqual(
            repeat.metadata["model_fingerprint_sha256"],
            self.model_4nm.metadata["model_fingerprint_sha256"],
        )

    def test_interactive_atomic_model_figure_has_physical_dark_scene(self) -> None:
        figure = make_atomic_model_figure(self.model_4nm)
        displayed_atoms = sum(len(trace.x) for trace in figure.data)
        self.assertEqual(displayed_atoms, len(self.model_4nm.atoms))
        self.assertEqual(figure.layout.paper_bgcolor, "rgb(0,0,0)")
        self.assertEqual(figure.layout.font.color, "rgb(255,255,255)")
        self.assertEqual(figure.layout.scene.aspectmode, "data")
        self.assertEqual(figure.layout.scene.dragmode, "orbit")
        self.assertFalse(figure.layout.scene.xaxis.showbackground)
        self.assertFalse(figure.layout.scene.xaxis.showgrid)
        self.assertTrue(figure.layout.scene.xaxis.showline)
        self.assertIn("Vacancy neighbours", [trace.name for trace in figure.data])

    @unittest.skipUnless(os.name == "nt", "Windows-specific Zarr path guard")
    def test_long_windows_zarr_path_is_rejected_before_simulation(self) -> None:
        _validate_zarr_output_path(Path("C:/cmep/plan.zarr"))
        long_path = Path("C:/") / ("a" * 180) / ("b" * 70 + ".zarr")
        with self.assertRaisesRegex(ValueError, "too long for reliable Windows"):
            _validate_zarr_output_path(long_path)

    def test_external_path_or_ase_atoms_can_use_mixed_elements(self) -> None:
        from ase import Atoms
        from ase.io import write

        source = Atoms(
            "AuAg",
            positions=[[1.0, 2.0, 3.0], [3.5, 2.0, 3.0]],
            cell=np.diag([8.0, 9.0, 10.0]),
            pbc=(True, False, False),
        )
        source.set_array("truth_atom_id", np.array([10, 20], dtype=np.int64))
        original_positions = source.positions.copy()

        prepared = prepare_atomic_model(
            None,
            input_model=source,
            recenter_model=False,
            pbc_policy="preserve",
            defect_mode="none",
        )
        np.testing.assert_allclose(source.positions, original_positions)
        np.testing.assert_allclose(prepared.atoms.positions, original_positions)
        np.testing.assert_array_equal(prepared.atoms.pbc, [True, False, False])
        np.testing.assert_array_equal(
            prepared.atoms.arrays["truth_atom_id"], [10, 20]
        )
        self.assertEqual(prepared.metadata["element_counts"], {"Ag": 1, "Au": 1})
        self.assertEqual(prepared.metadata["input_type"], "ase_atoms")
        self.assertIsNone(prepared.metadata["requested_diameter_nm"])
        self.assertFalse(prepared.metadata["defect_modification_applied"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mixed.extxyz"
            write(path, source)
            loaded = prepare_atomic_model(
                None,
                input_model=path,
                recenter_model=True,
                pbc_policy="disable",
                defect_mode="none",
            )
            np.testing.assert_allclose(loaded.atoms.positions.mean(axis=0), 0.0)
            self.assertFalse(any(loaded.atoms.pbc))
            self.assertEqual(loaded.metadata["input_type"], "atomic_model_file")
            self.assertEqual(Path(loaded.metadata["input_path"]), path.resolve())

    def test_external_preparation_policies_are_explicit(self) -> None:
        from ase import Atoms

        source = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [2.35, 0.0, 0.0]],
            cell=np.eye(3) * 5.0,
            pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "require_nonperiodic"):
            prepare_atomic_model(
                None,
                input_model=source,
                pbc_policy="require_nonperiodic",
                defect_mode="none",
            )
        with self.assertRaisesRegex(ValueError, "vacancy_neighbor_cutoff_angstrom"):
            prepare_atomic_model(
                None,
                input_model=source,
                pbc_policy="disable",
                defect_mode="vacancy_cluster",
                vacancy_fraction=0.5,
            )
        modified = prepare_atomic_model(
            None,
            input_model=source,
            pbc_policy="disable",
            defect_mode="vacancy_cluster",
            vacancy_fraction=0.5,
            vacancy_neighbor_cutoff_angstrom=2.6,
        )
        self.assertEqual(modified.metadata["removed_atom_count"], 1)
        self.assertEqual(modified.metadata["elements"], ["Si"])

    def test_element_specific_thermal_sigmas_are_validated(self) -> None:
        from ase import Atoms

        atoms = Atoms("AuAg", positions=[[0, 0, 0], [2.8, 0, 0]])
        config = PtychographyConfig(
            thermal_sigma_angstrom={"ag": 0.09, "Au": 0.08}
        ).validated()
        self.assertEqual(
            config.to_dict()["thermal_sigma_angstrom"],
            {"Ag": 0.09, "Au": 0.08},
        )
        self.assertEqual(
            _thermal_sigmas_for_atoms(config.thermal_sigma_angstrom, atoms),
            {"Ag": 0.09, "Au": 0.08},
        )
        with self.assertRaisesRegex(ValueError, "missing simulated elements: Ag"):
            _thermal_sigmas_for_atoms({"Au": 0.08}, atoms)

    def test_arbitrary_view_frame_is_right_handed_and_invertible(self) -> None:
        frame = make_view_frame("cross", [1, 1, 0], [0, 0, 1])
        matrix = frame.world_to_view_matrix
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)
        view = orient_atoms_for_abtem(self.model_4nm.atoms, frame, vacuum_angstrom=6.0)
        transformed = world_points_to_view(self.model_4nm.atoms.positions, view)
        restored = view_points_to_world(transformed, view)
        np.testing.assert_allclose(restored, self.model_4nm.atoms.positions, atol=1e-11)
        self.assertTrue((view.atoms.positions >= 0.0).all())
        self.assertFalse(any(view.atoms.pbc))

    def test_default_projection_frames_and_in_plane_rotation(self) -> None:
        plan = make_view_frame(
            "plan", [0, 0, 1], [0, 1, 0], projection_label="yx"
        )
        cross = make_view_frame(
            "cross", [0, -1, 0], [0, 0, 1], projection_label="zx"
        )
        np.testing.assert_allclose(plan.world_to_view_matrix, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(
            cross.world_to_view_matrix,
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            atol=1e-12,
        )
        self.assertEqual(plan.projection_label, "yx")
        self.assertEqual(cross.projection_label, "zx")

        rolled = make_view_frame(
            "plan",
            [0, 0, 1],
            [0, 1, 0],
            in_plane_rotation_deg=90.0,
            projection_label="yx",
        )
        np.testing.assert_allclose(
            rolled.world_to_view_matrix,
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            atol=1e-12,
        )
        self.assertNotEqual(view_frame_fingerprint(plan), view_frame_fingerprint(rolled))
        view = orient_atoms_for_abtem(self.model_4nm.atoms, rolled)
        self.assertEqual(view.metadata["projection_label"], "yx")
        self.assertEqual(view.metadata["in_plane_rotation_deg"], 90.0)
        np.testing.assert_allclose(
            view.metadata["image_row_positive_unit_vector_world"],
            [-1, 0, 0],
            atol=1e-12,
        )

    def test_validation_condition_ladder_is_explicit(self) -> None:
        base = PtychographyConfig(
            condition_name="instrument_model",
            device="cpu",
            frozen_phonon_configs=6,
            dose_electrons_per_angstrom2=50_000.0,
            probe_aberrations={"C10": 12.0},
            beam_tilt_mrad=(0.2, -0.1),
            partial_coherence_source_sigma_angstrom=0.15,
            scan_position_error_std_angstrom=0.04,
            detector_background_mean_counts=0.2,
        )
        conditions = make_validation_conditions(base)
        self.assertEqual(
            tuple(conditions),
            (
                "ideal_static",
                "thermal_only",
                "thermal_dose_limited",
                "instrument_model",
            ),
        )
        self.assertEqual(conditions["ideal_static"].frozen_phonon_configs, 1)
        self.assertIsNone(conditions["ideal_static"].dose_electrons_per_angstrom2)
        self.assertEqual(conditions["thermal_only"].frozen_phonon_configs, 6)
        self.assertIsNone(conditions["thermal_only"].dose_electrons_per_angstrom2)
        self.assertEqual(
            conditions["thermal_dose_limited"].dose_electrons_per_angstrom2,
            50_000.0,
        )
        self.assertIsNone(conditions["thermal_dose_limited"].probe_aberrations)
        self.assertEqual(
            conditions["instrument_model"].probe_aberrations,
            {"C10": 12.0},
        )
        self.assertGreater(
            conditions["instrument_model"].scan_position_error_std_angstrom, 0.0
        )
        with self.assertRaisesRegex(ValueError, "ideal_static requires"):
            PtychographyConfig(
                condition_name="ideal_static",
                dose_electrons_per_angstrom2=1_000.0,
            ).validated()
        with self.assertRaisesRegex(ValueError, "potential_parametrization"):
            PtychographyConfig(potential_parametrization="unknown").validated()

    def test_random_streams_are_reproducible_and_view_namespaced(self) -> None:
        namespace_a = "1" * 64
        namespace_b = "2" * 64
        seeds_a = _derived_seeds(17, namespace=namespace_a)
        self.assertEqual(seeds_a, _derived_seeds(17, namespace=namespace_a))
        self.assertNotEqual(seeds_a, _derived_seeds(17, namespace=namespace_b))
        self.assertEqual(len(set(seeds_a.values())), len(seeds_a))

    def test_resource_estimate_scales_without_a_cap(self) -> None:
        frame = make_view_frame("plan", [0, 0, 1], [0, 1, 0])
        config = PtychographyConfig(device="cpu")
        small = estimate_simulation_resources(
            orient_atoms_for_abtem(self.model_4nm.atoms, frame), config
        ).details
        large = estimate_simulation_resources(
            orient_atoms_for_abtem(self.model_8nm.atoms, frame), config
        ).details
        self.assertGreater(large["scan_positions"], small["scan_positions"])
        self.assertGreater(
            large["stored_4dstem_float32_gib"], small["stored_4dstem_float32_gib"]
        )
        self.assertIn("No atom", large["notes"][0])

    def test_truth_and_view_exports_are_cacheable(self) -> None:
        frame = make_view_frame("plan", [0, 0, 1], [0, 1, 0])
        view = orient_atoms_for_abtem(self.model_4nm.atoms, frame)
        with tempfile.TemporaryDirectory() as temp_dir:
            truth_paths = export_ground_truth(
                self.model_4nm, temp_dir, run_tag="au_test"
            )
            cached_paths = export_ground_truth(
                self.model_4nm, temp_dir, run_tag="au_test"
            )
            self.assertEqual(truth_paths, cached_paths)
            view_paths = export_oriented_view(
                view,
                temp_dir,
                run_tag="au_test",
                model_fingerprint=self.model_4nm.metadata[
                    "model_fingerprint_sha256"
                ],
            )
            self.assertTrue(all(path.exists() for path in truth_paths.values()))
            self.assertTrue(all(path.exists() for path in view_paths.values()))
            manifest = json.loads(
                truth_paths["manifest_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["final_atom_count"], len(self.model_4nm.atoms))
            self.assertEqual(manifest["parent_atom_count"], len(self.model_4nm.parent_atoms))
            self.assertIn("parent_extxyz", manifest["files"])

    def test_tiny_abtem_oracle_and_4dstem_cache(self) -> None:
        from ase import Atoms

        atoms = Atoms(
            "AuAg",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
            pbc=False,
        )
        atoms.new_array("truth_atom_id", np.array([0, 1], dtype=np.int64))
        frame = make_view_frame("tiny", [0, 0, 1], [0, 1, 0])
        view = orient_atoms_for_abtem(atoms, frame, vacuum_angstrom=4.0)
        config = PtychographyConfig(
            condition_name="ideal_static",
            device="cpu",
            potential_sampling_angstrom=0.5,
            potential_slice_thickness_angstrom=2.0,
            scan_step_angstrom=2.0,
            scan_margin_angstrom=1.0,
            detector_max_angle_mrad=30.0,
            frozen_phonon_configs=1,
            dose_electrons_per_angstrom2=None,
            reconstruction_iterations=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            oracle = export_oracle_potential(
                view,
                config,
                Path(temp_dir) / "oracle.npz",
                progress=False,
            )
            self.assertEqual(oracle.stack_slice_row_col.ndim, 3)
            self.assertEqual(
                oracle.stack_slice_row_col.shape,
                (
                    len(oracle.z_angstrom),
                    len(oracle.y_angstrom),
                    len(oracle.x_angstrom),
                ),
            )
            simulation = simulate_4dstem(
                view,
                config,
                Path(temp_dir) / "tiny.zarr",
                progress=False,
            )
            loaded = load_4dstem(simulation.output_path)
            self.assertEqual(tuple(loaded.diffraction_patterns.shape[:2]), (3, 3))
            self.assertEqual(loaded.metadata["stored_frozen_phonon_axis"], False)
            self.assertTrue(loaded.scan_positions_path.is_file())
            self.assertEqual(loaded.scan_positions_path, loaded.simulation_state_path)
            qc = analyze_4dstem_quality(
                loaded,
                Path(temp_dir) / "tiny_qc.npz",
                histogram_bins=12,
                progress=False,
            )
            self.assertEqual(qc.mean_diffraction_pattern.shape, tuple(loaded.diffraction_patterns.shape[-2:]))
            self.assertEqual(qc.scan_integrated_image.shape, (3, 3))
            self.assertEqual(qc.metadata["nan_value_count"], 0)
            self.assertEqual(qc.metadata["infinite_value_count"], 0)
            self.assertTrue(qc.metadata["no_visualization_or_data_cap_applied"])
            self.assertEqual(qc.metadata["scan_step_angstrom"], 2.0)
            self.assertEqual(len(qc.metadata["angular_sampling_mrad"]), 2)
            figure, axes = plot_4dstem_quality(qc)
            self.assertEqual(len(axes), 3)
            import matplotlib.pyplot as plt

            plt.close(figure)
            reconstruction = reconstruct_multislice_ptychography(
                loaded,
                view,
                config,
                Path(temp_dir) / "tiny_reconstruction.npz",
                object_step_size=0.25,
                probe_step_size=0.05,
                step_size_damping_rate=0.99,
                probe_correction_start_iteration=1,
                position_correction=False,
                verbose=False,
            )
            self.assertEqual(reconstruction.phase_stack_slice_row_col.ndim, 3)
            self.assertTrue(np.isfinite(reconstruction.phase_stack_slice_row_col).all())
            self.assertTrue(np.isfinite(reconstruction.error))
            self.assertTrue(
                reconstruction.metadata["abtem_fresnel_compatibility_shim_applied"]
            )
            controls = reconstruction.metadata["reconstruction_controls"]
            self.assertEqual(controls["object_step_size"], 0.25)
            self.assertEqual(controls["probe_step_size"], 0.05)
            self.assertEqual(controls["step_size_damping_rate"], 0.99)
            self.assertEqual(controls["probe_correction_start_iteration"], 1)
            self.assertFalse(controls["position_correction"])
            self.assertEqual(
                controls["abtem_pre_probe_correction_update_steps"],
                controls["scan_position_count"],
            )

    def test_tiny_instrument_condition_applies_physical_effects(self) -> None:
        from ase import Atoms

        atoms = Atoms(
            "AuAg",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
            pbc=False,
        )
        atoms.new_array("truth_atom_id", np.array([0, 1], dtype=np.int64))
        frame = make_view_frame(
            "tiny", [0, 0, 1], [0, 1, 0], projection_label="yx"
        )
        view = orient_atoms_for_abtem(atoms, frame, vacuum_angstrom=4.0)
        config = PtychographyConfig(
            condition_name="instrument_model",
            device="cpu",
            potential_sampling_angstrom=0.5,
            potential_slice_thickness_angstrom=2.0,
            scan_step_angstrom=2.0,
            scan_margin_angstrom=1.0,
            detector_max_angle_mrad=30.0,
            frozen_phonon_configs=2,
            thermal_sigma_angstrom={"Au": 0.08, "Ag": 0.09},
            dose_electrons_per_angstrom2=1_000.0,
            probe_aberrations={"C10": 10.0},
            beam_tilt_mrad=(0.1, -0.2),
            partial_coherence_source_sigma_angstrom=0.1,
            scan_position_error_std_angstrom=0.05,
            detector_background_mean_counts=0.1,
            detector_read_noise_std_counts=0.05,
            detector_gain_std_fraction=0.01,
            detector_dead_pixel_fraction=0.01,
            detector_saturation_counts=10_000.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            simulation = simulate_4dstem(
                view,
                config,
                Path(temp_dir) / "instrument.zarr",
                progress=False,
            )
            self.assertEqual(tuple(simulation.diffraction_patterns.shape[:2]), (3, 3))
            effects = simulation.metadata["condition"]["effects"]
            self.assertTrue(all(effects.values()))
            self.assertTrue(
                simulation.metadata["scan_position_model"][
                    "actual_positions_physically_used"
                ]
            )
            self.assertTrue(simulation.metadata["detector_response_model"]["enabled"])
            with np.load(simulation.scan_positions_path, allow_pickle=False) as data:
                nominal = data["nominal_positions_angstrom"]
                actual = data["actual_positions_angstrom"]
                gain = data["detector_gain_map"]
                live = data["detector_live_pixel_mask"]
            self.assertGreater(float(np.linalg.norm(actual - nominal)), 0.0)
            self.assertEqual(gain.shape, tuple(simulation.diffraction_patterns.shape[-2:]))
            self.assertEqual(live.shape, gain.shape)
            self.assertEqual(live.dtype, np.bool_)
            self.assertEqual(
                simulation.metadata["configuration"]["potential_parametrization"],
                "lobato",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
