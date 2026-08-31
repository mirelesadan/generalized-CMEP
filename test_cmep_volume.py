"""Focused numerical tests for the raw-volume preparation workflow."""

from __future__ import annotations

import base64
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cmep_alignment import (
    apply_calibration_preview,
    create_alignment_state,
    directional_scale_affine,
    load_alignment_state,
    preview_calibration,
    rotate_dataset_in_plane,
    save_alignment_state,
    slice_transformed_volume,
    transform_physical_points,
    transformed_geometric_center,
    translate_dataset_in_plane,
    union_depth_range,
)
from cmep_alignment_viewer import AlignmentViewer
from cmep_atom_localization import (
    AtomLocalizationResult,
    load_atom_localization,
    localize_atoms,
)
from cmep_atom_viewer import (
    prepare_cpu_atom_markers,
    prepare_gpu_atom_instances,
    prepare_gpu_atom_markers,
    write_cpu_atom_marker_viewer,
    write_gpu_atom_marker_viewer,
    write_gpu_atom_viewer,
)
from cmep_gpu_viewer import prepare_gpu_instances, write_gpu_volume_viewer
from cmep_likelihood import (
    LIKELIHOOD_SCHEMA,
    LikelihoodMapResult,
    compute_likelihood,
    create_likelihood_map,
    load_likelihood_map,
    make_likelihood_histogram,
)
from cmep_registration import (
    coarse_translation_offsets,
    create_world_grid,
    intensity_weighted_center_of_mass,
    optimize_correlative_alignment,
    print_registration_summary,
    rotation_affine,
    rotation_vector_affine,
    rotation_vector_axis_angle,
    sample_volume_on_grid,
)
from cmep_visualization import select_threshold_points, visualize_volume
from cmep_volume import (
    apply_physical_axis_flips,
    load_volume_input,
    map_stack_to_canonical,
    normalize_slices,
    process_volume,
    resample_stack_depth,
    validate_orientation,
)


HERE = Path(__file__).resolve().parent


class VolumeWorkflowTests(unittest.TestCase):
    def test_real_tiff_shapes_and_normalization(self) -> None:
        expected = {
            "Plan-view_slices_0.39.tif": (25, 620, 334),
            "Cross-section_slices_0.39.tif": (29, 513, 339),
        }
        for filename, shape in expected.items():
            with self.subTest(filename=filename):
                raw, info = load_volume_input(HERE / filename)
                normalized, _ = normalize_slices(raw)
                self.assertEqual(raw.shape, shape)
                self.assertEqual(normalized.shape, shape)
                self.assertEqual(normalized.dtype, np.float32)
                self.assertGreaterEqual(float(normalized.min()), 0.0)
                self.assertLessEqual(float(normalized.max()), 1.0)
                self.assertEqual(info["input_type"], "tiff")

    def test_pathological_slice_normalization(self) -> None:
        stack = np.array(
            [
                [[5.0, 5.0], [5.0, 5.0]],
                [[0.0, 1.0], [2.0, np.nan]],
                [[np.nan, np.inf], [-np.inf, np.nan]],
            ]
        )
        normalized, details = normalize_slices(stack)
        np.testing.assert_array_equal(normalized[0], 0.0)
        np.testing.assert_array_equal(normalized[2], 0.0)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertEqual(details["degenerate_or_nonfinite_only_slices"], [0, 2])

    def test_orientation_mappings_and_canonical_shapes(self) -> None:
        cases = {
            ("plan_view", "yx"): (("z", "y", "x"), (5, 4, 3)),
            ("plan_view", "xy"): (("z", "x", "y"), (4, 5, 3)),
            ("cross_section", "xz"): (("y", "x", "z"), (4, 3, 5)),
            ("cross_section", "zx"): (("y", "z", "x"), (5, 3, 4)),
            ("cross_section", "yz"): (("x", "y", "z"), (3, 4, 5)),
            ("cross_section", "zy"): (("x", "z", "y"), (3, 5, 4)),
        }
        stack = np.zeros((3, 4, 5), dtype=np.float32)
        for (kind, orientation), (labels, expected_shape) in cases.items():
            with self.subTest(kind=kind, orientation=orientation):
                mapping = validate_orientation(orientation, kind)
                self.assertEqual(
                    (mapping["slice"], mapping["row"], mapping["col"]), labels
                )
                canonical, _ = map_stack_to_canonical(stack, mapping)
                self.assertEqual(canonical.shape, expected_shape)

    def test_depth_interpolation_and_endpoint_order(self) -> None:
        stack = np.array([[[10.0]], [[20.0]], [[30.0]]], dtype=np.float32)
        result, coords, info = resample_stack_depth(
            stack, physical_depth=8.0, target_spacing=2.0
        )
        np.testing.assert_allclose(coords, [0.0, 2.0, 4.0, 6.0, 8.0])
        np.testing.assert_allclose(result[:, 0, 0], [30.0, 25.0, 20.0, 15.0, 10.0])
        self.assertEqual(info["original_stack_spacing"], 4.0)
        self.assertEqual(info["effective_resampled_depth_spacing"], 2.0)

    def test_bottom_left_origin_uses_coordinates_not_row_flip(self) -> None:
        source = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        normalized, volume, meta = process_volume(
            source,
            dataset_kind="plan_view",
            orientation="yx",
            pixel_size=2.0,
            physical_depth=4.0,
            length_unit="pm",
            lat_param=390.0,
        )
        self.assertEqual(volume.shape, (5, 4, 3))
        np.testing.assert_array_equal(meta["coordinates"]["x"], [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(meta["coordinates"]["y"], [6, 4, 2, 0])
        np.testing.assert_array_equal(meta["coordinates"]["z"], [0, 2, 4])
        np.testing.assert_array_equal(
            volume[:, :, -1], np.transpose(normalized[0], (1, 0))
        )
        np.testing.assert_array_equal(
            volume[:, :, 0], np.transpose(normalized[-1], (1, 0))
        )

    def test_physical_flip_targets_named_axis(self) -> None:
        volume = np.arange(24).reshape(2, 3, 4)
        flipped, applied = apply_physical_axis_flips(volume, {"y": True})
        np.testing.assert_array_equal(flipped, volume[:, ::-1, :])
        self.assertEqual(applied, {"x": False, "y": True, "z": False})

    def test_one_slice_rules(self) -> None:
        stack = np.ones((1, 2, 2), dtype=np.float32)
        result, coords, info = resample_stack_depth(
            stack, physical_depth=0.0, target_spacing=1.0
        )
        self.assertEqual(result.shape, (1, 2, 2))
        np.testing.assert_array_equal(coords, [0.0])
        self.assertIsNone(info["original_stack_spacing"])
        with self.assertRaisesRegex(ValueError, "one-slice"):
            resample_stack_depth(stack, physical_depth=1.0, target_spacing=1.0)

    def test_threshold_selection_uses_physical_coordinates_and_cap(self) -> None:
        volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 23.0
        meta = {
            "coordinates": {
                "x": np.array([10.0, 20.0]),
                "y": np.array([3.0, 2.0, 1.0]),
                "z": np.array([0.0, 5.0, 10.0, 15.0]),
            }
        }
        low = select_threshold_points(volume, meta, threshold=0.5, max_points=100)
        high = select_threshold_points(volume, meta, threshold=0.9, max_points=100)
        capped = select_threshold_points(volume, meta, threshold=0.0, max_points=5)
        self.assertGreater(low["displayed_count"], high["displayed_count"])
        self.assertTrue(np.all(low["intensity"] >= 0.5))
        self.assertTrue(set(low["x"]).issubset({10.0, 20.0}))
        self.assertEqual(capped["displayed_count"], 5)
        self.assertTrue(capped["visualization_downsampled"])

    def test_plotly_viewer_uses_native_slider_without_widget_bridge(self) -> None:
        volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 23.0
        meta = {
            "dataset_kind": "plan_view",
            "length_unit": "pm",
            "coordinates": {
                "x": np.array([0.0, 2.0]),
                "y": np.array([4.0, 2.0, 0.0]),
                "z": np.array([0.0, 3.0, 6.0, 9.0]),
            },
        }
        figure = visualize_volume(
            volume,
            meta,
            color="red",
            threshold=0.7,
            threshold_min=0.5,
            threshold_max=0.9,
            threshold_step=0.2,
            max_voxels=10,
            minimum_voxel_alpha=0.3,
            background_color="black",
            figure_size=700,
        )
        self.assertEqual(type(figure).__name__, "Figure")
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(
            [step.label for step in figure.layout.sliders[0].steps],
            ["0.50", "0.70", "0.90"],
        )
        self.assertFalse(figure.data[0].visible)
        self.assertTrue(figure.data[1].visible)
        self.assertTrue(figure.data[2].visible)
        self.assertEqual(figure.data[1].type, "mesh3d")
        self.assertEqual(len(figure.data[1].x) % 8, 0)
        self.assertEqual(len(figure.data[1].i) % 12, 0)
        self.assertEqual(len(figure.data[1].intensity), len(figure.data[1].x))
        self.assertEqual(
            list(figure.data[1].colorscale),
            [(0.0, "rgba(255,0,0,0.3)"), (1.0, "rgba(255,0,0,1)")],
        )
        self.assertAlmostEqual(figure.data[1].cmin, 0.7)
        self.assertEqual(figure.data[1].cmax, 1.0)
        self.assertEqual(
            list(figure.layout.sliders[0].steps[2].args[0]["cmin"]),
            [0.9, 0.9, 0.9],
        )
        self.assertEqual(figure.layout.scene.aspectmode, "data")
        self.assertEqual(figure.layout.scene.dragmode, "orbit")
        self.assertEqual(figure.layout.scene.bgcolor, "black")
        self.assertFalse(figure.layout.scene.xaxis.showbackground)
        self.assertFalse(figure.layout.scene.xaxis.showgrid)
        self.assertTrue(figure.layout.scene.xaxis.showline)
        self.assertEqual(figure.layout.scene.xaxis.linecolor, "rgb(255,255,255)")
        self.assertEqual(figure.layout.font.color, "rgb(255,255,255)")
        self.assertEqual(figure.layout.width, 700)
        self.assertEqual(figure.layout.height, 700)

    def test_threshold_count_controls_slider_and_single_threshold_mode(self) -> None:
        volume = np.linspace(0.0, 1.0, 11, dtype=np.float32).reshape(11, 1, 1)
        meta = {
            "coordinates": {
                "x": np.arange(11, dtype=np.float64),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
            }
        }
        slider_figure = visualize_volume(
            volume,
            meta,
            threshold=0.5,
            threshold_min=0.0,
            threshold_max=1.0,
            threshold_count=5,
            max_voxels=11,
        )
        self.assertEqual(
            [step.label for step in slider_figure.layout.sliders[0].steps],
            ["0.00", "0.25", "0.50", "0.75", "1.00"],
        )

        single_figure = visualize_volume(
            volume,
            meta,
            threshold=0.6,
            threshold_count=1,
            max_voxels=4,
        )
        self.assertEqual(len(single_figure.data), 1)
        self.assertEqual(len(single_figure.layout.sliders), 0)
        self.assertTrue(single_figure.data[0].visible)
        self.assertAlmostEqual(single_figure.data[0].cmin, 0.6)
        self.assertEqual(len(single_figure.data[0].x) // 8, 4)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            visualize_volume(volume, meta, threshold=0.5, threshold_count=0)
        with self.assertRaisesRegex(ValueError, "either"):
            visualize_volume(
                volume,
                meta,
                threshold=0.5,
                threshold_count=3,
                threshold_values=[0.0, 0.5, 1.0],
            )

    def test_background_inversion_and_alpha_validation(self) -> None:
        volume = np.ones((1, 1, 1), dtype=np.float32)
        meta = {
            "coordinates": {
                "x": np.array([0.0]),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
            }
        }
        figure = visualize_volume(
            volume,
            meta,
            threshold=0.5,
            threshold_min=0.5,
            threshold_max=1.0,
            threshold_step=0.5,
            max_voxels=1,
            background_color="#123456",
        )
        self.assertEqual(figure.layout.font.color, "rgb(237,203,169)")
        with self.assertRaisesRegex(ValueError, "minimum_voxel_alpha"):
            visualize_volume(
                volume,
                meta,
                threshold=0.5,
                threshold_min=0.5,
                threshold_max=1.0,
                minimum_voxel_alpha=1.1,
            )

    def test_gpu_instancing_and_self_contained_html(self) -> None:
        volume = np.array([0.4, 0.5, 0.75, 1.0], dtype=np.float32).reshape(
            4, 1, 1
        )
        meta = {
            "dataset_kind": "plan_view",
            "length_unit": "nm",
            "coordinates": {
                "x": np.array([0.0, 2.0, 4.0, 6.0]),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
            },
        }
        instances = prepare_gpu_instances(
            volume,
            meta,
            threshold=0.5,
            color="lightcoral",
            minimum_voxel_alpha=0.3,
            background_color="black",
            max_voxels=None,
            voxel_scale=0.5,
        )
        positions = np.frombuffer(
            base64.b64decode(instances["positions_b64"]), dtype="<f4"
        ).reshape(-1, 3)
        alphas = np.frombuffer(
            base64.b64decode(instances["alphas_b64"]), dtype="<f4"
        )
        self.assertEqual(instances["instance_count"], 3)
        self.assertEqual(instances["threshold_count"], 3)
        np.testing.assert_allclose(positions[:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(alphas, [0.3, 0.65, 1.0])
        np.testing.assert_allclose(instances["voxel_size"], [1.0, 0.5, 0.5])

        colored = prepare_gpu_instances(
            volume,
            meta,
            threshold=0.5,
            color="white",
            colormap="magma",
            minimum_voxel_alpha=0.3,
            background_color="black",
            max_voxels=None,
            voxel_scale=0.5,
        )
        color_lut = np.frombuffer(
            base64.b64decode(colored["colormap_rgb_b64"]), dtype=np.uint8
        ).reshape(256, 3)
        mapped_intensities = np.frombuffer(
            base64.b64decode(colored["intensities_b64"]), dtype="<f4"
        )
        self.assertTrue(colored["use_colormap"])
        self.assertEqual(colored["colormap_name"], "magma")
        np.testing.assert_array_equal(color_lut[0], [0, 0, 4])
        np.testing.assert_array_equal(color_lut[-1], [252, 253, 191])
        np.testing.assert_allclose(mapped_intensities, [0.5, 0.75, 1.0])

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = write_gpu_volume_viewer(
                volume,
                meta,
                output_path=Path(directory) / "viewer.html",
                threshold=0.5,
                color="lightcoral",
                colormap="magma",
                minimum_voxel_alpha=0.3,
                background_color="black",
                max_voxels=2,
                open_browser=False,
            )
            html = output.read_text(encoding="utf-8")
            self.assertIn("InstancedBufferGeometry", html)
            self.assertIn("instanceIntensity", html)
            self.assertIn("THREE.DataTexture", html)
            self.assertIn("powerPreference:\"high-performance\"", html)
            self.assertIn('"instance_count":2', html)
            self.assertIn('"colormap_name":"magma"', html)
            self.assertIn('"foreground_css":"rgb(255,255,255)"', html)

    def test_likelihood_formula_mask_and_round_trip(self) -> None:
        plan = np.array([0.1, 0.55, 1.0, 0.8], dtype=np.float32).reshape(2, 2, 1)
        cross = np.array([1.0, 0.91, 0.1, np.nan], dtype=np.float32).reshape(
            2, 2, 1
        )
        supplied_mask = np.ones(plan.shape, dtype=bool)
        plan_before = plan.copy()
        cross_before = cross.copy()

        likelihood, valid = compute_likelihood(
            plan,
            cross,
            supplied_mask,
            floor_plan=0.1,
            floor_cross=0.1,
            chunk_voxels=2,
        )
        self.assertEqual(likelihood.dtype, np.float32)
        np.testing.assert_array_equal(valid.reshape(-1), [True, True, True, False])
        np.testing.assert_allclose(
            likelihood.reshape(-1)[:3], [0.0, np.sqrt(0.45), 0.0], rtol=1e-6
        )
        self.assertTrue(np.isnan(likelihood.reshape(-1)[3]))
        np.testing.assert_array_equal(plan, plan_before)
        np.testing.assert_array_equal(cross, cross_before)

        aligned = {
            "coordinates": {
                "x": np.array([1.0, 1.5]),
                "y": np.array([2.0, 2.5]),
                "z": np.array([3.0]),
            },
            "volume_plan": plan,
            "volume_cross": cross,
            "valid_overlap_mask": supplied_mask,
            "length_unit": "nm",
        }
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = Path(directory) / "likelihood.npz"
            created = create_likelihood_map(
                aligned,
                floor_plan=0.1,
                floor_cross=0.1,
                output_path=output,
                chunk_voxels=2,
            )
            loaded = load_likelihood_map(output)
            self.assertTrue(created.map_path.is_file())
            self.assertTrue(created.metadata_path.is_file())
            self.assertEqual(loaded.metadata["schema_version"], LIKELIHOOD_SCHEMA)
            self.assertEqual(loaded.metadata["length_unit"], "nm")
            np.testing.assert_allclose(
                loaded.likelihood, created.likelihood, equal_nan=True
            )
            np.testing.assert_array_equal(
                loaded.valid_overlap_mask, created.valid_overlap_mask
            )

    def test_likelihood_input_validation(self) -> None:
        volume = np.ones((2, 2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "less than 1"):
            compute_likelihood(volume, volume, floor_plan=1.0)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            compute_likelihood(volume, np.ones((2, 2, 3), dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "Boolean"):
            compute_likelihood(volume, volume, np.ones(volume.shape, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Unknown Plotly colormap"):
            prepare_gpu_instances(
                volume,
                {
                    "coordinates": {
                        axis: np.arange(2, dtype=np.float64) for axis in "xyz"
                    }
                },
                threshold=0.5,
                color="white",
                colormap="definitely-not-a-colormap",
                minimum_voxel_alpha=0.3,
                background_color="black",
                max_voxels=None,
                voxel_scale=1.0,
            )

    def test_likelihood_histogram_is_thresholded_and_prebinned(self) -> None:
        likelihood = np.array(
            [0.1, 0.5, 0.6, 0.9, 1.0, np.nan], dtype=np.float32
        ).reshape(2, 3, 1)
        figure = make_likelihood_histogram(
            likelihood,
            threshold=0.5,
            bins=2,
            colormap="magma",
            background_color="black",
        )
        np.testing.assert_array_equal(figure.data[0].y, [2, 2])
        np.testing.assert_allclose(figure.data[0].x, [0.625, 0.875])
        self.assertIn("4 voxels", figure.layout.title.text)
        self.assertEqual(figure.layout.paper_bgcolor, "black")
        self.assertEqual(figure.layout.font.color, "rgb(255,255,255)")

    def test_subvoxel_atom_localization_and_saved_outputs(self) -> None:
        coordinates = np.linspace(0.0, 1.0, 21)
        x, y, z = np.meshgrid(
            coordinates, coordinates, coordinates, indexing="ij"
        )
        expected_center = np.array([0.483, 0.527, 0.491])
        plan = (
            0.08
            + 0.92
            * np.exp(
                -0.5
                * (
                    ((x - expected_center[0]) / 0.08) ** 2
                    + ((y - expected_center[1]) / 0.08) ** 2
                    + ((z - expected_center[2]) / 0.14) ** 2
                )
            )
        ).astype(np.float32)
        cross = (
            0.08
            + 0.92
            * np.exp(
                -0.5
                * (
                    ((x - expected_center[0]) / 0.08) ** 2
                    + ((y - expected_center[1]) / 0.14) ** 2
                    + ((z - expected_center[2]) / 0.08) ** 2
                )
            )
        ).astype(np.float32)
        valid = np.ones(plan.shape, dtype=bool)
        likelihood, overlap = compute_likelihood(
            plan,
            cross,
            valid,
            floor_plan=0.08,
            floor_cross=0.08,
        )
        coordinate_metadata = {axis: coordinates for axis in "xyz"}
        score_result = LikelihoodMapResult(
            likelihood,
            overlap,
            {
                "coordinates": coordinate_metadata,
                "length_unit": "nm",
                "floor_plan": 0.08,
                "floor_cross": 0.08,
            },
        )
        # A notebook reload creates a distinct class object while retaining
        # existing result instances. This stand-in reproduces that interface.
        reload_retained_score_result = SimpleNamespace(
            likelihood=score_result.likelihood,
            valid_overlap_mask=score_result.valid_overlap_mask,
            metadata=score_result.metadata,
            coordinates=score_result.coordinates,
            map_path=None,
        )
        aligned = {
            "coordinates": coordinate_metadata,
            "volume_plan": plan,
            "volume_cross": cross,
            "valid_overlap_mask": valid,
            "length_unit": "nm",
        }
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            prefix = Path(directory) / "localized_atoms"
            result = localize_atoms(
                reload_retained_score_result,
                aligned,
                lat_param=0.4,
                score_threshold=0.25,
                minimum_atom_separation=0.2,
                fit_radius=0.18,
                maximum_subvoxel_shift=0.08,
                initial_step=0.0125,
                position_tolerance=0.0015625,
                output_prefix=prefix,
                progress=False,
            )
            self.assertEqual(len(result.positions_xyz), 1)
            np.testing.assert_allclose(
                result.positions_xyz[0], expected_center, atol=0.006
            )
            self.assertEqual(result.atoms["status"][0], "refined")
            self.assertTrue(result.npz_path.is_file())
            self.assertTrue(result.csv_path.is_file())
            self.assertTrue(result.metadata_path.is_file())
            loaded = load_atom_localization(result.npz_path)
            np.testing.assert_allclose(loaded.positions_xyz, result.positions_xyz)
            self.assertEqual(
                loaded.metadata["source_likelihood_score_map_sha256"], None
            )

    def test_gpu_atom_spheres_have_no_display_cap(self) -> None:
        count = 12
        positions = np.column_stack(
            (
                np.linspace(0.0, 1.0, count),
                np.linspace(1.0, 2.0, count),
                np.linspace(2.0, 3.0, count),
            )
        )
        scores = np.linspace(0.4, 1.0, count, dtype=np.float32)
        result = AtomLocalizationResult(
            atoms={"position_xyz": positions, "likelihood_score": scores},
            candidates={"seed_score": scores},
            metadata={
                "length_unit": "nm",
                "score_threshold": 0.4,
                "coordinate_ranges": {
                    "x": [0.0, 1.0],
                    "y": [1.0, 2.0],
                    "z": [2.0, 3.0],
                },
            },
        )
        instances = prepare_gpu_atom_instances(
            result,
            sphere_radius=0.05,
            score_threshold=0.5,
            color="white",
            colormap="magma",
            minimum_sphere_alpha=0.3,
            background_color="black",
        )
        self.assertEqual(instances["localized_count"], count)
        self.assertEqual(instances["instance_count"], 10)
        self.assertEqual(instances["sphere_radius"], 0.05)
        self.assertEqual(instances["foreground_css"], "rgb(255,255,255)")

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = write_gpu_atom_viewer(
                result,
                output_path=Path(directory) / "atoms.html",
                sphere_radius=0.05,
                score_threshold=0.5,
                colormap="magma",
                open_browser=False,
            )
            html = output.read_text(encoding="utf-8")
            self.assertIn("THREE.SphereGeometry", html)
            self.assertIn("InstancedBufferGeometry", html)
            self.assertIn("Localized Atomic Centers", html)
            self.assertIn('"instance_count":10', html)
            self.assertIn("dataset.renderedInstances", html)

        cpu_markers = prepare_cpu_atom_markers(
            result,
            marker_size=4.5,
            score_threshold=0.5,
            color="white",
            colormap="magma",
            minimum_marker_alpha=0.3,
            background_color="black",
            initial_view_padding=0.08,
            initial_zoom_factor=0.78,
        )
        self.assertEqual(cpu_markers["localized_count"], count)
        self.assertEqual(cpu_markers["marker_count"], 10)
        self.assertEqual(cpu_markers["marker_size"], 4.5)

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = write_cpu_atom_marker_viewer(
                result,
                output_path=Path(directory) / "atom_markers.html",
                marker_size=4.5,
                score_threshold=0.5,
                colormap="magma",
                open_browser=False,
            )
            html = output.read_text(encoding="utf-8")
            self.assertIn('getContext("2d"', html)
            self.assertIn('"marker_count":10', html)
            self.assertIn('"marker_size":4.5', html)
            self.assertIn('canvas.dataset.renderer="canvas2d-cpu"', html)
            self.assertNotIn("WebGLRenderingContext", html)

        gpu_markers = prepare_gpu_atom_markers(
            result,
            marker_size=4.5,
            score_threshold=0.5,
            color="white",
            colormap="magma",
            minimum_marker_alpha=0.3,
            background_color="black",
            initial_view_padding=0.08,
            initial_zoom_factor=0.78,
        )
        self.assertEqual(gpu_markers["localized_count"], count)
        self.assertEqual(gpu_markers["marker_count"], 10)
        self.assertEqual(gpu_markers["instance_count"], 10)
        self.assertEqual(gpu_markers["render_mode"], "markers")

        reload_retained_result = SimpleNamespace(
            positions_xyz=result.positions_xyz,
            scores=result.scores,
            metadata=result.metadata,
        )
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = write_gpu_atom_marker_viewer(
                reload_retained_result,
                output_path=Path(directory) / "atom_gpu_markers.html",
                marker_size=4.5,
                score_threshold=0.5,
                colormap="magma",
                open_browser=False,
            )
            html = output.read_text(encoding="utf-8")
            self.assertIn("THREE.Points", html)
            self.assertIn("gl_PointSize", html)
            self.assertIn('"render_mode":"markers"', html)
            self.assertIn('"marker_count":10', html)
            self.assertIn("dataset.renderedMarkers", html)

    def test_directional_scale_preserves_perpendicular_direction(self) -> None:
        point1 = np.array([1.0, 2.0, 3.0])
        point2 = point1 + np.array([3.0, 4.0, 0.0])
        operation, details = directional_scale_affine(point1, point2, 10.0)
        transformed1 = operation @ np.append(point1, 1.0)
        transformed2 = operation @ np.append(point2, 1.0)
        perpendicular = np.array([-4.0, 3.0, 0.0])

        np.testing.assert_allclose(transformed1[:3], point1)
        self.assertAlmostEqual(np.linalg.norm(transformed2[:3] - transformed1[:3]), 10.0)
        np.testing.assert_allclose(operation[:3, :3] @ perpendicular, perpendicular)
        self.assertAlmostEqual(details["scale_factor"], 2.0)

    def test_alignment_preview_translation_and_state_round_trip(self) -> None:
        state = create_alignment_state("nm")
        points = {
            "plan": [[0.0, 0.0], [2.0, 0.0]],
            "cross": [[1.0, 1.0], [1.0, 3.0]],
        }
        preview = preview_calibration(
            state,
            plane="yx",
            depth=0.5,
            target_distance=4.0,
            points=points,
        )
        calibrated = apply_calibration_preview(state, preview)
        self.assertEqual(len(calibrated["calibrations"]), 1)
        self.assertAlmostEqual(
            preview["relative_alignment"]["rotation_degrees"], -90.0
        )
        plan_endpoints = transform_physical_points(
            preview["points_3d"]["plan"], calibrated["transforms"]["plan"]
        )
        cross_endpoints = transform_physical_points(
            preview["points_3d"]["cross"], calibrated["transforms"]["cross"]
        )
        np.testing.assert_allclose(cross_endpoints, plan_endpoints, atol=1e-12)
        self.assertAlmostEqual(
            np.linalg.norm(plan_endpoints[1] - plan_endpoints[0]), 4.0
        )
        np.testing.assert_allclose(
            preview["relative_alignment"]["endpoint_residuals"], [0.0, 0.0], atol=1e-12
        )

        translated = translate_dataset_in_plane(
            calibrated,
            dataset="plan",
            plane="yx",
            horizontal_delta=1.25,
            vertical_delta=-0.75,
            depth=0.5,
        )
        self.assertAlmostEqual(translated["transforms"]["plan"][0][3], 1.25)
        self.assertAlmostEqual(translated["transforms"]["plan"][1][3], -0.75)

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "state.json"
            current, snapshot = save_alignment_state(translated, path)
            loaded = load_alignment_state(current, length_unit="nm")
            self.assertTrue(snapshot.is_file())
            np.testing.assert_allclose(
                loaded["transforms"]["plan"], translated["transforms"]["plan"]
            )

    def test_in_plane_rotation_uses_transformed_geometric_center(self) -> None:
        metadata = {
            "coordinates": {
                "x": np.array([0.0, 1.0, 2.0]),
                "y": np.array([0.0, 2.0, 4.0]),
                "z": np.array([0.0, 3.0, 6.0]),
            }
        }
        state = create_alignment_state("nm")
        state["transforms"]["plan"][0][3] = 5.0
        state["transforms"]["plan"][1][3] = -1.0
        state["transforms"]["plan"][2][3] = 2.0
        center_before = transformed_geometric_center(
            metadata, state["transforms"]["plan"]
        )

        rotated = rotate_dataset_in_plane(
            state,
            metadata,
            dataset="plan",
            plane="yx",
            angle_degrees=90.0,
            depth=5.0,
        )
        center_after = transformed_geometric_center(
            metadata, rotated["transforms"]["plan"]
        )
        np.testing.assert_allclose(center_after, center_before, atol=1e-12)
        source_center = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
        transformed = transform_physical_points(
            source_center, rotated["transforms"]["plan"]
        )
        np.testing.assert_allclose(transformed[0], center_before, atol=1e-12)
        np.testing.assert_allclose(
            transformed[1], center_before + np.array([0.0, 1.0, 0.0]), atol=1e-12
        )
        np.testing.assert_allclose(
            rotated["transforms"]["cross"], np.eye(4), atol=1e-12
        )

    def test_transformed_physical_slice_and_union_depth(self) -> None:
        coordinates = {axis: np.arange(3, dtype=np.float64) for axis in "xyz"}
        volume = np.zeros((3, 3, 3), dtype=np.float32)
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    volume[x, y, z] = (x + 3 * y + 9 * z) / 26.0
        meta = {"coordinates": coordinates}
        identity = np.eye(4)
        sliced = slice_transformed_volume(
            volume,
            meta,
            identity,
            plane="yx",
            depth=1.0,
            threshold_floor=0.0,
        )
        self.assertEqual(sliced["displayed_point_count"], 9)
        self.assertEqual(len(sliced["hull"]), 4)
        np.testing.assert_allclose(
            np.sort(sliced["intensity"]), np.sort(volume[:, :, 1].reshape(-1))
        )

        shifted = create_alignment_state("nm")
        shifted["transforms"]["cross"][2][3] = 5.0
        depth_range = union_depth_range(
            {"plan": meta, "cross": meta}, shifted, "yx"
        )
        np.testing.assert_allclose(depth_range, [0.0, 7.0])

    def test_alignment_viewer_configuration_and_html(self) -> None:
        volume = np.ones((2, 2, 2), dtype=np.float32)
        meta = {
            "length_unit": "nm",
            "coordinates": {
                "x": np.array([0.0, 1.0]),
                "y": np.array([0.0, 1.0]),
                "z": np.array([0.0, 1.0]),
            },
        }
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            viewer = AlignmentViewer(
                {"plan": volume, "cross": volume},
                {"plan": meta, "cross": meta},
                state=create_alignment_state("nm"),
                state_path=Path(directory) / "alignment.json",
                colors={"plan": "red", "cross": "lightblue"},
                threshold_initial=0.5,
                threshold_min=0.2,
                threshold_max=0.8,
                threshold_count=1,
                depth_positions=50,
                minimum_voxel_alpha=0.3,
                background_color="black",
                voxel_scale=1.0,
            )
            html = viewer._html().decode("utf-8")
            self.assertIn("Calibrate Direction", html)
            self.assertIn("Save State", html)
            self.assertIn('id="rotation"', html)
            self.assertIn('/api/rotate', html)
            self.assertIn('"rotation_step_degrees":0.25', html)
            self.assertIn('data-plane="yx"', html)
            self.assertIn("InstancedBufferGeometry", html)
            self.assertEqual(viewer.depth_positions, 50)

    def test_registration_geometry_helpers(self) -> None:
        offsets_xyz = coarse_translation_offsets("zyx", radius=1, step=2.0)
        offsets_yx = coarse_translation_offsets("yx", radius=1, step=2.0)
        self.assertEqual(len(offsets_xyz), 27)
        self.assertEqual(len(offsets_yx), 9)
        np.testing.assert_array_equal(offsets_xyz[0], [0.0, 0.0, 0.0])
        self.assertEqual(
            len({tuple(offset.tolist()) for offset in offsets_xyz}), 27
        )

        rotation = rotation_affine("z", 90.0, center=[1.0, 1.0, 0.0])
        point = rotation @ np.array([2.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(point[:3], [1.0, 2.0, 0.0], atol=1e-12)

        vector_rotation = rotation_vector_affine(
            [1.0, 2.0, 2.0], center=[1.0, 1.0, 0.0]
        )
        expected_rotation = rotation_affine(
            [1.0, 2.0, 2.0], 3.0, center=[1.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(vector_rotation, expected_rotation, atol=1e-12)
        angle, axis = rotation_vector_axis_angle([1.0, 2.0, 2.0])
        self.assertAlmostEqual(angle, 3.0)
        np.testing.assert_allclose(axis, [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])

    def test_world_grid_sampling_respects_physical_transform(self) -> None:
        volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3) / 26.0
        meta = {
            "coordinates": {
                "x": np.arange(3, dtype=np.float64),
                "y": np.arange(3, dtype=np.float64),
                "z": np.arange(3, dtype=np.float64),
            }
        }
        transform = np.eye(4)
        transform[0, 3] = 1.0
        grid = create_world_grid([[1, 3], [0, 2], [0, 2]], spacing=1.0)
        sampled = sample_volume_on_grid(
            volume, meta, transform, grid, chunk_voxels=9
        )
        np.testing.assert_allclose(sampled, volume)

    def test_correlative_optimizer_recovers_mixed_axis_rotation_vector(self) -> None:
        coordinates = np.linspace(-4.0, 4.0, 13)
        x, y, z = np.meshgrid(
            coordinates, coordinates, coordinates, indexing="ij"
        )
        volume = (
            np.exp(
                -(
                    (x + 1.7) ** 2 / 0.45
                    + (y - 1.1) ** 2 / 0.75
                    + (z + 0.4) ** 2 / 0.35
                )
            )
            + 0.8
            * np.exp(
                -(
                    (x - 1.4) ** 2 / 0.8
                    + (y + 1.6) ** 2 / 0.4
                    + (z - 1.2) ** 2 / 0.6
                )
            )
            + 0.55
            * np.exp(
                -(
                    (x - 0.2) ** 2 / 0.3
                    + (y - 0.1) ** 2 / 0.5
                    + (z + 2.0) ** 2 / 0.7
                )
            )
        ).astype(np.float32)
        volume /= volume.max()
        metadata = {
            "length_unit": "nm",
            "lat_param": 1.0,
            "coordinates": {axis: coordinates for axis in "xyz"},
        }
        center = intensity_weighted_center_of_mass(
            volume, metadata, intensity_floor=0.03
        )
        imposed_rotation = np.array([4.0, -3.0, 2.0])
        state = create_alignment_state("nm")
        state["transforms"]["cross"] = rotation_vector_affine(
            imposed_rotation, center
        ).tolist()

        result = optimize_correlative_alignment(
            volume,
            metadata,
            volume,
            metadata,
            alignment_state=state,
            moving_dataset="cross",
            coarse_search_axes="",
            coarse_search_radius=0,
            coarse_translation_step=1.0,
            coarse_optimization_spacing=2.0 / 3.0,
            refinement_optimization_spacing=2.0 / 3.0,
            fine_translation_limit=0.0,
            fine_translation_initial_step=0.1,
            fine_translation_tolerance=0.1,
            fine_rotation_limit_degrees_xyz=(6.0, 6.0, 6.0),
            fine_rotation_initial_step_degrees_xyz=(2.0, 2.0, 2.0),
            fine_rotation_tolerance_degrees_xyz=(0.5, 0.5, 0.5),
            intensity_floor=0.03,
            chunk_voxels=10_000,
            native_validation=False,
            materialize_aligned_grid=False,
            progress=False,
        )
        optimization = result.summary["optimization"]
        np.testing.assert_allclose(
            optimization["optimized_rotation_vector_degrees_xyz"],
            -imposed_rotation,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            optimization["optimized_rotation_angle_degrees"],
            np.linalg.norm(imposed_rotation),
        )
        np.testing.assert_allclose(
            optimization["optimized_rotation_axis_vector_xyz"],
            -imposed_rotation / np.linalg.norm(imposed_rotation),
        )
        self.assertGreater(
            optimization["optimized_score"], optimization["initial_score"]
        )
        self.assertAlmostEqual(optimization["optimized_score"], 1.0, places=12)
        printed = io.StringIO()
        with redirect_stdout(printed):
            print_registration_summary(result)
        self.assertIn("rotation vector xyz", printed.getvalue())
        self.assertIn("rotation angle", printed.getvalue())

    def test_correlative_optimizer_recovers_known_coarse_translation(self) -> None:
        coordinates = np.arange(9, dtype=np.float64)
        x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
        volume = (
            np.exp(-((x - 2.0) ** 2 + (y - 5.0) ** 2 + (z - 4.0) ** 2) / 1.2)
            + 0.7
            * np.exp(-((x - 6.0) ** 2 + (y - 2.0) ** 2 + (z - 6.0) ** 2) / 0.8)
        ).astype(np.float32)
        volume /= volume.max()
        meta = {
            "length_unit": "nm",
            "lat_param": 1.0,
            "coordinates": {axis: coordinates for axis in "xyz"},
        }
        state = create_alignment_state("nm")
        state["transforms"]["cross"][0][3] = 1.0

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            result = optimize_correlative_alignment(
                volume,
                meta,
                volume,
                meta,
                alignment_state=state,
                moving_dataset="cross",
                coarse_search_axes="x",
                coarse_search_radius=1,
                coarse_translation_step=1.0,
                coarse_optimization_spacing=1.0,
                refinement_optimization_spacing=1.0,
                fine_translation_limit=0.25,
                fine_translation_initial_step=0.125,
                fine_translation_tolerance=0.125,
                fine_rotation_limit_degrees_xyz=(0.0, 0.0, 0.0),
                fine_rotation_initial_step_degrees_xyz=(0.5, 0.5, 0.5),
                fine_rotation_tolerance_degrees_xyz=(0.5, 0.5, 0.5),
                intensity_floor=0.05,
                chunk_voxels=1_000,
                native_validation=False,
                materialize_aligned_grid=True,
                result_state_path=Path(directory) / "optimized.json",
                aligned_grid_path=Path(directory) / "aligned.npz",
                progress=False,
            )
            optimization = result.summary["optimization"]
            np.testing.assert_allclose(
                optimization["best_coarse_offset_xyz"], [-1.0, 0.0, 0.0]
            )
            self.assertGreater(
                optimization["optimized_score"], optimization["initial_score"]
            )
            self.assertTrue(optimization["exact_refinement_score_improved"])
            self.assertGreater(
                optimization["exact_refinement_optimized_score"],
                optimization["exact_refinement_initial_score"],
            )
            self.assertAlmostEqual(
                result.optimized_transforms["cross"][0, 3], 0.0, places=6
            )
            self.assertTrue(result.state_path.is_file())
            self.assertTrue(result.aligned_grid_path.is_file())
            self.assertEqual(
                result.aligned_grid["volume_plan"].shape,
                result.aligned_grid["volume_cross"].shape,
            )

    def test_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_orientation("xx", "plan_view")
        with self.assertRaisesRegex(ValueError, "positive"):
            resample_stack_depth(
                np.ones((2, 2, 2), dtype=np.float32),
                physical_depth=0.0,
                target_spacing=1.0,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            load_volume_input(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
