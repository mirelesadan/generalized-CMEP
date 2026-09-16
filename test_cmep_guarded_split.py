import unittest

import numpy as np

from cmep_abtem_simulation import _split_projection_across_active_slices


class GuardedSplitProjectionTests(unittest.TestCase):
    def test_projection_is_split_only_over_unguarded_slices(self) -> None:
        amplitude = np.array(
            [[0.72, 0.91, 1.00], [1.08, 0.83, 0.65]], dtype=np.float64
        )
        phase = np.array(
            [[-1.10, -0.25, 0.00], [0.35, 1.20, 2.40]], dtype=np.float64
        )
        projection = (amplitude * np.exp(1j * phase)).astype(np.complex64)

        objects, metadata = _split_projection_across_active_slices(
            projection,
            8,
            vacuum_guard_slices=(2, 1),
        )

        self.assertEqual(objects.shape, (8, 2, 3))
        np.testing.assert_array_equal(objects[:2], 1.0 + 0.0j)
        np.testing.assert_array_equal(objects[-1:], 1.0 + 0.0j)
        np.testing.assert_allclose(
            objects[2:7],
            np.repeat(objects[2][np.newaxis], 5, axis=0),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.prod(objects.astype(np.complex128), axis=0),
            projection,
            atol=2e-7,
            rtol=2e-7,
        )
        self.assertEqual(metadata["active_slice_count"], 5)
        self.assertEqual(metadata["active_slice_range_stop_exclusive"], [2, 7])
        self.assertEqual(metadata["principal_root_degree"], 5)

    def test_guards_must_leave_an_active_slice(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one active"):
            _split_projection_across_active_slices(
                np.ones((2, 2), dtype=np.complex64),
                4,
                vacuum_guard_slices=(2, 2),
            )


if __name__ == "__main__":
    unittest.main()
