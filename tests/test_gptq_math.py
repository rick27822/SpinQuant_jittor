"""CPU regressions for GPTQ statistics, factor order, and explicit selection."""
import contextlib
import io
import pickle
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_spinquant.quant.gptq_math import HessianAccumulator, inverse_hessian_factor, GPTQQuantizationParameters
from jittor_spinquant.arguments import process_args_ptq, validate_rotation_training


class HessianTests(unittest.TestCase):
    def test_compact_parameters_preserve_original_column_mapping_after_serialization(self):
        parameters = GPTQQuantizationParameters(4, True, np.array([[0.1, 0.2, 0.3]]),
                                               np.zeros((1, 3)), np.array([2, 0, 2, 1]),
                                               {"act_order": True})
        restored = pickle.loads(pickle.dumps(parameters))
        scales, zeros = restored.expanded_parameters()
        np.testing.assert_allclose(scales, [[0.3, 0.1, 0.3, 0.2]])
        np.testing.assert_array_equal(zeros, np.zeros((1, 4)))
        self.assertEqual(restored.scale.shape, (1, 3))

    def test_every_batch_contributes_and_partition_does_not_change_statistics(self):
        first = np.array([[1, 2, 0], [0, 1, 3]], dtype=float)
        second = np.array([[4, 0, 2]], dtype=float)
        split = HessianAccumulator(3)
        split.add_batch(first[None])
        split.add_batch(second)
        whole = HessianAccumulator(3)
        whole.add_batch(np.concatenate([first, second]))
        np.testing.assert_allclose(split.hessian(), whole.hessian(), atol=1e-14)
        self.assertEqual((split.batches, split.tokens), (2, 3))
        self.assertGreater(np.max(np.abs(split.hessian() - 2 * second.T @ second)), 1)

    def test_empty_nonfinite_or_wrong_shape_calibration_rejected(self):
        acc = HessianAccumulator(3)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            acc.hessian()
        for values in (np.ones((2, 4)), np.zeros((0, 3)), np.full((2, 3), np.nan)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                acc.add_batch(values)

    def test_cholesky_factor_multiplies_in_the_correct_order(self):
        hessian = np.array([[4.0, 1.7, -0.2], [1.7, 3.0, 0.8], [-0.2, 0.8, 2.0]])
        upper, permutation, dead = inverse_hessian_factor(hessian, percdamp=0)
        np.testing.assert_allclose(upper.T @ upper, np.linalg.inv(hessian), atol=1e-12)
        np.testing.assert_array_equal(permutation, np.arange(3))
        self.assertFalse(np.any(dead))
        lower_inverse = np.linalg.inv(np.linalg.cholesky(hessian))
        self.assertGreater(np.max(np.abs(lower_inverse @ lower_inverse.T - np.linalg.inv(hessian))), 0.01)

    def test_act_order_and_damping_are_applied_to_the_same_matrix(self):
        hessian = np.array([[2.0, 0.3, 0.1], [0.3, 7.0, 0.4], [0.1, 0.4, 4.0]])
        upper, permutation, _ = inverse_hessian_factor(hessian, percdamp=0.03, actorder=True)
        np.testing.assert_array_equal(permutation, [1, 2, 0])
        damped = hessian[np.ix_(permutation, permutation)] + np.eye(3) * 0.03 * np.diag(hessian).mean()
        np.testing.assert_allclose(upper.T @ upper, np.linalg.inv(damped), atol=1e-12)

    def test_dead_columns_and_rank_deficiency(self):
        hessian = np.array([[2.0, 0, 2.0], [0, 0, 0], [2.0, 0, 2.0]])
        upper, _, dead = inverse_hessian_factor(hessian, percdamp=0.01)
        np.testing.assert_array_equal(dead, [False, True, False])
        self.assertTrue(np.all(np.isfinite(upper)))
        with self.assertRaisesRegex(ValueError, "positive definite"):
            inverse_hessian_factor(hessian, percdamp=0)


class SelectionTests(unittest.TestCase):
    def parse(self, *flags):
        return process_args_ptq(["--input_model", "unused", *flags])[2]

    def test_rtn_default_and_gptq_explicit_selection(self):
        self.assertEqual(self.parse().w_quant_method, "rtn")
        self.assertTrue(self.parse("--w_rtn").w_rtn)
        args = self.parse("--w_quant_method", "gptq", "--nsamples", "3", "--act_order")
        self.assertFalse(args.w_rtn)
        self.assertEqual(args.nsamples, 3)
        self.assertTrue(args.act_order)

    def test_conflicting_flags_and_invalid_gptq_options_fail(self):
        cases = (("--w_quant_method", "gptq", "--w_rtn"), ("--nsamples", "0"),
                 ("--gptq_blocksize", "0"), ("--percdamp", "nan"),
                 ("--percdamp", "-0.1"), ("--static_groups",))
        for flags in cases:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse(*flags)

    def test_gptq_cannot_silently_replace_rotation_training_ste(self):
        args = self.parse("--w_quant_method", "gptq", "--w_bits", "4")
        with self.assertRaisesRegex(ValueError, "post-training"):
            validate_rotation_training(args)


if __name__ == "__main__":
    unittest.main()
