"""Numerical boundaries and weighting rules used in manuscript reproduction."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("reproduce_paper", Path(__file__).resolve().parents[1] / "scripts/reproduce_paper.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ReportingTests(unittest.TestCase):
    def test_inverse_interpolation_and_censoring(self):
        reference = np.log2(report.BUDGETS) / 10 + .2
        self.assertAlmostEqual(report.inverse(.25, reference), np.sqrt(2))
        self.assertAlmostEqual(report.inverse(.25, reference, "linear"), 1.5)
        self.assertEqual(report.inverse(reference[0], reference), 1)
        self.assertAlmostEqual(report.inverse(reference[-1], reference), 5)
        self.assertTrue(np.isnan(report.inverse(.19, reference)))
        self.assertTrue(np.isnan(report.inverse(.5, reference)))
        with self.assertRaises(ValueError):
            report.inverse(.25, [.2, .2, .3, .4])

    def test_aulc_uses_log_budget(self):
        x = np.log2(report.BUDGETS)
        self.assertAlmostEqual(report.aulc(2*x+3), 3+x[-1])
        self.assertAlmostEqual(report.aulc(np.full(4,.25)), .25)
        with self.assertRaises(ValueError):
            report.aulc([1,2,3])

    def test_equal_hierarchical_weights_not_row_or_target_weights(self):
        # Fold 0 has one subset at r=0; fold 1 has three subsets at r=1.
        # Equal subset->fold averaging is .5, not pooled-row .75.
        rows=[]
        for fold,count,value in [(0,1,0.),(1,3,1.)]:
            for subset in range(count):
                rows.append(dict(candidate="model",budget=1,scan="scan",test_movie="movie",
                    repeat_orientation=0,outer_fold=fold,calibration_subset_id=str(subset),
                    mean_target_pearson=value,n_targets=10 if fold else 100))
        frame=pd.DataFrame(rows)
        _, total=report.aggregate(frame,["candidate","budget"],"mean_target_pearson")
        self.assertEqual(total.mean_target_pearson.iloc[0],.5)
        with self.assertRaises(ValueError):
            report.aggregate(pd.concat([frame,frame.iloc[[0]]]),["candidate","budget"],"mean_target_pearson")

    def test_inverse_displacement_is_additive(self):
        reference=[.2,.25,.30,.32]
        low,middle,high=[report.inverse(score,reference) for score in [.21,.24,.28]]
        self.assertAlmostEqual((high-middle)+(middle-low),high-low)


if __name__ == "__main__":
    unittest.main()
