"""Check regenerated manuscript tables against full-precision reference values."""
import importlib.util
import json
from pathlib import Path
import unittest
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('paper_value_report', ROOT/'scripts/reproduce_paper.py')
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class PaperValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Normal inherited permissions also support restricted Windows runners.
        cls.output = ROOT / 'outputs' / 'tests' / uuid.uuid4().hex
        cls.output.mkdir(parents=True)
        cls.addClassCleanup(cls.clean_output)
        cls.expected = json.loads((ROOT/'tests/fixtures/expected_paper_values.json').read_text())
        cls.result = report.build_tables(ROOT/'results', cls.output)

    @classmethod
    def clean_output(cls):
        if cls.output.resolve().parent != (ROOT/'outputs'/'tests').resolve():
            raise RuntimeError('Unexpected test output directory')
        for child in cls.output.iterdir():
            child.unlink()
        cls.output.rmdir()

    def test_all_pipeline_curves_and_aulc(self):
        rebuilt = pd.read_csv(self.output/'table8_complete_pipelines.csv').set_index('predictor')
        self.assertEqual(len(rebuilt), 9)
        for identifier, values in self.expected['table8'].items():
            with self.subTest(candidate=identifier):
                actual = rebuilt.loc[report.LABELS[identifier], ['b1','b2','b4','b5','aulc']].to_numpy(dtype=float)
                target = [values[key] for key in ['1','2','4','5','AULC']]
                np.testing.assert_allclose(actual, target, rtol=0, atol=1e-12)

    def test_spatial_feature_gains(self):
        actual = self.result['spatial_features']
        expected = self.expected['table5']
        self.assertEqual(len(actual), len(expected))
        # Published feature rows have the same explicit order in Table 5.
        np.testing.assert_allclose(actual[['b1','aulc']].to_numpy(dtype=float),
            [[row['b1'],row['AULC']] for row in expected], rtol=0, atol=1e-12)

    def test_reference_prices_and_censoring(self):
        names = {'low_rank':'Low-rank direct','direct':'Direct','mapped':'Mapped'}
        for expected in self.expected['table4']:
            rows = self.result['prices']
            actual = rows[(rows.reference == names[expected['reference']]) &
                          (rows.interpolation == expected['interpolation'])].iloc[0][f'b{expected["budget"]}']
            with self.subTest(reference=expected['reference'], scale=expected['interpolation'], budget=expected['budget']):
                if expected['position_displacement'] is None:
                    self.assertTrue(np.isnan(actual))
                else:
                    self.assertAlmostEqual(actual, expected['position_displacement'], places=11)
        ref = self.result['curves']['D1_SHRUNK']
        primary = self.expected['table4'][0]
        for candidate, field in [('D1','direct_inverse'),('D1_PAIR_CALIBRATED','mapped_inverse'),('D1_DXY','position_inverse')]:
            self.assertAlmostEqual(report.inverse(self.result['curves'][candidate][0],ref),primary[field],places=11)

    def test_paired_movie_ranges_for_both_location_budgets(self):
        mapping = {'delta_DXY':('None','Position'),'delta_RF':(None,'Location'),
                   'delta_DXY_given_RF':(None,'Position after location')}
        actual = self.result['locations']
        self.assertEqual(len(actual),10)
        for expected in self.expected['location']:
            if expected['contrast']=='retained_fraction':
                continue
            regime,feature=mapping[expected['contrast']]
            if regime is None:
                regime = {'RF_b':'Budget-matched','RF_hi':'Five repeats'}[expected['regime']]
            row=actual[(actual.budget==expected['budget']) & (actual.location_fit==regime) &
                       (actual.added_feature==feature)].iloc[0]
            with self.subTest(budget=expected['budget'],regime=regime,feature=feature):
                np.testing.assert_allclose(row[['gain','lower','upper']].to_numpy(dtype=float),
                    [expected[key] for key in ['gain','lower','upper']],rtol=0,atol=1e-12)

    def test_anatomical_and_graph_candidates_and_matched_contrasts(self):
        actual=self.result['anatomy'].query("policy == 'selected'").set_index(['panel','candidate'])
        for expected in self.expected['anatomy']:
            if expected['kind']=='candidate':
                values=actual.loc[(expected['panel'],expected['name']),['b1','aulc']]
            else:
                high,low=expected['name'].split('-')
                values=actual.loc[(expected['panel'],high),['b1','aulc']]-actual.loc[(expected['panel'],low),['b1','aulc']]
            with self.subTest(panel=expected['panel'],candidate_or_contrast=expected['name']):
                np.testing.assert_allclose(values.to_numpy(dtype=float),[expected['b1'],expected['aulc']],rtol=0,atol=1e-12)

    def test_common_validity_preserves_paired_comparison(self):
        cells=pd.read_csv(ROOT/'results/anatomy/cells.csv.gz')
        keys=['panel','scan','movie_rotation','outer_fold','orientation','budget','subset']
        counts=cells.groupby(keys)[['n_targets_total','n_targets_valid','n_targets_dropped']].nunique()
        self.assertTrue(counts.eq(1).all().all())
        self.assertTrue(cells.n_targets_dropped.eq(0).all())
        # Differences were independently aggregated from target-paired contrast cells.
        matched=pd.read_csv(ROOT/'results/anatomy/contrasts.csv',float_precision='round_trip')
        for row in matched.itertuples():
            candidates=self.result['anatomy']
            candidates=candidates[(candidates.panel==row.panel)&(candidates.policy==row.policy)].set_index('candidate')
            high,low=row.contrast.split('-')
            delta=candidates.loc[high,f'b{row.budget}']-candidates.loc[low,f'b{row.budget}']
            self.assertAlmostEqual(delta,row.value,places=12)

    def test_table_two_rounding_uses_full_precision(self):
        table=pd.read_csv(self.output/'table2_location_controls.csv')
        gain=float(table[table.location_fit.eq('Five repeats')].gain.iloc[0])
        self.assertEqual(f'{gain:.5f}','0.03072')
        quality=pd.read_csv(self.output/'location_fit_quality.csv').set_index('rf_budget')
        self.assertEqual(f'{quality.loc[5,"mean_selected_cv_pearson"]:.3f}','0.168')


if __name__=='__main__':
    unittest.main()
