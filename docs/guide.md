# Supporting guide

[← Project overview](../README.md)

[Methods, data, and repository guide](#methods-data-and-repository-guide) · [Getting started](#getting-started) · [Results dictionary](#results-dictionary) · [Technical reference](#technical-reference) · [Controls and extended results](#controls-and-extended-results)

## Methods, data, and repository guide

### Methods

#### Matched data access

- **Movie roles:** six observed movies rotate through four training movies, one validation movie, and one test movie.
- **Target calibration:** budgets of 1, 2, 4, or 5 repeats per training movie. A one-repeat budget therefore includes four movie presentations.
- **Repeat pools:** two disjoint five-repeat pools, with their calibration/population roles swapped across allocations.
- **Recording folds:** three outer target folds. Target recordings are excluded from shared-map fitting and context-only pseudo-target model selection.
- **Evaluation:** predict the held-out target response using test-movie context activity; inspect target test responses only when computing scores.

#### Estimators and selection

| Code identifier | Role |
|:--|:--|
| `D1` | Direct empirical target–context correlations |
| `D1_PAIR_CALIBRATED` | Shared correlation map |
| `D1_DXY` | Shared map with in-plane soma distance |
| `D1_SHRUNK` | Designated low-rank direct reference for equivalent-repeat comparisons |

The shared correlation map learns from context–context pairs to map limited-repeat correlations to population-pool correlations.

The selected policy chooses pair-map and decoder penalties using context-only validation. Fixed-penalty sensitivities and additional spatial arms are also included. Scores first average valid targets within a technical cell, then average calibration subsets, repeat allocations, folds, movies, and scans in that order. Pooling all observations directly would change the estimand.

Exact normalization, regularization, tie handling, reference inversion, and supplementary fitting conventions are documented in [reproducibility specification](#reproducibility).

### Data and scope

The recordings come from **MICrONS**, [DANDI dandiset 000402](https://dandiarchive.org/dandiset/000402/draft), subject 17797. Exact NWB asset identifiers, recording order, response windows, splits, and processed-array checksums are included under `configs/`.

| Included here | Needed for additional work |
|:--|:--|
| Retained primary and supplementary score tables | Raw NWBs or verified processed responses for primary refits |
| Recording identities, soma coordinates, movie roles, and repeat subsets | Frozen FNN core and aligned features for visual/location fits |
| Estimator, numerical functions, and tests | Matched CAVE annotations and reconstructed-source graph assets for anatomical/graph refits |
| Reproduction scripts and numerical reference values | The accompanying research write-up |

#### Limits that matter

- **One animal.** The 10 scans and 5 sessions are repeated measurements within one mouse. The 6,641 recording identities should not be read as that many unique biological neurons or independent animals.
- **Coordinates and coverage.** Soma positions are released functional x/y/z coordinates, not reconstructed cortical-surface or laminar coordinates. Anatomical and observed-input analyses use distinct, smaller cohorts.
- **Conditional uncertainty.** The location-control ranges describe reweighting the six observed movies while holding fitted predictions fixed. They do not include new animals or model refitting.
- **Versioned inputs are incomplete.** The analysis used a mutable DANDI draft release. Asset IDs and checksums detect changes, but a single immutable upstream version was not recorded.
- **Reproduction has levels.** Retained-result regeneration is available directly. Primary refitting needs response arrays; supplementary numerical functions are included, but a complete end-to-end supplementary refit pipeline is not supplied.

### Repository guide

```text
pricing-spatial-priors/
├── README.md                    # Project overview, results, and repository map
├── assets/readme/               # Pipeline diagram and result figures
├── configs/
│   ├── data/                    # NWB assets, array specifications, units, trials
│   └── splits/                  # Recording folds, movies, repeat pools, pair IDs
├── src/spatial_priors/
│   ├── estimator.py             # Direct, shared-map, and spatial models
│   ├── pipeline.py              # Selection, prediction, scoring, aggregation
│   ├── primitives.py            # Numerical building blocks
│   ├── reference.py             # Low-rank direct reference
│   └── supplementary.py         # Profiles, locations, class and graph features
├── scripts/                     # Data preparation, fitting, and figure generation
├── results/                     # Retained scores and supporting summaries
├── tests/                       # Numerical tests and full-precision fixtures
├── docs/guide.md                # Methods, setup, dictionary, and technical reference
├── CITATION.cff                 # Software citation metadata
└── LICENSE                      # MIT license
```

## Getting started

This page provides the entry commands. The [reproducibility specification](#reproducibility) is the detailed reference for numerical conventions and reproduction scope.

### 1. Regenerate tables and figures from the included scores

The documented reproduction environment is **Python 3.10**. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/reproduce_paper.py --check
python -m unittest discover -s tests -v
```

The reproduction script writes computed Tables 1, 2, and 4–8 and Figures 1–5 to `outputs/paper/`, including PNG and SVG figures. Table 3 specifies model inputs rather than computed results. This route needs **no external neural data or pretrained models**: it reaggregates retained predictions' scores and checks the manuscript's numerical values. It does not refit the predictors.

### 2. Refit the primary predictors

Follow [data setup](#data-setup-specification) to acquire, extract, and checksum-verify the response arrays. Then run:

```bash
python scripts/run_predictions.py \
  --data-root data/responses \
  --output outputs/primary
```

The default run fits `D1`, `D1_PAIR_CALIBRATED`, and `D1_DXY` across the full set of scans, rotations, allocations, folds, and budgets. Output directories must be new. A smaller, explicitly scoped example and additional arms are in [reproducibility specification](#refitting-with-external-response-arrays).

### Supporting specifications

- [Data setup](#data-setup-specification): public recordings, processed-array preparation, checksums, and supplementary assets
- [Reproducibility](#reproducibility): exact conventions, subset runs, supplementary scope, and reference inversion
- [Result dictionary](#results-dictionary): score columns, aggregation order, and diagnostic cohorts
- [Figure provenance](#figure-provenance): README assets and verification scope

The analysis used a mutable DANDI draft release. Asset IDs and checksums detect differences, but a single immutable upstream version was not recorded. A checksum mismatch requires investigation. Supplementary numerical functions are included, but complete end-to-end supplementary refitting also requires external assets and preprocessing.

## Results dictionary

These files contain the observations needed to reproduce the manuscript's tables, plots, and numerical summaries without refitting the neural predictors. Scores are Pearson correlations unless a column explicitly specifies R-squared, a count, a location dispersion, or a distance statistic. Blank values are unavailable, never zero.

### Primary and profile analyses

`spatial/cells.csv.gz` and `profiles/cells.csv.gz` contain one target-averaged score per scan, movie rotation, repeat allocation, outer recording fold, target budget, calibration subset, estimator, and regularization policy. Their columns are:

| Columns | Meaning |
| --- | --- |
| `scan`, `movie_rotation`, `test_movie` | Scan and held-out movie configuration. Session is the integer following `session_` in `scan`. |
| `repeat_orientation`, `outer_fold` | Repeat-pool allocation and outer target fold. |
| `budget`, `calibration_subset_id` | Repeats per training movie and the selected balanced subset. |
| `candidate`, `policy_arm` | Estimator and `selected` or `fixed` regularization. |
| `n_eligible_targets`, `n_common_valid_targets` | Target counts before and after common validity checks. |
| `n_excluded_nonfinite_labels`, `n_excluded_constant_labels` | Label-related exclusions. |
| `mean_target_pearson` | Mean across valid targets in this configuration. |

Average first across calibration subsets, then repeat allocations, outer folds, and test movies, and finally scans. `movie_rotation` and `test_movie` identify the same movie configuration, so retain one or both as grouping keys until the movie average. Weight scans equally for the primary result. For an equal-session sensitivity, average scans within session and then weight sessions equally. The normalized log-budget AULC is trapezoidal integration at budgets 1, 2, 4, and 5, divided by `log2(5)`. Do not pool all budgets before integration.

Estimator names follow the manuscript's Appendix E:

| Identifier | Estimator |
| --- | --- |
| `D1` | Direct correlations |
| `D1_PAIR_CALIBRATED` | Shared correlation map |
| `D1_DXY` | Map plus in-plane distance |
| `D1_DZ` | Map plus absolute depth separation |
| `D1_D3D` | Map plus three-dimensional distance |
| `D1_DZ_DXY` | Map plus depth separation and in-plane distance |
| `WF_D1_PAIR_CALIBRATED`, `WF_D1_DXY` | Corresponding maps with same-field fitting pairs and context |
| `D1_SHRUNK` | Designated low-rank direct reference |
| `F1`, `S1`, `A1`, `R2`, `FA1` | Visual, handcrafted visual, activity, anchor, and visual-activity profiles |

`curves.csv` files contain independently aggregated means; `spatial/scan_curves.csv` and `profiles/reference_scan_curve.csv` retain the scan-level curves. `spatial/coordinate_support.csv` describes the recorded coordinate ranges and pairwise distance associations. Reference inversions use aggregate scores and no extrapolation. The reference is `D1_SHRUNK` with the selected policy, including for comparisons involving spatial estimators.

### Fitted-location controls

`location/cells.csv.gz` contains both budgets. It uses the same grouping keys, plus `session`, and stores six matched score columns: `r_B` (map), `r_BG` (map plus position), `r_BR_b` and `r_BRG_b` (budget-matched location descriptor, without and with position), and `r_BR_hi` and `r_BRG_hi` (five-repeat location descriptor). `n_common_valid_targets` is shared across the six arms.

`location/contrasts.csv` records the resulting point estimates and conditional ranges. Regimes `RF_b` and `RF_hi` denote the two location fits. `delta_DXY` is `BG-B`; `delta_RF` is `BR-B`; `delta_DXY_given_RF` is `BRG-BR`; `retained_fraction` is `(BRG-BR)/(BG-B)`. The latter is a fraction, not a percentage.

For conditional intervals, first average to one six-column score vector per observed movie, retaining equal scan weighting. Sort the six movie identifiers. Draw 10,000 multinomial weight vectors with `numpy.random.default_rng(20260910).multinomial(6, [1/6]*6, size=10000)`, average the paired score vectors using those weights, and take the 2.5th and 97.5th percentiles with linear interpolation. Reuse each draw across arms and budgets. These ranges condition on fitted predictions and the observed movies.

`location/quality_scan.csv` contains each scan's mean selected location-fit cross-validation score and median leave-one-fold-out RMS dispersion. The manuscript reports the unweighted mean of the ten scan values for each location-fit budget. Individual fit diagnostics are not included here. `location/distance_association_scan.csv` contains sufficient sums to recalculate the within-scan Pearson association between soma and fitted-location distances. These recurring pair instances are descriptive observations, not independent biological replicates.

### Anatomical and observed-input analyses

`anatomy/cells.csv.gz` stores target-averaged scores in two distinct panels: `anatomy` (4,748 targets) and `anatomy_connectome` (317 targets). Its grouping keys are `panel`, `scan`, `session`, `movie_rotation`, `outer_fold`, `orientation`, `budget`, `subset`, `candidate`, and `policy`. `value` is the mean target Pearson score. The remaining columns give total, valid, and dropped target counts. All candidates use common valid targets within a configuration. Aggregate subsets, orientations, folds, movies, and scans in that order. Differences of matched candidate means reproduce the paired contrast summaries.

Candidate letters are `B` for the shared correlation map, `G` for in-plane distance, `T` for detailed class, `L` for coarse class, `Q` for degree and availability controls, and `C` for those controls plus observed shared-input overlap. The `conv` suffix denotes the conventional-class fixed-penalty sensitivity. Panels must remain separate.

`anatomy/curves.csv`, `contrasts.csv`, `scan_endpoints.csv`, and `contrast_scan_endpoints.csv` provide corresponding aggregate scores, differences, and endpoint summaries. `cohort_eligibility.csv.gz` identifies the full recording universe and the eligibility flags. Read nucleus and root identifiers as strings to avoid integer precision loss. `graph_coverage_scan.csv` distinguishes positive overlap, defined zero overlap, undefined denominators, and unknown eligibility/query status. `graph_training_support.csv` provides the number of defined overlap pairs and whether the overlap feature varies in each outer or pseudo fitting scope; 2 outer and 10 pseudo scopes have no varying overlap contribution.

### Appendix F diagnostics

`appendix_diagnostics/*_cells.csv.gz` contains the target-averaged configuration scores for full-context ridge, within-movie affine response-scale checks, activity/trajectory similarity, and the one-repeat activity-versus-all-geometry comparison. The corresponding `*_equal_session.csv` and `*_equal_scan.csv` files provide aggregate summaries. Apply the same subset, repeat-allocation, fold, and movie averages, then average scans within session and sessions equally where indicated. The affine result uses `within_movie_crossfit_mean_target_r2`; the activity identity check uses `a1_address_vs_rank64_trajectory_pearson`.

`appendix_diagnostics/repeat_reliability.csv` contains the equal-session reliability summary. Its per-recording input is not included here. These diagnostics use the estimators and target-data access specified in Appendix F; they do not provide matched alternatives to the primary in-plane estimator.

Full-precision reference values for automated checks are in [tests/fixtures/expected_paper_values.json](../tests/fixtures/expected_paper_values.json). Calculations use full precision; decimal formatting is applied only when presenting results.

## Technical reference

### Reproducibility

#### What can be reproduced directly

`python scripts/reproduce_paper.py --check` regenerates Tables 1, 2, and 4-8 and
Figures 1-5 from retained technical-cell scores in `results/`. Table 3 specifies
model inputs rather than computed results. This route recalculates learning
curves, matched contrasts, normalized log-budget AULC, reference-equivalent
repeats, and conditional movie-reweighting ranges. It requires no external
neural data and does not refit predictions.

Anatomical, graph, and Appendix F diagnostics are retained as matched score cells
and supporting summaries, including `results/appendix_diagnostics/`. The result
columns and their aggregation rules are described in
[results/README.md](#results-dictionary). Four- and five-repeat spatial
per-target predictions were not retained; their technical-cell scores are
included.

#### Refitting with external response arrays

Prepare the response arrays using [data.md](#data-setup-specification). The primary runner fits
the direct, mapped, and mapped-plus-position predictors using the supplied
split and pair metadata. A small run covering all five one-repeat calibration
subsets is:

```bash
python scripts/run_predictions.py --data-root data/responses --output outputs/smoke --scans session_5_scan_6 --rotations 0 --orientations 0 --folds 0 --budgets 1 --save-predictions
```

Without the subset arguments, the default run covers 1,440 selection
configurations and 5,760 calibration cells. Additional spatial controls and the
designated low-rank direct reference can be fitted with explicit arm lists:

```bash
python scripts/run_predictions.py --data-root data/responses --output outputs/spatial-controls --arms D1_PAIR_CALIBRATED D1_DXY D1_DZ D1_D3D D1_DZ_DXY WF_D1_PAIR_CALIBRATED WF_D1_DXY
python scripts/run_predictions.py --data-root data/responses --output outputs/reference --arms D1_SHRUNK
```

Existing output directories are protected against accidental overwriting.

#### Supplementary analyses and external assets

`src/spatial_priors/supplementary.py` implements visual, activity, and anchor
profiles; location fitting; class encoding; and graph features from prepared
arrays. These numerical functions and their tests are included, but complete
supplementary fitting requires external assets and preprocessing not supplied
as an end-to-end runner.

Visual and location fits require the frozen FNN core, aligned movie features,
and anchor identities. Anatomical controls require matched CAVE annotations,
nucleus identities, and duplicate exclusions; graph controls require the fixed
reconstructed-source bank and synapse/query-availability information. Exact
asset requirements and the DANDI version limitation are documented in
[data.md](#data-setup-specification). Reaggregating retained supplementary scores is distinct from
repeating these fits.

#### Core fitting and selection conventions

Recording identities, movie hashes, repeat pools, and calibration subsets in
`configs/splits/` define the data split. Deterministic sampled pair sets are
reconstructed and checked against their saved identities. Target recordings
are excluded from shared fitting and pseudo-target model selection. Held-out
test-movie target responses enter only the scoring function.

The runner defaults to context-only selection over pair penalties 1, 5, and 25
and decoder penalties 10, 30, and 100. Ties favor larger decoder, then larger
pair penalties. Reference ranks are 8, 16, 32, 64, 128, and full; reference ties
favor smaller rank, then larger decoder penalty. `--policy fixed` uses pair
penalty 5, decoder penalty 30, and reference rank 64.

Continuous pair features use fitting-pair means and population standard
deviations; binary features remain binary. The intercept is unpenalized. The
covariance diagonal is one, and the output gain follows the manuscript's
equation.

#### Aggregation rules

Technical-cell scores average valid target correlations. Subsequent means give
equal weight to calibration subsets, repeat allocations, folds, movies, and
scans in that order. Pooling all cells or target instances produces a different
estimand. Anatomical candidates share a common validity intersection, so
differences of matched candidate means reproduce their paired contrasts.

Normalized AULC integrates the four measured points over log2 budget and divides
by log2(5). The ten scans and five sessions are repeated measurements within one
animal; they are not independent biological replicates.

Location-control ranges use 10,000 shared multinomial weight draws over the six
observed movies with seed 20260910. They describe fixed-prediction sensitivity
to movie weights and do not include new animals, model refitting, or uncertainty
in reference-equivalent displacement. See the
[results data dictionary](#results-dictionary) for the exact resampling and
quantile conventions.

#### Reference-curve inversion and censoring

Scores are aggregated before inversion against the designated low-rank direct
reference. The reference curve is strictly increasing, so no monotonic
adjustment is required. Scores outside its measured range remain undefined;
inversion does not extrapolate. Alternative-reference rows and the linear-budget
sensitivity use the same fitted scores. Reference-equivalent repeats depend on
the selected reference and do not estimate experimental time saved.

#### Exact deterministic conventions

Predictions are cast to float32 before scoring. Scientific reductions use
float64 and single-thread BLAS; numerical libraries on different platforms can
still differ in their least significant bits.

Activity SVD components are oriented so their largest-absolute temporal loading
is positive, with sample-identifier tie breaking. Profiles are subsequently
centered across components. Location and visual-readout selection scores are
rounded to 12 decimals before deterministic tie handling: location ties use the
first row-major grid cell, and readout ties favor larger penalties.

Calculations use full-precision scores. The five-repeat location-control
positional gain at one decoder repeat is 0.030724611481405, displayed as
**0.03072** at five decimals. The five-repeat mean selected location CV
correlation is 0.168490721476197, displayed as **0.168** at three decimals.

#### Reproducibility checks

Run `python -m unittest discover -s tests -v` and
`python scripts/reproduce_paper.py --check`. The automated checks cover:

- Regenerated learning curves, AULC, spatial gains, reference prices and
  censoring, paired movie ranges, anatomical/graph contrasts, and display
  precision against `tests/fixtures/expected_paper_values.json`.
- All 540 saved outer/pseudo fitting pair identities, target exclusion from
  selection, and isolation of held-out target responses from predictions.
- Ridge/intercept and covariance-decoder identities, low-rank bounds,
  hierarchical aggregation, and reference inversion.
- Supplementary location fitting and projection, profile construction, anchor
  selection, class encoding, graph coverage conventions, and deterministic
  readout ties.

These checks exercise numerical invariants and retained-result reproduction.
They do not perform a full-study prediction refit or retrieve external assets.

### Data setup specification

The primary prediction runner needs ten raw response arrays and the small split
tables already included in `configs/`. Each array is float32 with axes
`[movie, repeat, time, recording]`. It contains six movies, ten repeats, and 62
time samples per presentation (85 for session 9/scan 3). The ten arrays together
occupy about 98 MiB. Raw NWBs are much larger and are not included in Git.

#### From existing processed arrays

If prepared arrays are available in `SCAN/processed/Y_clip_fluorescence_raw.npy`
under a local directory, copy and checksum-verify them with:

```bash
python scripts/prepare_data.py --replication-root /path/to/processed-source --output data/responses
```

The source is read-only. The output directory is ignored by Git. The optional
`--scans session_5_scan_6` prepares just the small-run scan.

#### From the public recordings

The recordings are from [DANDI dandiset 000402](https://dandiarchive.org/dandiset/000402/draft),
subject 17797. `configs/data/assets.csv` gives the ten exact NWB asset names and
asset identifiers used by the analysis. Download those files through DANDI,
preserving the listed subdirectories or placing their basenames together. Then:

```bash
python scripts/prepare_data.py --nwb-root /path/to/downloaded-nwb --output data/responses
```

The extractor uses the saved ROI ordering and response-window sample indices in
`configs/data/units.csv` and `trials.csv`. These specify the half-open
stimulus windows, truncation to the common presentation length, and first-field
time-window convention. It extracts the fluorescence-interface series without
global z-scoring. Training-only normalization happens inside the estimator.

`configs/data/responses.json` records expected shape, dtype, and SHA-256 for each
array. Preparation fails if its output differs. The analysis used a mutable
draft release; a single immutable dandiset version was not recorded.
Asset IDs and array checksums detect differences but do not reconstruct missing
upstream versions. A checksum failure must be investigated rather than accepted
as an equivalent dataset.

Soma coordinates, field labels, and recording identities required by the primary
runner are included in `configs/splits/neuron_folds.csv`. Coordinates are the
released functional x/y/z coordinates in micrometres. They are not reconstructed
cortical-surface or laminar coordinates.

#### Additional assets for supplementary fits

The summary-reproduction command requires none of these assets. Refitting the
supplementary models requires the corresponding external inputs:

- Visual profiles and fitted locations: the frozen foundation core from
  [cajal/fnn, pinned revision](https://github.com/cajal/fnn/tree/8f34c6753bd818c8fcd1daa91499a057252bea7e),
  aligned training-movie features, and the response-free anchor identities.
  The project uses only the frozen core, with new target-specific readouts.
- Cell-class controls: the matched morphology/synapse-informed class annotations,
  nucleus identities, and duplicate exclusions from the paper's specified CAVE
  materialization. Current live annotations need not match that snapshot.
- Observed shared input: the fixed reconstructed-source bank and its synapse
  counts/query-availability information. Missing coverage must remain distinct
  from measured zero overlap.

These upstream databases and checkpoints retain their own access and license
terms. This repository includes the retained supplementary score cells and
numerical feature functions; see the precise scope in [reproducibility specification](#reproducibility).

### Figure provenance

The README figures use retained results and the implemented split design. Research models were not refitted to generate these assets.

#### Assets

- `assets/readme/pipeline.svg`: a schematic of the implemented estimator and split design. Source definitions are `src/spatial_priors/estimator.py`, `src/spatial_priors/pipeline.py`, and `configs/splits/`.
- `assets/readme/main-results.png` and `.svg`: presentation figures generated from `results/spatial/curves.csv`, selecting `policy_arm == "selected"` at budgets 1, 2, 4, and 5. Lines connect measured budgets on a log₂ scale; the filled region in the gain panel is decorative area to zero, not an uncertainty interval.
- `assets/readme/chart-data.json`: the exact retained values plotted, with the derived map-plus-position minus map increment.

To rebuild these presentation assets in the documented reproduction environment:

```bash
python scripts/build_readme_assets.py
```

This is independent of the manuscript figure builder, `scripts/reproduce_paper.py`. The chart colors distinguish direct correlations (gray, circles, dashed), shared map (blue, squares, dash-dot), and map plus in-plane distance (orange, diamonds, solid). The main accuracy panel uses a focused y-axis; the gain panel includes zero. Both show descriptive means, without an animal-generalization confidence interval.

#### Headline evidence

| README statement | Evidence |
|:--|:--|
| 6,641 eligible recording identities in 10 scans | `configs/splits/neuron_folds.csv`, filtered to `analysis_eligible` |
| Five sessions in one mouse | Reproducibility specification; scan session IDs in `configs/splits/neuron_folds.csv` |
| Budget means repeats of each of four training movies | `configs/splits/movie_roles.csv`, `budget_subsets.csv`, and the estimator split design |
| Primary learning curves | `results/spatial/curves.csv`; regenerated from `results/spatial/cells.csv.gz` by `reproduce_paper.py` |
| Position gain positive in every scan/budget | `results/spatial/scan_curves.csv`; explicit check in `reproduce_paper.py` |
| AULC | Normalized trapezoidal integral over log₂ budgets 1, 2, 4, 5; `reproduce_paper.py:aulc` |
| Approximately 45% overstatement | At budget 1, `(mapped − direct) / (mapped + position − mapped) × 100 = 45.4115870671993%` |
| Equivalent-repeat displacements | Selected `D1_SHRUNK` curve in `results/profiles/curves.csv`; `reproduce_paper.py:inverse` and `tests/fixtures/expected_paper_values.json` |
| Depth and same-field comparisons | Selected arms in `results/spatial/curves.csv` |
| Fitted-location control | `results/location/contrasts.csv`, five-repeat descriptor at decoder budget 1 |
| Target test responses used only for scoring | `estimator.py:build_cell_arrays`, `pipeline.py:predict_cell` and `score_cell`, numerical isolation tests |

#### Verification scope

The retained-result reproduction command and all 26 included numerical tests have passed under Python 3.12 with available dependencies. The documented reproduction environment remains Python 3.10 with the pinned requirements. This check verifies retained-result aggregation and numerical invariants; it does not establish a fresh pinned-environment installation or a full-study prediction refit. Numeric precision is retained for calculation and rounded only for display.

### Citation metadata

Current software metadata lists anonymous authors. Named authorship and a manuscript link or identifier have not yet been supplied. Please credit the accompanying research and the MICrONS data release when using this work.

The machine-readable metadata is in [CITATION.cff](../CITATION.cff):

```yaml
cff-version: 1.2.0
message: "Please cite the accompanying manuscript and the MICrONS data release."
title: "Pricing Spatial Priors in Few-Shot Neural Prediction"
type: software
authors:
  - name: Anonymous authors
version: 0.1.0
license: MIT
keywords:
  - neural prediction
  - MICrONS
  - spatial priors
  - reproducibility
```

### MIT license

Project code is provided under the [MIT license](../LICENSE). Third-party data, annotations, and model assets retain their own terms.

```text
MIT License

Copyright (c) 2026 Anonymous contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Controls and extended results

### Primary results

![Two-panel figure. Left: mean target Pearson correlation at budgets 1, 2, 4, and 5 for direct correlations, a shared map, and the map with in-plane distance. Right: the incremental position gain decreases from 0.03071 at one repeat to 0.00949 at five repeats.](../assets/readme/main-results.png)

*Selected-regularization results, with equal weighting across scans after the prescribed nested aggregation. Lines connect the four measured budgets on a log₂ axis. These are retained prediction scores; no new fits were run to create this presentation.*

| Predictor | 1 repeat | 2 repeats | 4 repeats | 5 repeats | Log-budget AULC |
|:--|--:|--:|--:|--:|--:|
| Direct correlations | 0.21201 | 0.25426 | 0.29579 | 0.30710 | 0.26064 |
| Shared correlation map | 0.22596 | 0.25770 | 0.29407 | 0.30480 | 0.26448 |
| **Map + in-plane distance** | **0.25667** | **0.27815** | **0.30564** | **0.31429** | **0.28385** |
| **Position gain beyond the map** | **+0.03071** | **+0.02045** | **+0.01157** | **+0.00949** | **+0.01937** |

Scores are mean target Pearson *r*. AULC is the normalized trapezoidal area under the learning curve over log₂ budget. Full-precision sources: [spatial curves](../results/spatial/curves.csv), [retained score cells](../results/spatial/cells.csv.gz), and [aggregation definitions](#results-dictionary).

#### What the comparison shows

- **Position adds predictive value at every measured budget.** The map-plus-distance arm outperforms the map alone in all 10 scans at all four budgets. The aggregate increment is largest at one repeat and becomes smaller as target calibration increases.
- **The baseline changes the interpretation.** At one repeat, the total improvement over direct correlations is +0.04466, comprising +0.01395 from the shared map and +0.03071 from position beyond that map. Attributing the entire improvement to position would overstate its isolated contribution by about 45%.
- **Spatial gains can be expressed on a measured data scale.** Relative to the designated low-rank direct reference, the position increment corresponds to 0.825, 0.862, and 0.974 additional reference-equivalent repeats at budgets 1, 2, and 4. The five-repeat comparison is censored because it exceeds the measured reference range. These are reference-dependent score displacements, not estimates of experimental time saved.

#### Controls and interpretation

Additional analyses test depth separation, 3D distance, same-field context, fitted-location descriptors, visual/activity profiles, cell-class annotations, and observed shared-input features.

At one repeat, released depth-coordinate separation adds only **+0.00003** beyond the shared map, while the in-plane gain remains **+0.03108** when fitting and context are restricted to the same field. Adding the five-repeat fitted-location descriptor leaves an in-plane increment of **+0.03072**. These comparisons describe this study's coordinates, descriptors, and coverage; they do not establish a causal effect of cortical geometry.

See the [result dictionary](#results-dictionary) for distinct anatomical cohorts, feature definitions, and supplementary diagnostics. Figure provenance and the exact headline calculations are recorded in [README figure notes](#figure-provenance).

### Boundaries of the controls

The location-control ranges reweight the six observed movies while holding fitted predictions fixed. They do not include new animals or model refitting. The anatomical and observed-input analyses use distinct, smaller cohorts and should not be pooled with the primary cohort.

Released functional x/y/z coordinates are not reconstructed cortical-surface or laminar coordinates. The fitted-location descriptors, class annotations, and observed-input features test particular measurements with their documented coverage. They do not establish that spatial prediction gains are independent of every possible functional or anatomical explanation.

See [the result dictionary](#results-dictionary) for cohort definitions, feature meanings, conditional-range construction, and Appendix diagnostics, and [the reproducibility specification](#reproducibility) for aggregation and inversion conventions.
