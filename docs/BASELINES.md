# Baseline entry points

Every baseline produces an unknownness score in which larger values indicate stronger evidence for rejection. Each method calibrates its threshold independently from source validation scores at the same false-rejection budget.

## Shared source encoder

`dpr_rffi.baselines.posthoc` contains:

- `energy_unknown_score`
- `knn_unknown_score`
- `nndr_unknown_score`
- `fit_openmax` and `openmax_unknown_score`

## Trainable baselines

- OpenSVDD: `train_opensvdd_arpl`, followed by the fitting and scoring functions in `dpr_rffi.baselines.opensvdd`
- HyperRSI: `train_hyperrsi`, followed by hyperspherical feature inference and tail calibration
- MeDAE: `train_medae`, followed by reconstruction-aware feature inference and center scoring
- OSSEI: `train_ossei2025_source`, followed by its reconstruction and statistical score model

The modules retain the method-specific objectives and scoring heads used in the paper experiments. They share the source split, epoch budget, batch size, early-stopping convention, and target-evaluation code with DPR-RFFI.

## Calibration

Use `dpr_rffi.model.calibrate_source_threshold` for every method:

```python
threshold = calibrate_source_threshold(source_validation_score, source_frr=0.03)
rejected = target_score > threshold
```

Do not pool score thresholds across methods and do not use target labels or target score distributions for calibration.
