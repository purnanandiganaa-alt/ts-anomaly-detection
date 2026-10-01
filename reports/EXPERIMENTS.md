# Experiment log

Running record of experiments after the 13 Jul 2026 presentation checkpoint.
Each entry states what changed, what was measured, and what it means. Archived
artifacts (metrics, submissions, run logs) live in `outputs/experiments/<id>/`.

All F1 values are **validation** F1 unless a portal score is explicitly named.
The test set has no local labels.

---

## Reference point (pre-existing, 12 Jul 2026)

| | |
|---|---|
| Method | CNN autoencoder, per-sensor Mahalanobis (fit on confirmed-normal val windows), global F0.5 threshold |
| Val F1 | 0.6022 (precision 0.5998, recall 0.6047) |
| Val ROC-AUC | 0.7629 |
| Portal F1 | **0.63** |
| Submission | `outputs/cnn_predictions.csv` (restored and active) |

Documented run-to-run F1 wobble from training stochasticity alone: **±0.02**
(observed range 0.568-0.611 across identical configs). Any change smaller than
that is not evidence of improvement.

---

## Run 1 — threshold + per-run normalization A/B (15 Sep 2026)

**Archive:** `outputs/experiments/2026-09-15_run1_threshold_ab/`

**What changed:** none of the model. Executed the previously-written but never-run
4-variant comparison in `run_cnn.py`: {raw scores, per-run median/MAD normalized}
x {F0.5 threshold, F1 threshold}. All four share one trained network and one seed,
so differences are the variant, not training noise. Ran on the Apple M1 GPU (MPS)
after adding an `mps` tier to the device selection; the CUDA/cluster path is
unchanged.

**Motivation:** `plots/diagnostics/diagnostic_summary.json` showed F1 at the
F1-optimal threshold (0.6181) beating F1 at the F0.5 threshold (0.5731), and two
validation runs scoring F1 = 0.000 because their score peaks sat below a single
global threshold.

| Variant | F1 | Precision | Recall | ROC-AUC | Val flagged |
|---|---|---|---|---|---|
| raw + F0.5 (previous method) | 0.5829 | 0.577 | 0.589 | 0.7689 | 0.244 |
| **raw + F1** | **0.6067** | 0.505 | 0.759 | 0.7689 | 0.360 |
| per-run norm + F0.5 | 0.4050 | 0.690 | 0.287 | 0.6769 | 0.099 |
| per-run norm + F1 | 0.4585 | 0.380 | 0.577 | 0.6769 | 0.363 |

Checkpoint selection kept epoch 45 (F0.5 = 0.5794); per-checkpoint scores were
non-monotonic across the 50 epochs (0.531, 0.522, 0.528, 0.564, 0.524, 0.508,
0.468, 0.537, 0.579, 0.518).

### Findings

1. **Per-run normalization is rejected.** It cut ROC-AUC from 0.769 to 0.677 —
   it damaged the *ranking*, not merely the threshold, so no amount of threshold
   tuning recovers it. The median/MAD rescaling removes genuine cross-run signal.
   This contradicts the roadmap item and the "Fix" claim on slide 10 of
   `presentation_final.tex`, which must be corrected or reframed as a tested
   negative result.

2. **The F1 threshold helps, but less than predicted, and not beyond noise.**
   +0.024 within the run (0.5829 -> 0.6067), against the +0.045 the diagnostic
   implied. Versus the 0.6022 reference it is inside the ±0.02 wobble — not a
   demonstrable improvement.

3. **Validation and test score distributions do not match (new finding).**
   At the same absolute thresholds:

   | Submission | Test timesteps flagged | Val flagged at same threshold |
   |---|---|---|
   | Previous (portal F1 = 0.63) | **62.2%** | 24.4% |
   | raw + F1 (this run) | **84.2%** | 36.0% |
   | per-run norm + F1 | 36.7% | 36.3% |

   A threshold chosen on validation flags far more of test than of validation.
   The 84.2% file is close to "flag everything" and was **not** promoted to the
   portal submission. Note also that a 62%-flagging submission scored 0.63,
   which plain point-wise F1 on a ~24%-anomaly test set could not produce — so
   the portal metric is probably not point-wise F1 (possibly point-adjusted or
   event-wise). This is inference from the numbers, not confirmed.

**Outcome:** no change promoted. `outputs/cnn_predictions.csv` remains the
portal-scored 0.63 submission.

---

## Run 2 — Mahalanobis normal model: validation-normal vs train, and the verification that followed (15 Sep 2026)

**Archive:** `outputs/experiments/2026-09-15_run2_mahalanobis_source/`
Verification scripts were read-only and wrote nothing to `src/` or `outputs/`.

**What changed:** the source of the Gaussian "normal" error model.
**A** = confirmed-normal validation windows (existing behaviour).
**B** = all ~18k train windows.

Reconstruction errors are computed once and both normal models applied to the same
cached errors. Training reproduced run 1 exactly (final loss 0.007123, best
checkpoint epoch 45 at F0.5 = 0.5794). `outputs/cnn/model.pt` is now saved, which
made the follow-up verification possible without retraining.

| Variant | F1 | Precision | Recall | ROC-AUC | Val flagged | Test flagged |
|---|---|---|---|---|---|---|
| A val-normal, raw, F1 | 0.6067 | 0.505 | 0.759 | 0.7689 | 0.360 | 0.842 |
| A val-normal, raw, F0.5 | 0.5829 | 0.577 | 0.589 | 0.7689 | 0.244 | 0.617 |
| A val-normal, per-run norm, F0.5 | 0.4050 | 0.690 | 0.287 | 0.6769 | 0.099 | 0.102 |
| A val-normal, per-run norm, F1 | 0.4585 | 0.380 | 0.577 | 0.6769 | 0.363 | 0.367 |
| B train-fit, raw, F0.5 | 0.4829 | 0.369 | 0.700 | 0.6109 | 0.455 | 0.833 |
| B train-fit, raw, F1 | 0.4829 | 0.369 | 0.700 | 0.6109 | 0.455 | 0.833 |
| B train-fit, per-run norm, F0.5 | 0.4014 | 0.415 | 0.389 | 0.6387 | 0.225 | 0.226 |
| B train-fit, per-run norm, F1 | 0.4420 | 0.367 | 0.556 | 0.6387 | 0.363 | 0.359 |

> An earlier draft of this section explained B's failure with a single mechanism
> ("in-sample reconstruction error") and described A vs B as differing in one
> variable. Both statements were over-claimed and are corrected below.

---

### A vs B is NOT a clean one-variable experiment

Controlled with respect to the network: same trained weights, same cached
reconstruction errors, same validation data, same validation labels, same
threshold-selection procedure.

**But the two fits differ in three ways at once:**

1. **in-sample vs held-out** — train windows trained the network; validation windows did not
2. **contaminated vs filtered** — B uses all train windows (contamination included);
   A keeps only windows whose label is 0
3. **same-run vs different-run** — A is fitted on normal windows of the *very runs it
   then scores*; B shares no run with the data it scores

Any of the three could drive the gap. The two tests below give partial separation
but do not fully isolate them.

### Supporting evidence 1 — error distributions

Measured from the saved checkpoint:

| Set | n | per-window mean err | median | std | mean per-sensor variance |
|---|---|---|---|---|---|
| Train windows | 18,401 | 0.00668 | 0.00483 | 0.00558 | 0.000144 |
| Val **normal** windows | 5,430 | 0.02139 | 0.00989 | 0.02313 | 0.001727 |
| Val **anomalous** windows | 2,344 | 0.04609 | 0.01960 | 0.05788 | 0.009761 |

Median per-sensor ratio val-normal / train = **2.693** (17 of 18 sensors larger).
Fitted covariance scale: `log|Sigma|` = **-181.23** (train) vs **-152.20** (val-normal);
median Mahalanobis score **83.39** under B vs **7.30** under A.

### Supporting evidence 2 — contamination test (model C)

A third normal model **C** was fitted on *all* validation windows — held out from
training, but unfiltered and therefore ~30% anomalous at window level:

| Normal model | Held out? | Label-filtered? | ROC-AUC | F1 (F0.5 thr) |
|---|---|---|---|---|
| A — val-normal | yes | yes | 0.7689 | 0.5829 |
| C — all val windows | yes | **no** | 0.6929 | 0.5261 |
| B — train | **no** | no | 0.6109 | 0.4829 |

C is plausibly *more* contaminated than train, yet still scores above B. Contamination
alone therefore does not account for B's result.

### Supporting evidence 3 — leave-one-run-out (the most consequential finding)

Model A is fitted on normal windows of the same runs it scores. Test runs can never
have that. Each validation run was re-scored by a model fitted on the other 9 runs:

| | ROC-AUC | F1 | Precision | Recall | Flagged |
|---|---|---|---|---|---|
| A as the pipeline runs it | **0.7689** | **0.5829** | 0.577 | 0.589 | 0.244 |
| A under leave-one-run-out | **0.6985** | **0.5251** | 0.462 | 0.609 | 0.316 |

Per-run AUC (A -> LORO): run1 0.767->0.756, run2 0.933->0.930, run3 0.771->0.658,
run4 0.872->0.774, run5 0.810->0.753, run6 0.807->0.776, run7 0.799->0.724,
run8 0.759->0.724, run9 0.521->0.496, run10 0.891->0.870. Every run degrades.

Against the LORO baseline, B's deficit is **0.088 AUC**, not 0.158.

---

### Baseline discrepancy vs the 12 Jul model of record

| | Old record | Run 2 model A | Note |
|---|---|---|---|
| F1 @ beta=0.5 | 0.6022 | 0.5829 | like-for-like; run 2 is 0.019 **worse** |
| F1 @ beta=1.0 | not computed | 0.6067 | different threshold objective |
| ROC-AUC | 0.7629 | 0.7689 | threshold-free |
| Threshold | 43.298 | 42.773 | |

Verified identical between the two runs: code (`git diff 03c0850 HEAD` on `run_cnn.py`,
`config.py`, `metrics.py`, `models/cnn/` is empty), threshold-selection procedure, beta,
validation labels (19,086 positives — matches the organizers' manifest exactly),
windowing, preprocessing, random seed, checkpoint criterion, metric implementation.

The old run's test scores were recovered from git (`outputs/cnn/predictions.csv` at HEAD)
and compared with the current network's: **Spearman rank correlation 0.947**, ratio
new/old spanning **0.28-8.30** (1st-99th percentile), 90.9% of timesteps shifting rank
by more than 1,000 positions. The two runs produced genuinely different weights.

---

### CONFIRMED

- Old 0.6022 and run 2's 0.6067 are **different beta settings** and were not comparable.
- At matched beta = 0.5, run 2 model A = **0.5829** vs the old record **0.6022**.
- The old and new runs used **different trained weights despite the same seed**.
- A vs B share cached reconstruction errors, so the comparison **is controlled with
  respect to the network output**.
- Train reconstruction errors are **substantially tighter** than held-out
  validation-normal errors (variance 0.000144 vs 0.001727; `log|Sigma|` -181.23 vs -152.20).
- **Validation-normal fitting outperforms train fitting** (AUC 0.7689 vs 0.6109).
- **Leave-one-run-out reduces AUC from 0.7689 to 0.6985.**
- **Leave-one-run-out reduces F1 from 0.5829 to 0.5251.**
- The **portal metric and test prevalence are not documented anywhere in the repository**
  (checked: organizers' `manifest.json` recovered from commit 42901df, README,
  `data/README.md`, `HYBRID_MODEL_README.md`, notebooks, presentation, scripts).
  Validation prevalence is exactly 19,086/79,688 = **0.2395**. Test runs carry no
  `label_path`. The provided `sample_submission.csv` flags 10.01% of timesteps, but it
  is a format template, not ground truth.

### HYPOTHESES (not proven)

- **In-sample reconstruction-error distribution is likely an important reason train-fit
  Mahalanobis performs poorly.** Supported by the error-distribution and contamination
  tests, but not isolated from the other two differences.
- **Same-run Mahalanobis calibration likely causes optimism in the standard validation
  result.** The LORO drop is measured; attributing it specifically to same-run
  calibration is the interpretation.
- **The exact reason for the old-vs-new score difference is not fully proven** because
  historical cluster logs are unavailable (no `logs/` directory, no `.out`/`.err` ever
  committed). README documents the old record as "the latest GPU run" and the new runs
  were on Apple MPS, but the device is documented, not proven. The checkpoint epoch of
  the old run is unrecoverable.
- **Any claim that rate-matched thresholding improves the portal score is currently only
  a hypothesis** — it is neutral on validation by construction and has no local evidence.

### Status of per-run normalization

**Tested; not supported by the available validation evidence.** It lowered ROC-AUC in
both runs (0.7689 -> 0.6769 under model A; 0.6109 -> 0.6387 under model B, still far
below A). It has never been evaluated under leave-one-run-out. It should not be
described as a successful fix.

### Arithmetic bound on the portal metric question

For a submission flagging 62.22% of test timesteps, point-wise F1 = 2·TP/(0.6222N + pN):

| Assumed test prevalence | F1 ceiling at recall 1.0 | F1 at recall 0.76 |
|---|---|---|
| 0.2395 (= validation) | 0.556 | 0.423 |
| 0.30 | 0.651 | 0.495 |
| 0.40 | 0.783 | 0.595 |

Reaching the reported 0.63 requires test prevalence >= 0.286 at perfect recall, or
>= 0.440 at the model's validation recall of 0.76. So **either** test prevalence is much
higher than validation's, **or** the portal does not score point-wise F1. The repository
contains nothing that decides which.

**Outcome:** no change promoted. The pipeline keeps the validation-normal Mahalanobis
fit. `outputs/cnn_predictions.csv` remains the portal-scored 0.63 submission.

---

## Standing candidate submissions (not uploaded)

`outputs/candidates/`. None has local evidence of beating the active submission.

| File | Test flagged | Basis |
|---|---|---|
| *(active)* `outputs/cnn_predictions.csv` | 0.622 | portal F1 = 0.63, measured |
| `cnn_mahalA_val_raw_beta1.0.csv` | 0.842 | best val F1 (0.6067), but flags 84% of test |
| `cnn_mahalA_val_raw_beta1.0_ratematched.csv` | 0.360 | same scores, threshold set by quantile to reproduce the val flagged rate |
| `cnn_mahalB_train_*.csv` | 0.833 / 0.455 | rejected model B; kept for completeness |

**Open question blocking further portal work:** the portal metric is unknown. A
submission flagging 62% of test scored 0.63, which point-wise F1 on a ~24%-anomaly
test set cannot produce. Until that is resolved, threshold choice for the portal is
guesswork, and the rate-matched candidate is a hypothesis with no local support.

---

## Current scientific conclusion (as of 15 Sep 2026)

The current model has **no locally validated improvement** over the original
portal-scoring submission. Three hypotheses have now been tested — F1-vs-F0.5
threshold, per-run score normalization, and train-fitted Mahalanobis — and none of
them beats the existing method on validation.

The most important result from this investigation is not any of those three. It is
that **the previous validation protocol is optimistic**: the Mahalanobis normal model
is fitted using normal windows drawn from the same validation runs that are
subsequently evaluated. Removing that overlap costs 0.070 ROC-AUC and 0.058 F1.
Test runs never supply such windows, so the reported validation figures overstate
what transfers.

**Therefore future experiments should preferably use leave-one-run-out / grouped
validation when claiming generalization.** The run-grouped `GroupKFold` machinery
already written for `run_final_hybrid.py` is a suitable starting point. Comparisons
made under the old protocol — including every row in the results table of the README
and the presentation — should be read as optimistic estimates rather than transfer
estimates.

Two questions remain open and block further portal work:

1. **What metric does the portal compute?** Undocumented in the repository. Until it is
   known, threshold choice for submission is guesswork.
2. **What is the test anomaly prevalence?** Undocumented. It determines whether the
   observed 0.63 is even consistent with point-wise F1.
