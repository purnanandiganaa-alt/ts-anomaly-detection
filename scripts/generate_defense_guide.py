"""
Generate reports/presentation_defense_guide.pdf

A defense-oriented study guide: full pipeline reconstruction (what/why/effect/
alternatives), a worked numeric example on a real validation window, concept
primers for every method, a component dependency map, likely professor
questions with model answers, limitations, and a final cheat sheet.

Run:  python scripts/generate_defense_guide.py
"""

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = "reports/presentation_defense_guide.pdf"
NAVY = colors.HexColor("#1F3A5F")
ACCENT = colors.HexColor("#2E6DA4")
TEAL = colors.HexColor("#2A8C82")
GREEN = colors.HexColor("#3F7D3F")
GOLD = colors.HexColor("#8A6D0B")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EAF0F6")
GOLDBG = colors.HexColor("#F3EEDD")


def S():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("TitleBig", parent=s["Title"], fontSize=22, textColor=NAVY, spaceAfter=3))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], fontSize=10, textColor=GREY, spaceAfter=8))
    s.add(ParagraphStyle("H1", parent=s["Heading1"], fontSize=15, textColor=NAVY, spaceBefore=13, spaceAfter=5))
    s.add(ParagraphStyle("H2", parent=s["Heading2"], fontSize=12, textColor=ACCENT, spaceBefore=9, spaceAfter=3))
    s.add(ParagraphStyle("H3", parent=s["Heading3"], fontSize=10.5, textColor=TEAL, spaceBefore=6, spaceAfter=2))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=9.7, leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
    s.add(ParagraphStyle("Cap", parent=s["Normal"], fontSize=8.3, textColor=GREY, alignment=TA_LEFT, spaceAfter=10))
    s.add(ParagraphStyle("Q", parent=s["Normal"], fontSize=9.8, leading=13.5, textColor=NAVY, fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=1))
    s.add(ParagraphStyle("A", parent=s["Normal"], fontSize=9.6, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5))
    s.add(ParagraphStyle("Key", parent=s["Normal"], fontSize=9.7, leading=14, textColor=NAVY, backColor=LIGHT, borderPadding=6, spaceBefore=3, spaceAfter=9))
    s.add(ParagraphStyle("Cheat", parent=s["Normal"], fontSize=9.4, leading=13.5, textColor=colors.black, backColor=GOLDBG, borderPadding=5, spaceBefore=2, spaceAfter=6))
    s.add(ParagraphStyle("Cell", parent=s["Normal"], fontSize=8.1, leading=10.5))
    s.add(ParagraphStyle("CellH", parent=s["Normal"], fontSize=8.3, leading=10.5, textColor=colors.white, fontName="Helvetica-Bold"))
    return s


def bullets(items, st, style="Body"):
    return ListFlowable([ListItem(Paragraph(t, st[style]), leftIndent=10) for t in items],
                        bulletType="bullet", start="•", leftIndent=12, spaceAfter=6)


def P(t, st): return Paragraph(t, st["Cell"])
def PH(t, st): return Paragraph(t, st["CellH"])


def grid(data, widths, header=True):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C6D6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(style))
    return t


def _arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=GREY, strokeWidth=1))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    L = 5
    d.add(Polygon([x2, y2,
                   x2 - L * math.cos(ang - 0.4), y2 - L * math.sin(ang - 0.4),
                   x2 - L * math.cos(ang + 0.4), y2 - L * math.sin(ang + 0.4)],
                  fillColor=GREY, strokeColor=GREY))


def dependency_map():
    W, H = 470, 300
    d = Drawing(W, H)
    def box(x, y, w, h, txt, fill):
        d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=colors.white, strokeWidth=0.5, rx=3, ry=3))
        d.add(String(x + w / 2, y + h / 2 - 3, txt, textAnchor="middle", fontSize=7.4, fillColor=colors.white))
    # column 1: data
    box(5, 250, 150, 26, "raw CSV runs (data/raw)", GREY)
    box(5, 210, 150, 26, "windowing.py (per-run)", GREY)
    box(5, 170, 150, 26, "StandardScaler (train only)", GREY)
    # column 2: model
    box(175, 250, 130, 26, "model.py (ConvAE1D)", NAVY)
    box(175, 210, 130, 26, "train.py (+checkpoint sel.)", NAVY)
    box(175, 170, 130, 26, "predict.py (per-sensor err)", NAVY)
    # column 3: scoring/eval
    box(325, 250, 140, 26, "metrics.py: fit_mahalanobis", ACCENT)
    box(325, 210, 140, 26, "metrics.py: mahalanobis_scores", ACCENT)
    box(325, 170, 140, 26, "find_best_threshold (F-beta)", TEAL)
    # bottom
    box(120, 120, 150, 26, "run_cnn.py (orchestrates)", GREEN)
    box(300, 120, 165, 26, "cnn_predictions.csv (submission)", GREEN)
    box(120, 78, 150, 26, "config.py (all hyperparams)", GOLD)
    # arrows
    _arrow(d, 155, 263, 175, 263)
    _arrow(d, 80, 250, 80, 236)
    _arrow(d, 80, 210, 80, 196)
    _arrow(d, 240, 250, 240, 236)
    _arrow(d, 240, 210, 240, 196)
    _arrow(d, 305, 223, 325, 223)
    _arrow(d, 395, 250, 395, 236)
    _arrow(d, 395, 210, 395, 196)
    _arrow(d, 195, 170, 195, 146)
    _arrow(d, 395, 170, 395, 146)
    _arrow(d, 270, 133, 300, 133)
    _arrow(d, 195, 104, 195, 120)
    d.add(String(5, 60, "Grey = data handling   Navy = CNN model   Blue = scoring   Teal = thresholding   "
                        "Green = orchestration/output   Gold = config feeds everything",
                 textAnchor="start", fontSize=6.6, fillColor=GREY))
    return d


def build():
    st = S()
    doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            leftMargin=1.7 * cm, rightMargin=1.7 * cm, title="Presentation Defense Guide")
    e = []

    # ===== Title =====
    e.append(Paragraph("Presentation Defense Guide", st["TitleBig"]))
    e.append(Paragraph("Multivariate time-series anomaly detection — CNN autoencoder. "
                       "Validation F1 ≈ 0.60, competition-portal F1 = 0.63.", st["Sub"]))
    e.append(Paragraph("<b>How to use this:</b> read Sections 1–4 to master the pipeline, Section 6 to rehearse "
                       "answers, and the final cheat sheet in the last minutes before you present.", st["Body"]))

    # ===== The pitch =====
    e.append(Paragraph("1.  The pitch (memorise this)", st["H1"]))
    e.append(Paragraph("<b>30 seconds:</b> \"We detect anomalies in an 18-sensor industrial distillation process. "
        "There are no anomaly labels for training, so instead of a classifier we train a dilated-CNN autoencoder "
        "to reconstruct <i>normal</i> sensor windows; anything it reconstructs badly is flagged. We score with a "
        "Mahalanobis distance over per-sensor reconstruction error, threshold it on a small labeled validation "
        "set, and reach F1 = 0.63 on the competition portal.\"", st["Key"]))
    e.append(Paragraph("<b>2 minutes (add):</b> the training data is unlabeled and not guaranteed anomaly-free, "
        "which shaped every choice. We rebuilt the code from notebooks into a tested package, fixed six correctness "
        "bugs, and improved the model over five rounds (checkpoint selection, a dilated TCN architecture for wide "
        "temporal context, Mahalanobis scoring, a precision-weighted threshold). We also rigorously tested fusing "
        "the CNN with two tabular detectors (Isolation Forest, then LOF) using leakage-free cross-validation — both "
        "were rejected, so we ship the CNN alone: a focused model we fully understand.", st["Body"]))

    # ===== Pipeline reconstruction =====
    e.append(Paragraph("2.  The pipeline, end to end (what / why / effect / alternative)", st["H1"]))
    e.append(grid([
        [PH("Stage", st), PH("What it does", st), PH("Why", st), PH("Alternative & why not", st)],
        [P("<b>Load per-run</b>", st), P("Each of 28/10/53 runs kept separate", st),
         P("Runs are independent experiments", st), P("Concatenate all — would splice unrelated runs (a bug we fixed)", st)],
        [P("<b>StandardScaler</b> (train only)", st), P("Each sensor → mean 0, std 1", st),
         P("Stops high-variance sensors dominating MSE loss", st), P("Min-max — more outlier-sensitive; fitting on val/test leaks", st)],
        [P("<b>Windowing</b> 200×18, stride 10 (train) / 1 (eval)", st), P("Slice contiguous 200-step windows per run", st),
         P("Anomalies are slow drifts needing wide context", st), P("Point-wise/short windows — blind to drifts (notebook 04 showed F1 rises with window size)", st)],
        [P("<b>CNN autoencoder</b>", st), P("Compress→reconstruct normal windows", st),
         P("No labels → learn normal, flag deviations", st), P("Supervised classifier — impossible (no anomaly labels)", st)],
        [P("<b>Training</b> MSE + Adam, checkpoint sel.", st), P("50 epochs; keep best-validation epoch", st),
         P("Val score peaks (~epoch 45) before train loss does", st), P("Take last epoch — ships an over-trained, worse model", st)],
        [P("<b>Per-sensor error</b>", st), P("MSE over time → one value per sensor", st),
         P("Keeps which sensor is off", st), P("Flat mean error — dilutes one sensor's spike with another's noise", st)],
        [P("<b>Mahalanobis</b> (Ledoit-Wolf)", st), P("(x−μ)ᵀΣ⁻¹(x−μ) vs normal model", st),
         P("Weights each sensor by its own normal variability", st), P("Euclidean/flat MSE — ignores per-sensor scale & correlations", st)],
        [P("<b>Per-timestep expand</b>", st), P("Average overlapping window scores", st),
         P("Submission is per-timestep", st), P("Broadcast a block label — coarse, mislabels short anomalies", st)],
        [P("<b>Threshold</b> F-beta on val", st), P("Sweep PR curve, pick best cutoff", st),
         P("Turn a score into 0/1; β=0.5 curbs over-flagging", st), P("Fixed/quantile threshold — not tuned to the data", st)],
        [P("<b>Evaluate</b>", st), P("P/R/F1/ROC-AUC", st),
         P("AUC = ranking quality (the ceiling)", st), P("Accuracy — meaningless under class imbalance", st)],
    ], [2.6 * cm, 4.0 * cm, 4.0 * cm, 5.2 * cm]))

    e.append(PageBreak())

    # ===== Worked example =====
    e.append(Paragraph("3.  Worked numeric example (real validation window)", st["H1"]))
    e.append(Paragraph("A real 6-timestep slice from validation run 1 (t = 789–794), which is labeled anomalous. "
                       "We follow two sensors: <b>T701</b> (a temperature) and <b>FT703</b> (product-stream flow — "
                       "the strongest anomaly-signal sensor).", st["Body"]))
    e.append(Paragraph("Step 1 — raw values, then standardize with the train statistics "
                       "(T701: mean 370.4, std 58.4; FT703: mean 0.17, std 0.155). z = (x − mean) / std:", st["H3"]))
    e.append(grid([
        [PH("t", st), PH("T701 raw", st), PH("T701 z", st), PH("FT703 raw", st), PH("FT703 z", st), PH("label", st)],
        [P("789", st), P("445.5", st), P("+1.29", st), P("0.1", st), P("−0.46", st), P("1", st)],
        [P("790", st), P("445.5", st), P("+1.29", st), P("0.1", st), P("−0.46", st), P("1", st)],
        [P("791", st), P("445.5", st), P("+1.29", st), P("0.1", st), P("−0.46", st), P("1", st)],
        [P("792", st), P("445.5", st), P("+1.29", st), P("0.2", st), P("+0.19", st), P("1", st)],
        [P("793", st), P("445.5", st), P("+1.29", st), P("<b>1.1</b>", st), P("<b>+5.99</b>", st), P("1", st)],
        [P("794", st), P("445.6", st), P("+1.29", st), P("0.3", st), P("+0.83", st), P("1", st)],
    ], [1.3 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 1.8 * cm]))
    e.append(Paragraph("Note the signature: T701 sits flat and elevated (z ≈ +1.29); FT703 <b>spikes to z = +5.99</b> "
                       "at t = 793 — six standard deviations above normal flow.", st["Cap"]))

    e.append(Paragraph("Step 2 — the CNN reconstructs the window. Trained only on normal data, it 'expects' FT703 "
                       "to stay near its usual range, so it reconstructs the spike poorly. Per-sensor squared error "
                       "(illustrative, standardized units):", st["H3"]))
    e.append(grid([
        [PH("sensor", st), PH("actual z (t=793)", st), PH("CNN reconstructs", st), PH("per-sensor error", st)],
        [P("T701", st), P("+1.29", st), P("+1.27 (easy, flat)", st), P("≈ 0.04  (small)", st)],
        [P("FT703", st), P("+5.99", st), P("+0.35 (expects normal)", st), P("≈ 4.8  (large)", st)],
    ], [3.0 * cm, 3.6 * cm, 4.2 * cm, 4.0 * cm]))

    e.append(Paragraph("Step 3 — Mahalanobis distance vs the normal error model. Suppose the normal per-sensor error "
                       "has mean μ = [0.05, 0.20] and (diagonal, for illustration) covariance giving "
                       "Σ⁻¹ = diag(1000, 50) — T701's error normally varies little, FT703's more. For our vector "
                       "x = [0.04, 4.8]:", st["H3"]))
    e.append(Paragraph("diff = x − μ = [−0.01, 4.60].  &nbsp; d² = 1000·(−0.01)² + 50·(4.60)² = 0.1 + 1058 = "
                       "<b>1058</b>.  &nbsp; For a normal vector x = [0.06, 0.25]: d² = 0.1 + 0.13 = <b>0.23</b>.", st["Body"]))
    e.append(Paragraph("Step 4 — decision. The validation-selected threshold is ≈ 43 (raw Mahalanobis units). "
                       "1058 ≫ 43 → <b>flagged anomalous</b> (correct); 0.23 ≪ 43 → normal. The single FT703 spike "
                       "drove the score, exactly because Mahalanobis weights it by FT703's own variability instead "
                       "of averaging it away.", st["Key"]))

    # ===== Concept primers =====
    e.append(Paragraph("4.  Concept primer (every method, one bite each)", st["H1"]))
    e.append(bullets([
        "<b>Autoencoder</b> — a network that compresses input through a bottleneck and reconstructs it; trained on "
        "normal data, it reconstructs anomalies poorly, so reconstruction error is the anomaly score.",
        "<b>1-D convolution</b> — slides a small learned filter along time to detect local temporal shapes; sensors "
        "are the input 'channels'.",
        "<b>Dilated convolution</b> — spreads the filter's taps apart (gaps of 1,2,4,8) so stacking a few layers "
        "'sees' ~63 timesteps of context with few parameters — needed for slow drifts.",
        "<b>Residual connection</b> — output = F(x) + x; lets gradients flow through deep stacks and lets each block "
        "learn a refinement (from ResNet/TCN).",
        "<b>Batch normalization</b> — rescales layer activations per mini-batch to stabilise and speed training.",
        "<b>Max-pooling / bottleneck</b> — downsamples time; the narrow middle forces compression so the AE can't "
        "just copy input to output (which would reconstruct anomalies too).",
        "<b>StandardScaler (z-score)</b> — (x − mean)/std per feature, fit on training data only to avoid leakage.",
        "<b>Mahalanobis distance</b> — distance from a point to a distribution that accounts for each dimension's "
        "variance and the correlations between them: (x−μ)ᵀΣ⁻¹(x−μ).",
        "<b>Ledoit-Wolf covariance</b> — a shrinkage estimator of Σ that stays well-conditioned (invertible) even "
        "with few samples relative to dimensions.",
        "<b>Precision / Recall / F1</b> — precision = fraction of flags that are correct; recall = fraction of real "
        "anomalies caught; F1 = their harmonic mean. <b>F-beta</b> with β&lt;1 weights precision more (β=0.5 here).",
        "<b>Precision-Recall curve & ROC-AUC</b> — sweeping every threshold traces P vs R; ROC-AUC summarises the "
        "threshold-free ranking quality (our ceiling ≈ 0.76).",
        "<b>Isolation Forest</b> — ensemble of random trees; anomalies isolate in few splits (global, tree-based). "
        "Baseline; never earned fusion weight.",
        "<b>Local Outlier Factor (LOF)</b> — flags points whose local neighbourhood density is much lower than their "
        "neighbours' (local, density-based). Round-5 baseline; also rejected.",
        "<b>GroupKFold cross-validation</b> — k-fold CV where whole runs stay together in a fold, so no run leaks "
        "between train and test — used to choose the fusion weight honestly.",
    ], st))

    e.append(PageBreak())

    # ===== Dependency map =====
    e.append(Paragraph("5.  Component dependency map", st["H1"]))
    e.append(dependency_map())
    e.append(Paragraph("Figure. config.py feeds all hyperparameters. run_cnn.py orchestrates: window & scale the "
                       "data, train the CNN (with validation checkpoint selection), score per-sensor error, fit the "
                       "Mahalanobis normal model, threshold on validation, and write the submission.", st["Cap"]))

    # ===== Q&A =====
    e.append(Paragraph("6.  Likely questions & model answers", st["H1"]))
    qa = [
        ("Why an autoencoder and not a classifier?",
         "There are no anomaly labels in the training data, so a supervised classifier can't be trained. An "
         "autoencoder learns 'normal' from unlabeled data and flags what it can't reconstruct."),
        ("How is your anomaly score computed exactly?",
         "Per-sensor reconstruction error (MSE over time, one value per sensor) turned into a Mahalanobis distance "
         "from a Gaussian model of normal error, so each sensor is weighted by its own normal variability and "
         "sensor correlations are accounted for."),
        ("Why Mahalanobis instead of plain reconstruction error?",
         "Flat error averages away which sensor is off — a genuine spike on a precise sensor gets buried under a "
         "noisy sensor's ordinary error. Mahalanobis rescales per sensor; it moved F1 from ~0.48 to ~0.60."),
        ("Your training data may contain anomalies — isn't that a problem?",
         "Yes, and we acknowledge it: the loss keeps falling while validation peaks around epoch 45, the signature "
         "of the AE slowly learning to reconstruct anomalies too. We mitigate with best-checkpoint selection, and "
         "the top future-work item is contamination-robust training (down-weighting likely-anomalous windows)."),
        ("How do you choose the threshold, and isn't that overfitting to validation?",
         "We sweep the precision-recall curve on the 10-run validation set and pick the F-beta-optimal cutoff. It is "
         "a risk — everything is tuned on those 10 runs — but the portal F1 (0.63) exceeds validation (0.60), so we "
         "are not badly over-fit. A run-grouped CV (already built) would give a fully honest estimate."),
        ("Why F-beta = 0.5 and not F1?",
         "The score distributions overlap heavily, so plain F1 kept choosing a trigger-happy threshold flagging "
         "65–88% of timesteps. β=0.5 weights precision to curb that. (Diagnostic caveat: if the portal grades plain "
         "F1, an F1-tuned threshold scores ~0.045 higher — a change we are testing.)"),
        ("What limits your F1 — why not 0.9?",
         "ROC-AUC ≈ 0.76: the score ranking genuinely overlaps, so no threshold separates cleanly. F1 is bounded by "
         "score separability, which is a model/score problem, not a threshold problem."),
        ("Why dilated convolutions?",
         "The anomalies are slow drifts. A plain kernel-3 conv sees only 3 timesteps; stacking dilations 1-2-4-8 "
         "gives a ~63-step receptive field with no extra parameters, so the model can actually see the drift."),
        ("Why did you drop the hybrid / Isolation Forest?",
         "Across three fusion attempts (IF, raw-LOF, standardized-LOF) with leakage-free run-grouped CV, the tabular "
         "detector always got ~0 weight and never beat the CNN alone — its AUC (~0.67) is well below the CNN's "
         "(~0.77). Shipping the CNN alone is the honest, stronger choice."),
        ("How do you prevent data leakage?",
         "The scaler is fit on train only; windows never cross run boundaries; the Mahalanobis model uses only "
         "confirmed-normal windows; and fusion weights are chosen with GroupKFold so no run appears in both train "
         "and test folds."),
        ("Why per-run windowing?",
         "Each run is an independent experiment; concatenating them before windowing would create windows splicing "
         "two unrelated runs — a fake transient. We window each run separately (a bug we found and fixed)."),
        ("Is your validation estimate trustworthy?",
         "Reasonably: the portal F1 (0.63) is slightly above validation (0.60), so validation is if anything "
         "pessimistic. The weakness is that only 10 labeled runs drive every tuning decision."),
    ]
    for q, a in qa:
        e.append(Paragraph("Q: " + q, st["Q"]))
        e.append(Paragraph("A: " + a, st["A"]))

    # ===== Limitations =====
    e.append(Paragraph("7.  Limitations & future work (say these before they ask)", st["H1"]))
    e.append(bullets([
        "<b>Score-separability ceiling</b> (AUC ≈ 0.76) — the main limit; needs a better score, not a better cutoff.",
        "<b>Training-set contamination</b> — 'normal' data isn't guaranteed clean; contamination-robust training "
        "(kNN-density loss weighting, CLEANet/RiAD-style) is the highest-leverage next step.",
        "<b>Single global threshold across runs</b> — diagnostics show raw scores vary ~100× per run, so two "
        "validation runs score F1=0 despite correct within-run ranking; per-run score normalization is being tested.",
        "<b>Everything tuned on 10 validation runs</b> — use the run-grouped CV for an unbiased estimate.",
        "<b>Metric mismatch</b> — we optimise F0.5 while the portal may grade F1; aligning them is ~free F1.",
        "<b>Run-to-run variance</b> — identical configs vary ±0.02 F1; seed-ensembling would stabilise and lift it.",
    ], st))

    # ===== Cheat sheet =====
    e.append(PageBreak())
    e.append(Paragraph("8.  Cheat sheet (last-minute review)", st["H1"]))
    e.append(Paragraph("<b>One-sentence project:</b> an unsupervised dilated-CNN autoencoder that learns normal "
                       "18-sensor behaviour and flags deviations via Mahalanobis distance; portal F1 = 0.63.", st["Cheat"]))
    e.append(Paragraph("<b>Pipeline in 9 words:</b> load → scale → window → reconstruct → per-sensor error → "
                       "Mahalanobis → expand → threshold → predict.", st["Cheat"]))
    e.append(Paragraph("<b>Key numbers:</b> 18 sensors · window 200, stride 10/1 · 50 epochs, best ≈ epoch 45 · "
                       "encoder dilations 1-2-4-8 (RF ≈ 63) · bottleneck 64×100 · β = 0.5 · val F1 ≈ 0.60, "
                       "ROC-AUC ≈ 0.76, portal F1 = 0.63.", st["Cheat"]))
    e.append(Paragraph("<b>Three 'why' anchors:</b> (1) no labels → autoencoder; (2) slow drifts → dilated + window "
                       "200; (3) one sensor matters → per-sensor Mahalanobis, not flat error.", st["Cheat"]))
    e.append(Paragraph("<b>The ceiling line:</b> \"F1 is capped by score separability (AUC 0.76); threshold tuning "
                       "can't beat overlap — only a better score can.\"", st["Cheat"]))
    e.append(Paragraph("<b>The hybrid line:</b> \"We tested fusion three ways with leakage-free CV; the tabular "
                       "detector always got ~0 weight, so CNN-alone is the honest answer.\"", st["Cheat"]))
    e.append(Paragraph("<b>If stuck, retreat to:</b> \"We learn normal, measure surprise per sensor, and pick a "
                       "threshold on validation — and we validated every claim honestly, including the negatives.\"", st["Cheat"]))
    e.append(Paragraph("<b>Biggest weakness to own first:</b> training data may contain anomalies (contamination); "
                       "next step is robust training. Say it before they do.", st["Cheat"]))

    doc.build(e)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
