# Layer-wise probe agreement as a confidence signal

**Setup.** Qwen3-VL-2B-Instruct, frozen. One full-depth forward pass per
(image, question, candidate) with `output_hidden_states=True` gives the
last-token representation at all 29 depths (embedding output + 28 decoder
layers) in a single backbone call. A separate StandardScaler+LogisticRegression
probe is fit per depth on a 4,000-question A-OKVQA training subset (16,000
candidates) and evaluated on the full 1,145-question validation split (matches
the paper's Appendix C linear-probe protocol, just one probe per layer instead
of one probe at a fixed depth).

For each validation question this gives 29 independent layer votes for the
answer. Per-question signals:
- **plurality_frac**: fraction of the 29 layers whose predicted candidate
  equals the modal (most common) vote
- **agree_with_final**: fraction of layers agreeing with the last layer's vote
- **vote_entropy**: entropy (bits) of the 29-layer vote distribution
- baseline: the **final layer's own softmax margin** (top-1 minus top-2
  candidate probability) — the standard single-layer confidence signal

Scripts: `extract_all_layers.py` (caching) → `fit_layer_probes.py` (per-layer
probes + agreement analysis) → `combine_signals.py` (does agreement add to the
margin baseline?) → `make_figures.py`. Cached features and full results/figures
are under `outputs/`.

## Findings

1. **Reproduces the paper's Appendix Fig. 12 pattern** on this subset: probe
   accuracy rises from chance (24.5%) to ~83% by layer ~17 and plateaus
   (`outputs/figures/layer_accuracy_curve.png`).

2. **The hypothesis holds**: when more layers agree, the final answer is more
   often correct — a clean, monotone relationship (59% → 76% → 93% accuracy
   across increasing agreement bins;
   `outputs/figures/accuracy_by_agreement_bin.png`). All three agreement
   signals score well above chance at predicting correctness (AUROC
   0.74–0.76 vs. 0.5 chance).

3. **But agreement is not a substitute for the final layer's own softmax
   margin**, which alone reaches AUROC 0.83 — clearly stronger than any
   agreement metric alone (`outputs/figures/auroc_comparison.png`,
   `outputs/figures/risk_coverage.png`). Restricting agreement to only the
   "already-plateaued" late-half layers (14–28) is *worse* (AUROC 0.69) than
   using all 29 layers, including the near-chance early ones — an interesting
   detail: the noisy early-layer votes apparently still carry independent
   signal that isn't present once everything downstream has converged.

4. **Agreement gives a small, consistent complementary boost on top of the
   margin baseline**: combining margin with `agree_with_final` or with the
   late-half plurality fraction raises 5-fold CV AUROC from 0.833 (margin
   alone) to ~0.840, a modest but real gain
   (`outputs/figures/combined_signal_auroc.png`,
   `outputs/agreement_analysis/combined_signal_results.json`).

## Honest takeaway for the paper

The original framing ("if layers agree → model knows the answer") is
supported, but the more precise and more interesting claim is: **cross-layer
probe agreement is correlated with correctness and adds a small amount of
information beyond the standard softmax-margin confidence — it is a
complementary signal, not a replacement.** This is a defensible, appropriately
hedged extension of the existing layer-wise probe analysis (Appendix Figs
11–12) rather than an overclaim.

## Part 2: agreement-gated adaptive-depth inference

**Idea.** Instead of pruning every example to the same fixed decoder depth
(the paper's existing contiguous keep-first-N pruning), stop the forward pass
*per example* as soon as a stopping rule fires, using the layer-wise probes as
cheap intermediate exit classifiers. This is simulated entirely from the
per-layer votes/margins already cached in Part 1 — no new backbone passes are
needed, because decoder layers are causal: the hidden state read at layer L
from a full-depth pass is identical to what an actually-truncated L-layer
model would produce (this is exactly the argument your paper already makes
for the half-depth backbone). Script: `adaptive_depth_policy.py` →
`compare_to_fixed_baseline.py` → `make_adaptive_figures.py`.

Three gating policies, all causal (layer L's decision only uses information
from layers ≤ L), with a full-depth fallback if the rule never fires:
- **agreement-gated**: stop once the last W layer-probes agree, i.e. the exact
  mechanism the framing originally proposed.
- **confidence-gated**: stop once the current layer's own softmax margin
  exceeds a threshold τ (a standard early-exit criterion, included as a
  natural baseline/alternative given Part 1's finding that margin is the
  stronger single signal).
- **agreement AND confidence**: require both.

Each policy is compared against the fixed-depth curve at the *same average
depth* (linear interpolation between adjacent integer depths), so the
comparison is apples-to-apples on compute, not just accuracy
(`outputs/adaptive_depth/comparison.json`,
`outputs/figures/adaptive_depth_frontier.png`,
`outputs/figures/adaptive_depth_frontier_zoom.png`).

**Result.**
- **Confidence-gated (margin-threshold) adaptive depth clearly beats fixed
  uniform pruning** across most of the compute range: e.g. +6.5 points at
  ~26% of full depth (67.1% vs. an interpolated 60.6% fixed baseline at the
  same average 7.3 layers), +5.2 points at ~37% depth, and it reaches the
  full 28-layer accuracy (82.8%) using only ~18.3 layers on average (65% of
  full depth) — i.e. it needs noticeably less average compute than any fixed
  depth to hit the same accuracy.
- **Pure agreement-gated adaptive depth does *not* clearly beat fixed
  pruning** — it's roughly tied with the fixed-depth curve throughout (gains
  and losses both within ~±1–4 points, no consistent direction), including
  one range (avg depth ~16.4) where it's ~1.2 points *worse* than fixed.
- **Combining agreement AND confidence tracks confidence-gating alone**, with
  no clear further improvement — consistent with Part 1's finding that
  agreement is a weak complementary signal on top of margin, not an
  independent source of gains.

## Honest takeaway for the paper

The originally-proposed mechanism (stop when layers agree) does validate the
*correctness* story from Part 1 but does **not**, on its own, produce a
better compute/accuracy trade-off than the paper's existing fixed-depth
pruning. The genuinely positive, publishable result that falls out of this
extension is the more general one: **per-example adaptive-depth inference,
gated on the layer probes' own confidence, Pareto-dominates fixed uniform
pruning** — the same per-layer-probe infrastructure your paper already builds
(Appendix Figs. 11–12) turns into a real efficiency method once the gating
signal is confidence rather than agreement. Frame it as "layer-wise probes
enable adaptive per-example depth, and confidence is the effective gating
signal, with agreement as a validated but secondary correctness correlate" —
this keeps every claim backed by a number above and avoids overselling the
agreement mechanism specifically.

## Part 3: replication on CUB-200

**Setup.** Same method as Parts 1-2, same Qwen3-VL-2B-Instruct backbone, same
scripts pattern (`extract_cub200_layers.py` -> `fit_layer_probes.py` ->
`adaptive_depth_policy_cub200.py` -> `compare_to_fixed_baseline_cub200.py` ->
`make_adaptive_figures_cub200.py`), applied to `Donghyun99/CUB-200-2011`
(4000 train / 1145 validation images, matching the A-OKVQA sample sizes for a
directly comparable compute budget). **Important protocol caveat**: CUB-200 is
natively 200-way, but scoring all 200 species per image would cost ~50x more
backbone compute than this A-OKVQA-scale run. Following the paper's own stated
alternative protocol ("the candidate strings are class names, using either
four sampled choices or all 200 CUB-200 species"), each image gets 1 gold
species + 3 randomly sampled distractor species (n_choices=4), not the full
200-way task. This makes the task much easier than the paper's headline
200-way CUB-200 numbers (~11-14% for a frozen linear probe) — final-layer
accuracy here is 93.9% — so absolute accuracies are **not** comparable to
Table 1; only the *shape* of the adaptive-depth vs. fixed-depth comparison is.

**Results** (`outputs/cub200/adaptive_depth/comparison.json`,
`outputs/cub200/figures/adaptive_depth_frontier*.png`):

- **The Part 2 pattern reproduces on CUB-200**: confidence-gated (margin
  threshold) adaptive depth beats fixed uniform pruning across the
  low-to-mid compute range, and by a *wider* margin than on A-OKVQA — e.g.
  +7.4 points at avg depth 5.4 (86.0% vs. 78.6% fixed-interpolated, tau=0.04),
  +6.3 points at avg depth 6.6, +3.5 points at avg depth 9.0. It also reaches
  full-depth accuracy (93.9%) at avg depth ~18.4 (66% of 28 layers), similar
  compute savings to A-OKVQA's ~65%.
- **Agreement-gated (window agreement) is again inconsistent**, and on
  CUB-200 it is markedly *worse* than fixed pruning at low depth: W=3 scores
  58.7% at avg depth 4.2 vs. 77.0% fixed-interpolated at the same depth (-18.3
  points) — a larger downside than anything seen on A-OKVQA. W=2 and W>=5 are
  roughly neutral-to-slightly-positive, so the failure is specific to
  small-window early stopping, not agreement-gating in general.
- **combined_or_W3 inherits window agreement's early-depth failure** (same
  -18 point gap at avg depth ~4.2, since OR-combining fires as soon as either
  rule fires and the agreement rule fires first and badly here).
  **combined_and tracks confidence-gating closely** and is never worse than
  margin-alone by more than ~1 point, consistent with Part 2.

**Honest takeaway.** The CUB-200 replication does *not* support the "Natural
next steps" hypothesis that a larger answer set would make *agreement* a
stronger gating signal — if anything agreement-gating is less reliable here
(a sharper early-depth failure mode) than on A-OKVQA. But it strongly
reinforces the Part 2 finding that **confidence (softmax margin), not
agreement, is the signal that makes per-example adaptive depth beat fixed
pruning** — the effect replicates on a second dataset with a different
answer-set size and an even larger low-depth accuracy gain. Because this run
used the 4-sampled-choices variant rather than the full 200-way task, a
faithful test of the large-label-space hypothesis would need to rerun with
all 200 candidates per image (~50x the compute of this run).

## Natural next steps (not yet run)

- Rerun the CUB-200 replication above with the full 200-way candidate set
  (not the 4-sampled-choices variant) to properly test whether a large label
  space changes the agreement-vs-confidence picture — this run's version used
  4 sampled choices for compute-budget reasons and left that question open.
- Repeat with Qwen3-VL-8B and/or InternVL3 (cross_family_probe already has the
  InternVL3 extraction code) to check the finding isn't backbone-specific.
- Full calibration analysis (reliability diagrams, ECE) rather than just AUROC
  and risk-coverage.
- Measure *wall-clock* latency for the adaptive-depth policy (this analysis
  uses layer count as a compute proxy, matching the paper's own convention,
  but an actual early-exit implementation should confirm real speedup net of
  the per-layer probe-evaluation overhead).
- A learned gating policy (e.g. train a tiny classifier on early-layer
  features to predict "will deeper layers change the answer") could plausibly
  beat both hand-set-threshold policies above.
