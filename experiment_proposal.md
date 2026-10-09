# Proposed real-data experiment: "which nudge do we scale?"

A design memo, not committed. Companion script: `megastudy_prototype.py`.

---

## 1. Why SPRINT should stay the anecdote and not be the validation

The `sprint_talk.ipynb` experiment works, but four things make it hard to defend as the
paper's conditional-validity evidence:

1. **`M = 4`, `k = 2`.** There are only six possible selection events, and 92% of re-runs
   pick `75+`. Selection barely bites.
2. **The truth is assumed.** Setting `mu = theta_hat` and resampling
   `T ~ N(theta_hat, Sigma_hat)` is a simulation wearing a data analysis's clothes. A
   referee will say you put the answer in by hand — and the whole result (naive conditional
   coverage 0.52 for `65-69`) rests on `65-69` truly being worst, which is an artifact of
   one plug-in.
3. **Naive looks fine.** Marginal coverage 0.907, and naive conditional coverage is 0.91,
   0.91, 0.93 in three of four groups. The failure lives in a 6%-probability cell. That is a
   thin reed for a JRSSB applied section.
4. **SPRINT's own interaction tests found no heterogeneity**, so "the subgroup that benefits
   most" is arguably estimating noise. That invites the objection that the estimand itself
   is uninteresting.

Keep SPRINT as the opening slide — it is a recognisable trial and the
`(-78.95, 1.06)` polyhedral interval is a great visual. Put the validation somewhere else.

---

## 2. The proposal

**Setting.** A behavioural-science *megastudy*: one population, `M` randomised
intervention arms plus a holdout, binary outcome.

**Estimand.** `theta_j = P(Y = 1 | arm j) - P(Y = 1 | holdout)`, `j = 1..M`.

**Selection.** Top-`k` arms by `theta_hat` (`k = 1` and `k = 3`).

**Report.** A confidence interval for each selected arm, valid *conditional on the selected
set*.

### Why this is genuinely a conditional question

The entire stated purpose of a megastudy is "test many interventions at once, then scale the
winner." That makes the identity of the winner the whole point:

- **The thing that gets deployed is the arm that won.** A guarantee that holds on average
  over which arm *might* have won does not price the programme that will actually be run.
- **Costs differ by arm.** A text reminder and a $10 incentive have different cost per
  marginal vaccination. The cost-effectiveness calculation needs *the selected arm's own*
  effect, not an average over counterfactual winners.
- **The confirmatory trial is powered off the selected arm's estimate.** This is the
  sharpest version: winner's curse → systematically underpowered confirmatory studies, and
  the error is conditional by construction.

This is the "it matters what was selected" property you asked for, and it is not a
statistician's framing — it is how those papers describe their own pipeline.

---

## 3. The validity check: design-based subsampling, no assumed truth

This is the part that fixes SPRINT's problem 2. **Nothing is simulated from a model.**

Let arm `j` have `N_j` participants and `X_j` successes in the published megastudy. Define
the target as the full-sample rate

```
p_j = X_j / N_j          (a known finite-population parameter — not an estimate of anything)
theta_j = p_j - p_0
```

Now draw `R` "small megastudies". For replicate `r`, sample `n_j = ceil(f * N_j)`
participants from arm `j` without replacement — i.e. draw

```
X_j^(r) ~ Hypergeometric(N_j, X_j, n_j)
```

which needs **only the published arm-level counts**, no individual records. Compute
`theta_hat^(r)` and `Sigma_hat^(r)` from the pilot alone, with the finite-population
correction

```
Var(X_j^(r)/n_j) = (1 - f) * p_j(1 - p_j) / n_j * N_j/(N_j - 1)
```

and the shared holdout giving an equicorrelated `Sigma` (off-diagonal = variance of the
holdout rate — the repo's methods already accept a general `Sigma`). Then select top-`k` and
build intervals using pilot data only, exactly as a smaller megastudy would.

**Report**, for each arm `j`, over the `R` replicates:

- selection frequency;
- **conditional coverage** of `theta_j` among replicates that selected `j`;
- median interval length;
- mean signed error of `theta_hat_j` among replicates that selected `j` (= the realised
  winner's curse, in percentage points);
- implied power of a confirmatory trial sized off the naive estimate.

**Why this is airtight.** The target is a known number. The randomness is a real random
split of real participants. The only assumption is exchangeability of participants within an
arm — which the original randomisation delivers. This is a *design-based* experiment, not a
model-based simulation, and that distinction is worth a sentence in the paper.

Choose `f` so that the per-arm standard error is comparable to the spread of true effects
(`f ≈ 0.05–0.15` for a 700k-person megastudy). Sweep `f` to show the transition.

**Optional external check.** If a follow-up megastudy exists that re-ran the winning arms,
one extra column — "independent replication estimate" — next to the naive and conditional
intervals from round 1 is the single most persuasive number you can put in the paper.

---

## 4. Predicted numbers

`megastudy_prototype.py` runs the Gaussian analogue at megastudy-plausible calibration
(`M = 22`, `k = 3`, true effects 2.0 pp down to -0.2 pp, per-arm SE 0.6–1.0 pp,
`alpha = 0.10`, `B = 1200`). **This is a feasibility calibration, not real megastudy data** —
I could not reach the network from this session. It predicts what the real experiment
should produce.

### Conditional coverage by arm (nominal 0.90)

| arm | true effect (pp) | sel. freq. | Standard | Polyhedral | Randomized PSI | naive bias (pp) |
|---|---|---|---|---|---|---|
| 0 | 2.00 | 0.63 | 0.937 | 0.896 | 0.897 | +0.31 |
| 1 | 1.70 | 0.48 | 0.872 | 0.890 | 0.908 | +0.58 |
| 3 | 1.30 | 0.23 | 0.757 | 0.891 | 0.882 | +0.81 |
| 7 | 0.90 | 0.18 | 0.640 | 0.896 | 0.907 | +1.31 |
| 10 | 0.60 | 0.09 | 0.429 | 0.857 | 0.917 | +1.52 |
| 11 | 0.50 | 0.04 | **0.196** | 0.804 | 0.863 | +1.38 |
| 14 | 0.30 | 0.02 | **0.077** | 0.962 | 0.955 | +1.70 |
| 18 | 0.10 | 0.03 | **0.088** | 0.794 | 0.872 | +2.13 |

Marginal coverage over all reported winners: Standard 0.742, Polyhedral 0.896,
Randomized PSI 0.904. Median length: 2.28 / 5.03 / **3.28** pp.

### The slide that makes the conditional point

Add two *marginally valid* comparators, computed from the same replicates:

- **calibrated-marginal**: `theta_hat ± c * se` with `c` chosen so marginal coverage over
  winners is exactly 0.900 (gives `c = 2.15`);
- **FCR-BY** (Benjamini–Yekutieli 2005): level `1 - alpha*k/M`, `z = 2.47`.

| arm | true effect (pp) | calibrated-marginal | FCR-BY | Randomized PSI |
|---|---|---|---|---|
| 0 | 2.00 | 0.982 | 0.987 | 0.897 |
| 1 | 1.70 | 0.967 | 0.988 | 0.908 |
| 3 | 1.30 | 0.899 | 0.935 | 0.882 |
| 9 | 0.70 | 0.879 | 0.955 | 0.934 |
| 11 | 0.50 | **0.543** | **0.652** | 0.863 |
| 14 | 0.30 | **0.423** | **0.692** | 0.955 |
| 18 | 0.10 | **0.412** | **0.735** | 0.872 |
| | marginal coverage | 0.900 | 0.950 | 0.904 |
| | median length (pp) | 2.97 | **3.41** | **3.28** |

Read the last two rows together with the last three data rows. This is the argument:

> A procedure with **exactly 90% marginal coverage** covers the truth 98% of the time when
> the winner deserved to win and 41% of the time when it won by luck. FCR-BY intervals are
> **longer** than ours (3.41 vs 3.28 pp) and still cover only 65–74% on the arms that win by
> luck. You cannot buy conditional validity with a wider one-size-fits-all interval, because
> the needed width depends on what was selected — and you cannot tell from the data which
> case you are in. That is what conditioning on the selection event is for.

### The consequence slide

From the same replicates: the reported best arm has mean estimate 2.54 pp against mean truth
1.44 pp — an overstatement of **1.11 pp (×1.77)**. A two-arm confirmatory trial sized for
80% power off that naive estimate attains **44% power** (median; sized at the paper's
`alpha = 0.10`, two-sided — at `alpha = 0.05` it is 39%).

That is one sentence any audience understands, and it is the reason a practitioner should
care about the method rather than about coverage tables.

---

## 5. Data options

Minimal requirements: per-arm `(n_j, X_j)` counts; `M >= 15`; randomised allocation; a
spread of true effects (one or two good arms, many mediocre); total `n` large enough that
`f ≈ 0.05–0.15` leaves per-arm SE comparable to the effect spread.

Candidates, best first (**availability needs checking — I had no network access here**):

1. **Walmart flu-vaccination megastudy** (Milkman et al., *PNAS* 2022): ~22 text-message
   arms + holdout, `n ≈ 690,000`, uptake ≈ 30%, arm effects ≈ 0–1.1 pp. Binary outcome, huge
   `n` for an essentially-exact truth, counts published. **Recommended.**
2. **24 Hour Fitness "StepUp" exercise megastudy** (Milkman et al., *Nature* 2021): 53 arms,
   `n ≈ 61,000`. `M = 53` is the most dramatic, and the *published* analysis is already in
   the noise-dominated regime — so the single-run table on the real full data is itself a
   headline, with subsampling validating the method at the same signal-to-noise.
3. **Penn Medicine / Geisinger text megastudy** (Milkman et al., *PNAS* 2021): 19 arms,
   `n ≈ 47,000`. Smaller, same story.
4. **Upworthy Research Archive** (~32,000 randomised headline tests, 4–14 arms each). Use if
   you want thousands of *independent* top-`k` problems and therefore tiny Monte Carlo error
   on the coverage curve. Weaker as a scientific story, and arms within a package get near-
   equal impressions, so the precision heterogeneity is milder.

Swap in whatever you can actually obtain — the experiment design in §3 is dataset-agnostic.

---

## 6. Referee objections and answers

| Objection | Answer |
|---|---|
| "This is just subsampling, not a real second study." | The target is a known finite-population parameter; the randomness is a real random split of real participants; the FPC is applied. Design-based, not model-based. Contrast with the plug-in truth in the current SPRINT experiment. |
| "Arms are exchangeable, so conditional = marginal." | False here: the arms have genuinely different effects, so coverage is indexed by the *true rank of the winner*, which varies. The tables in §4 are the proof. Sweep `f` to show it is not knife-edge. |
| "Why not just shrink / use empirical Bayes?" | Shrinkage gives a better point estimate, not a valid interval for the selected arm. Worth one row in the table. |
| "Why not FCR control?" | Answered above: longer intervals, still 65–74% conditional coverage where it matters. This comparison should be in the paper. |
| "Non-diagonal `Sigma`." | The shared holdout makes `Sigma` equicorrelated. This is a feature — a realistic non-diagonal `Sigma` that the existing code already handles. Verify `PolyhedralTopKInference` on it. |

---

## 7. Compute

Measured in this session with the repo's own code at `M = 22`, `k = 3`, `grid_size = 500`:
randomized selection 0.002 s, randomized CI **0.154 s**, polyhedral CI 0.006 s. So
`R = 2000` replicates × 3 methods ≈ **6 minutes single-threaded**. At `M = 53` expect
roughly 3–4×. Cost is not a constraint; `R = 10,000` is affordable, which matters because
the interesting cells (arms selected 2–4% of the time) need the replicates.

---

## 8. Alternatives, ranked

If the megastudy data does not pan out:

1. **Preclinical target screens with an independent replication** (DepMap Avana vs KY
   libraries; GDSC vs CCLE drug sensitivity). Truth comes from a different lab, which is the
   most convincing check possible. "We took the top 5 hits forward" is a real, expensive
   decision, and the preclinical reproducibility crisis is pre-sold. Weaker on the
   estimand's cleanliness (no randomisation).
2. **Hospital / school league tables** (CMS Hospital Compare, state assessment files). The
   precision heterogeneity is extreme and the conditional question is unimpeachable ("this
   hospital was named top-5 — is it good?"). Kahneman's small-schools example makes it
   instantly legible. Costs: observational estimand, and CMS already shrinks, so you must
   handle "why not just shrink".
3. **GWAS / eQTL winner's curse with a replication cohort.** The failure mode already has a
   name and a literature, which is both the attraction and the risk (existing
   winner's-curse corrections become required comparators).
4. **SPRINT, expanded.** Zero data friction — the patient-level file is already in `result/`
   and has `SUB_CKD`, `SUB_CVD`, `SUB_SENIOR`, `RACE_BLACK`, `FEMALE`, `SBPTERTILE`,
   `SMOKE_3CAT`, `ASPIRIN`, `STATIN`, `NOAGENTS`, `NEWSITEID`. Go from 4 age groups to the
   ~14 pre-specified subgroups of the published forest plot; "the subgroup that benefits
   most" is real practice and a notorious source of false claims. **But** 460 events total
   is too few for the §3 split design — a 25% pilot leaves ~115 events across 14 subgroups.
   Use this to upgrade the single-run table, not as the validation.

---

## 9. Unrelated, but you should know

`result/baseline.csv` and `result/outcomes.csv` are **tracked in git** (commit `92699ce`,
"Add files via upload") and contain individual-level SPRINT records — 9,361 participants
with age, race, blood pressure, labs, clinic site ID, and event times. There is no
`.gitignore`. The repository's own README says:

> Do not commit restricted individual-level SPRINT files to a public repository.

Since `github.com/snigdhagit/winners-with-confidence` is public, this is likely a BioLINCC
data-use-agreement problem, and deleting the files in a new commit would not fix it — the
blobs stay in history. Removing them needs a history rewrite plus a force-push, and probably
a note to BioLINCC. Say the word and I will prepare the rewrite.
