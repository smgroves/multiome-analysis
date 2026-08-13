# Organoid perturbation-walk predictions: does GEMM's fitted network reproduce RORB's real effect?

*(Scripts referenced below live under `network-inference-DIRECT-NET/claude_analysis/04_rorb_perturbation_validation/`.
Companion doc: `../domain_shift_diagnostic/FINDINGS.md` covers the separate question of why
organoid/mets_compiled validate poorly against GEMM's fitted rules in the first place —
the rewiring finding referenced below (that doc's §2/§9) is background for this one, not
duplicated here.)*

## 1. Validating "RORB knockdown moves NE toward Intermediate": archetype-level walk simulation + real organoid data (`run_organoid_perturbation_walks.py`, `validate_organoid_rorb_archetype_shift.py`)

`../domain_shift_diagnostic/FINDINGS.md` §9's leaf-conditional test showed genuine rewiring at the gene level; a follow-up direct
simulation of RORA_RORB's 9 fitted target genes (`diagnose_rorb_perturbation_transfer.py`,
not in this file's numbered sections) found only weak, non-significant sign agreement
between GEMM's predicted per-gene shift and organoid's real shGFP-vs-shRORB shift (6/9
genes, r=0.06-0.42). That result doesn't resolve the user's actual claim, which is at the
**archetype/state-space level**, not the single-gene level: "RORB knockdown moves NE
toward Intermediate archetypes." Two independent checks, deliberately kept separate so
they can agree or disagree informatively:

**Terminology note**: the rest of this section uses the network's original `Arc_N`
labels throughout the prose (as they appear in `6667/attractors/`), but the archetypes
have confirmed biological names: `Arc_1`=**Intermediate**, `Arc_2`=**nonNE2**,
`Arc_3`=**Secretory**, `Arc_4`=**nonNE1**, `Arc_5`=**NE1**, `Arc_6`=**NE2** (`Generalist_NE`/
`Generalist_nonNE` keep their own names). The walk plots (`walk_axis_position_*.png`)
label archetypes with these biological names directly.

**(A) GEMM's own archetype-level perturbation data, re-examined.** `attractors_filtered.txt`
(the basin set used for every perturbation/walk analysis on this network) has 5 usable
basins: `Generalist_NE`, `Generalist_nonNE`, `Arc_3`, `Arc_5`, `Arc_6` (`Arc_1`/`2`/`4`,
including an ad-hoc, unreplicated `Arc_1`="Intermediate" label found in one unrelated
plotting script, were excluded by the existing filtering step and have no computed
perturbation data). RORA_RORB knockdown significantly destabilizes `Generalist_NE`
(-0.334) and `Arc_5` (-0.415) per `6667/perturbations/.../perturbation_stats.csv`, and
existing long-walk population dynamics (`6667/walks/long_walks/4000_step_walks/`) starting
from `Arc_5` show unperturbed occupancy of 12.8% `Arc_5`/9.7% `Generalist_NE` collapsing to
2.8%/0.6% under knockdown — a near-total collapse of NE-basin occupancy. No walk starting
from `Generalist_NE` itself was ever aggregated (documented as a future option, not
executed, per direct feedback — the Arc archetypes are the more interesting comparison
point and this isn't needed for what follows).

**(B) Seeding BoBa-T's actual walk/perturbation machinery with organoid_shGFP's own real
cells, under GEMM's fitted rules — result: the destabilization direction reproduces.**
Bypassed the `parent_heatmap` continuous-validation machinery entirely and used
`bb.rw.long_random_walks` (the discrete async walk BoBa-T actually uses for perturbation
simulation) seeded from real organoid data: binarized organoid_shGFP cells grouped by
organoid's own `predicted.id` label (Seurat-based archetype calls), collapsed to one
representative discrete state per label (`bb.tl.find_avg_states`), then walked forward
4000 steps x 100 iterations under GEMM's fitted rules — unperturbed and with `RORA_RORB`
clamped off for the whole walk — measuring occupancy against GEMM's fixed 5-basin
`attractor_dict` (same basins as (A), not a new organoid-specific attractor set).

Face-validity check first: organoid's own `Neuroendocrine1`-labeled state turns out to be
*closer* to GEMM's `Generalist_NE`/`Arc_5` basins (Hamming distance 10) than organoid's own
`Generalist NE`-labeled state is (distance 20) — organoid's broader "Generalist NE" label
apparently captures a less extreme NE identity than its "Neuroendocrine1" label does, at
least in this Boolean-state-space sense.

Result, starting from `Neuroendocrine1` (the organoid state closest to GEMM's NE basins):

| basin | unperturbed | RORA_RORB knockdown |
|---|---|---|
| `Generalist_NE` | 10.46% | 0.40% |
| `Arc_5` | 4.48% | 0.16% |
| `Generalist_nonNE` | 0.16% | 0.42% |

A near-total collapse of NE-basin occupancy — directionally, and even proportionally,
comparable to GEMM's own native `Arc_5`-seeded result in (A). Starting from organoid's
`Intermediate`-labeled state shows the same qualitative pattern one step further along:
knockdown *decreases* `Generalist_NE`/`Arc_3`/`Arc_6` occupancy (1.65%→0.83%,
1.76%→0.68%, 1.53%→1.01%) while *increasing* `Generalist_nonNE` (7.37%→8.20%) — i.e. even
starting already past "NE," knockdown pushes further away from NE-associated basins and
toward the nonNE basin. **Starting from organoid's own literal `Generalist NE`-labeled
state gives the same direction but a much smaller magnitude** (1.43%→0.21% `Generalist_NE`),
consistent with that state simply starting farther from any GEMM basin to begin with (per
the face-validity check).

**Caveat that motivated a follow-up**: none of the organoid-seeded walks land cleanly on a
*specific* named alternative GEMM basin within the fixed radius — occupancy mostly shifts
into `"None"` (unclassified state-space region, e.g. `Neuroendocrine1`'s `None` share rises
from 84.0% to 98.4%) rather than converging on `Generalist_nonNE`/`Arc_3` specifically. That
leaves open a real question: does "moving into None" mean the walk drifts *randomly* away
from NE, or does it move measurably *toward* a specific archetype without quite reaching
it (including the 3 archetypes excluded from the filtered basin set — `Arc_1`, ad-hoc
-labeled "Intermediate" in one script, `Arc_2`, `Arc_4` — which still have a well-defined
theoretical average state to measure distance to, per `6667/attractors/average_states.txt`,
even though they were never used as basins).

**Follow-up (`diagnose_walk_archetype_distance.py`): it moves toward a specific cluster,
but not the one informally called "Intermediate."** For every walk (100 per condition),
computed the mean Hamming distance across the whole trajectory to each of the network's
8 archetype average states, and the minimum (closest-approach) distance reached; compared
unperturbed vs. knockdown per archetype via Mann-Whitney U, Bonferroni-corrected across
the 8 archetypes. Result, consistent across 3 of the 4 organoid starting states
(`Neuroendocrine1`, `Generalist NE`, `Neuroendocrine2`):

- **Significantly closer under knockdown** (both by mean distance across the walk and by
  closest approach, p as low as 1e-13): `Generalist_nonNE`, `Arc_2`, `Arc_4`.
- **Significantly farther under knockdown**: `Generalist_NE`, `Arc_5`, `Arc_6`, and —
  notably — **`Arc_1` itself** (e.g. `Neuroendocrine1` start: distance to `Arc_1` goes
  from 21.3 to 25.6, p=7e-11; `Generalist NE` start: 24.2 to 28.3, p=2e-12).
- `Arc_3` shows no consistent signal (significant but tiny for one start, not significant
  for others).
- Starting from organoid's own `Intermediate`-labeled state, no archetype shows a
  significant shift either direction — consistent with that population already sitting
  wherever it's going to sit.

**Resolved by placing every archetype on one shared NE↔nonNE axis, rather than treating
Hamming distance to each archetype as independent.** `Arc_1` is confirmed (by the user,
who has direct knowledge of this network's archetype identities) to genuinely be
"Intermediate" — not an unvalidated ad-hoc label as an earlier draft of this section
assumed. Defined the NE↔nonNE combinatorial axis as the 46 genes where `Generalist_NE`
and `Generalist_nonNE` actually differ (of the network's 53), and positioned every
archetype along it by % agreement with `Generalist_NE` on those 46 genes:

| archetype | % NE-like along the axis |
|---|---|
| `Generalist_NE` | 100% |
| `Arc_6` | 95.7% |
| `Arc_5` | 91.3% |
| **`Arc_1` ("Intermediate")** | **65.2%** |
| `Arc_3` | 58.7% |
| `Arc_4` | 6.5% |
| `Arc_2` | 4.3% |
| `Generalist_nonNE` | 0% |

All 8 archetypes fall on a single ordered spectrum — `Arc_1` genuinely sits about
two-thirds of the way from NE toward nonNE, confirming it as a real intermediate point,
not an isolated or unrelated state. **The walks' apparent movement "away from `Arc_1`
and toward `Generalist_nonNE`/`Arc_2`/`Arc_4` instead" is not a different destination — it's
the *same axis*, same direction, just traveling past the 65% way-point all the way to the
~95-100% nonNE end.** A 4000-step unconstrained walk has time to reach much further along
this trajectory than `Arc_1` represents. This resolves the geometric puzzle too: a point
65% along a 46-bit-long combinatorial path is still 16 bits from one end and 30 from the
other in raw Hamming terms (both individually large numbers) even though it is a genuine,
correctly-ordered interpolation — a continuous PCA/UMAP embedding compresses that same
46-dimensional path into 1-2 visual dimensions, where "65% along" simply looks "roughly in
the middle"; there's no contradiction, just what happens when a long high-dimensional path
gets projected onto a low-dimensional picture.

**Directly confirmed by tracking walk trajectories over time, not just whole-walk summary
stats (`diagnose_walk_temporal_ordering.py`).** If the walk genuinely passes *through*
`Arc_1` en route to the nonNE cluster, distance-to-`Arc_1` should dip to a minimum
*earlier* in the walk than distance-to-nonNE-cluster does. Tested directly: for every
walk, found the step of closest approach to `Arc_1` and to the nonNE cluster
(`Generalist_nonNE`/`Arc_2`/`Arc_4`, min distance), and compared their timing
(Wilcoxon signed-rank, paired within each walk). Confirmed for all three NE-starting
organoid populations under knockdown: `Neuroendocrine1` (89% of 100 walks reach `Arc_1`
first, mean step 876 vs. 2734, p=2e-14), `Generalist NE` (74%, step 1448 vs. 2544,
p=9e-9), `Neuroendocrine2` (81%, step 690 vs. 1330, p=6e-5). The mean distance-vs-step
curve (plotted for `Neuroendocrine1`) shows the expected shape directly: distance to
`Arc_1` dips to a minimum around step ~900, then *rises again* after step ~2000 as the
walk continues past it, while distance to the nonNE cluster keeps decreasing the whole
time and only bottoms out much later — a visually and statistically clean "passes through
Intermediate on the way to nonNE," not a coincidental proximity along the axis. The same
ordering appears (more weakly) in the *unperturbed* walk too, with much smaller amplitude
(NE-cluster distance only rises to ~19 vs. ~28 under knockdown; nonNE-cluster distance
only falls to ~31 vs. ~23) — knockdown doesn't create this temporally-ordered passage, it
amplifies it, letting the walk actually complete the transit instead of mostly wobbling
near its NE-like starting region. (One exception: starting *from* organoid's own
`Intermediate`-labeled state shows the reverse timing — unsurprising, since that
population doesn't need to travel through `Arc_1`'s neighborhood again, and its own
representative state isn't itself especially close to `Arc_1` specifically to begin with.)

**Updated read**: GEMM's simulated dynamics, applied to organoid's real states, correctly
identify *both* the right axis (NE↔nonNE) *and* the right direction (toward nonNE) under
RORA_RORB knockdown, *and* visit the Intermediate way-point in the correct temporal order
before continuing on — they just don't stop at the `Arc_1`/"Intermediate" way-point within
this walk length, continuing on to a near-complete nonNE identity instead. A plausible
reading: the long simulated walk approximates the *asymptotic* fate of sustained RORB
loss, while the real, finite-duration shRNA experiment (§C, below) captures a partial,
earlier snapshot along the same trajectory — consistent with, not contradicting, (C)'s
finding of a real but incomplete (not yet 100%) shift toward Intermediate in the real
data.

**(C) Independent, non-simulation check in organoid's own real data — result: a real,
substantial, statistically robust shift.** Fully decoupled from BoBa-T's rules: compared
organoid's own Seurat-based archetype metadata across the real experimental conditions
(`condition`: shGFP vs. shRORB1 vs. shRORB2). Every NE-ness score (`prediction.score.
Generalist.NE`, `NE1_score1`, `NE2_score1`) decreases from shGFP to both shRORB
conditions; every Intermediate-ness score except one near-zero exception increases;
all 10 comparisons remain significant after Bonferroni correction. Most strikingly, the
categorical `predicted.id` proportions show a clean, dose-like gradient:

| `predicted.id` | shGFP | shRORB1 | shRORB2 |
|---|---|---|---|
| `Generalist NE` | 75.0% | 67.1% | 63.7% |
| `Intermediate` | 4.8% | 14.4% | 29.3% |

A negative control (randomly splitting shGFP into two halves and running the identical
test) gives 0/5 significant results, confirming this isn't a large-N statistical
artifact.

**Overall verdict**: (C) firmly confirms the premise in organoid's real data — a
substantial, dose-like shift from NE-like toward Intermediate identity. (B) shows GEMM's
fitted rules, applied to organoid's own real starting states, correctly identify both the
right axis and the right direction under knockdown: measurable, statistically significant
movement away from NE-associated archetypes (`Generalist_NE`/`Arc_5`/`Arc_6`) and toward
the nonNE end of the *same* NE↔nonNE spectrum that `Arc_1`/"Intermediate" sits on (65%
of the way along it) — the walks simply travel further along that spectrum (to
`Generalist_nonNE`/`Arc_2`/`Arc_4`, ~95-100% nonNE) than the `Arc_1` way-point within a
4000-step simulation, plausibly approximating a more complete/asymptotic version of the
same shift the real, finite-duration knockdown only partially achieves. This is a genuine,
non-trivial confirmation that the rule set's perturbation-response logic transfers at the
archetype level — both the axis and the direction — even though `../domain_shift_diagnostic/FINDINGS.md` §9 already established it
doesn't transfer at the absolute-expression-value level.

Outputs: `organoid_seeded_walk_basin_occupancy.csv`, `walk_archetype_distance_shift.csv`,
`walk_temporal_ordering.csv`, `walk_temporal_distance_Neuroendocrine1_{kd,unperturbed}.png`
(simulation), `organoid_rorb_archetype_shift.csv`,
`organoid_predicted_id_proportions_by_condition.csv` (real data).

**RadViz-style 2D projection** (`plot_radviz_archetype_projection.py`,
`plot_radviz_archetype_kde_reference.py`): a hexagonal Hamming-distance projection onto 6
of bobaT's 8 fitted archetype average states (vertices: `Arc_2/3/4/5/6` + `Generalist_
nonNE`; `Generalist_NE` and `Arc_1`/Intermediate plotted as circles inside, not vertices,
since they weren't chosen to anchor the hexagon). Two companion plots, deliberately kept
separate so no single plot mixes "which condition" coloring with "which archetype"
coloring:
- `radviz_archetype_projection_Neuroendocrine1.png` — simulation only (100 individual
  walks + mean path, knockdown vs. unperturbed, organoid `Neuroendocrine1` start); no real
  cells plotted, so the only color-coded distinction is knockdown (purple) vs. unperturbed
  (grey).
- `radviz_archetype_kde_reference.png` — real organoid_shGFP cells only (no walks),
  summarized as a 50%-highest-density-region contour per organoid `predicted.id` group
  (`Generalist NE` n=6670, `Neuroendocrine1` n=1748, `Intermediate` n=423, `Neuroendocrine2`
  n=28; `Generalist nonNE` n=16, `Stress` n=8, and `nonNE1` n=1 are too few for a KDE and
  are shown as raw points instead), against the same fixed GEMM anchors.

**Two notable divergences, both consistent with (not contradicting) findings already
established above, not artifacts of this specific plot:**
1. Organoid's own `Intermediate`-labeled cells project *away* from GEMM's own
   `Arc_1`/Intermediate marker, and the NE-labeled groups (`Generalist NE`,
   `Neuroendocrine1`, `Neuroendocrine2`) cluster together off to the `nonNE1`/`Generalist_
   nonNE` side of the hexagon rather than near the `NE1`/`NE2` vertices. This is the same
   genuine cross-context rewiring already established in `../domain_shift_diagnostic/FINDINGS.md` §2/§3 (organoid's own regulatory
   relationships don't project onto GEMM's specific fitted axes the way GEMM's own data
   does) — expected, given the whole premise of this investigation, not a new problem.
2. The simulated mean walk paths look visibly smoother/cleaner than the real-cell KDE
   contours' shape. This is partly an artifact of what's being averaged, not evidence the
   walk is "more real": the mean path is an average of 100 stochastic replicates that all
   share one starting state and one fitted rule set (inherently low-variance — see how much
   rougher the underlying 100 individual walks look in `radviz_archetype_projection_
   Neuroendocrine1.png` before averaging), whereas the KDE contour is a density estimate
   over thousands of genuinely heterogeneous real single cells with real biological and
   technical noise. The two are not an apples-to-apples smoothness comparison; only the
   *position* relative to the fixed GEMM anchors is directly comparable between them,
   which is why the two plots share anchor geometry but were deliberately not overlaid into
   one.
- organoid_shGFP is the untreated control condition (no RORB knockdown), so nonNE
  representation being almost absent here (`Generalist nonNE` n=16, `nonNE1` n=1) is
  expected, not missing data — the dose-like shift toward nonNE/Intermediate only shows up
  under `shRORB1`/`shRORB2` (per the (C) table above: `Generalist NE` 75%→67%→64%,
  `Intermediate` 4.8%→14.4%→29.3%).

Hexagon orientation and vertex/circle colors in all three RadViz scripts were matched to
the user's own reference figure (flat top edge: `nonNE1`/`Generalist_nonNE`; vertex colors
`tab:red`/`lightcoral`/`tab:purple`/`darkred`/`tab:green`/`orange`; `Generalist_NE`="0.4"
grey, `Arc_1`/Intermediate=`tab:blue`) rather than an independently-chosen palette.

**ASCL1 knockout added as a positive control** (`run_ascl1_positive_control_walks.py`):
ASCL1 is a canonical NE master regulator, so its knockout is expected to drive an
unambiguous, large NE→nonNE shift — a sanity check on how large a *clearly real* effect
looks by this method, next to RORB's smaller, hypothesis-specific one. New long walks
(`off_nodes=["ASCL1"]`, same starting states already used for RORA_RORB, so the two
perturbations are directly comparable) were generated for the same organoid-seeded starts
(`Neuroendocrine1`, `Generalist NE`, `Neuroendocrine2`, `Intermediate`) and the same
GEMM-native `Arc_5` basin. All four plot scripts (`plot_radviz_archetype_projection.py`,
`plot_radviz_archetype_projection_gemm_native.py`, `diagnose_walk_axis_position.py`,
`diagnose_walk_temporal_ordering.py`) gained an opt-in `ascl1` CLI flag that writes a
separate `*_with_ASCL1` output rather than overwriting the original RORB-only files.
Result: on the hexagon and per-target-distance views, ASCL1 shows a visibly larger/faster
shift than RORB, as expected; on the single collapsed NE↔nonNE axis-position number,
though, ASCL1 ends up close to RORB rather than far beyond it — an honest divergence
between the two summary views (the axis position tracks generic %NE-likeness, while the
hexagon's per-anchor projection is sensitive to which *specific* nonNE archetype a state
lands nearest, which ASCL1 knockout appears to affect more than the aggregate axis does),
not a contradiction.

## 2. Pseudotime-calibrated axis, new single-real-cell starts, and RORB-overexpression rescue (`pseudotime_axis_utils.py`, `diagnose_walk_pseudotime_axis.py`, `run_new_start_walks.py`, `plot_new_start_*.py`, `plot_kde_reference_additional.py`)

**The NE<->nonNE axis's poles were wrong -- fixed using real pseudotime, not assumption.**
The original axis (§1) poled on `Generalist_NE`/`Generalist_nonNE`. Checked directly
against real GEMM cells' own `palantir_pseudotime` (`palantir_data.csv` joined to
`data/AA_clusters_splitgen.csv` phenotype labels): the true pseudotime extremes are
`Arc_5`/**NE1** (mean 0.013) and `Arc_4`/**nonNE1** (mean 0.532) -- the Generalists sit
well short of both ends (0.057 and 0.466). `Arc_1`/Intermediate landing near the true
middle (0.243) is a reassuring consistency check. Rebuilt the axis on NE1/nonNE1 (41 of 53
genes differ between them, vs. 46 for the Generalist pair), and calibrated the resulting
0-100 raw position to real pseudotime via isotonic regression fit on real GEMM cells
(binned first to avoid the noisy step-function artifact raw per-cell isotonic produces,
which showed up as vertical spikes in individual walk lines). New calibrated-axis plots
(`walk_pseudotime_axis_*.png`, with/without ASCL1) live in `axis_position_plots/
pseudotime_calibrated_axis/`; the original raw-Hamming-%-based plots were moved (not
deleted) to `axis_position_plots/raw_hamming_axis/`.

**New starting points: single real cells, not pooled populations.** For "the GEMM NE1
cell closest to the NE1 average" and equivalent nonNE queries: every archetype average
state checked is achieved *exactly* (Hamming distance 0) by at least one real, phenotype
-labeled GEMM cell -- so these starts are literally real cells' own discrete profiles, not
synthesized attractor states or population averages. Ran fresh long walks (4000 steps x
100 iters) from:
- **GEMM NE1 real cell** (idx `8914049663948766`), unperturbed vs. RORA_RORB knockdown --
  same qualitative NE->nonNE collapse as every other NE-start result in this doc. Outputs
  in `gemm_ne1_start/`.
- **GEMM's 3 nonNE archetypes** (`Generalist_nonNE`, `Arc_4`/nonNE1, `Arc_2`/nonNE2), each
  its own real cell, unperturbed vs. RORA_RORB **overexpression** (`on_nodes`, not
  `off_nodes`) -- outputs in `gemm_nonne_start_oe/`.
- **Organoid's real shRORB1+shRORB2 "Generalist nonNE" cells** (pooled, `find_avg_states`
  -collapsed, since there's no single-archetype analog for organoid's own real perturbed
  data), unperturbed vs. RORA_RORB overexpression -- outputs in
  `organoid_shrorb_nonne_oe/`.

**Result: RORB overexpression rescues nonNE states back toward NE, consistently, across
every nonNE start tested (4 of 4).** On the calibrated pseudotime axis, unperturbed walks
from a nonNE start drift *further* into nonNE territory (final pseudotime 0.18-0.27);
RORA_RORB overexpression instead pulls them back down to 0.09-0.13 (near
Intermediate/Secretory), a large, consistent, statistically robust reversal in every case
(p as low as 6.8e-17) -- including organoid's own real shRORB-perturbed cells, the closest
thing to a direct experimental analog available here.

**KDE reference plots extended to organoid's shRORB1/shRORB2 and to GEMM's own real
cells** (`plot_kde_reference_additional.py`, outputs in `hexagon_plots/kde_references/`,
alongside the original shGFP-only plot, untouched). Two findings:
- organoid shRORB1->shRORB2 shows the same dose-like spread/shift toward nonNE already
  established quantitatively (§1); visually confirmed here on the hexagon.
- **GEMM's own real cells mostly don't separate by fine phenotype label either.**
  Projecting GEMM's real cells by their own true phenotype (`Generalist_NE`, `Arc_1`
  -`Arc_6`) shows most subtypes (`Generalist_NE`, `Arc_5`/NE1, `Arc_6`/NE2, `Arc_1`
  /Intermediate, `Arc_3`/Secretory, `Arc_4`/nonNE1) overlapping in nearly the same region
  rather than each sitting near its own vertex -- only `Generalist_nonNE`/`Arc_2` form a
  distinct, separate cluster. This is the same phenomenon as the domain-shift rewiring
  finding (`../domain_shift_diagnostic/FINDINGS.md` §2/§9), just visualized directly: even
  GEMM's own fine-grained labels don't cleanly separate under this Hamming-distance
  projection -- it isn't something specific to organoid's domain shift.

**Checked whether the palantir calibration itself was causing the archetype "smushing" --
it isn't.** (`diagnose_walk_axis_position_ne1nonne1_raw.py`, outputs in
`axis_position_plots/ne1_nonne1_raw_axis/`.) Re-plotted the NE1/nonNE1 axis with the same
poles but *no* pseudotime calibration (raw %-agreement, 0-100). Same collapse persists:
`Arc_4`/`Arc_2` (nonNE1/nonNE2) are both exactly 0 on the raw axis too, and checking why
directly -- they differ by only 2 of the network's 53 genes total (`NFYC`, `THRB`), and
both of those happen to be genes where NE1 and nonNE1 *agree* with each other, so they
fall outside this axis's 41-gene definition entirely. `Generalist_NE`/`NE1` are similarly
close (97.6 vs. 100) in the raw metric. This is a property of the fitted archetypes
themselves (the nonNE side of the state space is much less differentiated than the NE
side, at least along genes this specific axis inspects) -- not an artifact introduced by
calibrating to pseudotime.

