# Build specification and review record

This file documents how the repository was built. It serves as the provenance record referenced by the AI assistance disclosure in `README.md`.

The artifact was built with GitHub Copilot in agent mode, working from the specification in Part 1 and the review cycle in Part 2. The specification fixes the mathematics, the experimental design, and the accuracy constraints in advance; the review record documents what was wrong in the generated output and what was corrected.

---

## Part 1 — Initial specification

### Claim

> A fixed-size associative memory can store many key–value associations in the same weight matrix, but retrieval degrades through interference between overlapping keys — not because it runs out of discrete slots.

Every cell, control, and plot must serve this claim.

**Audience:** ML-literate reader who knows linear algebra and basic attention, but has not thought carefully about fast-weight memory.
**Prerequisites:** dot products, outer products, cosine similarity, rough familiarity with attention.

### Substrate — exact mathematics

Implemented in NumPy. This math was specified in advance and not left to the model:

- Keys `k_i ∈ R^d`, values `v_i ∈ R^d`, all unit-normalized.
- Write: `M ← λM + v_i k_iᵀ`, from `M = 0`.
- Read: `v̂ = M k_q`

The teaching decomposition:

```
v̂ = M k_j = v_j·(k_j·k_j) + Σ_{i≠j} v_i·(k_i·k_j)
             └── signal ──┘   └──── crosstalk ────┘
```

With decay, each stored contribution carries weight `λ^(N-1-i)`; the decomposition sums exactly to the read in both cases.

Expected crosstalk magnitude for independent keys scales as `sqrt(N/(d-1))` — plotted as an analytic expectation, never as a fit.

### Correlated-key generation

`make_vectors(n, d, rho, seed)` returns unit vectors with mean pairwise cosine ≈ `rho`, by the shared-component construction:

`k_i = sqrt(rho)·c + sqrt(1-rho)·u_i`, where `c` is a unit direction and each `u_i` is unit and orthogonal to `c`.

Norm is exactly `rho + (1-rho) = 1` since `c·u_i = 0`. For `i ≠ j`, `k_i·k_j = rho + (1-rho)(u_i·u_j)` with `E[u_i·u_j] = 0`, so `E[k_i·k_j] = rho`. Calibration is asserted by test against `rho ∈ {0.0, 0.3, 0.6, 0.9}`.

`make_problem(n, d, rho, seed)` returns `(keys, values)` where keys carry the requested `rho` and **values are always generated at `rho=0`** with a derived seed. This is required: `top1_accuracy` ranks by dot product against stored values, so correlated values would degrade accuracy independently of key interference and confound the experiment.

### Controls

Four, each mapped to one real variable, all recomputing under one second:

| Control | Range | Variable |
| --- | --- | --- |
| `N` | 1–200 | associations stored |
| `d` | 16–512 | memory dimension |
| `rho` | 0.0–0.95 | key overlap — the control that proves the claim |
| `λ` | 0.5–1.0 | decay / forgetting |

### Truth beside estimate

Every update displays: true value against retrieved value component-wise; cosine similarity as a number; crosstalk norm; top-1 accuracy across all `N`; and the signal/crosstalk decomposition.

The notebook opens with a preset already computed — no blank canvas, no Run button as the first interaction.

### Notebook structure

1. Header — claim verbatim, audience, prerequisites, five learning objectives
2. Live preset
3. Guided walkthrough — one, five, twenty associations
4. The decomposition, with analytic overlay
5. Interactive sandbox with three prescribed experiments
6. Misconception buster
7. BDH module
8. Limitations
9. References

### BDH module — accuracy rules

**No BDH equation, parameter count, benchmark result, or architectural claim may be written from memory.** The section was generated as a scaffold with `<!-- TODO-VERIFY: ... -->` markers naming what must be checked against the primary sources, covering: which system is meant; whether BDH-CQ has a direct role (stated plainly if not, rather than inventing one); why fast-weight associative memory appears in BDH; what changes during inference; and where the toy corresponds to and diverges from BDH.

A disclaimer states that this is an independent educational toy, not an official BDH model or reimplementation, producing no BDH results.

### Hard constraints given to the model

- Never fabricate a citation, benchmark number, equation, or BDH detail — use `TODO-VERIFY` instead.
- No decorative controls; every control maps to a real variable.
- No scripted animation presented as computation.
- State every cap and approximation in the notebook.
- Dense, mechanism-first prose. No padding, no undefined buzzwords, no repeated adjectives.
- Memory logic lives in `fastweights.py`, imported by the notebook, never reimplemented in cells.
- Build in slices and stop for review.

---

## Part 2 — Review record

Each slice was reviewed before the next was built. Corrections directed:

**Slice 1 — `fastweights.py`, `test_fastweights.py`**

Core math verified correct on inspection: decay indexing `λ^(N-1-i)` matched the stated construction, and the decomposition summed exactly to the read. Corrections directed:

1. The correlated-key generator was absent entirely — the single component the central claim depends on. Specified and added, with a calibration test.
2. `decompose_read` rebuilt from the stored lists rather than from `matrix`, allowing silent divergence. A consistency test was added locking them together, with and without decay.
3. Construction was N Python-loop writes each allocating a fresh `d × d` matrix, and `top1_accuracy` was an N-length matvec loop — both would have breached the interactivity budget. Batched paths added (`from_arrays`, vectorized scoring), with a test asserting equivalence to the incremental route.
4. The class docstring did not distinguish mechanism from scaffolding, inviting the objection that storing all keys and values contradicts a fixed-size memory. Clarified.
5. Validation was reported as "syntax passes." Dependencies were installed and `pytest` was actually run before approval.

**Slice 2 — variable confound**

`make_problem` was specified and added after identifying that generating values at the same `rho` as keys would have varied two things at once. Two approximations were documented: the `(d-1)` subspace at `rho=0`, and the equicorrelation Gram structure.

Benchmark obtained at `N=200`, `d=512`: `from_arrays` 3.3 ms min, `top1_accuracy` 3.8 ms min.

**Slice 3 — notebook sections 1–4**

Preset output was checked against the analytic model independently: `sqrt(39/127)` ≈ 0.554 predicted crosstalk against 0.596 measured, and `1/sqrt(1 + 0.596²)` ≈ 0.859 predicted cosine against 0.866 measured — consistent within single-draw variance. Corrections directed:

1. Learning objective 4 was vague and omitted BDH, which the track requires be woven in with its own objective. Replaced with two objectives covering the slot-model refutation and the BDH connection.
2. The misconception the claim denies — that memory is a set of slots that fills and fails — was never named. Made explicit.
3. Hedged language ("can add crosstalk", "can decline") understated what the mathematics guarantees. Replaced with the mechanism: random unit keys in finite dimensions have pairwise dot products of mean zero and typical magnitude `1/sqrt(d)`, so every write contributes a nonzero cross-term.
4. The signal term was described as staying at unit magnitude, true only at `λ=1`. Decay caveat added.
5. The preset showed cosine 0.866 with 100% accuracy, framing the opening as "nothing is wrong" — against the claim. Reframed so the gap is the lesson: the retrieved vector has already degraded while ranking still succeeds, making accuracy a late indicator. Crosstalk norm added as a displayed metric.

**Slice 4 — sections 5–7**

BDH scaffold approved: every substantive claim correctly fenced behind `TODO-VERIFY`, nothing invented. Corrections directed:

1. The sandbox described its controls but prescribed no experiments, making it a dashboard rather than a lesson. Three experiments added, each stating the manipulation and what to watch, with guidance to watch crosstalk norm rather than accuracy.
2. The misconception buster's parameters risked collapsing the contrast — at `d=128`, the high-`N` case would have shown crosstalk ≈ 0.88 and degraded alongside the low-`N` case, proving nothing. `d=512` specified for both cases, holding dimension constant so it is not a confound, and both result sets printed side by side.

Resulting contrast: `N=10, rho=0.9` gives cosine 0.336 and 20% accuracy; `N=100, rho=0` gives cosine 0.913 and 100% accuracy. Verified against the analytic prediction for the failing case.

**Slice 5 — sections 8–9**

Limitations and placeholder references added. Notebook complete at 21 cells, executing end to end.

---

## Part 3 — Written without AI assistance

TODO — record here, before submission:

- BDH module content, written from the Dragon Hatchling paper and BDH-CQ technical report by hand after resolving every `TODO-VERIFY` marker
- `references.md` entries, each verified against the source
- The one-page concept summary PDF
