# Associative Memory and Fast Weights — an interactive explainer

> A fixed-size associative memory can store many key–value associations in the same weight matrix, but retrieval degrades through interference between overlapping keys — not because it runs out of discrete slots.

An interactive notebook explainer built for the DataForge 2026 Pathway Track. Every result is live NumPy computation on a real fast-weight memory; the reader manipulates the mechanism's actual variables and watches the claim hold or break.

**Public artifact:** TODO — JupyterLite / GitHub Pages URL
**Repository:** TODO — repo URL

## Intended learner and prerequisites

**Audience:** An ML-literate reader who knows linear algebra and basic attention, but has not thought carefully about fast-weight memory.

**Prerequisites:** Vector dot products, outer products, cosine similarity, and rough familiarity with attention.

## Learning objectives

1. Trace a key–value write into a fixed `d × d` weight matrix and read the association back.
2. Separate the desired signal from crosstalk caused by other stored keys.
3. Compare measured crosstalk with the approximate `sqrt(N / (d - 1))` scaling for this construction.
4. Explain why retrieval fails from key overlap rather than from exhausting a fixed number of slots.
5. Locate this same outer-product write inside Pathway's Dragon Hatchling architecture, and state where the toy diverges from it.

## Architecture

| File | Role |
| --- | --- |
| `fastweights.py` | The memory mechanism. Hebbian outer-product writes (`M ← λM + v kᵀ`), matrix reads (`M q`), exact signal/crosstalk decomposition, correlated-key generation, batched construction, and top-1 retrieval accuracy. |
| `test_fastweights.py` | Eight tests: single-association recall, monotonic degradation under overlap, orthogonal-key control, generator calibration against requested `rho`, cosine helper correctness, decomposition-vs-read consistency with and without decay, incremental-vs-batched equivalence, and isolation of key correlation in the accuracy sweep. |
| `notebook.ipynb` | The nine-section explainer. Imports all mathematics from `fastweights.py`; no math is reimplemented in cells. |
| `references.md` | Claim → citation → link record. |
| `docs/BUILD_PROMPT.md` | Build specification and review record. See AI assistance disclosure below. |
| `requirements.txt` | Pinned dependencies. |

The memory mechanism is `FastWeightMemory.matrix` alone, which stays `d × d` regardless of how many associations are written. The class also retains `keys` and `values` as lists, but these are teaching scaffolding for ground-truth comparison and decomposition only — they play no role in reads.

## What is live and what is not

Everything in the notebook is live NumPy computation. There are no precomputed results, no synthetic numbers presented as measurements, no scripted animations, and no illustrative figures standing in for real behaviour. Cell outputs are committed so the notebook is readable without execution, but every cell recomputes identically on run with the default seed.

## Reproducing the results

**Codespaces or local:**

```bash
pip install -r requirements.txt
pytest -v                      # 8 tests, all passing
jupyter lab notebook.ipynb     # run all cells top to bottom
```

The default seed is fixed, so reported figures reproduce exactly. Sandbox caps are `N ≤ 200` and `d ≤ 512`, chosen to keep every control interactive.

**Measured performance** at the worst-case corner (`N=200`, `d=512`): `from_arrays` 3.3 ms minimum / 5.2 ms mean, `top1_accuracy` 3.8 ms minimum / 6.5 ms mean. Full recomputation on a slider move stays roughly two orders of magnitude inside the one-second interactivity target.

## Key results a reader can verify

**Opening preset** (`N=40`, `d=128`, `rho=0`, no decay): cosine 0.866, crosstalk norm 0.596, top-1 accuracy 100%. The retrieved vector has already degraded measurably while ranking still succeeds for every key — interference is continuous and begins at the second write; retrieval failure is a late symptom.

**The slot model refuted** (both at `d=512`):

| Configuration | Cosine | Crosstalk norm | Top-1 accuracy |
| --- | --- | --- | --- |
| `N=10`, `rho=0.90` | 0.336 | 2.790 | 20% |
| `N=100`, `rho=0.00` | 0.913 | 0.440 | 100% |

A memory holding ten times more data retrieves perfectly, while the smaller one fails four times out of five. If capacity were a count of discrete slots, the ten-item case could not be the failing one. What differs is key overlap, not item count.

Both figures are consistent with the analytic model: at `N=10`, `rho=0.9`, nine cross-terms of overlap ≈ 0.9 predict a crosstalk norm near 2.7 and cosine `1/sqrt(1 + 2.79²)` ≈ 0.34.

## Stated limits and approximations

Full detail is in the notebook's limitations section. In brief:

- **Dense and linear.** This toy uses dense linear reads. BDH uses sparse non-negative activations. TODO — state the divergence precisely after primary-source verification.
- **Synthetic keys.** Keys here are generated, not learned. Real models learn keys, which changes the interference picture.
- **Equicorrelation structure.** All key overlap comes from a single shared direction, so the key Gram matrix is rank-one-plus-identity in expectation. Real correlation structure is richer.
- **The `(d-1)` subspace.** At `rho=0` the generated keys span the subspace orthogonal to the shared direction, so expected crosstalk scales as `sqrt(N/(d-1))` rather than `sqrt(N/d)`. Negligible at `d=512`, visible at `d=16`.
- **Mechanism isolation.** No sequence model, no training, no language. This isolates one mechanism deliberately.
- **Caps.** `N ≤ 200`, `d ≤ 512`, stated in the notebook.

## BDH module status

TODO — Section 7 is a verification scaffold. Every substantive claim about BDH and BDH-CQ is fenced behind a `TODO-VERIFY` marker pending direct reading of the Dragon Hatchling paper and the BDH-CQ technical report. No BDH equation, benchmark, or architectural claim has been written from memory.

This notebook contains an independent educational toy implementation of a fast-weight associative memory. It is **not** an official BDH model, **not** a reimplementation of BDH, and produces **no** BDH results.

## Primary sources

TODO — at least three primary papers from 2022–2026, each cited beside the specific claim it supports. See `references.md`. Every title, author, and year to be verified against the source before submission.

## AI assistance disclosure

Code and notebook drafts were generated with GitHub Copilot in agent mode from specifications recorded in `docs/BUILD_PROMPT.md`.

All mathematics, experimental design, parameter choices, and technical claims were specified and verified by me. The build record documents the review cycle, including corrections I identified and directed: the missing correlated-key generator, a variable confound in which correlated values would have degraded accuracy independently of key interference, a decomposition-versus-matrix drift risk, unvectorized code that would have broken the interactivity budget, weak parameter choices that collapsed the misconception demonstration, and hedged prose that understated what the mathematics guarantees.

Analytic predictions were checked independently against measured output: the preset crosstalk norm against `sqrt((N-1)/(d-1))`, and the resulting cosine against `1/sqrt(1 + ||crosstalk||²)`.

TODO — add any further tools used, and note that the BDH module content is written from primary sources by hand.

## Credits and licenses

TODO — MIT license for this repository. Record licenses for NumPy, Matplotlib, ipywidgets, pytest. Record any borrowed code, figures, or fonts, or state plainly that there are none.
