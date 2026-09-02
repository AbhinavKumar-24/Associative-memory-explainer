# Build Prompt — DataForge 2026 Pathway Track

Paste everything below the line into GitHub Copilot Chat (agent mode) in your Codespace.

---

You are helping me build a submission for the DataForge 2026 Pathway Track ("Explain the Frontier"). Build it incrementally, file by file, so I can review and edit each piece. Ask me before making assumptions about anything you cannot verify.

## What I am building

An interactive notebook explainer on **Associative Memory and Fast Weights**, connected to Pathway's Dragon Hatchling (BDH) architecture.

**The one-sentence falsifiable claim the artifact must teach:**

> A fixed-size associative memory can store many key–value associations in the *same* weight matrix, but retrieval degrades through **interference between overlapping keys** — not because it runs out of discrete slots.

Every cell, control, and plot must serve this claim. Cut anything that doesn't.

**Audience:** a data scientist or ML-literate student who knows linear algebra and basic attention, but has not thought carefully about fast-weight memory.

**Prerequisites to state explicitly:** vector dot products, outer products, cosine similarity, rough familiarity with attention.

## The substrate (real computation — no scripted animation)

Implement a genuine fast-weight associative memory in NumPy. This is the exact math; do not substitute your own:

- Keys `k_i ∈ R^d` and values `v_i ∈ R^d`, all unit-normalized.
- **Write (Hebbian outer-product):** `M ← M + v_i @ k_i.T`, starting from `M = 0`.
- **Read:** `v̂ = M @ k_q`

The teaching moment comes from expanding the read:

```
v̂ = M @ k_j = v_j·(k_j·k_j) + Σ_{i≠j} v_i·(k_i·k_j)
              └── signal ──┘   └──── crosstalk / interference ────┘
```

The notebook must **decompose and plot these two terms separately**. That decomposition is the core insight — a learner should see the crosstalk term grow while the signal term stays constant.

For random unit keys, expected crosstalk magnitude scales roughly as `sqrt(N/d)`. Plot the empirical curve against this analytic expectation so the learner sees theory beside measurement.

## Controls (few, each mapped to one real variable)

Use `ipywidgets`. Every control must recompute in under one second — keep `d ≤ 512` and `N ≤ 200` and say so in the notebook.

1. **`N` — number of stored associations** (slider, 1–200). Primary control.
2. **`d` — memory dimension** (slider, 16–512). Shows capacity scales with dimension.
3. **`ρ` — key correlation** (slider, 0.0–0.95). Generates keys with controlled pairwise overlap instead of pure random. **This is the control that proves the claim**: at fixed `N` and `d`, raising `ρ` degrades retrieval, demonstrating interference is about key overlap, not slot count.
4. **Optional: decay `λ`** — `M ← λM + v k.T`. Shows forgetting as an explicit trade-off against interference.

## Truth beside estimate (required)

On every update, display:
- A side-by-side view of the true value `v_j` and the retrieved `v̂` for one probed key (bar plot or heatmap of components).
- Cosine similarity between them, as a number.
- Top-1 retrieval accuracy across all `N` stored keys (does `argmax` over stored values return the right one?).
- The signal-vs-crosstalk decomposition plot described above.

The notebook must **open with a preset already computed and rendered** — no blank canvas, no "Run" button as the first interaction.

## Notebook structure

Build `notebook.ipynb` in this order:

1. **Title + the one-sentence claim, stated verbatim**, plus audience, prerequisites, and 3–4 numbered learning objectives.
2. **Live preset** — memory already built and plotted before the learner touches anything.
3. **Guided walkthrough** — write one pair, read it back perfectly. Then write five. Then twenty. Prose explains what is happening at each step.
4. **The decomposition** — reveal the signal/crosstalk split and the `sqrt(N/d)` expectation.
5. **Interactive sandbox** — all sliders released together.
6. **The misconception buster** — an explicit cell where the learner sets `N` low but `ρ` high and watches retrieval fail anyway. Add a short prompt asking the learner to explain in their own words why this happens.
7. **The BDH module** (see below).
8. **Limitations** (see below).
9. **References.**

## The BDH module — accuracy rules

This section connects the concept to BDH. It must be woven in with its own learning objective, not tacked on.

**Critical: do NOT write any BDH equation, parameter count, benchmark number, or architectural claim from memory.** Instead, generate this section as a scaffold with clearly marked placeholders:

```
<!-- TODO-VERIFY: [what I need to check in the primary source] -->
```

Scaffold it to cover:
- Which system is meant (BDH, and where BDH-CQ is or is not relevant — if BDH-CQ has no direct role here, say so plainly rather than inventing a link).
- Why associative/fast-weight memory shows up in BDH at all: BDH reformulates attention as *synaptic* memory updated by Hebbian writes as the model reads. Placeholder for the paper's actual formulation.
- What is *actually changing* in BDH during inference — recurrent/synaptic state, not trained parameters.
- Where my toy matches BDH and where it does not. At minimum: my toy is dense and linear; BDH uses sparse non-negative (ReLU) activations, and BDH-GPU is a ReLU-low-rank formulation with linear attention. Placeholder for exact details.
- A pointer to the primary sources (the Dragon Hatchling paper and the BDH-CQ technical report) rather than secondhand summaries.

Add a prominent disclaimer cell, in this spirit:

> This is an independent educational toy implementation of a fast-weight associative memory. It is **not** an official BDH model, not a reimplementation of BDH, and produces no BDH results. All BDH claims in this notebook are cited to published primary sources.

## Limitations section (required, honest)

Cover at least:
- Dense linear toy vs. BDH's sparse non-negative activations.
- Keys here are random or synthetically correlated; real models *learn* keys, which changes the interference picture.
- No sequence model, no training, no language — this isolates one mechanism.
- Caps stated: max `N`, max `d`, and why.

## References

Create a `references.md` with a strict format: claim → citation → link. Populate it with **placeholders only**, plus this candidate list for me to verify myself before committing. Do not assert any of these are correct, do not invent DOIs, and do not add papers I have not confirmed:

- Dragon Hatchling paper (Pathway) — primary source, must verify title/authors/year
- BDH-CQ technical report — primary source, must verify
- Candidates in the fast-weight / linear-attention line, 2022–2026 window, all requiring verification: Mamba (Gu & Dao); Gated Linear Attention (Yang et al.); DeltaNet / delta-rule linear transformers (Yang et al.); Titans: Learning to Memorize at Test Time (Behrouz et al.)

The track requires **at least three primary papers from 2022–2026**, cited beside the specific claims they support.

## Repo files to create

```
README.md            # see spec below
notebook.ipynb       # the explainer
fastweights.py       # memory implementation, importable and unit-testable
test_fastweights.py  # pytest: perfect recall at N=1, monotonic degradation, orthogonal-key control
requirements.txt     # numpy, matplotlib, ipywidgets, pytest — pinned versions
references.md
LICENSE              # MIT
AI_DISCLOSURE.md     # what was AI-generated, what I wrote and modified
.gitignore
```

Put the memory logic in `fastweights.py`, not buried in notebook cells — the notebook imports it. This makes it testable and makes the code defensible when judges ask me to trace it.

## README spec

Must explain, in this order: the one-sentence claim; intended learner and prerequisites; learning objectives; artifact architecture; the role of every major component; **which parts are live vs. precomputed vs. synthetic vs. illustrative** (here: everything is live computation, state that plainly); how to reproduce results; setup instructions for Codespaces and local; credits and licenses; AI-assistance disclosure.

## Public artifact URL

The track requires a public URL that opens without sign-in. Set up **JupyterLite deployed to GitHub Pages** via a GitHub Action so the notebook runs in-browser with no login. Include the workflow file and setup steps in the README. If JupyterLite cannot support `ipywidgets` for this use case, tell me and propose an alternative rather than shipping something broken.

## Hard constraints

- **Never fabricate** a citation, benchmark number, equation, or BDH detail. Use `TODO-VERIFY` placeholders instead. Incorrect technical claims carry a major scoring penalty.
- No decorative sliders. Every control maps to a real variable in the math.
- No scripted animation presented as computation.
- State every cap and approximation in the notebook itself.
- Write prose that is dense and mechanism-first. No padded sentences, no undefined buzzwords, no repeated adjectives.
- Keep the code readable enough that I can explain every line under questioning.

Start with `fastweights.py` and `test_fastweights.py`, show me the code, and wait for my review before building the notebook.