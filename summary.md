---
beads_snapshot: 46f87877392f56aba7cb4cc0756f0bb3034896f04a7a4000e887d4c8e69e6d5c
beads_epic: astabot-pavi-0mp
generated_at: 2026-05-12T23:08:00Z
issue_count: 9
ready_count: 5
---

# Implement RNN with context-inference for dynamic routing task

## Mission
Goals:
1. Implement a RNN model task trained on the dynamic routing task as specified in latent_circuit_dynamic_routing_spec.md

Key details to adhere to:
2. Implementation of the Dynamic Routing task should match as close to experiment as possible. Experiment details available in experiment_task_details/EXPERIMENT.md
3. RNN architecture should infer context from repeat cues as opposed to having an explicit context input. Design details present in design folder.

Notes: Use the conda environment 'latent_circuit' for this project.

## Research Question & Scope
**Question:** Can a recurrent neural network with no explicit context input learn to perform the dynamic routing task by inferring the current block rule from the conjunction of reward and own-lick feedback delivered during each trial's response window, and reach the spec's Stage 4 performance criteria (intra- and inter-modal d' > 1.5 on at least 4/6 blocks across consecutive evaluation sessions)?

**In scope:** Phase 1 of `latent_circuit_dynamic_routing_spec.md` only — the ground-truth `DynamicRoutingRNN` (N=200, ReLU, tau=100 ms, sigma_rec=0.15, non-negative `W_in`, scalar sigmoid readout), the dynamic-routing task pipeline (6-block sessions, 5 instruction trials per block with `late_autoreward` semantics, catch trials at p=0.1, 125-step trial timeline), the 7-channel input convention (no explicit context channel), the full 5-stage curriculum with d'-based advancement gates and windowed BPTT, post-training validation checks #1–#6 from the spec, and a ~50-seed ensemble. Uses the `latent_circuit` conda env and PyTorch.

**Out of scope:** Phase 2 (LatentCircuit fitting) and Phase 3 (connectivity-conjugation, projections, perturbations, decoder comparisons, solution-space PCA), reproducing the Mante-task reference results, fitting against real mouse data, architectural variants beyond the spec (LSTMs/GRUs/Transformers, Dale's-law constraint, attention, alternative N/tau), task variants (coherence gradient, removing instruction trials, adding a context channel), and production engineering (distributed training, deployment, hyperparameter sweeps beyond curriculum gates).

**Success criteria:** Implementation matches the spec's architecture and hyperparameters; closed-loop reward/lick feedback is correct; the curriculum advances through all 5 stages using the spec's d' gates and regression criterion; post-training the spec's validation checks pass (hits ~0.8+, FAs ~0.2 or less, catch licks near zero, ~6–10 PCs > 90% variance, measurable block-transition context update); per-checkpoint training logs exist for post-hoc inspection; runs reproducibly in the `latent_circuit` conda env. See `astabot-pavi-x8g` for full bulleted lists.

## Operational Definitions
Sixteen terms fixed in `astabot-pavi-5ro` to give downstream tasks a shared vocabulary. Key ones:

- **Dynamic routing task** — block-based, 5 trial conditions (vis1/vis2/aud1/aud2/catch), lick-iff-rewarded-target, context never explicitly cued.
- **Block** — 5 instruction trials (newly-rewarded target, `late_autoreward`) followed by ~85 regular trials in shuffled 20-trial sub-blocks; 6 blocks per session, alternating rewarded target.
- **Instruction trial (late_autoreward)** — non-contingent reward scheduled for end of response window; if the network licks first, the scheduled reward is cancelled and a contingent reward fires at the lick.
- **Repeat cues** — the mission's term for the 5 instruction trials at each block start; bound here to "instruction trials" so vocabularies don't drift.
- **Catch trial** — no stimulus on u[0]–u[3] for the entire stimulus window; any lick counts as a false alarm (validation check #6).
- **Trial timeline** — 125 steps at dt=20 ms: quiescent (steps 0–74), stim on (75–99), stim off but response window continues (100–124). u[6] = 1 only on steps 80–124.
- **Closed-loop reward delivery** — on first z(t)>0.5 in the response window, fire u[5] for 2 timesteps; if stimulus is the rewarded target, also fire u[4] for those 2 timesteps. No re-firing within the same response window.
- **Context belief** — operationally implicit in hidden state y(t): "correct" iff a hypothetical stimulus at t+1 would yield the right z under the current block rule. No labeled channel, no auxiliary loss.
- **Hit / FA rate** — per stimulus × block-type, computed on post-instruction trials only, any-timestep z>0.5 within the response window.
- **d'_intra / d'_inter** — signal-detection-theoretic discriminability per block, with Z-inverse and 1/(2N) edge clipping; gates Stages 3–4 advancement.
- **Stage advancement / regression criteria** — exact gates for each of Stages 0–4 and the conditions under which a seed reverts to Stage 0.
- **Truncated BPTT window** — y persists across all trials in a block; autograd graph detaches at every 15–20-trial window boundary. Stages 0–2 use W=1.
- **Ensemble seed** — 50-seed target; reductions allowed but must be documented.
- **Validation check (post-training)** — the six spec analyses with fixed thresholds (hits ≥ ~0.8, FAs ≤ ~0.2, catch near zero, 6–10 PCs > 90% variance) constitute the project's only post-training gates.
- **latent_circuit conda env** — the single environment for all Python entry points; no second env or pip-only install path.

See `bd show astabot-pavi-5ro --json` for full operational definitions and rationales.

## Related Work
Long-form review at `literature_review.md`; structured citations and gaps on `astabot-pavi-q5k`. Eight headline findings:

- Mante, Sussillo, Shenoy & Newsome (2013, Nature) established the framework the project sits in: macaque PFC context-dependent integration is reproduced by a trained continuous-time RNN whose mechanism is a line attractor that selects task-relevant inputs and integrates them to choice.
- Langdon & Engel (2025, Nature Neuroscience) is the latent-circuit framework that the spec's Phase 2 will plug Phase 1 into. Applied to RNNs trained on a context-dependent task, it recovers a suppression mechanism in which contextual representations inhibit irrelevant sensory pathways — directly predicting the kind of mechanism the spec expects to find post-training.
- Low-rank RNN theory (Mastrogiuseppe & Ostojic 2017 Neuron; Beirán et al. 2020 Neural Comp; Dubreuil et al. 2022 Nat Neurosci) shows that context-dependent / flexible tasks require multi-population structure with gain-modulated low-rank connectivity — independent theoretical support for the latent-circuit hypothesis.
- Wang et al. (2018, Nature Neuroscience) "Prefrontal cortex as meta-RL" and Duan et al. (2016, arXiv) "RL²" show that an RNN whose inputs include reward and previous action can encode an implicit belief about the current task in its recurrent state — the architectural paradigm Phase 1 inherits, restricted to a 2-MDP family.
- Hajnal et al. (2024, Nat Comm) is the closest published behavioral analogue of the dynamic-routing task: head-fixed mice attend to the rewarded modality among identical visual+auditory stimuli with no external context cue, and the authors model ACC with an RNN that develops context-gated mutual inhibition between modality-selective ensembles — essentially the latent-circuit mechanism the spec predicts.
- Slow-point / fixed-point analysis tools (Sussillo & Barak 2013; Golub & Sussillo 2018 FixedPointFinder) and the dynamical-motif catalog (Driscoll, Shenoy & Sussillo 2024) provide off-the-shelf machinery for validation checks #4 and #5.
- Line-attractor regularization (Koppe, Beutelspacher & Durstewitz 2019) and curriculum-dependent emergence of long timescales (Khajehabdollahi et al. 2023) directly address the spec's largest known training risk — the vanishing-context-gradient problem documented in `background_knowledge.txt`.
- Closed-loop reward-driven RNN training is comparatively under-studied. Song, Yang & Wang (2016, eLife) is the canonical demonstration and Ger & Barak (2025) is the most recent systematic analysis. The spec's late_autoreward + sigmoid-threshold-gated reward delivery has no direct published analogue.

## Hypotheses
Five hypotheses derived from the literature review's five gaps. All open, all blocked-by the literature review (now closed) and inputs from scope.

### H1 (astabot-pavi-gw7): Headline — curriculum suffices
**Statement:** An RNN with the spec's architecture and no explicit context channel, trained via the spec's 5-stage curriculum, will satisfy Stage 4 advancement criteria (d'_intra > 1.5 AND d'_inter > 1.5 on ≥ 4/6 blocks across ≥ 5 consecutive evaluation sessions). Predicts: across 50 seeds, ≥ 1 seed reaches Stage 4 and passes validation checks #1–#6.

### H2 (astabot-pavi-yla): Negative — curriculum alone insufficient
**Statement:** Without an explicit line-attractor regularizer (Koppe 2019) or context-attractor weight init (the rank-1 W_rec update in `background_knowledge.txt`), the spec's 5-stage curriculum produces d'_inter < 1.5 on all of N ≥ 10 seeds within the stage-advancement budget. H1 and H2 are deliberately in tension; both are testable on the same training-and-evaluate pipeline.

### H3 (astabot-pavi-e38): Ensemble convergence to one motif
**Statement:** Across ≥ 20 Stage-4-passing seeds, ≥ 80 % share a single dominant latent-circuit motif: two context-selective sub-populations whose activity gates the modality-specific readout pathway via mutual inhibition, qualitatively matching Langdon & Engel (2025) / Hajnal et al. (2024).

### H4 (astabot-pavi-7k0): All 5 instruction trials matter
**Statement:** In a Stage-4-passing RNN, post-switch d' increases monotonically with the number of instruction trials presented (k=1..5), and Stage-4 criteria are met only at k ≥ 4. Tested by ablation replay.

### H5 (astabot-pavi-bqs): Truncated BPTT under closed-loop reward is stable
**Statement:** With Adam (lr=1e-3 → 1e-4), grad-clip 1.0, the spec's regularizers, and W=15–20-trial windows, training is numerically stable: bounded gradient norms, ≥ 90 % windows with non-increasing steady-state loss, ≤ 10 % seeds regressing from Stage 1+ to Stage 0.

## Experimental Designs
None yet — to be created via `plan` once a hypothesis is picked up.

## Results Summary
None yet.

## Open Questions
Tracked implicitly in the 5 hypothesis statements above; will be re-summarized after `analysis` tasks close.

## Status
- Closed: 3 — IDs: astabot-pavi-x8g, astabot-pavi-5ro, astabot-pavi-q5k
- In progress: 0 — IDs: (none)
- Ready: 5 hypotheses (+ epic astabot-pavi-0mp) — IDs: astabot-pavi-gw7, astabot-pavi-yla, astabot-pavi-e38, astabot-pavi-7k0, astabot-pavi-bqs
- Blocked: 0

### Next Steps
- astabot-pavi-gw7 [hypothesis]: 5-stage curriculum reaches Stage 4 on dynamic routing without explicit context channel — frame as falsifiable prediction; `plan` will then unlock its experiment_design → evidence_gathering → analysis chain.
- astabot-pavi-yla [hypothesis]: 5-stage curriculum alone insufficient for stable context substrate; line-attractor regularization or context-attractor init required — paired-comparison design comparing with/without attractor fix.
- astabot-pavi-e38 [hypothesis]: Stage-4 ensemble solutions converge on a single latent-circuit motif — depends on H1 succeeding for ≥ 20 seeds; can be deferred until H1 produces a passing ensemble.
- astabot-pavi-7k0 [hypothesis]: All 5 instruction trials per block contribute to context-belief update — ablation-replay design on a passing Stage-4 RNN.
- astabot-pavi-bqs [hypothesis]: Truncated BPTT under closed-loop reward is numerically stable with the spec's optimizer settings — can be measured from H1 training logs; lowest-cost hypothesis to test.
