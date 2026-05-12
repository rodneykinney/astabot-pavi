# Literature Review: Context-Inferring RNNs for the Dynamic Routing Task

Scope: prior work bearing on Phase 1 of `latent_circuit_dynamic_routing_spec.md` — a 200-unit ReLU RNN that performs the dynamic routing task with no explicit context channel and infers the current block rule from closed-loop reward and own-lick feedback plus the 5 instruction trials at each block boundary.

Organized into five themes that match how the implementation, training tricks, and post-training validation will draw on the literature.

---

## 1. Latent circuits and low-rank RNNs for context-dependent decision making

This is the framework the project will eventually plug into (Phase 2/3), and it constrains the architecture and training of Phase 1.

**Mante, Sussillo, Shenoy & Newsome (2013), Nature.** *Context-dependent computation by recurrent dynamics in prefrontal cortex.* The founding work: macaques flexibly select and integrate visual motion and color toward a choice, FEF/PFC activity is captured by a trained continuous-time RNN, and the trained network reveals a line-attractor mechanism that selects task-relevant inputs and integrates them to choice. Every modern RNN-modeling paper on context-dependent behavior, including this project's spec, sits downstream of this.

**Langdon & Engel (2025), Nature Neuroscience.** *Latent circuit inference from heterogeneous neural responses during cognitive tasks.* The spec's Phase 2 explicitly fits a 9-node latent circuit to the Phase 1 RNN via this framework. The paper applies latent-circuit inference to RNNs trained on a context-dependent task and discovers a "suppression mechanism in which contextual representations inhibit irrelevant sensory responses" — establishing both the inference machinery and the kind of mechanism the Phase 1 model is expected to discover end-to-end without supervision.

**Mastrogiuseppe & Ostojic (2017), Neuron.** *Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks.* Theoretical foundation: in networks with random plus low-rank connectivity, the collective dynamics live in a subspace whose dimension equals the rank. This is the analytic backbone of low-rank RNN theory.

**Dubreuil, Valente, Beirán, Mastrogiuseppe & Ostojic (2022), Nature Neuroscience.** *The role of population structure in computations through neural dynamics.* Shows that flexible input-output mappings (i.e. context-dependent tasks like Mante 2013) cannot be implemented by a single global population — they require multiple sub-populations whose gain modulations shape a flexibly switching dynamical landscape. Directly relevant to the latent-circuit hypothesis: the project's RNN is expected to spontaneously organize into context-selective sub-populations that gate visual vs auditory readout.

**Beirán, Dubreuil, Valente, Mastrogiuseppe & Ostojic (2020), Neural Computation.** *Shaping Dynamics With Multiple Populations in Low-Rank Recurrent Networks.* Companion theoretical paper: a rank-R, P-population low-rank RNN can approximate any R-dimensional dynamical system — sets the expectation for what dimensionality the Phase 1 RNN's task-related dynamics should sit in (the spec targets 6-10 PCs > 90% variance).

---

## 2. Meta-RL and belief-state RNNs (context inference without an explicit cue)

The architecture in `design/` — input = (sensory channels, reward, own-lick, trial-phase), no context channel — is structurally meta-RL.

**Wang, Kurth-Nelson, Kumaran, Tirumala, Soyer, Leibo, Hassabis & Botvinick (2018), Nature Neuroscience.** *Prefrontal cortex as a meta-reinforcement learning system.* The clearest conceptual ancestor: an RNN receives observations + previous reward + previous action as inputs, is trained across a distribution of tasks, and its recurrent state comes to represent a belief about the current task. The "fast" inner-loop RL is implemented in the activity dynamics; the "slow" outer-loop is the gradient training. The Phase 1 RNN does the same — but with a single MDP family (visual-rewarded vs auditory-rewarded blocks) and a supervised lick-objective rather than policy-gradient RL.

**Duan, Schulman, Chen, Bartlett, Sutskever & Abbeel (2016), arXiv.** *RL²: Fast Reinforcement Learning via Slow Reinforcement Learning.* The machine-learning precursor of Wang 2018. Establishes that a recurrent network whose inputs include reward and previous action can learn an inner RL algorithm in its activations, and that this generalizes to held-out MDPs. The Phase 1 RNN's "context is encoded implicitly in the recurrent state" reduces to RL² on a 2-MDP family.

---

## 3. Mouse cross-modal / dynamic-routing experimental analogues

The spec asks the model to match the Allen Institute dynamic-routing task at the trial level. Closely related behavioral paradigms and their RNN models are the right comparators:

**Hajnal, Tran, Szabó, Albert, Safaryan, Einstein, Vallejo Martelo, Polack, Golshani & Orbán (2024), Nature Communications.** *Shifts in attention drive context-dependent subspace encoding in anterior cingulate cortex in mice during decision making.* The closest published analogue of the dynamic-routing task: head-fixed mice are presented identical auditory and visual stimuli in two contexts and must attend to the rewarded modality without an external context cue, exactly the regime the Phase 1 RNN must operate in. They model the ACC response with an RNN showing context-gated mutual inhibition between modality-selective ensembles — essentially the latent-circuit mechanism the spec predicts.

**Orlandi, Abdolrahmani, Aoki, Lyamzin & Benucci (2021), Nature Communications.** *Distributed context-dependent choice information in mouse posterior cortex.* A mouse visual decision task; an RNN trained on the animals' choices reproduces context-dependent decision dynamics in posterior cortex. Validates that mouse-scale context-dependent tasks can be captured by trained RNNs and that the resulting dynamics sit in a low-dimensional subspace.

**Wimmer, Schmitt, Davidson, Möhrlein, Halassa et al. (2015), Nature.** *Thalamic control of sensory selection in divided attention.* Classic mouse cross-modal attention task with explicit cued context. Sets the experimental benchmark for sensory-selection mechanisms — and clarifies that the dynamic-routing task is the harder, uncued version.

---

## 4. Curriculum, truncated BPTT, slow points, attractor regularization

These are the practical training-stability tools the spec lists as Phase 1 pitfalls.

**Sussillo & Barak (2013), Neural Computation.** *Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks.* The slow-point / fixed-point methodology for reverse-engineering trained RNNs. Validation check #5 (PCA on trial-averaged responses) and any future latent-circuit fit rely on this analytical lineage.

**Golub & Sussillo (2018), Journal of Open Source Software.** *FixedPointFinder.* TensorFlow toolbox to find fixed and slow points of trained RNNs. A reusable building block for post-training analysis, even if re-implemented in PyTorch for this project.

**Koppe, Beutelspacher & Durstewitz (2019), arXiv.** *Inferring Dynamical Systems with Long-Range Dependencies through Line Attractor Regularization.* Direct precedent for the most important Phase 1 training trick: regularize the RNN's Jacobian to encourage a line attractor along the dimension carrying the slow context variable. The project's `background_knowledge.txt` already documents that without such an attractor the context signal decays to ~0.8 % after one trial — line-attractor regularization is one published fix.

**Yang, Joglekar, Song, Newsome & Wang (2019), Nature Neuroscience.** *Task representations in neural networks trained to perform many cognitive tasks.* Methodological template for curriculum training of RNNs on cognitive tasks and for cluster analysis of unit selectivity. The spec's per-unit "mixed selectivity" validation (#4) is downstream of this.

**Driscoll, Shenoy & Sussillo (2024), Nature Neuroscience.** *Flexible multitask computation in recurrent networks utilizes shared dynamical motifs.* Identifies recurring dynamical motifs — ring attractors, line attractors, decision boundaries — that trained RNNs reuse across tasks, and shows that ReLU networks implement motifs via clustered sub-populations. The latent-circuit framework expected to emerge in Phase 1 is one such motif (modality-selective context populations gating a decision motif).

**Khajehabdollahi, Zeraati, Giannakakis, Schafer, Martius & Levina (2023), ICLR.** *Emergent mechanisms for long timescales depend on training curriculum and affect performance in memory tasks.* Shows that curriculum order shapes whether RNNs use line attractors vs other mechanisms to retain information across long delays. Reinforces that the spec's 5-stage curriculum is not aesthetic — it will determine *what kind of solution* the network finds.

**Haviv, Rivkind & Barak (2019), ICML.** *Understanding and Controlling Memory in Recurrent Neural Networks.* Mechanistic analysis of how RNNs store information across delays and how regularizers can be designed to control memory horizon — useful for diagnosing the project's known vanishing-context-gradient problem.

---

## 5. Closed-loop reward-driven training of RNNs on cognitive tasks

The Phase 1 spec is unusual in that reward and own-lick channels are driven by the network's own behavior during training, not teacher-forced. This is the smallest body of literature directly relevant to the project but the most important methodologically.

**Song, Yang & Wang (2016), eLife.** *Reward-based training of recurrent neural networks for cognitive and value-based tasks.* Trains continuous-time RNNs on cognitive tasks with REINFORCE — the network's outputs gate the reward signal, exactly as in the Phase 1 spec. Establishes that closed-loop reward-driven training is tractable on the scale of the spec's tasks. The non-supervised contingency between output and reward is the structural feature the project inherits.

**Ger & Barak (2025), arXiv.** *Learning Dynamics of RNNs in Closed-Loop Environments.* Most recent systematic analysis of how training dynamics differ when the input distribution is determined by the network's own output. Directly relevant for diagnosing instabilities that arise in Stages 0-2 of the spec's curriculum.

**Barak, Sussillo, Romo, Tsodyks & Abbott (2013), Progress in Neurobiology.** *From fixed points to chaos: three models of delayed discrimination.* Three RNN models of monkey delayed discrimination, illustrating how different attractor structures (fixed points, line attractors, chaotic transients) can implement the same delay computation. Sets expectations for the range of solutions a curriculum like the spec's might select among.

**Ehrlich, Stone, Brandfonbrener, Atanasov & Murray (2020), eNeuro.** *PsychRNN: An Accessible and Flexible Python Package for Training Recurrent Neural Network Models on Cognitive Tasks.* Reference implementation of the supervised-training side of this literature; an explicit comparator for the project's training code.

**Kononov, Pospelov, Anokhin, Nekorkin & Maslennikov (2025), arXiv.** *Emergence of hybrid computational dynamics through reinforcement learning.* Shows that RL-trained RNNs spontaneously develop hybrid attractor + transient dynamics, providing a recent example of the kind of solution structure the Phase 1 RNN may find.

---

## Synthesis: what this literature implies for Phase 1

1. **The architecture is well-supported.** A 200-unit ReLU RNN with reward + own-lick + trial-phase as auxiliary inputs is essentially a Wang-2018 / RL² meta-learner restricted to a 2-MDP family. The literature has not previously combined this architecture with the dynamic-routing task and a Mante-2013-style latent-circuit analysis target — that is the project's novelty.

2. **The expected solution mechanism is the latent-circuit one.** Dubreuil 2022 + Langdon & Engel 2025 + Hajnal 2024 all converge on the same prediction: context-selective sub-populations will form and gate modality-specific decision pathways by mutual inhibition. Validation check #4 (mixed selectivity) and the eventual Phase 2 latent-circuit fit are direct probes of this prediction.

3. **The known hard problem is the slow-context-variable bottleneck.** The spec's pitfalls and the project's own `background_knowledge.txt` describe exactly the vanishing-context-gradient problem that Koppe et al. (2019), Driscoll et al. (2024), and Khajehabdollahi et al. (2023) have published mitigations for: line-attractor regularization or curriculum-staged emergence of long timescales. The project should expect either (a) to use one of these published fixes (the context-attractor initialization in `background_knowledge.txt` is conceptually a variant of Koppe 2019), or (b) to demonstrate empirically that the spec's curriculum alone is sufficient.

4. **Closed-loop training is comparatively under-studied.** Song-Yang-Wang (2016) and Ger-Barak (2025) are essentially the only systematic references; the project will be operating at the leading edge of this sub-literature and should expect to encounter undocumented instability modes.

5. **Post-training validation has off-the-shelf tools.** FixedPointFinder, Driscoll 2024's motif catalog, and the Mante-2013 PCA conventions can all be reused directly for validation checks #4 and #5.

## Gaps that motivate downstream hypotheses

- **G1: No published Phase-1 + dynamic-routing combination.** No prior work has trained an RNN end-to-end on the spec's exact task with no explicit context channel and closed-loop reward — every closely-related model (Hajnal 2024, Orlandi 2021, Mante 2013) gives the network either a context cue, a different task structure, or teacher-forced reward. Phase 1 is genuinely a new demonstration.

- **G2: It is unknown whether the spec's curriculum alone solves the slow-context-variable problem.** The literature offers principled fixes (line-attractor regularization, gradual-timescale curricula), but the spec proposes a 5-stage advancement gate without an explicit attractor regularizer. The project's own v9–v11 logs already suggest plain curriculum is insufficient. A first hypothesis to test is: "the spec's 5-stage curriculum, as written, does NOT reliably produce a line-attractor context substrate; it needs either explicit Jacobian regularization (Koppe 2019) or weight-perturbation initialization (the project's context-attractor trick)."

- **G3: It is unknown how solution diversity across the 50-seed ensemble compares to the multi-population low-rank theory.** Dubreuil 2022 predicts that flexible context-dependent tasks need multi-population structure; whether 50 unconstrained seeds produce 50 minor variations of one mechanism or 50 distinct mechanisms is an empirical question this project is well-placed to answer.

- **G4: The role of the 5-trial instruction preamble in driving context-belief updates is untested.** No prior model uses 5 repeated unrewarded-target trials (with late_autoreward semantics) as the only context-disambiguating signal. Whether all 5 are necessary, how the context belief evolves trial-by-trial inside the preamble, and how quickly the post-preamble d' reaches threshold are open empirical questions on which the project can yield direct measurements (validation check #3).

- **G5: No published precedent for late_autoreward + own-lick gating with a sigmoid scalar readout.** Most reward-based RNN work uses a multi-class softmax or REINFORCE policy gradient; the spec's sigmoid `z > 0.5` lick threshold with closed-loop reward gating has no direct analogue in the literature, and its stability under truncated BPTT is therefore unknown a priori.
