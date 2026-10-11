# Product Requirements Document (PRD)
## Chess AI from Scratch: Research, Training, and Pattern Discovery

**Document version:** 1.0  
**Date:** 10 October 2026  
**Project type:** Independent AI research and engineering project  
**Primary audience:** An LLM acting as research advisor, ML architect, and implementation partner  
**Status:** Research completed; implementation planning and experimentation to follow

---

## 1. Executive Summary

The project investigates whether a chess-playing AI can be trained from randomly initialized neural-network weights using large-scale chess game data, potentially progressing from one million games to hundreds of millions or billions.

The project has two primary ambitions:

1. **Chess-playing capability:** Build a neural network that learns chess from data and progressively improves its playing strength.
2. **Chess knowledge discovery:** Investigate whether the model learns useful strategic, positional, tactical, or statistical patterns that can be extracted, validated, and potentially used to improve chess understanding.

The initial research indicates that training a chess model from scratch is feasible. However, training on historical human games alone primarily teaches the model to imitate human decisions. This does not automatically produce a strong chess engine or guarantee novel strategic discoveries.

The project must distinguish between human-move prediction, engine-guided learning, position evaluation, self-play reinforcement learning, and genuine pattern discovery.

**The central strategic decision is to begin with a small, reproducible experiment and expand only when measured results justify additional data, model complexity, and computational expenditure.**

The first experiment should use approximately one million publicly available Lichess games, train a small neural network from random initialization, establish reliable evaluation benchmarks, and compare alternative board representations.

No large-scale training run should be approved before the initial experiments demonstrate measurable improvements in playing strength.

## 2. Project Context

### 2.1 Original idea

The original hypothesis is that exposing a neural network to an enormous number of chess games may allow it to learn more than opening sequences and common move patterns.

Potentially useful learned capabilities include:

- Recognizing tactical motifs and recurring positional structures.
- Learning relationships between board configurations and game outcomes.
- Estimating positional advantages.
- Recognizing common human mistakes and decision patterns.
- Learning strategic concepts that are difficult to express as manually written rules.
- Discovering candidate relationships between position features, move choices, and winning outcomes.
- Developing a representation of chess positions that supports stronger decision-making.

These are research hypotheses, not guaranteed outcomes.

### 2.2 What the project is not

The project must not assume that:

- A billion games necessarily produce a stronger model than ten million carefully selected games.
- Predicting the human move is equivalent to finding the best move.
- A model with more parameters is automatically stronger.
- A high move-prediction score implies strong tournament performance.
- A statistical correlation is a new chess principle.
- Training a model from random weights means that it learns without external supervision.
- A neural network trained on PGN alone will necessarily outperform Stockfish or Leela Chess Zero.

The project must use experiments to distinguish demonstrated capabilities from speculation.

### 2.3 Definition of “from scratch”

For this project, the default definition is:

**A neural network initialized with random weights and trained on chess-specific data, without relying on a pretrained general-purpose language model.**

The project may still use external software, public chess games, and optionally Stockfish-generated training labels.

Every experiment must explicitly state whether it uses:

- Human game moves as labels.
- Game results as labels.
- Engine evaluations or engine-selected moves.
- Self-play outcomes.
- Tree search or other inference-time search.

This distinction is necessary to interpret the results correctly.

## 3. Research Findings and Current Conclusions

The following findings are derived from the initial research report. They should guide the project but must be re-verified against the original papers before being used in a formal publication.

### 3.1 Human-game imitation is feasible

Models such as Maia and Chess-GPT demonstrate that neural networks can learn meaningful chess behaviour from recorded games.

The report cites approximately 55–57% top-1 human-move agreement for a Maia-3 model, and reports that Chess-GPT learned legal chess moves at a high rate while achieving club-level playing strength in its reported setup.

Implication: a baseline model trained on PGN is a reasonable starting point.

Limitation: human moves contain mistakes, inconsistencies, and biases. The model may reproduce these rather than learn objectively stronger play.

### 3.2 Engine-labelled training can produce stronger play

The report describes ChessBench as an example in which historical game positions were labelled using Stockfish action-value information.

The report cites a 270-million-parameter model trained on positions derived from ten million Lichess games and reports a 2895 Lichess blitz rating in the relevant evaluation setting.

The essential distinction is that the historical games supplied the positions, while the engine supplied the learning targets.

Implication: engine-guided supervision should be evaluated as a potential route toward stronger play.

Limitation: this approach inherits knowledge from an existing engine and requires additional computation for generating labels.

### 3.3 Self-play can produce superhuman chess

AlphaZero and Leela Chess Zero demonstrate that neural networks trained through self-play, reinforcement learning, and search can achieve very strong chess performance.

Implication: self-play is a plausible later-stage approach if the project aims for high playing strength.

Limitation: reproducing this methodology can require substantial compute, infrastructure, experimentation, and engineering.

### 3.4 Billions of games are not the appropriate starting point

The report identifies large public chess datasets, including Lichess game archives. It also notes that dataset counts change over time and that game quantity alone does not determine training value.

Implication: measure data-scaling behaviour before committing to large-scale processing and training.

The initial data experiments should compare subsets of approximately one million, three million, and ten million games.

### 3.5 Pattern discovery requires independent validation

A model may develop useful internal representations, but identifying those representations does not establish that they correspond to novel or strategically useful knowledge.

Raw win-rate correlations are particularly vulnerable to confounding factors such as player strength, opening choice, colour, position advantage, and game selection.

Implication: pattern discovery must be implemented as a separate research and validation pipeline rather than treated as an automatic consequence of training.

### 3.6 Current project conclusion

The project is worth attempting as a staged independent research effort.

The recommended sequence is:

1. Build the data pipeline and evaluation infrastructure.
2. Train a small PGN-only baseline.
3. Compare board-state and move-sequence representations.
4. Measure actual playing strength.
5. Investigate model representations and candidate patterns.
6. Evaluate engine-labelled training.
7. Consider self-play and search.
8. Scale only when measured evidence supports it.

The project has not yet established that a new model will outperform established chess engines or discover genuinely novel chess principles.

## 4. Product Vision

Develop a reproducible experimental framework for training chess models from scratch, evaluating their playing strength, and investigating what they learn from large-scale chess data.

The framework should support multiple training strategies and make it possible to compare them under controlled conditions.

The long-term ambition is to produce a chess model whose capabilities and learned patterns can be studied scientifically, rather than merely building another move-prediction demonstration.

## 5. Goals and Non-Goals

### 5.1 Primary goals

**G1. Train a chess model from random initialization.**

Create a functional neural network that accepts a chess position and predicts a move.

**G2. Establish reliable benchmarks.**

Measure held-out move-prediction accuracy and actual playing strength separately.

**G3. Quantify the effect of dataset scale.**

Determine whether increasing the dataset from one million to three million and ten million games produces measurable improvements.

**G4. Compare model architectures.**

Evaluate a structured board-state model against a move-sequence Transformer.

**G5. Investigate stronger supervision.**

Determine whether engine evaluations and engine-selected moves improve playing strength enough to justify their cost.

**G6. Build a pattern-discovery workflow.**

Generate candidate patterns, test them against confounders, and validate them on independent positions.

**G7. Preserve reproducibility.**

Record data provenance, model configurations, training settings, evaluation settings, checkpoints, and experiment results.

### 5.2 Non-goals for the first milestone

The initial implementation is not required to:

- Train on a billion games.
- Outperform Stockfish.
- Reproduce AlphaZero at its original computational scale.
- Implement distributed multi-GPU training immediately.
- Guarantee the discovery of previously unknown chess principles.
- Build a production chess website or multiplayer platform.
- Generate an entire production-ready engine before validating the baseline.

These may become future objectives if the experimental evidence supports them.

## 6. Target Users and System Responsibilities

### Primary user

An independent developer conducting a chess AI research project.

### LLM responsibilities

The LLM assisting this project must act as:

1. **Research advisor:** Explain existing work, identify gaps, and verify claims.
2. **ML architect:** Recommend representations, model architectures, loss functions, and training methods.
3. **Software engineering partner:** Design modules, write implementation code, review errors, and maintain a coherent repository architecture.
4. **Experiment designer:** Formulate hypotheses, define controls, select metrics, and interpret results.
5. **Research critic:** Identify weak evidence, confounding variables, data leakage, and unsupported conclusions.
6. **Project planner:** Prioritize work according to research value, available compute, and implementation effort.

The LLM must not assume that a proposed feature is necessary simply because it appears in this PRD. It should explain important trade-offs and recommend the smallest experiment capable of answering each research question.

## 7. Functional Requirements

### FR-1: Dataset acquisition

The system must support importing publicly available PGN files, beginning with a suitable Lichess dataset.

It must:

- Preserve original source files and checksums.
- Record dataset version and retrieval date.
- Validate PGN syntax and game metadata.
- Identify duplicates and invalid games.
- Filter unsupported variants for the initial standard-chess experiment.
- Track player ratings, time controls, results, and other relevant metadata when available.
- Document licensing and permitted uses.
- Support reproducible subsets of the dataset.

### FR-2: Position extraction

Convert games into training examples containing the board state, legal moves, played move, game result, and relevant metadata.

The representation must correctly handle:

- Side to move.
- Piece placement.
- Castling rights.
- En passant.
- Promotions.
- Move legality.
- Repetition and draw-related state where relevant to the selected model.
- Game termination.

Use a trusted chess library such as python-chess for initial parsing and legality checks.

### FR-3: Data splitting and leakage prevention

Separate training, validation, and test data before model selection.

The split must minimise contamination from:

- Duplicate games.
- Repeated positions across splits.
- Closely related games.
- Shared events or player identities where relevant.
- Temporal overlap when measuring generalisation to later games.

The experiment report must describe the split policy and acknowledge residual leakage risks.

### FR-4: Baseline models

Implement at least two models:

**Model A: Board-state policy network**

- Receives a structured chess-board representation.
- Predicts a distribution over candidate moves.
- Optionally predicts win, draw, or loss probabilities.
- Uses legal-move masking at inference time.

**Model B: Move-sequence Transformer**

- Receives an encoded move history or PGN sequence.
- Predicts the next move.
- Uses a clearly documented move vocabulary and decoding procedure.
- Validates generated moves against legal chess rules.

Start with a small model. The initial board-state model should target approximately 3–10 million parameters, subject to hardware constraints.

### FR-5: Training and experiment tracking

The training pipeline must support:

- Random initialization.
- Reproducible configuration files and random seeds.
- Checkpoint saving and resumption.
- Training and validation metrics.
- Learning-rate scheduling.
- Mixed-precision training when supported.
- Early stopping or explicit training budgets.
- Logging of compute usage and wall-clock time.

The initial stack should prioritise Python, PyTorch, python-chess, NumPy, and lightweight experiment tracking.

Distributed training and more complex data systems should be introduced only when justified.

### FR-6: Evaluation

The system must maintain separate evaluation suites for:

**Move prediction**
- Top-1 accuracy.
- Top-5 accuracy.
- Cross-entropy.
- Results by player-rating band.
- Results by opening, middlegame, and endgame.
- Results by tactical or strategic position category where labels are available.

**Playing strength**
- Match results against fixed baselines.
- Estimated rating with uncertainty.
- Win, draw, and loss rates.
- Performance under documented time controls.
- Results across colour, opening, and position conditions.

**Pattern discovery**
- Candidate-pattern reproducibility.
- Generalisation to unseen positions.
- Robustness after controlling for confounders.
- Comparison against engine analysis.
- Independent validation and correction for multiple statistical tests.

All engine comparisons must document the engine version, settings, hardware, and search budget.

### FR-7: Engine-guided supervision

Support a later experiment in which Stockfish provides evaluations or candidate move labels for selected positions.

Compare the resulting model against the PGN-only baseline.

Record:

- Number of labelled positions.
- Engine settings and analysis budget.
- Label-generation time and cost.
- Model training cost.
- Improvement in playing strength.
- Whether the improvement justifies the added expense.

Do not add engine labelling to the first experiment unless it is required to resolve a specific blocker.

### FR-8: Pattern-analysis tools

Provide tools to:

- Extract and compare internal model representations.
- Cluster similar positions.
- Find candidate tactical or positional features.
- Compare human choices with engine-preferred moves.
- Investigate conditions under which a pattern appears.
- Produce representative board positions for manual inspection.
- Test hypotheses on held-out positions.

The system must distinguish descriptive observations from validated discoveries.

### FR-9: Scale-up decision

Before expanding the dataset or model, produce a report containing:

- Measured scaling curves.
- Marginal improvement from additional data.
- Training and preprocessing cost.
- Playing-strength improvement.
- Prediction improvement.
- Uncertainty and reproducibility.
- Projected cost of the next experiment.

No scale-up decision should be based solely on dataset size, training loss, or prediction accuracy.

## 8. Non-Functional Requirements

### Reproducibility

Every experiment must be repeatable from recorded configuration, dataset version, code revision, and model checkpoint.

### Correctness

Board-state encoding, legal move generation, move decoding, and game-result handling must have unit tests.

### Resource efficiency

The project should begin with a single-machine workflow and avoid distributed infrastructure until a measured bottleneck requires it.

### Observability

The system must expose data-processing throughput, training speed, GPU memory use, storage consumption, failures, and evaluation metrics.

### Scientific integrity

All results must be labelled as one of:

- Verified external finding.
- Direct experimental observation.
- Calculation from stated assumptions.
- Estimate or projection.
- Unverified hypothesis.

Failed experiments and negative results must be retained rather than silently discarded.

### Data governance

Use data with appropriate permissions, follow relevant dataset licences and service terms, and minimise unnecessary retention of identifiable player metadata.

## 9. Proposed Technical Architecture

Use modular components with clearly defined interfaces.

```text
chess-ai-research/
├── configs/
│   ├── data.yaml
│   ├── board_model.yaml
│   └── transformer.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── parse_pgn.py
│   │   ├── validate.py
│   │   ├── deduplicate.py
│   │   ├── encode_board.py
│   │   └── build_dataset.py
│   ├── models/
│   │   ├── board_policy.py
│   │   ├── board_value.py
│   │   └── move_transformer.py
│   ├── training/
│   │   ├── train.py
│   │   ├── losses.py
│   │   └── checkpointing.py
│   ├── evaluation/
│   │   ├── move_prediction.py
│   │   ├── engine_matches.py
│   │   └── rating_estimation.py
│   ├── patterns/
│   │   ├── embeddings.py
│   │   ├── clustering.py
│   │   └── validation.py
│   └── inference/
│       ├── predict_move.py
│       └── legal_moves.py
├── tests/
├── experiments/
├── reports/
├── requirements.txt
└── README.md
```

This structure is a proposed starting point, not a requirement to create every module immediately.

## 10. Experimental Roadmap

### Phase 0: Research and environment setup

**Objective:** Confirm the research assumptions and prepare a reproducible development environment.

Tasks:
- Verify primary sources for the central published claims.
- Inspect available hardware and choose a training environment.
- Create the repository and dependency configuration.
- Define a minimal dataset manifest.
- Establish basic tests.

Exit criteria: the environment works and the first experiment is fully specified.

### Phase 1: Dataset feasibility

**Objective:** Demonstrate that the data can be processed reliably.

Tasks:
- Download a representative Lichess PGN sample.
- Parse and validate games.
- Remove or account for duplicates.
- Extract positions and labels.
- Measure processing speed and storage.
- Verify position encodings against known examples.

Exit criteria: a reproducible dataset containing approximately one million games, with a documented quality report.

### Phase 2: Baseline move prediction

**Objective:** Establish the first learned model.

Tasks:
- Implement a simple statistical baseline.
- Train a small board-state policy network.
- Train a move-sequence Transformer if compute permits.
- Compare training and held-out performance.
- Check legal move handling.
- Record training cost and throughput.

Exit criteria: reproducible test metrics and a model capable of predicting moves from valid positions.

### Phase 3: Playing-strength evaluation

**Objective:** Determine whether the trained model plays competent chess.

Tasks:
- Build a legal move-selection interface.
- Run matches against fixed baselines.
- Use controlled time limits and openings.
- Estimate strength with uncertainty.
- Separate prediction quality from match performance.

Exit criteria: a documented estimate of playing strength and an evaluation process that can compare future models fairly.

### Phase 4: Data-scaling experiment

**Objective:** Test whether additional human games materially help.

Train comparable models using approximately:
- 1 million games.
- 3 million games.
- 10 million games.

Control model architecture, training procedure, evaluation conditions, and compute budgets as far as practicable.

Exit criteria: measured scaling curves for prediction performance, playing strength, and cost.

### Phase 5: Pattern discovery

**Objective:** Investigate what the model has learned.

Tasks:
- Extract internal representations.
- Cluster positions and inspect representative examples.
- Compare model decisions with human moves and engine analysis.
- Control for rating, opening, position advantage, and other confounders.
- Test candidate findings on independent positions.

Exit criteria: a small set of clearly documented candidate findings, including negative results and uncertainty. Novel strategic knowledge is not a mandatory outcome.

### Phase 6: Engine-guided learning

**Objective:** Test whether stronger labels improve play.

Tasks:
- Select a manageable subset of positions.
- Generate engine evaluations or candidate move labels.
- Train a model from random initialization on those labels.
- Compare it with the PGN-only model.
- Measure label-generation and training costs.

Exit criteria: an evidence-based decision on whether engine supervision is worthwhile.

### Phase 7: Self-play and search

**Objective:** Investigate reinforcement learning or search-based improvement.

This phase should begin only after the preceding evaluation pipeline is trustworthy.

Tasks:
- Define the self-play objective.
- Select an appropriate policy/value architecture.
- Investigate search and self-play data generation.
- Compare improvements against the engine-guided baseline.
- Estimate infrastructure and compute costs before scaling.

Exit criteria: a reproducible pilot demonstrating that the additional complexity offers a measurable benefit.

### Phase 8: Scale-up decision

Consider hundreds of millions or billions of games only when experiments show that additional scale is likely to improve the project's primary objective.

The decision must include an explicit budget, expected gain, alternative experiments, and a clear stopping condition.

## 11. Initial Success Metrics

The first milestone should be considered successful when:

1. A reproducible pipeline can process approximately one million games.
2. Dataset validity, duplication, and storage costs are measured.
3. A randomly initialized neural network learns to predict moves on held-out data.
4. Legal move generation and decoding pass the relevant tests.
5. A playing-strength evaluation harness produces reproducible results.
6. The project can compare at least two model or data configurations.
7. The resulting evidence supports a rational next-step decision.

These are engineering and research milestones. They do not imply a target Elo rating or guarantee novel discoveries.

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| The model imitates mistakes rather than strong play | Compare PGN-only training with engine-labelled training. |
| More data produces diminishing returns | Measure scaling curves before increasing dataset size. |
| High prediction accuracy is mistaken for playing strength | Maintain independent match-based evaluation. |
| The model appears to discover a pattern that is already known | Review chess literature and existing engine analysis. |
| Pattern findings are caused by confounders | Use controlled comparisons and independent validation. |
| Data leakage inflates results | Deduplicate games and positions and document split policies. |
| Engine labels are too expensive | Label a selected subset and measure marginal benefit. |
| Infrastructure complexity delays experiments | Begin with a single-machine pipeline and scale incrementally. |
| Public data has restrictive or unclear terms | Review source licences and applicable terms before use. |
| Compute estimates prove inaccurate | Measure real throughput and cost during early experiments. |
| Results cannot be reproduced | Version data, code, configurations, and checkpoints. |

## 13. Evidence and Citation Requirements

The existing research report was prepared on 10 October 2026. Its quantitative estimates must be treated as dated findings, not timeless constants.

The LLM must:

- Consult the original research report before answering project-specific questions.
- Retrieve the original papers or official sources before repeating major performance claims.
- Distinguish published results from this project's own measurements.
- Preserve the original experimental conditions when reporting Elo or accuracy.
- Avoid directly comparing ratings from different pools or rating systems without an appropriate methodology.
- Identify any unresolved or conflicting evidence.
- Never invent benchmarks, citations, dataset counts, or cost figures.

The project's source of truth should be the research report, the repository, experiment records, and verified primary literature. General model knowledge is supplementary and must be labelled when it changes a source-derived conclusion.

## 14. Immediate Next Actions

The LLM should prioritise the following work in order:

1. Review the complete research report and extract its recommendations, references, implementation roadmap, and cost assumptions.
2. Create a concise technical decision record describing the initial architecture, dataset, evaluation strategy, and reasons for those choices.
3. Ask for hardware details only if they are unavailable and materially affect the design.
4. Specify the first experiment, including input representation, model size, data split, training objective, evaluation metrics, and expected resource usage.
5. Design the repository and implement the dataset pipeline.
6. Build and validate the baseline model.
7. Record the results before proposing larger experiments.

Do not begin by implementing distributed training, downloading billions of games, or designing a complicated self-play system.

## 15. Instructions for the LLM Taking Over This Project

You are assisting with a serious, staged chess AI research project.

Your responsibility is to help turn the existing research into reproducible experiments and evidence-based engineering decisions.

At the start of your work:

1. Read this PRD and the complete research report.
2. Identify what has been researched, what has been decided, what has been implemented, and what remains unverified.
3. Do not assume that any implementation has already been completed unless the repository or experiment records demonstrate it.
4. Maintain a clear distinction between established findings, hypotheses, engineering estimates, and measured project results.
5. Recommend the smallest experiment that can resolve the next important uncertainty.
6. Explain technical decisions in terms of expected benefit, cost, complexity, and alternatives.
7. When writing code, provide complete and testable changes that fit the existing architecture.
8. When interpreting results, identify confounders and avoid conclusions unsupported by the evidence.
9. Update the experiment log and decision record as the project evolves.
10. Do not recommend scaling to hundreds of millions or billions of games without measured evidence that additional scale is justified.

For each proposed experiment, state:

- Research question.
- Hypothesis.
- Required data.
- Model and training configuration.
- Evaluation procedure.
- Compute and storage requirements.
- Success and failure criteria.
- Risks and limitations.
- Decision that will follow from the result.

**The immediate objective is not to build the largest chess AI possible. It is to establish a reliable experimental foundation that reveals which training strategies actually improve chess-playing ability and which methods produce useful, independently validated insights.**

---

## 16. Definition of Done for the Initial Research Milestone

The initial milestone is complete when the project has:

- A validated dataset pipeline.
- A reproducible one-million-game dataset subset.
- A baseline model trained from random initialization.
- A separate held-out move-prediction benchmark.
- A legal move-generation and inference interface.
- A reproducible playing-strength evaluation procedure.
- A documented cost and throughput report.
- A comparison of at least two configurations, where feasible.
- A decision record describing whether to continue with data scaling, engine supervision, or pattern analysis.

The next phase must be selected using the observed results, not the assumption that larger training runs are inherently better.