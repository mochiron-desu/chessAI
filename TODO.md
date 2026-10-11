# ChessLab: Project TODO & Research Backlog

This document tracks all active, planned, and backlogged engineering and research tasks for the ChessLab framework. It is kept synchronized with [ACHIEVEMENTS.md](file:///home/surajwork/Coding/misc/chessAI/ACHIEVEMENTS.md) and [docs/PRD.md](file:///home/surajwork/Coding/misc/chessAI/docs/PRD.md).

---

## 1. Immediate Active Focus (Current Sprint)

- [ ] **TODO-01: Inference-Time Tree Search Harness**
  - **Goal**: Implement lightweight tree search (2–3 ply Minimax / Alpha-Beta with policy move-ordering and value-leaf evaluation) utilizing the trained 1M ResNet.
  - **Expected Benefit**: Eliminate 1-ply and 2-ply blunders; test the hypothesis that calculation adds +300–500 Elo with zero model retraining.
  - **Deliverable**: `src/chesslab/search/minimax.py`, `tests/test_search.py`, tournament match comparing searched vs. searchless model.

- [ ] **TODO-02: Stockfish Calibration Ladder Benchmark**
  - **Goal**: Measure exact BayesElo against Stockfish 16 at fixed node limits (200, 500, 1000, 2000 nodes).
  - **Deliverable**: Automated match runner in `scripts/run_tournament.py` with Stockfish UCI process; BayesElo report with 95% confidence intervals.

---

## 2. Upcoming Research Milestones (Medium Term)

- [ ] **TODO-03: Quality-Filtered Dataset Pipeline (Phase B Blunder Curation)**
  - **Goal**: Stream games and evaluate human moves with Stockfish at shallow depth (~10ms/pos), filtering out human moves with centipawn loss $>25$ cp.
  - **Hypothesis**: Training on blunder-free human moves improves playing strength without copying engine-specific biases.
  - **Deliverable**: Quality-filtered shard generation and comparison against raw 1M baseline.

- [ ] **TODO-04: Data-Scaling Curve Experiment (1M vs. 3M Positions)**
  - **Goal**: Test whether increasing dataset from 1M to 3M games produces measurable improvements in playing strength or hits human-imitation saturation.
  - **Deliverable**: 3M-game streaming shard set, training curve comparison, and scaling report.

- [ ] **TODO-05: Move-Sequence Transformer Comparison (Model B per PRD FR-4)**
  - **Goal**: Train a small decoder-only Transformer on UCI move sequences to compare board-state (Model A) vs. move-sequence (Model B) inductive biases.
  - **Deliverable**: `src/chesslab/models/move_transformer.py`, training script, and side-by-side predictive evaluation.

- [ ] **TODO-06: Pattern Discovery & Concept Probing (Phase 5 per PRD FR-8)**
  - **Goal**: Implement linear probes and hidden-state clustering to verify what concepts (material balance, piece safety, forks, pins) are linearly decodable from ResNet layers.
  - **Deliverable**: `src/chesslab/patterns/probing.py`, cluster visualization script, and false-discovery-rate (FDR) controlled report.

---

## 3. Engineering & Infrastructure Backlog

- [ ] **TODO-07: Interactive Position Viewer & Inference Web GUI**
  - **Goal**: Lightweight local HTML/SVG dashboard allowing the user to set up a board position, inspect model policy probabilities, view the top candidate moves, and see the value head WDL evaluation.
  - **Deliverable**: `scripts/serve_viewer.py` with minimal vanilla web UI.

- [ ] **TODO-08: Compiled PGN Parsing Bridge (Rust Shakmaty / PGN-Reader)**
  - **Goal**: Prepare a Rust-accelerated parser for potential scaling beyond 10M games if Python throughput becomes a bottleneck.

---

## 4. Completed Tasks Archive

For empirical results, configurations, and metrics, see [ACHIEVEMENTS.md](file:///home/surajwork/Coding/misc/chessAI/ACHIEVEMENTS.md).

- [x] **DONE-01: Research Assessment & Epistemic Audit** (Completed 2026-10-10)
- [x] **DONE-02: CPU-Native ResNet Architecture & Test Suite** (Completed 2026-10-11, 26 unit tests passing)
- [x] **DONE-03: Zero-Disk-Overflow Streaming PGN Ingestion Pipeline** (Completed 2026-10-11)
- [x] **DONE-04: 10,000-Game Smoke Test Ingestion & Verification** (Completed 2026-10-11)
- [x] **DONE-05: 1,000,000-Game Streaming Ingestion at Scale** (Completed 2026-10-11, 2.65M scanned, 1M valid in 68.5 min, 348 MB footprint)
- [x] **DONE-06: 1,000,000-Position ResNet Training Run** (Completed 2026-10-11, 3 epochs, Val Top-1: 38.72%, Val Top-5: 76.06%, WDL: 59.46%)
- [x] **DONE-07: Generalization Evaluation on 25k Held-Out Test Set** (Completed 2026-10-11, Top-1: 38.34%, Top-5: 76.04%, 0.38% gap)
- [x] **DONE-08: 1,000-Puzzle Tactical Benchmark Suite** (Completed 2026-10-11, Top-1: 30.90%, Top-5: 64.80%, Forks: 45.9%)
- [x] **DONE-09: Paired Head-to-Head Tournament Harness** (Completed 2026-10-11, +301.3 ± 249.8 Elo vs baseline, 0 illegal moves)
