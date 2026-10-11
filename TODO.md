# ChessLab: Project TODO & Research Backlog

This document tracks all active, planned, and backlogged engineering and research tasks for the ChessLab framework. It is kept synchronized with [ACHIEVEMENTS.md](file:///home/surajwork/Coding/misc/chessAI/ACHIEVEMENTS.md) and [docs/PRD.md](file:///home/surajwork/Coding/misc/chessAI/docs/PRD.md).

---

## 1. Immediate Active Focus (Current Sprint)

- [ ] **TODO-04: Data-Scaling Curve Experiment (1M vs. 3M Positions)**
  - **Goal**: Test whether increasing dataset from 1M to 3M games produces measurable improvements in playing strength or hits human-imitation saturation.
  - **Deliverable**: 3M-game streaming shard set, training curve comparison, and scaling report.

- [ ] **TODO-05: Move-Sequence Transformer Comparison (Model B per PRD FR-4)**
  - **Goal**: Train a small decoder-only Transformer on UCI move sequences to compare board-state (Model A) vs. move-sequence (Model B) inductive biases.
  - **Deliverable**: `src/chesslab/models/move_transformer.py`, training script, and side-by-side predictive evaluation.

---

## 2. Upcoming Research Milestones (Medium Term)

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
- [x] **DONE-10: Inference-Time Tree Search Harness** (Completed 2026-10-11, Milestone M8 in ACHIEVEMENTS.md; previously referred to as TODO-01 during planning)
- [x] **DONE-11: Search Correctness Hardening & Metric Resolution** (Completed 2026-10-11, Milestone M9 in ACHIEVEMENTS.md; TT halfmove clock, repetition taint isolation, mate-score ply normalization, 50 tests passing; raw illegal-rate mathematically and empirically proved)
- [x] **DONE-12: Stockfish 16 Calibration Ladder Benchmark Pilot** (Completed 2026-10-11, Milestone M10 in ACHIEVEMENTS.md; 16 configurations, 192 paired games, verified node limits 200-2000, zero illegal moves)
- [x] **DONE-13: Quality-Filtered Dataset Pipeline & Empirical Comparison** (Completed 2026-10-11, Milestone M11 in ACHIEVEMENTS.md; 41.9k evaluated, 25k kept, 40.4% blunder rejection rate; proved predictive accuracy gain [+1.68% test Top-1] does not imply playing strength gain [-70.4 ± 122.9 Elo])

---

## 5. Next Session Handoff

1. **What has been completed**:
   - `DONE-11` (M9): TT correctness fixes (halfmove clock in key, repetition taint exclusion, mate-score ply shifting, 50 unit tests passing in `tests/test_search.py`). Empirical resolution of raw unmasked illegal rate in `data/verify_illegal_metric.json`.
   - `DONE-12` (M10): Stockfish 16 local setup (`tools/stockfish16/stockfish`) and 192-game pilot calibration ladder across 16 matrix cells (`data/calibration/pilot_d{0,1,2,3}_sf{200,500,1000,2000}.json`). Node limits verified (200, 500, 1000, 2000). 0 illegal moves recorded.
   - `DONE-13` (M11): Quality-filtered streaming pipeline (`scripts/run_filtered_pipeline.py`). Evaluated 41,979 positions with Stockfish 16 (nodes=500, ~2.8 ms/pos); filtered blunders >25 cp (40.4% rejection rate); generated `data/shards/train_filtered_25k.bin`. Completed controlled comparison (Raw 25k vs Filtered 25k) across test.bin (+1.68% Top-1 for Filtered), tactical puzzles (+0.2% Top-1, +2.5% Top-3), and head-to-head match (40.0% score, -70.4 ± 122.9 Elo).
2. **Current model checkpoints**:
   - 1M baseline: `checkpoints/run_1m/final_ema_model.pt`.
   - Filtered 25k: `checkpoints/filtered_25k/best_model.pt`.
   - Raw 25k baseline: `checkpoints/raw_25k/best_model.pt`.
3. **Verified empirical performance**:
   - Held-out test set (25,368 unseen): Filtered 25k: Top-1 = 22.79%, Top-5 = 53.44% vs Raw 25k: Top-1 = 21.11%, Top-5 = 52.07%.
   - Head-to-head (Depth 1, 20 games): Filtered 25k scored 40.0% (4W-8D-8L, -70.4 ± 122.9 Elo) vs Raw 25k.
   - Tactical puzzles (1,000 rated): Filtered 25k: Top-1 = 13.8%, Top-3 = 33.0% vs Raw 25k: Top-1 = 13.6%, Top-3 = 30.5%.
4. **Highest-priority unfinished task**: `TODO-04: Data-Scaling Curve Experiment (1M vs. 3M Positions)`.
5. **Exact next action**: Stream 3M-game dataset using `scripts/run_1m_pipeline.py --max-games 3000000 --output-dir data/3m` to test whether scaling from 1M to 3M positions overcomes human-imitation saturation.
6. **GPU Acceleration Option**: For scaling to 3M positions or training deeper architectures without CPU bottlenecks, make use of [colab/train_gpu.ipynb](file:///home/surajwork/Coding/misc/chessAI/colab/train_gpu.ipynb) on Google Colab (with T4/A100 GPU and Automatic Mixed Precision).

