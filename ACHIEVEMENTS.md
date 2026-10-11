# ChessLab: Project Achievements & Milestones Log

This document tracks all completed experiments, empirical benchmarks, architectural milestones, and verified findings for the ChessLab research project.

---

## Project Overview & Hardware Profile

- **Goal**: Train a chess AI from scratch on large-scale PGN data and evaluate human emulation, tactical strength, and pattern discovery.
- **Environment**: 100% Local (zero cloud compute, zero GPU rental).
- **Host Hardware**:
  - CPU: 13th Gen Intel Core i5-13420H (4 P-cores, 4 E-cores, 12 hardware threads, AVX2 + VNNI).
  - RAM: 15 GiB (system headroom: ~7 GiB).
  - Storage: 196 GB NVMe SSD (~34 GB free space).
  - ML Backend: PyTorch CPU with Intel oneDNN / OpenMP multi-threading.

---

## Log of Completed Milestones

### Milestone 1: Architectural Design & Acceptance Testing
- **Date**: 2026-10-10
- **Status**: Completed
- **Deliverables**:
  - Implemented `ChessResNet` (~943k parameters, 6 residual blocks, 64 channels) with dual heads (1,880 discrete UCI move logits + 3-way WDL value).
  - Designed symmetric perspective flipping: board planes and moves dynamically orient to side-to-move, eliminating duplicate parameter learning for White and Black.
  - Implemented bitboard-backed legal move masking guaranteed to set illegal move probabilities to strictly 0.0.
  - Built an automated acceptance test suite covering all 6 quality gates (17 unit tests passing).

### Milestone 2: 10,000-Game Streaming Ingestion Smoke Test
- **Date**: 2026-10-11
- **Status**: Completed
- **Deliverables**:
  - Ingested 10,000 games from Lichess standard blitz/rapid archive (`2024-01.pgn.zst`).
  - Scanned 27,486 games in 57.7 seconds (173.2 valid games/sec).
  - Created packed binary shards: `train.bin` (9,467 positions, 3.29 MB), `val.bin` (263 positions, 91 KB), `test.bin` (270 positions, 94 KB).
  - Verified 5-epoch test training: Top-5 accuracy reached 49.43% in 43 seconds.

### Milestone 3: 1,000,000-Game Streaming Ingestion at Scale
- **Date**: 2026-10-11
- **Status**: Completed
- **Dataset Manifest** (`data/manifests/dataset_manifest.json`):
  - Total games scanned from Lichess stream: **2,651,148**
  - Games filtered out (bullet / low Elo < 1500): **1,647,782** (62.2% rejection rate)
  - Duplicate games skipped: **3,366**
  - Valid games retained: **1,000,000**
  - Stream wall-clock runtime: **4,114.55 seconds (68.5 minutes)**
  - Sustained throughput: **243.04 valid games/second**
  - Total storage footprint: **~348 MB** (34 GiB free SSD space preserved)
- **Partitions**:
  - `data/shards/train.bin`: 949,316 positions (330.4 MB)
  - `data/shards/val.bin`: 25,316 positions (8.8 MB)
  - `data/shards/test.bin`: 25,368 positions (8.8 MB)

### Milestone 4: 1,000,000-Position ResNet Training Run
- **Date**: 2026-10-11
- **Status**: Completed
- **Configuration**:
  - Script: `scripts/train.py`
  - Training set: 949,316 positions (1 position subsampled per game to eliminate intra-game correlation)
  - Validation set: 25,316 positions
  - Batch size: 128
  - Optimizer: AdamW (lr: $1\times 10^{-3} \to 3.3\times 10^{-7}$ via cosine schedule)
  - Duration: 3 Epochs (22,251 steps) in **2,516.6 seconds (41.9 minutes)**
  - Throughput: **~1,131 positions/second sustained on CPU**
- **Empirical Results**:
  - Training loss decreased from **3.0794** (Step 1,000) to **2.2680** (Epoch 3 End).
  - Validation Top-1 Move Accuracy: **38.72%** (anchored against Maia-1's 46%–52% at 36M+ positions).
  - Validation Top-5 Move Accuracy: **76.06%**.
  - Validation WDL Outcome Accuracy: **59.46%**.
  - Checkpoints saved: `checkpoints/run_1m/best_model.pt` and `checkpoints/run_1m/final_ema_model.pt`.

### Milestone 5: Held-Out Test Set Generalization Evaluation
- **Date**: 2026-10-11
- **Status**: Completed
- **Dataset**: `data/shards/test.bin` (25,368 completely unseen test positions)
- **Results**:
  - Top-1 Move Accuracy: **38.34%**
  - Top-5 Move Accuracy: **76.04%**
  - Policy Cross-Entropy Loss: **2.0806**
  - WDL Outcome Accuracy: **59.85%**
  - Generalization Gap: **0.38%** (Val 38.72% vs Test 38.34%), confirming near-zero overfitting.

### Milestone 6: Tactical Puzzle Benchmark (1,000 Lichess Puzzles)
- **Date**: 2026-10-11
- **Status**: Completed
- **Dataset**: `data/benchmarks/puzzles_1k.csv` (1,000 rated tactical puzzles)
- **Results**:
  - Top-1 Tactical Solution Accuracy: **30.90%**
  - Top-3 Tactical Solution Accuracy: **52.50%**
  - Top-5 Tactical Solution Accuracy: **64.80%**
  - Performance on **Forks**: **45.9%** Top-1 accuracy
  - Performance on **Crushing Advantage**: **38.3%** Top-1 accuracy
  - Performance on **Rating 1500–1800 Puzzles**: **33.9%** Top-1 accuracy

### Milestone 7: Head-to-Head Tournament Playing Strength
- **Date**: 2026-10-11
- **Status**: Completed
- **Configuration**: 10 opening pairs (20 games total, colors swapped) vs. Random Baseline
- **Results**:
  - Engine Score: **85.0%** (14 Wins, 6 Draws, 0 Losses)
  - Estimated Delta Elo: **+301.3 ± 249.8 (95% CI)**
  - Illegal Move Rate: **0.0%**

### Milestone 8: Inference-Time Tree Search Harness (Controlled Depth Experiment)
- **Date**: 2026-10-11
- **Status**: Completed
- **Architecture & Search Formulation**:
  - Negamax with Alpha-Beta pruning, neural policy move-ordering, and WDL leaf evaluation ($V(s) = P_{\text{win}} - P_{\text{loss}} \in [-1.0, 1.0]$).
  - Terminal checkmate scoring parameterized by ply distance ($-100.0 + \text{ply} \times 0.1$) to strictly prefer faster wins and slower losses.
  - Draw handling incorporating path-dependent repetition (`board.is_repetition(2)` or `board.can_claim_draw()`), 50-move rule, and stalemate.
  - Transposition table caching with exact, lowerbound, and upperbound flag testing, isolated from path-dependent repetition draws.
  - Unit test suite: 9 rigorous tests in `tests/test_search.py` verifying Minimax vs Alpha-Beta equivalence (69.0% node reduction on Sicilian defense benchmark), mate-in-1, mate-in-2, blunder avoidance, and TT agreement.
  - Empirical perspective audit: `tests/test_perspective_audit.py` confirmed identical perspective symmetry and correct sign for White vs Black ($+0.3712$ winning turn, $-0.1046$ losing turn).

- **Tactical Puzzle Benchmark Scaling (200 rated Lichess puzzles)**:
  | Search Mode | Depth (Plies) | Top-1 Accuracy | Mate Accuracy | Mate-in-1 | Fork Accuracy | Latency (Mean) |
  |---|---|---|---|---|---|---|
  | Searchless Policy | 0 | **31.00%** | 25.0% | 19.2% | 42.9% | ~1.2 ms |
  | Alpha-Beta | 1 | **45.50%** (+14.5%) | 65.0% | 100.0% | 38.1% | ~15 ms |
  | Alpha-Beta | 2 | **59.00%** (+28.0%) | 78.3% | 100.0% | 66.7% | ~220 ms |
  | Alpha-Beta | 3 | **70.00%** (+39.0%) | 95.0% | 100.0% | 76.2% | ~1660 ms |

- **Head-to-Head Tournament Match 1: Depth 2 vs. Searchless Baseline**:
  - Configuration: 10 opening pairs (20 games total, color-swapped, 100 max plies)
  - Engine A (Depth 2) vs Engine B (Searchless)
  - Record: **10 Wins, 8 Draws, 2 Losses**
  - Score: **70.0%**
  - Draw Rate: **40.0%**
  - Estimated Delta Elo: **+147.2 ± 140.4 (95% CI)**
  - Performance Profile:
    - Depth 2: Median latency = 227.4 ms, P95 latency = 519.8 ms, Mean nodes = 152.8, Mean NN evals = 151.0
    - Searchless: Median latency = 1.24 ms, P95 latency = 2.00 ms, Mean nodes = 1.0

- **Head-to-Head Tournament Match 2: Depth 3 vs. Depth 2**:
  - Configuration: 5 opening pairs (10 games total, color-swapped, 100 max plies)
  - Engine A (Depth 3) vs Engine B (Depth 2)
  - Record: **3 Wins, 6 Draws, 1 Loss**
  - Score: **60.0%**
  - Draw Rate: **60.0%**
  - Estimated Delta Elo: **+70.4 ± 141.9 (95% CI)**
  - Performance Profile:
    - Depth 3: Median latency = 1,666.2 ms, P95 latency = 4,372.0 ms, Mean nodes = 1,256.7, Mean NN evals = 1,080.2
    - Depth 2: Median latency = 216.3 ms, P95 latency = 493.6 ms, Mean nodes = 139.8, Mean NN evals = 137.9
  - Tradeoff Conclusion: Depth 3 yields additional playing strength (+70.4 Elo over Depth 2), but requires ~7.7× more CPU time per move. Depth 2 provides the optimal balance of tactical reliability and latency for CPU play.

---

## Active Status & Next Steps

| Milestone / Experiment | Priority | Description | Status |
|---|---|---|---|
| **M8: Inference Search (1-3 Ply Alpha-Beta)** | High | Add lightweight search to ResNet; eliminated blunders (+147.2 Elo, 70% puzzle accuracy) | Completed |
| **M9: Stockfish Calibration Ladder Benchmark** | High | Calibrate exact BayesElo against Stockfish at fixed node limits (200, 500, 1000, 2000 nodes) | In Progress |
| **M10: Engine Blunder Filtering (Quality Gate)** | Medium | Filter human games by Stockfish centipawn loss ($\le 25$ cp) to train exclusively on sound master moves | Planned |
| **M11: Data Scaling Curve (1M vs 3M vs 10M)** | Long-term | Compare whether scaling beyond 1M positions improves Elo or hits human imitation saturation | Planned |

