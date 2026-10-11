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

---

## Active Status & Next Steps

| Milestone / Experiment | Priority | Description | Status |
|---|---|---|---|
| **M8: Inference Search (2-3 Ply Minimax / MCTS)** | High | Add lightweight search to the 1M ResNet to eliminate tactical blunders and boost Elo | Planned |
| **M9: Engine Blunder Filtering (Quality Gate)** | Medium | Filter human games by Stockfish centipawn loss ($\le 25$ cp) to train exclusively on sound master moves | Planned |
| **M10: Data Scaling Curve (1M vs 3M vs 10M)** | Long-term | Compare whether scaling beyond 1M positions improves Elo or hits human imitation saturation | Planned |
