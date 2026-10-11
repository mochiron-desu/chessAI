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

---

### Milestone 9: Search Correctness Hardening & Illegal-Move Metric Empirical Resolution
- **Date**: 2026-10-11
- **Status**: Completed
- **Search Defects Audited & Fixed**:
  - **Transposition Table Fifty-Move Clock**: Added `board.halfmove_clock` to `get_position_key()` in `src/chesslab/search/minimax.py`. Prevents 50-move rule draws from cross-pollinating identical layouts with different clocks.
  - **Path-Dependent Repetition Isolation**: Subtrees influenced by repetition draws (`is_repetition(2)` or `can_claim_threefold_repetition()`) tracked via `_rep_taint` and excluded from position-independent TT storage.
  - **Mate Score Normalization**: Added `score_to_tt()` and `score_from_tt()` with ply adjustment (`MATE_PLY_PENALTY`) ensuring mate distance from root remains invariant across TT hits.
  - **Root Claimable Draw Protection**: Root positions with claimable draws search legal replies rather than returning null move.
  - **Regression Test Suite**: Added 12 new unit/regression tests to `tests/test_search.py`. Total test suite expanded from 38 to 50 tests; 100% passing (`50 passed in 122.45s`).
- **Empirical Resolution of 100% Raw Unmasked Illegal-Move Rate** (`data/verify_illegal_metric.json`):
  - **Mathematical Proof**: Loss is $\mathcal{L} = -\log \frac{\exp(z_{\text{target}})}{\sum_{m \in \mathcal{M}_{\text{legal}}} \exp(z_m)}$. Illegal moves masked to $-\infty$ receive strictly zero gradient: $\forall k \notin \mathcal{M}_{\text{legal}}, \frac{\partial \mathcal{L}}{\partial z_k} \equiv 0.0$.
  - **Empirical Verification (E1)**: On 5,000 held-out test positions (`final_ema_model.pt`), mean legal logit is $-8.20$, while unsuppressed illegal logit reaches $+5.89$. Raw argmax illegal rate is 100.0%, masked argmax illegal rate is 0.0%.
  - **Gradient Measurement (E2)**: Masked CE max absolute gradient on illegal logits is exactly $0.000000$. Full (unmasked) CE yields non-zero negative gradients.
  - **Controlled Training (E3)**: Identical 400-step training from scratch. Masked CE yielded 98.38% raw illegal rate; unmasked CE dropped raw illegal rate to 17.40%.

### Milestone 10: Stockfish 16 Calibration Ladder Benchmark (Pilot Matrix)
- **Date**: 2026-10-11
- **Status**: Completed
- **Opponent Engine**: Stockfish 16 (NNUE `nn-5af11540bbfe.nnue`, 1 thread, 16 MB hash) installed locally at `tools/stockfish16/stockfish` (SHA-256: `00628bd9c9855c1b7ff93d7f8d51b413586cdc6336fe637f9a14a61531a05aca`).
- **Protocol**: 16 matrix cells (ChessLab Depths 0, 1, 2, 3 vs Stockfish Node Budgets 200, 500, 1000, 2000). 6 opening pairs (12 games per cell, colors swapped, max 120 plies). 192 games played total.
- **Node Budget Enforcement Verified**:
  - Target 200 nodes: Stockfish mean actual nodes = 202.1 to 203.7.
  - Target 500 nodes: Stockfish mean actual nodes = 500.6 to 501.5.
  - Target 1000 nodes: Stockfish mean actual nodes = 1000.7 to 1001.2.
  - Target 2000 nodes: Stockfish mean actual nodes = 2000.6 to 2001.5.
  - Node limits isolated exclusively to Stockfish subprocess; ChessLab node/NN counts profiled independently.
- **Results Matrix (12 Games per Cell)**:
  | ChessLab Depth | SF 200 Nodes | SF 500 Nodes | SF 1000 Nodes | SF 2000 Nodes | Median Latency | Mean Nodes/Move |
  |---|---|---|---|---|---|---|
  | **Depth 0 (Searchless)** | 0.0% (0-0-12) | 0.0% (0-0-12) | 0.0% (0-0-12) | 0.0% (0-0-12) | 2.6 ms | 1.0 |
  | **Depth 1 (Alpha-Beta)** | 0.0% (0-0-12) | 0.0% (0-0-12) | 0.0% (0-0-12) | 0.0% (0-0-12) | 53.8 ms | 22.2 |
  | **Depth 2 (Alpha-Beta)** | 8.3% (0-2-10) | 4.2% (0-1-11) | 0.0% (0-0-12) | 0.0% (0-0-12) | 276.5 ms | 155.1 |
  | **Depth 3 (Alpha-Beta)** | 4.2% (0-1-11) | 8.3% (0-2-10) | 0.0% (0-0-12) | 0.0% (0-0-12) | 2788.7 ms | 1516.4 |
- **Key Findings & Evidence**:
  - **Illegal Move Rate**: 0 illegal moves in 192 games (0.0%).
  - **Engine Dominance**: Stockfish 16 NNUE even at 200–500 nodes severely outclasses raw imitation-trained ChessLab (P95+ win rate).
  - **Resistance from Search**: Searchless and Depth 1 achieved 0 draws across all node budgets (0.0% score). Depths 2 and 3 achieved 6 draws total against 200 and 500 node Stockfish, showing tree search provides tactical resistance against shallow engine search.
  - **Latency / Compute Profile**: Depth 3 requires ~2.8s per move on CPU (~10× Depth 2), establishing Depth 2 as the practical CPU sweet spot.

---

---

### Milestone 11: Quality-Filtered Dataset Pipeline & Controlled Empirical Comparison
- **Date**: 2026-10-11
- **Status**: Completed
- **Pipeline Implementation** (`scripts/run_filtered_pipeline.py`):
  - Streaming from Lichess standard blitz/rapid archive; evaluated candidate positions with local Stockfish 16 at 500 nodes (~2.8 ms/pos).
  - Calculated centipawn loss (CPL) strictly from moving player's perspective ($S$ before move vs $S'$ after move).
  - Terminal checkmates scored $+10,000$ (0 CPL); draws scored $0.0$; mate scores normalized with `mate_score=10000`.
  - Preserved strict split boundary via SHA-256 game hash (`salt="chesslab_2026"`). Only `train` partition was filtered. Held-out reference shards (`val.bin`, `test.bin`) kept completely fixed and untouched.
- **Pilot & 25k Shard Empirical Results** (`data/filtered_pipeline_25k.json`):
  - Games scanned: 122,756; PRD filters rejected: 78,430; Non-train split skipped: 2,288.
  - Candidate positions evaluated: 41,979.
  - Kept positions ($\le 25$ cp): **25,000 (59.6% retention rate)** in `data/shards/train_filtered_25k.bin` (SHA-256: `124d26e5b0139a45276195fa6b1519dde02882114eaeced074729654af88da0d`).
  - Blunders rejected ($> 25$ cp): **16,979 (40.4% blunder rate)**.
  - Throughput: 128.7 evaluations/sec on single CPU thread in 326.2 seconds. Engine failures: 0.
  - CPL Distribution: 0 cp (exact top engine move): 20,333 (48.4%); 1–10 cp: 1,758 (4.2%); 11–25 cp: 2,909 (6.9%); 26–50 cp: 4,632 (11.0%); 51–100 cp: 5,434 (12.9%); $>100$ cp: 6,913 (16.5%).
- **Strictly Controlled Model Comparison (Filtered 25k vs Raw 25k Baseline)**:
  - Identical initialization (seed 42), architecture (ChessResNet 6 blocks, 64 channels), optimizer (AdamW, lr $10^{-3} \to 3.3\times 10^{-7}$), batch size 128, 3 epochs.
  - **Training Loss**: Filtered 25k achieved lower loss (**2.3684** vs 2.6519 for Raw 25k).
  - **Validation Set (25,316 positions)**: Filtered Top-1: **22.76%** (+1.27% over Raw 21.49%), Top-5: **53.21%** (+1.16% over Raw 52.05%).
  - **Held-Out Test Set (25,368 positions)** (`data/predictive_filtered_vs_raw_25k.json`):
    - Raw 25k: Top-1 = **21.11%**, Top-5 = **52.07%**, Policy Loss = **2.8844**.
    - Filtered 25k: Top-1 = **22.79%** (+1.68%), Top-5 = **53.44%** (+1.37%), Policy Loss = **2.8373** (-0.0471).
  - **Tactical Puzzle Benchmark (1,000 rated Lichess puzzles)**:
    - Raw 25k: Top-1 = **13.60%**, Top-3 = **30.50%**, Top-5 = **41.00%**, Mate-in-2 = **3.0%**.
    - Filtered 25k: Top-1 = **13.80%** (+0.20%), Top-3 = **33.00%** (+2.50%), Top-5 = **42.90%** (+1.90%), Mate-in-2 = **8.0%** (+5.0%).
  - **Head-to-Head Tournament Matches**:
    - Searchless (20 games, color-swapped): **0 Wins, 20 Draws, 0 Losses** (Score: 50.0%, 0.0 Elo).
    - Depth 1 (20 games, color-swapped, max 100 plies) (`data/match_filtered25k_vs_raw25k_d1.json`):
      - Filtered 25k vs Raw 25k: **4 Wins, 8 Draws, 8 Losses** (Score: **40.0%**, Delta Elo: **-70.4 ± 122.9**).
  - **Research Conclusion & Limitation**:
    - **Predictive vs Playing Strength Divergence**: While blunder filtering clearly improves predictive imitation accuracy (+1.68% test Top-1) and reduces cross-entropy loss, it does **not** translate into an immediate head-to-head playing strength advantage at shallow search ($\Delta \text{Elo} = -70.4 \pm 122.9$, statistically indistinguishable from parity, $p > 0.05$).
    - Filtering out human mistakes exposes the model to a narrower distribution of positions, potentially reducing its familiarity with sharp counter-blunder responses when opponents play unsound moves.

---

### Milestone 12: Colab Notebook Synchronization & Interface Alignment
- **Date**: 2026-10-11
- **Status**: Completed
- **Motivation & Root Cause Analysis**:
  - `colab/generate_data.ipynb` crashed with `TypeError: filter_game() got an unexpected keyword argument 'min_elo'` during streaming ingestion because `filter_game` in `src/chesslab/data/stream_parser.py` hardcoded 1500 Elo without accepting `min_elo`, and `pipeline.py` ignored `cfg.min_elo`.
  - `colab/train_gpu.ipynb` contained a phantom import `from chesslab.inference.eval_interface import ChessEvaluator`, passed invalid argument `evaluator` to `UCIEngine.__init__`, accessed `puzzle_results['total']` instead of `'total_evaluated'`, and missed repository root on `sys.path`.
- **Engineering Changes & Evidence**:
  - `src/chesslab/data/stream_parser.py`: Updated `filter_game(headers, min_elo=1500)` to accept configurable threshold.
  - `src/chesslab/data/pipeline.py`: Forwarded `min_elo=cfg.min_elo` to `filter_game`.
  - `src/chesslab/inference/engine_uci.py`: Updated `UCIEngine.__init__` to accept `model: Optional[ChessResNet] = None`.
  - `scripts/eval_puzzles.py`: Added `"total"` key alongside `"total_evaluated"` in benchmark return dictionary.
  - `scripts/__init__.py`: Created package marker.
  - `colab/generate_data.ipynb`: Added `check_game_filter` backwards-compatible fallback and added repository root to `sys.path`.
  - `colab/train_gpu.ipynb`: Removed `ChessEvaluator`, updated `UCIEngine` instantiation with `model=model`, handled `total_evaluated`, and added repository root to `sys.path`.
  - Unit tests: 50 passed in 161.4s, including newly added custom `min_elo` test assertions.

---

## Active Status & Next Steps

| Milestone / Experiment | Priority | Description | Status |
|---|---|---|---|
| **M8: Inference Search (1-3 Ply Alpha-Beta)** | High | Add lightweight search to ResNet; eliminated blunders (+147.2 Elo, 70% puzzle accuracy) | Completed |
| **M9: Search Hardening & Metric Resolution** | High | Fixed TT halfmove/repetition/mate bugs (50 tests passing); resolved raw illegal rate | Completed |
| **M10: Stockfish 16 Calibration Ladder** | High | 192-game pilot benchmark against Stockfish 16 (200-2000 nodes); verified node limits | Completed |
| **M11: Engine Blunder Filtering (Quality Gate)** | High | Streamed 25k blunder-filtered shard ($\le 25$ cp); proved predictive gain does not imply playing strength gain | Completed |
| **M12: Colab Notebook Synchronization** | High | Resolved all signature mismatches, phantom imports, and filter arguments across Colabs | Completed |
| **M13: Data Scaling Curve (1M vs 3M vs 10M)** | High | Compare whether scaling dataset from 1M to 3M games produces measurable playing strength gains | Next Action |


