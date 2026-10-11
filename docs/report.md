# Can We Train a Chess AI From Scratch on Hundreds of Millions or Billions of PGN Games?

**A feasibility study, literature review, and implementation blueprint**
Prepared 10 October 2026. Source retrieval and verification were done on that date; anything newer is not covered.

---

## How to read this report (evidence legend)

Every quantitative or factual claim carries one of these tags so you can tell evidence from arithmetic from opinion.

| Tag | Meaning |
|---|---|
| **[V]** | Verified against a primary or official source during this research pass. The source key (e.g. `[S6]`) points to the bibliography in Section 17. |
| **[C]** | My own calculation from stated assumptions. Formulas are shown so you can re-derive them. |
| **[E]** | Estimate or engineering judgment. Treat as a hypothesis to be measured, not a fact. |
| **[M]** | Recalled from memory and **not** re-verified in this pass. Confirm before relying on it. |

Three standing caveats:

1. **Elo numbers from different pools are not comparable.** A Lichess blitz rating (Glicko-2, earned against humans and bots) is a different scale from a BayesElo tournament rating (earned against a fixed set of engines), and neither is a FIDE rating. Where this report quotes both, they are labeled and never mixed.
2. **I did not find any verified result in which a small team trained a superhuman chess engine from scratch.** Where this report projects what a small team could achieve, it says so explicitly and labels the projection **[E]**.
3. **Dataset sizes drift.** Lichess grows by roughly a hundred million games per month **[V]** `[S5]`, so any total quoted here is a snapshot.

---

## 1. Executive summary

**The short answer.** Yes, it is feasible to train a chess model from random initialization on PGN data, and the published evidence says what it will and will not learn.

1. **What PGN-only supervised learning gives you is a model of human play, not a strong engine.**
   - Models trained purely to predict the move a human played reach about **55–57% top-1 move agreement** with humans of matched rating. The best published figure I found is 57.1% for a 79M-parameter model (Maia-3) **[V]** `[S11]`. The earlier Maia family reported 46% to over 52% **[V]** `[S9]`.
   - A GPT-style model trained on raw PGN text learns legal chess (over 99% legal moves reported) and an internal board representation, but plays at club level, not master level. Its 16-layer variant won 64% of games against Stockfish level 0 **[V]** `[S8]`.
   - A transformer trained on games from players capped at 1000 or 1300 rating can play *better than every player in its training set* (about 1500) if sampled at low temperature. This is error-averaging, not new insight. A model trained on players capped at 1500 did not exceed its data **[V]** `[S16]`.
2. **Grandmaster-level searchless play was achieved, but with engine labels, not with human moves.** The 270M-parameter model with a Lichess blitz rating of 2895 was trained on Stockfish 16 action-value labels over 10 million Lichess games (15.3 billion labeled data points). Its behavioral-cloning ablation (predict the human move) is a different, weaker objective **[V]** `[S6]`. The games supplied the *positions*; the engine supplied the *targets*.
3. **Superhuman play from scratch needs reinforcement learning with search, or distillation from something that used it.** AlphaZero beat Stockfish 8 after about 4 hours using 5,000 first-generation TPUs for self-play generation and 64 second-generation TPUs for training **[V]** `[S12]`. Stockfish's own evaluation network is trained on Leela Chess Zero self-play data, so even the strongest "classical" engine now depends on reinforcement-learning output **[V]** `[S20]`.
4. **More games stops helping for human-move imitation well before one billion.** Small, well-designed models already approach the ceiling imposed by human inconsistency. Maia-3's authors attribute the flat accuracy at low ratings to irreducible randomness in weak play (aleatoric uncertainty), and find that scale and better inductive bias help mostly for *strong* play **[V]** `[S11]`. Billions of games are available (Lichess alone has more than 7 billion rated standard games **[V]** `[S2]`), but quality, rating balance, and label strength matter more than raw count.
5. **Pattern discovery is a different project from move prediction, and the strongest published route is indirect.** The credible discoveries come from *interpreting* engine-strength networks (AlphaZero concept probing, learned look-ahead in Leela) and from *contrasting* human behavior with engine evaluation. Mining raw PGN win correlations mostly rediscovers rating, opening theory, and who was already winning.
6. **A billion-game run is not justified by current evidence.** The cheapest informative experiments cost tens to a few hundred dollars. The decision to scale should rest on measured scaling curves from Phases 1–5 (Section 12).

**Recommendation in one paragraph.** Attempt it as a staged project. Start with 1 million Lichess games, build a clean pipeline and a rigorous evaluation harness first, compare a board-state policy/value network against a move-sequence transformer, and add engine labels before considering self-play. Treat "discover new chess patterns" as a separate, validated analysis layer on top of the trained models. Do not buy compute for 100M+ games until a scaling curve on 1M, 3M, and 10M games shows a measurable gain in *playing strength*, not only in prediction accuracy.

---

## 2. Direct answer to "is it feasible?"

Feasibility depends on which objective you mean. These are different problems and the report analyzes them separately throughout.

| Objective | Feasible from PGN alone, trained from scratch? | Best evidence | Confidence |
|---|---|---|---|
| Learn legal chess from move text | **Yes** (over 99% legal moves) | Chess-GPT `[S8]` **[V]** | High |
| Predict the move a human of rating *r* will play | **Yes**, about 55–57% top-1 | Maia-3 `[S11]` **[V]** | High |
| Play like a human at a given rating | **Yes**, within the limits of imitation | Maia family `[S9]` `[S10]` `[S11]` **[V]** | High |
| Play above the average rating of the training data | **Partially**, via low-temperature averaging | Transcendence `[S16]` **[V]** | Medium |
| Evaluate positions (value) | **Weakly**: game result is a noisy label | No direct PGN-only strength result found | Low |
| Play at master or grandmaster strength | **No** from human moves alone; **yes** with engine labels | ChessBench `[S6]` **[V]** | High |
| Play at superhuman strength | **No** from PGN alone; needs self-play with search or engine distillation | AlphaZero `[S12]`, Leela `[S13]` **[V]** | High |
| Discover genuinely new strategy | **Unproven** from PGN alone; plausible via model interpretability and human-vs-engine contrast | `[S17]` `[S18]` `[S19]` **[V]** | Low to medium |

**The distinction that governs everything:** predicting moves humans played, predicting moves a strong engine prefers, evaluating positions, discovering statistically meaningful patterns, and playing stronger chess are five different objectives. One network can be trained for several of them with multiple heads, but they pull in different directions, and a gain on one is not evidence of a gain on another.

---

## 3. Definitions: what "from scratch" can mean

| # | Definition | Starting point | Teacher signal | Search | What it can reach (evidence) |
|---|---|---|---|---|---|
| 1 | **Supervised imitation on PGN** | Random weights | Human move (and optionally result) | None at train time | Human-like, club to strong-amateur play. Maia family, Chess-GPT. |
| 2 | **Language-model-style next-move prediction** | Random weights | Next token of the PGN text | None | Legal play, about 1500-class strength. Chess-GPT `[S8]`. Board state is learned implicitly. |
| 3 | **Supervised learning on engine labels over PGN-sourced positions** | Random weights | Stockfish value or action-value | None at inference | Grandmaster-level blitz play (2895 Lichess). ChessBench `[S6]`. Not "from scratch" in the strict sense because the knowledge is distilled. |
| 4 | **Self-play reinforcement learning with search** | Random weights, rules only | Search visit counts and game outcomes | MCTS in the loop | Superhuman. AlphaZero `[S12]`, Leela `[S13]`. No human games needed. |
| 5 | **Hybrid: PGN pretraining, then self-play or distillation** | Weights pretrained on PGN | Mixed | Optional | Plausible shortcut; evidence for chess specifically is thin (Section 7). |

**"From scratch" is a spectrum.** Definitions 1, 2, and 4 start from random weights and use no external model. Definition 3 starts from random weights too, but imports a teacher's knowledge through its labels. If your goal is a *research claim about what data alone can teach*, only 1, 2, and 4 count. If your goal is *a strong playing network*, 3 is the most efficient route.

**Board representation versus move history.** A model can be given the board explicitly (as planes or square tokens), or only the move sequence (as text or move tokens). The move-only route works: Chess-GPT recovers the board state internally, shown by linear probes and by successful edits of that internal state `[S8]` **[V]**. The Chessformer authors argue the move-sequence formulation is misaligned with chess because chess is Markovian given the current position, and cite evidence that move-token models spend capacity reconstructing the board at every step `[S11]` **[V]**.

**Legality.** Neither route guarantees legal output. In practice, always compute the legal move set with a rules library at inference time and mask the model's output to it. The Chess-GPT figure of over 99% legal moves is a measurement of the model, not a license to omit the mask `[S8]` **[V]**.

---

## 4. Existing research and state of the art

### 4.1 Summary table

| System | Year | Training data / labels | From scratch? | Search | Reported compute | Headline result | Relevance to this project |
|---|---|---|---|---|---|---|---|
| **AlphaZero** `[S12]` | 2017/2018 | Self-play only | Yes (rules only) | MCTS | 5,000 TPU v1 for generation, 64 TPU v2 for training, 700k steps at batch 4,096 **[V]** | Surpassed Stockfish 8 after about 4 hours; beat it 28–0–72 (W–L–D) in a 100-game match after about 9 hours **[V]** | Defines the self-play ceiling. Not reproducible by a small team at original scale. |
| **Leela Chess Zero** `[S23]` `[S13]` | 2018– | Volunteer self-play; over 2.5 billion games by June 2024 **[V]** | Yes (original runs) | MCTS | Distributed volunteer compute | Comparable to Stockfish in strength; transformer architecture since 2022 **[V]** `[S14]` | Best open reference for self-play at scale. Its self-play data is also the source of Stockfish's training data `[S20]`. |
| **Stockfish + NNUE** `[S20]` | 2020– | Search-labeled and Leela-derived positions | No (hybrid) | Alpha-beta | Thousands of CPU cores for testing | Strongest classical engine; current architecture family SFNNv9, first layer width 3072 **[V]** | Shows the practical frontier is "search plus a network trained on RL data". Also your evaluation opponent. |
| **Giraffe** `[S15]` | 2015 | Positions from computer games, then TD-Leaf self-play | Network yes; positions imported | Probabilistic search | About 72 hours of training on a single machine **[V]** | About FIDE International Master level (about 2400) **[V]** | Early proof that a learned evaluator can reach master level; far below modern engines. |
| **DeepChess** | 2016 | Pairwise comparison of positions from engine games | Network yes | Alpha-beta with learned comparison | Not recorded here | Reported competitive with strong conventional engines of its era **[M]** | Historical interest only; verify before citing. |
| **Maia (1)** `[S9]` | 2020 | Lichess human games, 12 million per rating level **[V]** | Network yes | None | Not recorded here | 46% to over 52% move-matching, best at the rating it was trained on **[V]** | The original human-behavior baseline. |
| **Maia-2** `[S10]` | 2024 | Lichess human games, rating-conditioned | Yes | None | Not recorded here | 52.0% move-matching on the ALLIE test set **[V]** `[S11]` | Unified skill-aware model. |
| **ALLIE** (Zhang et al., ICLR 2025) | 2025 | Human games as move-token sequences, 355M parameters | Yes | Optional adaptive search | Not recorded here | 55.7% (policy) and 55.9% (with search) **[V]** `[S11]` | Language-model formulation of human play. Cited by `[S11]`; I did not read the primary paper. |
| **Chess-GPT** `[S8]` | 2024 | Lichess PGN text, next-character prediction | Yes | None | Not recorded here | Over 99% legal moves; 64% win rate against Stockfish level 0 for the 16-layer model **[V]** | Direct precedent for a PGN-text language model. |
| **Transcendence** `[S16]` | 2024 | Lichess games from Jan–Oct 2023, truncated by player rating | Yes | None | Not recorded here | Models limited to ≤1000 or ≤1300 data reached about 1500 at low temperature **[V]** | Answers "can imitation exceed its data?" |
| **ChessBench / searchless chess** `[S6]` | 2024 | 10M Lichess games; Stockfish 16 action-values at 50 ms per state-action; 15.3B labels **[V]** | Network yes; labels imported | None at inference | Details not verified in this pass | 2895 Lichess blitz Elo; 95.4% on the puzzle set **[V]** `[S6]` | The strongest evidence that supervision from an engine can replace search. |
| **Chessformer / Maia-3 / Leela-CF** `[S11]` | 2026 | Lichess blitz (Jan 2023 – Jul 2025), and Leela self-play from an April 2024 run **[V]** | Network yes | None for Maia-3; MCTS in full Leela | Maia-3 79M: 8 A100s for about a week **[V]** | 57.1% move-matching (79M); 2374 tournament Elo searchless policy agent; adds over 100 Elo to full Leela **[V]** | The current state of the art for both human prediction and searchless strength. |

### 4.2 Findings that shape the design

**AlphaZero versus supervised learning from PGN are different methods.** AlphaZero trains policy and value networks from search visit counts and game outcomes generated by the network itself. Supervised PGN training predicts what a person chose. The first can exceed any human; the second is bounded by its teacher unless a de-noising effect applies. Do not read AlphaZero's results as evidence about PGN training.

**The 2895 result is engine distillation.** The searchless transformer was trained to predict Stockfish 16's action-values. The authors state that "strong chess performance only arises at sufficient scale" and that a good approximation of Stockfish's search can be distilled into the network but perfect distillation remains out of reach `[S6]` **[V]**. A 9M-parameter model reached 2054 Lichess blitz Elo, a 136M model 2156, and the 270M model 2895 as reported in a secondary summary of the paper's table; These per-size figures come from a secondary summary (liner.com) and were not checked against the paper's own table.

**Oracle strength matters more than network type.** When the Leela team distilled self-play games, networks trained on games from CNNs and from transformers reached "virtually identical" strength, suggesting the oracle's quality dominates `[S11]` **[V]**. Chessformer then exceeded the model that generated its training data (Leela-CF, a 191M searchless model, trained on 100M-parameter-transformer self-play at 600 nodes per move) `[S11]` **[V]**. Its searchless policy agent scored a tournament Elo of 2374 ± 37 versus 2299 ± 36 for the 270M ChessBench model, **on a tournament scale anchored to the ChessBench paper's reported value, not on the Lichess scale** `[S11]` **[V]**.

**Architecture matters at fixed compute.** Chessformer's Geometric Attention Bias, which makes the positional encoding depend on the board, matched an absolute-position baseline with about 60% of the parameters and compute, and added 83 tournament Elo at comparable scale `[S11]` **[V]**. The Lc0 team likewise reports its transformer matches AlphaZero-class strength at 8× fewer FLOPs `[S13]` **[V]**. A generic language-model architecture is therefore not the efficient choice for a board game.

**History helps only a little, and mostly for weak players.** For human-move prediction, input history improved accuracy from 54.0% (no history) to 55.4% (7 past positions) at 5M parameters, with no gain from 7 to 31 past positions, and the gain was concentrated at low ratings `[S11]` **[V]**. For engine-strength play, the current position plus rule-state (castling rights, move clocks, repetition flags) is nearly sufficient.

**Do published chess language models show strength or only sequence prediction?** Both, at different levels. The Chess-GPT results show real play (it beats a calibrated weak engine) with a measurable internal board model. They do not show master strength. The 2895 model is strong because of engine labels. No result I found demonstrates master-level play from PGN-text prediction alone.

**Interpretability evidence for pattern discovery.**
- Probing AlphaZero found human chess concepts encoded in its layers (McGrath et al., PNAS 2022; cited in `[S11]`) **[V]**.
- A later method extracted concepts from AlphaZero that were both *teachable to another network* and *novel relative to human games*; four top grandmasters improved on the resulting puzzles in a preliminary study `[S17]` **[V]**. The authors call it a proof of concept at small scale.
- A Leela transformer showed evidence of learned look-ahead (Jenner et al., NeurIPS 2024; cited in `[S11]`) **[V]**.
- A cross-layer transcoder on Maia-3 found interpretable features for forks, pins, and back-rank danger, plus some features that were not interpretable `[S11]` **[V]**.

**Relevance to this project.** The discoveries all came from models that were already strong (self-play or engine-distilled), not from raw PGN statistics. That shapes the pattern-discovery design in Section 8.

### 4.3 Comparability warnings

- **Elo pools.** ChessBench's 2895 is a Lichess blitz rating earned against humans and bots; the same paper's tournament Elo is a separate scale from agents playing each other and "cannot be directly compared" `[S6]` **[V]**. Chessformer re-anchors its tournament Elo to the ChessBench paper's value `[S11]` **[V]**.
- **Time controls and opponents.** Beating Stockfish "level 0" by a margin (Chess-GPT) and holding a Lichess blitz rating are different measurements. The authors describe Stockfish level 0 as roughly 1300 Elo `[S8]` **[V]**, while a community post about the same model claims about 1500 Elo for it; these are different calibrations and the discrepancy is unresolved **[V]** (the post is not a primary source).
- **Metrics.** Move-matching accuracy (Maia) and puzzle accuracy (ChessBench) can rise while playing strength does not. Section 11 covers how to test them separately.

---

## 5. Dataset availability, what PGN teaches, and licensing

### 5.1 Sources

| Source | Scale and unit | Metadata | Access | License / terms | Notes |
|---|---|---|---|---|---|
| **Lichess open database (standard rated games)** `[S1]` `[S2]` | "Over 7 billion rated standard games" on the site banner (snapshot around Aug–Sep 2025) **[V]**. Roughly 100 million games per month in recent years **[V]** `[S5]`. At the snapshot rate, around 8B or more by Oct 2026 **[E]**, not verified. | Ratings, result, time control, ECO and opening name, termination, clocks (since April 2017), and a minority of games with engine analysis (about 6%, per the dataset card quoting Lichess) **[V]** `[S3]` | Monthly `.pgn.zst` files with torrents and SHA-256 checksums **[V]** `[S1]` | **CC0** ("research, commercial purpose, publication, anything you like") **[V]** `[S1]` | Dominated by bullet and blitz. Rating distribution skews to the middle and low end. Bots and cheaters are present; filter them. |
| **Lichess evaluations** `[S1]` | 416,442,401 positions evaluated with Stockfish (file updated 2026-10-05) **[V]** | FEN, depth, nodes, principal variations, centipawn or mate score | One `.jsonl.zst` file | CC0 **[V]** | Free engine labels, crowd-sourced from browser analysis. Variable depth; pick the deepest entry per position. |
| **Lichess puzzles** `[S1]` | 6,157,341 rated, tagged puzzles (updated 2026-10-05); generated from 600 million analyzed games **[V]** | FEN, solution, rating, themes (fork, pin, discoveredAttack, mate-in-N) | CSV | CC0 **[V]** | Ready-made tactical-motif benchmark. |
| **Lichess variants** `[S1]` | Antichess about 37.9M, Atomic about 29.5M, Chess960 about 29.8M, Crazyhouse about 30.7M games **[V]** | Same fields | Same format | CC0 **[V]** | Must be **excluded** from standard-chess training. Chess960 has different castling and starting positions. |
| **Hugging Face mirror (Lichess)** `[S3]` | One monthly config shows 87,649,571 games, 183,971,368,905 bytes, 62,651,992,665 byte download **[V]** | Parquet columns for headers plus movetext | HF `datasets` | CC0 | Marked "work in progress, breaking changes may occur" by its maintainers. Use the official files for reproducibility. |
| **Lichess Elite Database** (community project) | Filtered standard games, 2400+ rated vs 2200+ rated, no bullet **[V]** `[S4]` (forum mention) | Same | Community-hosted | Verify | A convenient high-quality subset. I did not verify its current size or hosting. |
| **ChessBench** `[S6]` `[S7]` | 10M Lichess games, 15.3B Stockfish-16 action-value labels **[V]** | Per state-action values | DeepMind repository | Repository says portions are CC0 from Lichess **[V]** `[S7]`; check the dataset's own terms | Reusing it saves roughly 212,000 CPU-hours of labeling **[C]**: 15.3e9 labels × 0.05 s = 7.65e8 core-seconds. |
| **Leela Chess Zero training data** | Volunteer self-play, over 2.5 billion games by June 2024 **[V]** `[S23]` | Policy visit distributions, outcomes | Project servers | Stockfish's README says nets are "trained on data provided by the Leela Chess Zero project, which is made available under the Open Database License (ODbL)" **[V]** `[S20]` | ODbL has share-alike and attribution terms; read them before redistributing derived data. |
| **Chess.com** | Public data via the Published-Data API: player profiles, monthly game archives in PGN, leaderboards **[V]** `[S21]` | Ratings, results, opening, time class | Per-player monthly archives; no bulk dump found | Serial requests are not rate-limited; parallel requests are **[V]** `[S21]`. I did **not** find a statement that permits redistribution or ML training | Read Chess.com's current Terms of Service and API terms before any use beyond personal experiments. |
| **TWIC (The Week in Chess)** | Weekly PGN bundles of master games | Titles, events | Free download | Terms not verified **[M]** | Small (millions at most), high quality, over-the-board. Check redistribution terms. |
| **FICS and other historical servers** | Historical online games | Ratings | Various | Terms not verified **[M]** | Older, smaller, mixed quality. |
| **CCRL / engine-vs-engine collections** | Engine games | Engine names | Public | Terms not verified **[M]** | Useful for engine-play style, not for human behavior. |

**"Billions of games" versus a usable dataset.** The Lichess figure is real and downloadable. A storage and processing plan for it is in Section 9. But "available" does not mean "useful": most of it is bullet and blitz at middling ratings, and one billion games at about 75 plies each is 75 billion positions of which a large share are repeated opening positions or near-duplicates. Platforms advertising game counts that exist only behind a login, or in unpublished archives, are not datasets you can process.

**Are billions necessary?** The evidence does not support it for human-move prediction:
- Maia reached 46–52% with 12 million games per rating bin **[V]** `[S9]`.
- Maia-3's 5M-parameter model reached 55.4% and the 79M model 57.1%, a gain of 1.7 points from 16× the parameters `[S11]` **[V]**. The authors expect a ceiling around 50% at low ratings because play there is inconsistent `[S11]` **[V]**.
- ChessBench's strength came from 10 million games with strong labels, not from billions of games with weak ones `[S6]` **[V]**.

For engine-quality targets, the limiting resource is label quality and compute, not the count of human games. A rating-balanced 10M-game subset with engine labels is likely a better use of compute than 1B raw games **[E]**. The experiment that settles it for your setting is a data-scaling curve (Section 12, Phase 7).

### 5.2 What does a PGN actually contain, and what does it teach?

| PGN field | Useful for | Bias or risk |
|---|---|---|
| **Move sequence** | Next-move targets; opening repertoires; reconstructing every position | Contains human errors; early moves are heavily duplicated |
| **Result** (`1-0`, `0-1`, `1/2-1/2`) | Weak value label | Includes resignations, timeouts, and abandonments; correlated with rating gap and clock, not only move quality |
| **WhiteElo / BlackElo** | Rating-conditioned models; stratified sampling | Platform-specific, not FIDE; inflated or deflated by pool; provisional ratings |
| **TimeControl, Event** | Style by time control | Bullet play is premove-heavy and error-rich; models trained on bullet learn "bullet chess" |
| **Clock annotations** (`%clk`) | Time-pressure filtering (Maia-3 removes moves made with under 30 seconds left `[S11]` **[V]**) | Leaks no chess information but confounds quality |
| **ECO / Opening** | Splits by opening family; opening-aware evaluation | Derived label, safe as metadata, risky as an input feature (leakage of the answer's cluster) |
| **Termination** | Filtering abnormal endings | "Time forfeit" in a winning position teaches nothing about the position |
| **Engine analysis tags** (about 6% of Lichess games) **[V]** `[S3]` | Free labels | Selection bias: analyzed games are not a random sample |
| **Player names / IDs** | Player-level splits; stylometry | Privacy risk; individual players are identifiable from a few games `[S11]` **[V]** |

**What a model can learn from each, in plain terms:**

1. **Opening repertoires and move order.** Easily learned by memorization; the main reason top-k accuracy is high in the first 10 moves. Maia-3 evaluation drops the first 10 moves from the test set because they "can be easily memorized" `[S11]` **[V]**.
2. **Tactical motifs.** Learned implicitly when the target is an engine-strength move; probing a Maia-3 transcoder found fork and pin features `[S11]` **[V]**. Under human-move targets, tactics are learned only as well as the players executed them.
3. **Positional concepts (pawn structure, king safety, space).** Learned as statistical regularities; weakly validated by game results.
4. **Endgames.** Sparse in blitz data; engine labels or tablebases are the reliable source.
5. **Rating-dependent behavior.** Learned when rating is an input; Chess-GPT learned to estimate player skill from the move sequence without being told `[S8]` **[V]**.
6. **Common mistakes.** Learnable and valuable for coaching: a model conditioned on rating predicts blunders better than a context-free engine `[S9]` **[V]**.

**Correlation versus a reliable chess principle: how a model can be fooled.** A move can look "good" because it appears in winning games for reasons that have nothing to do with its quality:

- **Position confounding.** Players who are already winning tend to play simple, forcing, or "winning technique" moves. The move co-occurs with a win because the *position* was winning.
- **Rating confounding.** Stronger players choose some moves more often, and stronger players win more often against weaker ones. Pooling ratings makes those moves look good in absolute terms.
- **Opponent error.** A speculative sacrifice that wins because the opponent failed to refute it looks sound in a win-rate table and is unsound against best defense.
- **Survivorship and selection.** Games that end by resignation or time loss censor the continuation; the dataset shows what happened, not what would have been best.
- **Clock effects.** In time trouble, simple moves are played because there is no time to calculate, and games are decided by flagging.
- **Opening theory.** Moves deep in memorized lines are good because of prior analysis, and their win rate says little about a model's understanding.

**Mitigation:** condition on rating and position difficulty, use engine evaluation as an independent judge, and never treat a win-rate difference as evidence without a controlled comparison (Section 8.3).

### 5.3 Legal and data-governance considerations

This section is not legal advice.

| Item | What I verified | What to check |
|---|---|---|
| **Lichess game data** | CC0, usable for research, commercial purposes, publication, redistribution `[S1]` **[V]** | Nothing further for the PGNs themselves. Usernames remain in headers; see privacy below. |
| **Chess.com data** | The API documents public, read-only access; serial requests are unlimited `[S21]` **[V]** | **Terms of service and data-use terms for ML training and redistribution were not verified.** Treat as unverified until you read them. |
| **Leela training data** | Described by the Stockfish project as ODbL `[S20]` **[V]** | ODbL share-alike and attribution obligations apply to derived databases. |
| **Stockfish software** | GPLv3; if you distribute it you must include the license and source `[S20]` **[V]** | Binary redistribution obligations. Using Stockfish as an external opponent over UCI is common practice; consult the license if you bundle it. |
| **Leela Chess Zero engine and network weights** | Not verified in this pass **[M]** | Engine license and network-weight license are separate; read both. |
| **python-chess, shakmaty, other GPL libraries** | Not verified in this pass **[M]** | Linking GPL libraries into distributed software triggers copyleft; the license of *model weights trained with them* is a separate question without settled law. |
| **TWIC and commercial databases** | Not verified **[M]** | Individual game moves are generally treated as facts, but compilations, annotations, and redistribution terms can be restricted. Check each source. |

**Privacy and memorization.** Individual players can be identified from a few of their games using behavioral stylometry (McIlroy-Young et al., NeurIPS 2021, cited in `[S11]`) **[V]**. A model trained on PGNs with real usernames can memorize or reproduce player-specific behavior. Replace usernames with salted hashes before building training shards, and do not publish player-level derived datasets without review.

**Dataset license is not software license.** CC0 on the data does not set the terms for the tools you use to process it, or for any engine labels you generate. Keep a per-component license table in your repository.

---

## 6. Architecture comparison

### 6.1 Move-sequence Transformer (language-model style)

**Setup.** A decoder-only Transformer predicts the next move token given previous moves (teacher forcing, cross-entropy).

| Design choice | Options and notes |
|---|---|
| **Tokenization** | (a) **Character level** over PGN text, as in Chess-GPT `[S8]` **[V]**; no vocabulary design, but about 600 characters per game **[E]**. (b) **UCI move tokens**: ALLIE uses 1,968 tokens for the UCI moves `[S11]` **[V]**. About 75 tokens per game **[E]**. (c) SAN: shorter strings but context-dependent (disambiguation, check symbols), so the same token can denote different moves. UCI is safer for training. |
| **Special moves** | Castling as king-two-squares, en passant as a diagonal pawn move, promotion as an extra piece suffix. Standard in UCI and in the Chessformer implementation `[S11]` **[V]**. |
| **Context window** | Games are short by language-model standards. A full game fits in 256 move tokens. Character-level needs about 1,024. |
| **Position history vs full history** | Full history lets the model estimate player skill and style. Chess-GPT does this implicitly `[S8]` **[V]**. For engine-strength targets, history adds little `[S11]` **[V]**. |
| **Board reconstruction** | The board is a deterministic function of the move history, but the network must compute it internally at every layer-position. This is wasteful, and `[S11]` cites evidence that move-token models "simply reconstruct the board state at each token" instead of specializing **[V]**. |
| **Legality** | Mask to legal moves from a rules engine at inference. Count illegal-sampling rate as a diagnostic, not as a solved problem. |
| **Value** | Not natural in next-move prediction. A separate head can predict the result, but it is a noisy target. |
| **Generalization** | Strong when positions resemble the training distribution; weak on unusual positions, because the history prefix is out of distribution **[E]**. |

**Verdict.** Good for a quick baseline and for studying emergent world models. Not the efficient route to strength.

### 6.2 Board-state policy and value networks

**Inputs.** Explicit position representation:
- 12 piece-type-by-color planes over 64 squares, flipped to the side-to-move's perspective (the Chessformer layout `[S11]` **[V]**).
- Rule state: castling rights (4 flags), en passant square, side to move, halfmove clock (Chessformer divides by 100 `[S11]` **[V]**), repetition flags for the current and past positions.
- Optional history of the last *n* positions. For human-move prediction, n = 7 gave the main gain `[S11]` **[V]**.
- Legal move mask on the policy output.

**Outputs.** A policy over moves (source-destination square pairs or a fixed move vocabulary) and a value, ideally win/draw/loss probabilities rather than a scalar. Leela outputs win/draw/loss probabilities `[S26]` **[V]**.

**Why the value head needs more than the board.** Draw probability depends on the halfmove clock, repetition history, and whether a draw is nearly forced by the 50-move rule or three-fold repetition. Two positions with identical piece placement can have different true values if one is one move from a 50-move draw. Without those inputs the value head is a biased estimator in drawn endgames.

**Architectures compared:**

| Architecture | Evidence |
|---|---|
| **Residual CNN** | AlphaZero-style `[S12]`; Maia (1) used it `[S9]` **[V]**. Reliable, easy to train. Leela's 195M CNN scored 2096 (policy) and 2168 (value-maximizing) tournament Elo in the Chessformer comparison `[S11]` **[V]**. |
| **Transformer, square tokens (64 tokens)** | Chessformer's 191M model scored 2374 (policy) and 2466 (value) on the same scale, with fewer FLOPs than the competing models `[S11]` **[V]**. |
| **Transformer, linearized board plus FEN-like tokens (77 tokens)** | Used by the ChessBench models; position encoding is 1-D, which Chessformer argues is misaligned with board geometry `[S11]` **[V]**. |
| **Graph neural networks / relational nets** | **No head-to-head result found in this pass.** Square-token attention with a board-aware position bias captures the same relations without a separate graph library; treat GNNs as an experiment, not a default **[E]**. |

**Verdict.** The best starting point is a small square-token model (about 5–25M parameters) with a rule-state input vector, a policy head, a WDL value head, and optional auxiliary targets (material balance, check flag). The Chessformer paper reports that its smallest sensible variant (about 3–5M parameters) already reaches 54.8–55.4% move-matching `[S11]` **[V]**.

### 6.3 Hybrids and search

| Combination | Practicality | Expected return for a small team |
|---|---|---|
| PGN pretraining + engine-label fine-tuning | Practical. Labels can come from Stockfish at about 50–100 ms per position or from the Lichess evaluation file (416M positions, free) `[S1]` **[V]** | **High**: this is the ChessBench route at small scale |
| Policy/value network + MCTS (Leela-style inference) | Practical for inference; moderate engineering (batching, tree reuse) | High for strength; changes the evaluation question |
| Policy/value network + alpha-beta | Research-intensive; NNUE-style evaluators need speed-optimized architectures | Low for a first project |
| Self-play RL from random weights | Practical only at small scale for a small team; costs in Section 10 | Medium: tests whether the pipeline is correct, unlikely to be strong |
| Opening book and endgame tablebases | Easy to add at inference time; tablebases are large downloads **[M]** | High for playing-strength testing, but contaminate "from scratch" claims |

### 6.4 Comparative table

| Approach | Training objective | Data needed | Compute (relative) | Strengths | Limits | Best use |
|---|---|---|---|---|---|---|
| Move-sequence Transformer | Next-move cross-entropy | Games only | Low–medium | Simple; probes emergent world models | Wasteful board reconstruction; no value; unusual positions weak | Baseline; interpretability studies |
| Board-state policy/value CNN or Transformer, human labels | Cross-entropy on human move + result | Games only | Low–medium | Efficient; human-like at a given rating | Capped by human move quality | Human-behavior model, coaching tools |
| Board-state network, engine labels | Regression/classification on engine value or action-value | Games for positions + labeling cost | Medium + labeling | Reaches strong searchless play | Labels cost CPU time; "not from scratch" | Strong searchless engine |
| Policy/value + MCTS (AlphaZero-style) | Visit-count policy, outcome value | None (self-play) | High–very high | Superhuman at scale | Generation cost dominates | Research-scale engine |
| Hybrid: pretrain on PGN, then self-play | Both | Games + self-play | High | May reach useful strength sooner | Little chess-specific evidence | Phase-6 experiment |

---

## 7. Supervised learning, reinforcement learning, and self-play

### 7.1 Supervised learning from historical games

**Construct examples.** For each game and each ply: input = position (plus history if used), targets = played move (policy), game outcome (value), optional rating-conditioning embeddings, optional engine value. Maia-3 samples 32 positions per game at random and rebalances the dataset so each rating bin is equally represented `[S11]` **[V]**.

**Can supervised learning exceed the average player?** Two verified mechanisms:
- **Low-temperature de-noising.** Models trained on players up to 1000 or 1300 reached about 1500 by sampling the most likely move, which behaves like a majority vote over many mistake-prone experts. The authors state the effect is denoising of errors, not novel reasoning `[S16]` **[V]**. A model trained on players up to 1500 did not exceed its data `[S16]` **[V]**.
- **Engine labels.** Replacing human move targets with engine action-values lifts the ceiling to the engine's strength minus a distillation gap `[S6]` **[V]**.

**What is not shown:** a PGN-only model trained on strong players surpassing them. Plan on a model near the *consistent* strength of its data, plus a modest uplift at low temperature.

### 7.2 Reinforcement learning

Chess is a deterministic, two-player, zero-sum, perfect-information game with sparse terminal rewards (win +1, draw 0, loss −1). Challenges: the reward arrives after about 80–100 plies; random initial play rarely produces decisive games that teach anything; and shaped rewards (material) can teach the wrong objective.

**Practical conclusions:**
- Pure model-free policy gradient from random weights is not the route that worked. The route that worked combines search with a learned prior (AlphaZero) `[S12]` **[V]**.
- Giraffe used TD-Leaf with positions imported from computer games and reached about 2400 in 72 hours on a single machine `[S15]` **[V]**, which shows a modest, cheap route to master level exists but is far below modern engines.
- Historical games can bootstrap the policy so that early self-play is not random. I found no controlled study isolating this benefit at scale **[V]** (absence of evidence in this pass).

### 7.3 Self-play systems at implementation level

**Loop.**
1. The current network provides a prior policy *p(a|s)* and a value *v(s)*.
2. MCTS runs *N* simulations per move, selecting actions by a value-plus-prior-driven rule, and returns visit counts *π*.
3. A move is sampled from *π* (with temperature early in the game for exploration). The game continues to termination.
4. Training targets: policy ← visit distribution *π*; value ← final game outcome (and, in Leela, a moves-left head, visible as a separate loss term in open training code `[S27]` **[V]**).
5. The network is updated from a replay buffer; the new network generates later games.

AlphaZero removed the gating step of earlier versions: self-play games are generated by the latest parameters, with no evaluation tournament before replacing the generator `[S12]` **[V]**.

**Published principles versus project-specific details.** The loop above is described in the AlphaZero paper `[S12]`. Leela adds distributed volunteer generation, WDL value outputs, and many engineering enhancements; the transformer design (`[S13]` `[S14]`) is Leela-specific. Treat these as implementation choices, not requirements.

**Cost structure.** Generation dominates. The Lc0 team states that the most expensive part of the process "by far" is generating training games because it needs hundreds of model evaluations per position `[S11]` **[V]**. A secondary summary estimated AlphaZero's cost to beat Stockfish at about 20,000 TPU-hours `[S24]` **[V]** (secondary source; the number is the summary author's arithmetic).

**Collapse and instability.** Mitigations: keep a replay buffer with recent games, use temperature and Dirichlet noise at the root for exploration, validate by playing fixed-baseline matches regularly (not only against the previous version), and keep the network's loss and the match results both under watch.

### 7.4 Six training paths compared

| Path | Expected performance | Data | Compute | Engineering complexity | Overfitting risk | My assessment |
|---|---|---|---|---|---|---|
| 1. PGN pretrain → SFT on stronger games | Human-like; strong-club to 2000-class **[E]** | Games | Low | Low | Opening memorization | Good Phase-2 baseline |
| 2. PGN pretrain → self-play RL | Unknown; plausibly faster early progress than random init **[E]** | Games + self-play | High | High | Distribution shift between human and self-play positions | Phase-6 experiment |
| 3. PGN + Stockfish labels | Strong searchless play; 2895 Lichess blitz at 270M with 15.3B labels `[S6]` **[V]** | Games + labels | Medium + labeling | Medium | Low if test positions are held out by game | **Best expected return** for the engine objective |
| 4. PGN pretrain → search-based policy improvement | Strong; closest to Leela's inference-time use | Games + search | Medium–high | Medium–high | Low | Natural next step after path 3 |
| 5. Self-play from scratch, no games | Superhuman at AlphaZero/Leela scale `[S12]` `[S13]` **[V]** | None | Very high | High | Low (data is unlimited) | Only if resources match |
| 6. Games → engine distillation → self-play | Combines paths 3 and 5 | Games + labels + self-play | High | High | Low | Large-project path |

**Cheapest informative comparison for a small team:** path 1 versus path 3 on the same 1M games, then path 4 on the winner.

---

## 8. Can the model discover new chess patterns?

### 8.1 Be precise about what "discovery" means

Three levels, in increasing order of difficulty and value:

1. **Interesting correlation.** A feature or move pattern that co-occurs with winning, or with errors, in the data.
2. **Useful predictive feature.** The feature improves out-of-sample prediction of moves, errors, or outcomes after controlling for rating, opening, time control, and position evaluation.
3. **Genuinely stronger strategy.** Playing by the pattern gains strength in controlled play against a fixed opponent, or teaches human experts something they did not know.

Level 3 has been demonstrated in chess only once, to my knowledge, and not from PGN statistics: concepts extracted from AlphaZero's internal representations that were teachable and not present in human games helped four grandmasters on puzzles in a preliminary study `[S17]` **[V]**. The same paper calls its result a proof of concept at small scale.

### 8.2 Methods, and what each can and cannot establish

| Method | Can establish | Cannot establish |
|---|---|---|
| **Clustering position embeddings** (k-means, HDBSCAN on a trained network's hidden states) | That a recurring structural family exists in the data | That the network "understands" why the family matters; embeddings may cluster on opening, rating, or pawn skeleton |
| **Nearest-neighbor retrieval** | Analogous positions across games; good for human inspection | Causal or strategic equivalence |
| **Frequent-sequence and association-rule mining** (on SAN or on abstracted features) | Which move sequences or feature sets co-occur | Quality of the pattern; rules mined without controls reproduce opening theory and rating effects |
| **Decision trees / interpretable surrogates** | A compact, human-readable approximation of a model's behavior | That the model uses those rules; surrogates fit behavior, not mechanism |
| **Feature attribution and sensitivity** (gradients, occlusion by removing a piece) | Which squares and pieces influenced one prediction | Global rules; attribution methods can disagree with each other |
| **Linear probes and activation analysis** | That a concept is linearly decodable from a layer; intervention tests can show causal use (Chess-GPT board edits `[S8]` **[V]**) | That the model *relies* on it unless intervention confirms |
| **Sparse autoencoders and transcoders** | A dictionary of interpretable features; Maia-3 features for forks, pins, and back-rank danger `[S11]` **[V]** | That unlabeled features are meaningful; some features were not interpretable `[S11]` **[V]** |
| **Contrastive win/loss analysis** | A difference between groups | Causation; every confounder in Section 5.2 applies |
| **Conditional statistics of move choice given board features** | Rating- and context-specific tendencies when properly stratified | Optimality; humans choose frequently what is not best |
| **Causal inference** (matching or inverse-propensity weighting on rating, opening, clock) | A defensible effect estimate under stated assumptions | Anything if unmeasured confounders such as player intention matter |
| **Comparing human move distributions to engine-optimal moves** | Where humans systematically deviate from the engine, by rating | Whether the deviation is a *mistake* in practical terms (it may be a safe, simple choice) |

**Example.** A cluster of "opposite-side castling with pawn storms" positions may show a higher win rate for the attacker. That finding could reflect the attackers being stronger, an earlier tactical oversight, or a time-trouble effect. It becomes a chess claim only after it survives the validation procedure below.

### 8.3 A validation procedure for a candidate pattern

1. **Extract the candidate** from a *discovery* split only (for example, games from months 1–6).
2. **State a measurable hypothesis** before looking at the validation data. Template: "In positions satisfying predicate *P*, the move class *M* has centipawn loss at least *δ* lower than alternatives for players rated *r1–r2*." Write the predicate as code.
3. **Hold out** by time and by player. A random split of positions leaks, because positions from one game are highly correlated `[S6]` **[V]** (ChessBench's statement that nearly every new game contains many unseen board states supports splitting by game, not position).
4. **Control for confounders:** rating of both players, color, opening family, time control, move number, and an engine-evaluated measure of position difficulty. Use stratification, matching, or regression with those covariates.
5. **Compare against established heuristics.** The pattern must add predictive information beyond material balance, king safety, mobility, and pawn-structure features.
6. **Check with a strong engine.** Evaluate representative positions at a fixed node or depth budget, and record the engine's evaluation of the pattern move versus alternatives. A pattern that humans like and the engine rejects is a *human-behavior* finding, not a chess-strength finding.
7. **Run controlled experiments where possible.** For a claimed stronger plan: build two versions of a fixed engine or policy network, one constrained or biased toward the pattern, and play a large match with balanced openings (Section 11).
8. **Measure the effect on a decision metric** such as centipawn loss, blunder rate, or win probability from an engine, not only on game outcomes.
9. **Correct for multiple comparisons.** If you test *m* independent hypotheses at α = 0.05, you expect 0.05 × *m* false positives; with 10,000 candidate features that is about 500 **[C]**. Use a false discovery rate procedure (Benjamini–Hochberg) and report the adjusted threshold, and keep a log of every hypothesis tested, not only the successes.
10. **Document counterexamples.** Publish positions where the pattern fails; a pattern with no counterexamples is almost certainly under-tested.

### 8.4 Human behavior versus objective strength

These are separate targets. Train and evaluate them separately.

| Target | Defined as | Training signal | Evaluation |
|---|---|---|---|
| **Moves humans play** | *P(move | position, rating)* | Human move labels | Move-matching accuracy, cross-entropy, calibration `[S9]` `[S11]` **[V]** |
| **Objectively strong moves** | Engine-preferred action | Engine value or action-value | Puzzle accuracy; match play vs fixed engines `[S6]` **[V]** |
| **Moves that are hard to defend against** | Moves whose refutation has low probability of being found by a human of rating *r* | Human-behavior model × engine evaluation | Hold-out games: realized win rate versus predicted "trappiness" |
| **Maximizes practical chances at rating *r*** | argmax over moves of (engine value adjusted by *r*-dependent error probability) | Combination of the two models | Match play against rating-matched human-behavior models |
| **Time-control-specific play** | Same as above, conditioned on time control and clock | Clock-aware features | Evaluation by time-control slice |

Combined architecture: a shared trunk with a human-policy head (rating-conditioned), an engine-value head, and optionally an engine-policy head, evaluated separately. Maia-3 shows that one square-token architecture can serve human emulation and engine strength, but through separately trained models `[S11]` **[V]**, not one set of weights.

### 8.5 The most promising path to useful patterns

1. Train or reuse a rating-conditioned human-behavior model (a Maia-style network; code and weights exist for Maia-2 and Maia-3 `[S10]` `[S11]` **[V]**).
2. Obtain engine evaluations for the same positions (Lichess's evaluation file for a free subset `[S1]` **[V]**, or your own Stockfish run).
3. Compute, per position, the human model's probability of each legal move and the engine's centipawn loss for each. Define:
   - **Expected human loss** = Σ over legal moves of *P_human(m | s, r)* × *CPL(m)*.
   - **Surprise** of an actual move = −log *P_human(m | s, r)*.
4. Mine two candidate sets:
   - **Systematic human error regions:** positions or feature predicates where expected human loss is high at rating *r* but low at rating *r + 400*.
   - **High-surprise, high-value moves:** actual moves with high surprise and a large *positive* engine-value swing, played by strong players.
5. Validate each set with the Section 8.3 procedure.
6. For teachability, adapt the puzzle-transfer design of `[S17]`: present positions with the discovered idea to rating-matched players and measure improvement against a control group. This step needs human participants and is outside Phase 1–5 scope.

---

## 9. Data engineering at scale

### 9.1 Pipeline

| # | Stage | Practical approach | Failure to watch for |
|---|---|---|---|
| 1 | Acquisition | Download official monthly files with SHA-256 verification and torrents `[S1]` **[V]** | Partial downloads; mixed variants |
| 2 | Raw archive | Keep `.pgn.zst` unmodified in object storage; record file hashes and download date | Silent file replacement; use immutable paths |
| 3 | PGN parsing | Stream-decompress and parse in a compiled language (Rust `pgn-reader` is a common choice; the Lichess forum recommends it for compressed databases `[S4]` **[V]**) | Python parsing becomes the bottleneck at 100M+ games |
| 4 | Malformed/truncated games | Quarantine and count; never silently drop | Truncated final game of a file |
| 5 | Variant filter | Keep only standard chess (no `Variant` header other than Standard; exclude Chess960 starting FENs) | Variant games leaking into standard training |
| 6 | Duplicate detection | Hash movetext and key header fields | See 9.3 |
| 7 | Legality verification | Replay every move with a rules library; discard games with illegal moves | Library differences on rare rules (e.g., repetition claims) |
| 8 | Metadata normalization | Standard time-control buckets, rating bins, result codes; hash usernames | Provisional ratings; missing Elo |
| 9 | Position extraction | Emit positions or defer to load time (see 9.2) | Storage explosion if all positions are materialized |
| 10 | Partitioning | Split by game ID hash and by month (Section 9.3) | Leakage |
| 11 | Feature encoding | Fixed encoder with version number; unit-tested on known FENs | Silent encoder changes between runs |
| 12 | Serialization | Parquet/Arrow for metadata and moves; binary arrays for training | Format lock-in |
| 13 | Sharding | Fixed-size shards (for example, 256–512 MB) with shuffle at shard and buffer level | Poor shuffling creates temporal correlation |
| 14 | GPU-efficient loading | Multi-process loader, pinned memory, pre-encoded batches | Loader slower than the GPU |
| 15 | Versioning | Content-hash the shard manifests; log config and git commit with every run | Irreproducible results |

### 9.2 Storage formats

| Format | Pros | Cons | Use when |
|---|---|---|---|
| **Raw PGN** | Universal; keeps clocks and annotations; official source format | Text parsing is slow; verbose (about 2 KB per game uncompressed, **[C]** from 130 GB / 68M games `[S4]`) | Archival only |
| **Compressed PGN (`.pgn.zst`)** | About 250 bytes per game in the Sept 2020 file (16.8 GB / 68M games, **[C]** from `[S4]`) | Not random-access | Raw archive |
| **Parquet** | Columnar, typed, analytic queries (DuckDB, Arrow); the HF mirror used about 715 bytes per game including headers (62.65 GB / 87.65M, **[C]** from `[S3]`) | Not designed for random access to individual plies | Metadata and curated game sets |
| **Arrow / memory-mapped arrays** | Zero-copy reads; fast | Larger on disk | Hot training shards |
| **Binary move arrays** (16-bit move index per ply, plus a header) | About 160 bytes per game **[C]**: 75 plies × 2 bytes plus about 10 bytes of header | Boards must be reconstructed at load time | Best compromise for game-level storage |
| **Binary position records** (packed board plus rule state plus targets, about 40 bytes) | Fastest training loading; no replay needed | 40 B × 75 plies ≈ 3 KB per game, **about 19× the move-array form** **[C]** | When the loader is the bottleneck |
| **WebDataset / tar shards** | Sequential streaming at scale; works across object stores | Weak random access | Multi-node training |

**When to keep PGN versus structured examples.** Always keep the original PGNs (immutable, versioned, hashed). Derive structured shards from them reproducibly, and *do not* delete the PGNs when you build shards. Materialize position-level records only when profiling shows board reconstruction is the data-loading bottleneck. The 3 GB-per-million-games cost of position records is irrelevant at 1M games and significant at 1B (3 TB **[C]**).

### 9.3 Deduplication and leakage prevention

| Concern | What can go wrong | Approach |
|---|---|---|
| **Exact duplicate games** | Same game appears in two archives | Hash of the full move list (64-bit is enough to start; 128-bit if you expect billions) |
| **Same moves, different headers** | Rematches or re-uploads | Dedupe on move hash after header normalization |
| **Transpositions and repeated positions** | Same position reached by different move orders; very common in the opening | Do **not** dedupe positions for training (it distorts frequencies), but **do** ensure no test position (by Zobrist hash) appears in training when you want a "novel position" test |
| **Same player, tournament, or opening family** | Memorization of individual style | Hold out whole players or events for one split |
| **Temporal leakage** | Overlapping archives or later games reusing earlier lines | Split by month; keep validation and test strictly after training |
| **Derived-label leakage** | Engine annotations in the PGN, or evaluation labels computed on test positions | Compute labels per split and never reuse a test position's label for training |

**Why random position-level splits mislead.** Neighboring positions in a game share nearly all pieces. If positions are split at random, the validation set contains positions one ply away from training positions, and accuracy is inflated. The same applies to engine evaluations of near-identical positions. Split by game at minimum.

**Recommended splits:**

| Split | When | Trade-off |
|---|---|---|
| **Game-level hash split** | Default for iteration | Cheap; still leaks players and openings |
| **Player-level** | Test generalization across individuals | Reduces training data; strong players are rare |
| **Tournament/event-level** | Over-the-board data | Small data |
| **Opening-family** | Test novelty handling | Biases hold-out toward unusual lines |
| **Temporal** | The final test | Honest deployment simulation; distribution shift is real |

**Memory for dedup at scale.** A hash set of *n* 64-bit hashes takes 8*n* bytes plus table overhead; 1 billion games is 8 GB for raw hashes and about 16–32 GB with overhead **[C]**. Use an external sort-merge on disk if RAM is limited.

### 9.4 Scale calculations

**Assumptions** (state and revise after Phase 1 measurements):

| Quantity | Value used | Source / basis |
|---|---|---|
| Average game length | **75 plies** (sensitivity: 60–90) | **[E]** Consistent with ChessBench's ratio of 15.3B action-values to about 750M positions at an average of about 20 legal moves per position **[C]**: 10M games × 75 = 750M positions; 15.3B / 750M = 20.4 |
| Compressed PGN size | **250–350 bytes per game** | **[C]** 16.8 GB / 68M games = 247 B (Sept 2020) `[S4]`; newer files carry more clock data, so the range goes up |
| Uncompressed PGN | **about 2 KB per game** | **[C]** 130 GB / 68M = 1.9 KB `[S4]`; 184 GB / 87.65M = 2.1 KB `[S3]` |
| Positions per game | **75** (one per ply) | **[E]** |
| Move tokens per game | **75** (move-level) or **about 600** (character-level) | **[E]** |
| Move-array storage | **about 160 bytes per game** | **[C]** |
| Position-record storage | **about 40 bytes per position** | **[E]** (packed 8×8 board, rule state, move, result) |
| Parsing throughput | **Python: about 200 games/s/core; Rust: about 15,000 games/s/core** | **[E]**, order-of-magnitude guesses to be **measured** in Phase 1 |
| Network | 100 Mbit/s (12.5 MB/s) | Reader's connection; scale linearly |

**Storage, positions, and time:**

| Scale | Compressed PGN | Uncompressed PGN | Positions | Move tokens | Char tokens | Move-array store | Position-record store | Download at 100 Mbit/s |
|---|---|---|---|---|---|---|---|---|
| **100k games** | 25–35 MB | 0.2 GB | 7.5M | 7.5M | 60M | 16 MB | 0.3 GB | about 3 s |
| **1M** | 0.25–0.35 GB | 2 GB | 75M | 75M | 0.6B | 0.16 GB | 3 GB | 20–28 s |
| **10M** | 2.5–3.5 GB | 20 GB | 750M | 750M | 6B | 1.6 GB | 30 GB | 3.3–4.7 min |
| **100M** | 25–35 GB | 200 GB | 7.5B | 7.5B | 60B | 16 GB | 300 GB | 33–47 min |
| **1B** | 250–350 GB | 2 TB | 75B | 75B | 600B | 160 GB | 3 TB | 5.6–7.8 h |

*Worked check:* 300 GB ÷ 12.5 MB/s = 24,000 s = 6.7 h **[C]**.

**Parse and validate CPU time:**

| Scale | Python at 200 games/s/core | Rust at 15,000 games/s/core |
|---|---|---|
| 100k | 500 s (8 min) | 7 s |
| 1M | 5,000 s (1.4 core-hours) | 67 s |
| 10M | 14 core-hours | 11 min |
| 100M | 139 core-hours | 1.9 core-hours |
| 1B | 1,389 core-hours | 18.5 core-hours (about 1.2 h on 16 cores) |

**Sensitivity.** At 1B games, changing average length from 60 to 90 plies moves positions from 60B to 90B (±20%); changing bytes per game from 250 to 350 moves compressed storage from 250 GB to 350 GB (±17%). Parsing throughput uncertainty is the largest factor: an order-of-magnitude error in the Rust estimate changes 1B-game parsing from 1.9 to 185 core-hours **[C]**. Measure it.

**Memory.** Streaming parsing needs megabytes. A shuffle buffer of 50M position records at 40 bytes is 2 GB **[C]**. The dedup hash set for 1B games is 8–32 GB (Section 9.3). Unique-position counting at 1B games needs a probabilistic structure (HyperLogLog) or disk-based sorting **[E]**.

**Checkpoint storage.** Full training state (weights, optimizer moments, EMA) is roughly 12–16 bytes per parameter **[E]**: 25M parameters ≈ 0.3–0.4 GB; 79M ≈ 1–1.3 GB; 270M ≈ 3.2–4.3 GB. Keeping 20 checkpoints of a 270M model is 65–86 GB **[C]**.

**The distinction that matters: retaining versus training on 1 billion games.** Keeping the raw archive costs hundreds of gigabytes. A *single epoch* over all 75 billion positions with a 25M-parameter model costs about 6,700 A100-hours **[C]** (Section 10), or roughly $10,000–$23,000 at the dated cloud prices below. The storage is cheap; the compute is not.

**Data-loading overhead.** The 4M-parameter Chessformer ablation trained at an implied 8,300 positions per second on one A100 **[C]** (see Section 10.2). A Python board-reconstruction loop cannot supply that; use a compiled encoder or pre-encoded shards **[E]**.

---

## 10. Model size, compute, time, and cost

### 10.1 Hardware scenarios

| Scenario | Hardware | Feasible experiments | Models that fit | Compromises |
|---|---|---|---|---|
| **1. No dedicated GPU** | 4–8 CPU cores, 16 GB RAM | Pipeline development; tiny baselines on ≤5M positions; Stockfish matches for evaluation | 0.5–5M parameters | At 25–65 positions per second **[C]** (2–5M parameters at about 50 GFLOP/s sustained; 6 × N × 64 FLOPs per position), 1M positions per epoch takes about 4–11 hours |
| **2. Consumer GPU, 8–16 GB** | e.g., RTX-class card | Phase 1–4; board-state networks up to about 25–50M parameters; 1M-game experiments | Up to 50M | At 20–40 TFLOP/s sustained, a 25M-parameter square-token model processes 100M positions per epoch in 6.7–13 hours **[C]** |
| **3. Workstation, 24–48 GB** | 4090 / A6000 / L40S class | Phase 1–5; 10M-game runs; models up to 100–300M with bf16 | 100–300M | Training memory is about 16 bytes per parameter **[E]** (4.3 GB for 270M); throughput, not VRAM, limits you |
| **4. Multi-GPU workstation** | 4–8× A100/H100 | Replicate Maia-3-class runs: the 79M model trained on 8 A100s for about a week, the 23M model on 8 A100s for a few days `[S11]` **[V]** | Up to about 300M | Data loading and gradient synchronization |
| **5. Small multi-node cluster** | 32–128 GPUs | ChessBench-scale distillation (Section 10.3); 1–10M self-play games | 270M–1B | Engineering for distributed training and self-play generation |
| **6. Industrial scale** | Thousands of accelerators or volunteer fleets | AlphaZero-scale self-play: 5,000 TPUs for generation `[S12]` **[V]**; Leela's 2.5B+ self-play games `[S23]` **[V]** | 100M–1B+ | Organization, not hardware, becomes the bottleneck |

### 10.2 Model size and scaling

| Range | Role | Evidence |
|---|---|---|
| **1–10M parameters** | Baselines, human prediction, fast iteration | Chessformer 3M: 54.8%, 5M: 55.4% human move-matching `[S11]` **[V]**. A 4M-parameter model with the board-aware position bias reached 58.5% policy accuracy on Leela-self-play targets `[S11]` **[V]** |
| **10–100M** | Strong human emulation; the 79M Maia-3 hit 57.1% `[S11]` **[V]**; 100M-parameter Leela transformers generated the self-play data used by `[S11]` **[V]** | |
| **100M–1B** | Engine-strength searchless play: 191M (Leela-CF) and 270M (ChessBench) `[S6]` `[S11]` **[V]** | |
| **Beyond 1B** | No chess-specific justification found in this pass | |

**Why chess scaling differs from language modeling.**
1. **Human-move prediction saturates early** because play is inconsistent: Maia-3's low-rating accuracy plateaus near 50% `[S11]` **[V]**. Language modeling has a long tail of knowledge that rewards parameters; human chess moves have a noise floor.
2. **Engine-distillation keeps improving with scale.** Strength appears only at sufficient size and data `[S6]` **[V]**; the target is a deterministic function (the engine's evaluation), so there is no noise floor, only approximation error.
3. **Inductive bias buys compute.** A better position encoding delivers the same performance at about 60% of the compute `[S11]` **[V]**.
4. **Contexts are tiny** (64 square tokens), so model size dominates cost, not sequence length.

**Calibration point for throughput.** The Chessformer ablation trained a 4.01M-parameter model for 1.4M steps at batch size 2,048 on one A100 in about four days `[S11]` **[V]**. That is 2.87 billion training samples in 96 hours, about **8,300 samples per second** **[C]**. The paper lists 228M "FLOPs" per forward pass for that model; if that is a MAC count, training cost per sample is about 3 × 2 × 228M = 1.37 GFLOP, giving about 11 TFLOP/s achieved, or about **3.7% of the A100's 312 TFLOP/s bf16 peak** **[C]**. If it is already a FLOP count, utilization is about 1.8%. **Small chess networks run far below peak utilization; budget 2–15% rather than 40%.** (The FLOP-versus-MAC definition was not confirmed in the paper.)

### 10.3 Training-compute formulas and worked examples

**Formula.** For a dense network, training FLOPs ≈ 6 × *N* × *T* × *S*, where *N* is parameters, *T* is tokens per sample (64 for square-token boards; about 1 per move for move-token sequences), and *S* is the number of samples **[C]**. Wall-clock = FLOPs ÷ sustained FLOP/s.

**Example A: 25M-parameter square-token model, 1M games.**
- Samples *S* = 75M positions, *T* = 64.
- FLOPs = 6 × 25×10⁶ × 64 × 75×10⁶ = 7.2×10¹⁷.
- At 30 TFLOP/s sustained (about 10% of A100 peak): 24,000 s = **6.7 hours per epoch**.
- Ten epochs: about 67 hours, or about **$100–$230** at $1.49–$3.43 per A100-hour **[C]**.

**Example B: 50M-parameter move-token decoder, 1M games.**
- *T* = 1, *S* = 75M tokens: FLOPs = 6 × 50×10⁶ × 75×10⁶ = 2.25×10¹⁶ per epoch.
- At 30 TFLOP/s: 750 s (about 12.5 minutes) per epoch; character-level (about 600 tokens per game, 8× more) is about 100 minutes per epoch **[C]**.

**Example C: ChessBench-scale distillation (270M, 77 tokens per sample).**
- One pass over 15.3B action-value samples: 6 × 2.7×10⁸ × 77 × 1.53×10¹⁰ = 1.9×10²¹ FLOPs.
- At 125 TFLOP/s sustained (40% of A100 peak, optimistic for a model this size): 1.5×10⁷ s ≈ **4,200 A100-hours per epoch**, about **$6,300–$14,500** at $1.49–$3.43 **[C]**.
- **I could not verify the paper's step count or batch size in this pass**, so I do not give a total for the published run. Multiply by the number of epochs to get a total.
- Add **labeling cost**: 15.3B labels × 50 ms = 7.65×10⁸ core-seconds ≈ **212,000 CPU core-hours** **[C]**. At an assumed $0.03–$0.05 per vCPU-hour **[E]** (price not verified), that is about $6,400–$10,600. A single-best-move-plus-value label (one 100 ms search per position) for 1M games is 75M × 0.1 s = 2,083 core-hours, about $60–$100 **[C]** (price assumption as above).

**Per-scale training cost for a 25M-parameter model, one epoch, 30 TFLOP/s sustained:**

| Scale | FLOPs (9.6×10⁹ × positions) | A100-hours | Cost at $1.49–$3.43 per A100-hour |
|---|---|---|---|
| 100k games (7.5M positions) | 7.2×10¹⁶ | 0.67 | $1–$2 |
| 1M | 7.2×10¹⁷ | 6.7 | $10–$23 |
| 10M | 7.2×10¹⁸ | 67 | $99–$229 |
| 100M | 7.2×10¹⁹ | 667 | $994–$2,288 |
| 1B | 7.2×10²⁰ | 6,667 | $9,934–$22,868 |

At 10 TFLOP/s, multiply every row by 3; at 60 TFLOP/s, halve it. For a 100M-parameter model, multiply by 4 **[C]**.

### 10.4 Self-play generation cost model

For *E* evaluations per game and *F* FLOPs per evaluation, GPU-seconds per game = *E* × *F* ÷ sustained FLOP/s.

| Setup | E (100 plies × sims) | F = 2 × N × 64 | FLOPs per game | GPU-seconds per game at 30 TFLOP/s | Cost for N games |
|---|---|---|---|---|---|
| 10M-param net, 200 simulations per move | 20,000 | 1.28 GFLOP | 2.56×10¹³ | 0.85 | **1M games ≈ 237 A100-hours ≈ $350–$810** |
| 50M-param net, 400 simulations per move | 40,000 | 6.4 GFLOP | 2.56×10¹⁴ | 8.5 | **100M games ≈ 237,000 A100-hours ≈ $350,000–$810,000** |

**These are lower bounds on cost.** MCTS is partly CPU-bound, evaluation batches are small and irregular, and real utilization may be several times lower **[E]**. The estimate says self-play at a research scale of a million games is within a serious individual's budget, and Leela- or AlphaZero-scale generation is not.

### 10.5 What to spend compute on

| Option | Bottleneck | Expected value for this project |
|---|---|---|
| More historical games | Diminishing returns for human-move targets `[S11]` **[V]** | Low beyond a few million rating-balanced games |
| More epochs on the same data | Positions within a game are correlated; overfitting to openings | Moderate; monitor held-out games |
| Larger model | Compute; small models are inefficient on GPUs (Section 10.2) | High for engine-label targets `[S6]` **[V]** |
| Better data filtering | Engineering time | **High**: rating balance and time-pressure filtering are standard in Maia-3 `[S11]` **[V]** |
| Stronger engine labels | CPU labeling cost | **Highest** for engine strength |
| Search at inference | Inference cost per move | High for strength; changes evaluation |
| More self-play | Generation cost | Only after baselines |
| Evaluation and tuning | Engine matches take CPU hours | Frequently underbudgeted; see Section 11 |

### 10.6 Cost analysis (cloud, dated)

| Item | Price observed | Source and date |
|---|---|---|
| A100 80GB, single GPU | **$1.49–$3.43 per GPU-hour** (from specialist providers to hyperscaler 8-GPU nodes) | `[S22]`, March 2026 **[V]** |
| H100 SXM, specialist clouds | **about $2.4–$4 per GPU-hour**; RunPod $3.29 and Lambda $3.99 on one list | `[S22]`, 2026 **[V]** |
| H100 8-GPU node, hyperscalers | **$55.04** (AWS), **$88.49** (Google Cloud), **$98.32** (Azure) per node-hour | `[S22]` **[V]** |
| H100 one-year contract index | $1.70 per GPU-hour (Oct 2025) rising to $2.35 (Mar 2026) | `[S22]` (citing SemiAnalysis) **[V]** |

Prices moved up about 40% in six months in the contract index `[S22]` **[V]**, so quote ranges, not points, and re-check before committing.

**Representative experiment costs** (computed from the dated ranges above, A100-equivalent **[C]**):
- Maia-3-class 79M run (8 A100 for about 7 days = 1,344 A100-hours): **$2,000–$4,600**.
- 4M-parameter ablation (1 A100 for 4 days = 96 hours): **$143–$330**.
- Failed runs and repeats: budget **2–3×** the nominal cost **[E]**.
- Evaluation matches against engines: CPU-bound, not GPU-bound; budget separately (Section 11).

**Rent versus buy.** Break-even hours = purchase price ÷ the hourly rate of the rented equivalent. As an illustration with an assumed $2,000 card and a $1.99 per hour rental, break-even is about 1,000 hours (42 days of continuous use) **[C]** (purchase price illustrative, not verified). Rent for bursts (single-GPU Phase 3–6 runs); buy only if you will keep a card busy for months.

**Free-tier GPUs.** Do not plan long runs on them: availability, session limits, and eviction are not guaranteed, and they interrupt multi-day training. They are adequate for pipeline testing and small baselines only.

---

## 11. Evaluation framework

### 11.1 Predictive metrics

| Metric | How | Why it can mislead |
|---|---|---|
| **Top-1 / top-k move accuracy** | Fraction of positions where the human move is the model's top-1 / in its top-k | Capped by human inconsistency: about 50–57% is near the ceiling for human data `[S11]` **[V]** |
| **Cross-entropy and perplexity** | Mean negative log-likelihood of the played move | Insensitive to *which* errors the model makes |
| **Calibration** | Expected calibration error on predicted move probabilities; reliability curves | A calibrated model of human errors is not a strong player |
| **Breakdowns** | By game phase (move number), rating band, opening family, time control, clock state | A global average hides failures at the extremes. The ALLIE test set has only 205 positions above 2850 rating, a margin of error over 7% there `[S11]` **[V]** |
| **Rare and unfamiliar positions** | Positions whose Zobrist hash never appears in training; positions with unusual material | Required to detect memorization |

**Why high move-prediction accuracy does not imply strong play.** A model that matches a 1500-rated player's move 55% of the time reproduces 1500-rated play, including blunders. Maia's move-matching at one rating is highest when the model was trained at that rating `[S9]` **[V]**, which means the metric *rewards* reproducing weakness. For strength, measure strength.

### 11.2 Playing strength

**Protocol (repeatable):**

1. **Fixed opponents.** A ladder of Stockfish settings at fixed node limits (not fixed time, which depends on hardware) plus, optionally, other engines. Record the exact engine version, hash size, thread count, node limit, and hardware for every match. Calibrate the ladder so your agent scores roughly 20–80% against each rung; Elo error bars widen sharply outside that range.
2. **Opening diversity with color balance.** Play games in pairs from the same opening with colors swapped, as `[S11]` did using the "UHO Lichess 4852 v1" opening set of 2.6 million unbalanced human openings `[S11]` **[V]**. A balanced book reduces variance; an unbalanced book tests robustness. State which you used.
3. **Sufficient games.** Use the confidence interval formula below to set the number of games before you start.
4. **Rating computation.** BayesElo was used by `[S6]` and `[S11]` **[V]**; Ordo is an alternative **[M]**. Report the 95% interval.
5. **Tooling.** `cutechess-cli` or `fastchess` for UCI matches **[M]**; OpenBench was used for distributed engine testing in `[S11]` **[V]**.
6. **Search and inference compute.** Report whether the agent used search, how many nodes, and per-move time. Elo is meaningless without the compute that produced it.

**Confidence-interval arithmetic** **[C]**. Let the score per game be *s* (win = 1, draw = 0.5, loss = 0) and the draw rate *d*.
- Variance per game at *s* ≈ 0.5: σ² ≈ (1 − *d*) / 4.
- Standard error of the mean score over *N* games: SE = σ / √*N*.
- Elo as a function of score: Elo = −400 log₁₀(1/*s* − 1), with slope dElo/d*s* = 400 / (ln 10 × *s* (1 − *s*)) ≈ **695 Elo per unit score** at *s* = 0.5.
- **Example:** *N* = 1,000 games, *d* = 0.3: σ² = 0.175, σ = 0.418, SE = 0.0132, **1σ ≈ ±9 Elo, 95% ≈ ±18 Elo**.
- **Cross-check against the literature:** *N* = 200 games gives 1σ ≈ ±20.5 and 95% ≈ ±40 Elo, consistent with the ±36–42 intervals in `[S11]`'s tables (200 games per pairing) **[V]**.
- **Games needed for a ±10 Elo (95%) interval** at *d* = 0.3: SE needed = 10/1.96 = 5.1 Elo → 0.00734 in score → *N* = 0.175 / 0.00734² ≈ **3,250 games** **[C]**.
- For A/B comparisons between two network versions, a sequential probability ratio test stops matches early when a difference is clear. Stockfish's own testing framework relies on this approach **[M]**.

**Human opponents.** Lichess lets registered bot accounts play rated games through its bot interface **[M]**. A published Lichess blitz rating is the most intuitive number, but it depends on the pool, time control, and number of games, and is not comparable to tournament Elo `[S6]` **[V]**. Verify Lichess's bot rules before running one.

**Do not mix scales.** Never put a Lichess rating, a BayesElo number, and a FIDE rating on the same axis. State the pool every time.

### 11.3 Generalization and robustness suite

| Test | Method |
|---|---|
| Unseen openings | Hold out opening families (ECO groups) from training |
| Rare tactical positions | Lichess puzzles stratified by theme and rating `[S1]` **[V]** |
| Unusual but legal positions | Positions from random legal playouts, filtered to avoid trivially illegal states; compare with engine labels |
| Out-of-range ratings | Evaluate on rating bands absent or rare in training (Maia-3's augmented test set exists for this reason `[S11]` **[V]**) |
| Long-term strategic positions | Positions with engine evaluations near zero and long forced sequences; qualitative review plus engine centipawn loss |
| Endgames | Positions with ≤7 pieces checked against tablebases **[M]** |
| Repetition and draw-related states | Positions with halfmove clock above 80 or repeated positions; check value-head draw probability |
| Adversarial or unfamiliar move sequences | Random-but-legal prefixes of 10–20 plies followed by evaluation |
| Out-of-distribution sources | Positions from engine games (CCRL) or from a different platform |

### 11.4 Pattern-discovery evaluation

| Benchmark | Construction | Pass criterion |
|---|---|---|
| Known tactical motifs | Lichess puzzle themes `[S1]` **[V]** | Per-theme accuracy within a stated margin of a baseline |
| Strategic clustering | Cluster positions; label a random sample blind | Inter-annotator agreement of cluster purity above a pre-set threshold |
| Human error tendencies | Predict blunders by rating on held-out games | AUC/log-loss above a rating-only baseline |
| Human-vs-engine discrepancies | Rank positions by expected human loss (Section 8.5) | Rank correlation with realized centipawn loss on held-out games |
| Candidate positional features | Feature predicates tested per Section 8.3 | Survives temporal and player hold-out with FDR control |
| Hypotheses that survive independent validation | The final list | Reproduced by a second person from the repository |

**Distinguishing discovery from artifacts.**
- **Memorization:** the pattern appears only in positions whose hash occurs in training. Test on hash-disjoint positions.
- **Leakage:** any feature computed from the result, the time forfeit, or a later position. Audit the feature code for lookahead.
- **Coincidence:** permute labels (shuffle results within rating bands) and run the same discovery procedure; the rate of "discoveries" on shuffled data is your false-positive baseline.

---

## 12. Implementation roadmap

Scale only when a decision gate is passed. Durations are for one person working part-time unless stated **[E]**.

### Phase 1: Dataset feasibility study (about 3–5 days)
- **Objective:** a reproducible pipeline from official files to validated, deduplicated shards, with a dataset report.
- **Deliverables:** a script that downloads one month of Lichess standard games with checksum verification; a dataset report listing games per second, invalid-game rate, duplicate rate, unique positions (exact count at 1M games), average length, storage overhead, and preprocessing time.
- **Tools:** Python, a rules library, a compiled PGN reader (Rust), zstd, DuckDB/Parquet.
- **Hardware:** laptop is enough; 1M games parse in minutes in Rust.
- **Success:** report generated by one command; counts reproducible across two runs.
- **Failure:** illegal-game rate unexplained, or throughput too low to process 10M games in a day.
- **Risks:** variant leakage; hidden duplicate months.
- **Gate:** the numbers in Section 9.4's assumption table are replaced by measurements.

### Phase 2: Baseline move predictor (about 1–2 weeks)
- **Objective:** train from random initialization on 1M games; compare at least two representations.
- **Deliverables:** (a) a square-token or CNN board-state network with rule-state inputs and a policy head; (b) a move-sequence decoder on UCI tokens; (c) baselines: uniform over legal moves, an opening-frequency lookup, and an *n*-gram model.
- **Tools:** PyTorch, bf16, `torch.compile`, a checkpointing utility.
- **Hardware:** consumer GPU (Scenario 2).
- **Success:** held-out top-1 well above the frequency baseline; both models trained to convergence with the same budget; a table of results by phase and rating band.
- **Failure:** top-1 below the *n*-gram baseline, a sign of a data or encoder bug.
- **Risks:** leakage by position-level split; mixing time controls.
- **Gate:** one representation clearly ahead, or both within noise (then keep the simpler one).

### Phase 3: Pattern-analysis system (about 1–2 weeks)
- **Objective:** inspect what the networks represent and find reproducible regularities.
- **Deliverables:** embedding extraction; clustering with retrieval of representative positions; a position viewer; linear probes for known concepts (piece locations, check, material).
- **Tools:** scikit-learn, UMAP/HDBSCAN, an SVG board renderer.
- **Hardware:** GPU for embedding; CPU for analysis.
- **Success:** probes recover board state with accuracy in line with `[S8]`-style findings; clusters reviewed blind.
- **Failure:** clusters collapse to rating or opening.
- **Gate:** at least one candidate hypothesis written in the Section 8.3 template.

### Phase 4: Chess-specific evaluation (about 1 week)
- **Objective:** measure prediction and playing strength separately.
- **Deliverables:** a harness that plays UCI matches against a fixed Stockfish ladder, computes BayesElo with intervals, and records engine version, settings, hardware, and node budget; puzzle accuracy on a fixed Lichess puzzle sample.
- **Hardware:** CPU cores for engine games; a GPU for the network.
- **Success:** reruns of the same match reproduce within the confidence interval.
- **Failure:** results change with hardware.
- **Gate:** a numeric strength estimate with an interval for each Phase-2 model.

### Phase 5: Stronger supervision (about 1–2 weeks plus labeling time)
- **Objective:** test whether engine labels improve *strength*, and at what cost.
- **Deliverables:** (a) the same network trained with engine best-move and value labels on 1M games (about 2,083 core-hours **[C]**, Section 10.3); (b) optionally, the Lichess evaluation file as a free value-label source `[S1]` **[V]**; (c) a comparison of Elo gain per unit of compute against Phase 2.
- **Success:** a measurable Elo gain larger than the confidence interval.
- **Failure:** accuracy on engine labels improves but match strength does not (Section 14 covers this negative result).
- **Gate:** continue to Phase 6 only if the label route works.

### Phase 6: Self-play (about 3–6 weeks)
- **Objective:** test whether search-based policy improvement adds strength beyond the supervised baseline.
- **Deliverables:** a batched MCTS with the trained network as prior; a replay buffer; a loop that generates 100k–1M games (about 24–237 A100-hours at the Section 10.4 assumptions **[C]**); periodic fixed-baseline matches.
- **Success:** a statistically significant gain over the Phase-5 model at equal inference compute.
- **Failure:** strength plateaus or oscillates; value loss diverges.
- **Risks:** self-play instability; CPU-bound search.

### Phase 7: Scale-up decision
Define the criteria *before* looking at results.
- **Expand data** if a curve over 100k, 300k, 1M, 3M, and 10M games shows held-out *strength* (not only accuracy) still rising at the largest size after controlling for rating balance.
- **Expand model** if loss decreases with parameter count at fixed data and strength follows.
- **Expand compute for self-play** only if Phase 6 produced a significant, reproducible gain.
- **Stop** if accuracy gains do not move Elo.

---

## 13. Technology stack and engineering design

### 13.1 Stack

| Component | Minimal prototype | Larger scale |
|---|---|---|
| Language / ML | Python, PyTorch (bf16, `torch.compile`) | Same, with distributed data parallel |
| Chess rules | `python-chess` (GPL, **[M]**) for tooling | A compiled library for hot loops (Rust, e.g., `shakmaty`, **[M]**) |
| PGN reading | `python-chess` for small files | Rust PGN reader on compressed streams `[S4]` **[V]** |
| Storage / analytics | Parquet plus DuckDB for dataset reports | Arrow, WebDataset-style shards, object storage |
| Models | Own PyTorch modules; HF Transformers only if you want a stock GPT baseline | Same |
| Engines | Stockfish over UCI for labels and opponents | A fleet with OpenBench `[S11]` **[V]** |
| Experiment tracking | A CSV plus git commit hash per run | MLflow / Weights & Biases |
| Distributed training | Not needed | PyTorch DDP/FSDP |
| Profiling | `torch.profiler`, simple timers | Same |

Not needed until later: distributed training, WebDataset, a hand-written C++ MCTS, hyperparameter search frameworks.

### 13.2 Modules and interfaces

1. **acquire**: download and verify. `fetch(month) -> Path`
2. **parse**: stream PGN. `read_games(path) -> Iterator[Game]`
3. **validate**: legality and filters. `validate(Game) -> ValidGame | Reject(reason)`
4. **encode**: board to tensor. `encode(board, history) -> Tensor` (versioned)
5. **dataset**: shards and manifests. `open_split(name) -> IterableDataset`
6. **models**: network definitions. `forward(planes, rule_state, rating) -> (policy_logits, wdl_logits)`
7. **train**: loop, schedule, EMA. `fit(cfg)`
8. **checkpoint**: save and resume with config and data manifest hashes
9. **infer**: `Policy.select(board) -> Move`
10. **legal**: legal-move generation and masking (the only code allowed to define legality)
11. **search**: optional MCTS. `Search.run(board, net, budget) -> visit_counts`
12. **patterns**: embeddings, clustering, discrepancy analysis, statistical tests
13. **eval**: predictive metrics, matches, puzzles, Elo
14. **track**: metrics and artifacts logging

### 13.3 Repository layout

```
chess-pgn-lab/
├── configs/                 # YAML experiments
├── data/                    # raw/, shards/, manifests/ (not committed)
├── src/chesslab/
│   ├── data/                # acquire.py parse.py validate.py dedup.py split.py shard.py
│   ├── encode/              # planes.py tokens.py
│   ├── models/              # cnn.py square_transformer.py move_gpt.py
│   ├── train/               # loop.py ema.py ckpt.py
│   ├── infer/               # policy.py legal.py
│   ├── search/              # mcts.py
│   ├── eval/                # predictive.py match.py puzzles.py elo.py
│   ├── patterns/            # embed.py cluster.py discrepancy.py stats.py
│   └── track/
├── tests/                   # encoder round-trips, legality, split-leakage tests
└── scripts/
```

### 13.4 Minimal experiment configuration

```yaml
experiment: phase2_board_cnn_1m
data:
  source: lichess_standard
  months: ["2026-03"]            # discovery / training month(s)
  test_months: ["2026-06"]       # temporal hold-out
  filters: {time_control: [blitz, rapid], min_elo: 1500, termination: normal,
            drop_clock_under_s: 30, skip_first_plies: 0}
  n_games: 1000000
  split: {by: game_hash, val_frac: 0.01}
  rating_balance: true
model:
  type: square_transformer        # or cnn, move_gpt
  d_model: 256
  layers: 8
  heads: 8
  history_positions: 7
  inputs: [planes12, castling, ep, halfmove_clock, repetition, rating_embed]
  heads_out: [policy, wdl]
train:
  precision: bf16
  batch_size: 2048
  optimizer: adamw
  lr: 5e-4
  schedule: cosine
  steps: 300000
  ema: 0.999
  seed: 0
eval:
  every_steps: 10000
  predictive: [top1, top5, ce, calibration]
  match: {opponent: stockfish, nodes: [200, 500, 1000], games: 400, pairs: true}
```

### 13.5 Pseudocode

**Data pipeline**

```python
def build_shards(raw_paths, out_dir, cfg):
    seen = HashSet(bits=64)                      # use external sort beyond ~1B games
    writers = {s: ShardWriter(out_dir/s, shard_mb=cfg.shard_mb) for s in SPLITS}
    stats = Counter()
    for game in stream_pgn_zst(raw_paths):       # compiled reader
        stats["read"] += 1
        if game.variant != "Standard":      stats["variant"] += 1;  continue
        if not passes_filters(game, cfg):   stats["filtered"] += 1; continue
        moves = replay_and_validate(game)        # rules library; None if illegal
        if moves is None: quarantine(game);  stats["illegal"] += 1; continue
        h = hash64(moves)
        if h in seen:                       stats["dup"] += 1;      continue
        seen.add(h)
        split = assign_split(game, cfg)          # game-id hash plus month
        writers[split].write(Record(
            moves=encode_moves(moves), result=game.result,
            white_elo=bucket(game.white_elo), black_elo=bucket(game.black_elo),
            time_control=normalize_tc(game.time_control),
            user_hash=salted_hash(game.users)))
    write_manifest(out_dir, stats, cfg, git_commit())
```

**Training loop**

```python
for step, b in enumerate(loader):    # planes, rule_state, rating, move_idx, result, legal_mask
    with autocast(dtype=bfloat16):
        pol, wdl = model(b.planes, b.rule_state, b.rating)
        pol = pol.masked_fill(~b.legal_mask, float("-inf"))   # legality at train and test
        loss = ce(pol, b.move_idx) + cfg.lam_v * ce(wdl, b.result)
    loss.backward()
    clip_grad_norm_(model.parameters(), cfg.clip)
    opt.step(); opt.zero_grad(set_to_none=True); sched.step(); ema.update(model)
    if step % cfg.eval_every == 0:
        log(run_heldout_eval(ema.model)); save_checkpoint(step, ema, opt, cfg, data_manifest_hash)
```

---

## 14. Risks, failure modes, and mitigation

| Risk | Detection | Mitigation |
|---|---|---|
| Emphasizing dataset size over quality | Scaling curve flat in strength while data grows | Rating-balanced sampling, time-pressure filtering `[S11]` **[V]**; run the Phase-7 curve before buying compute |
| Duplicate or correlated games | Duplicate rate in the dataset report; near-duplicate hash on first *k* plies | Dedupe by move hash; split by game; hold out players |
| Learning human mistakes as strategy | Engine evaluation of "common" winning moves; centipawn loss by move class | Use engine labels as the strength objective; treat human-only models as models of behavior |
| Overfitting to common openings | Accuracy by move number; test on hash-disjoint positions | Skip early plies in evaluation `[S11]` **[V]**; hold out opening families |
| Weak performance on unfamiliar positions | Unusual-position suite (Section 11.3) | More diverse sources; engine-labeled positions; self-play |
| Illegal move generation | Illegal-sampling rate | Always mask with a rules engine; log the unmasked rate as a diagnostic |
| Missing repetition and draw conditions | Value calibration on high-halfmove-clock positions | Include halfmove clock and repetition flags `[S11]` **[V]**; separate draw-focused tests |
| Evaluation leakage | Hash-overlap audit between train and test | Game-level and temporal splits; leakage unit tests |
| Prediction gain without Elo gain | Match results flat while accuracy rises | Track both from Phase 2; stop or change objective (negative-result section below) |
| Excess compute on model scaling | Elo per A100-hour curve flattening | Run small ablations first; remember small models run at 2–15% utilization (Section 10.2) |
| Self-play instability | Oscillating fixed-baseline match results; value loss spikes | Replay buffer, root noise, fixed-baseline gating, smaller learning rate |
| Search cost | Per-move latency and node counts in reports | Report compute with every Elo number; budget before building |
| Incorrect causal conclusions | Re-run discovery on shuffled labels | Section 8.3 controls |
| Insufficient evidence for discovered patterns | Multiple-comparison log; FDR-adjusted p-values | Pre-register hypotheses; report counterexamples |
| Privacy and license violations | Review of headers and source terms | Hash usernames; per-source license table (Section 5.3) |

### Negative results: what to conclude

If a larger dataset improves prediction accuracy but **not** playing strength:
1. **Do not conclude the larger dataset is worthless.** Conclude that the target (human moves) saturates, as `[S11]` reports for low-rated play **[V]**.
2. **Check that evaluation measures what you intend.** Verify that the match protocol has enough games (Section 11.2) and the opponent ladder is in the 20–80% score range.
3. **Switch the objective, not the data size.** Replace human-move targets with engine labels or search targets, the route that produced the strength gains in `[S6]` and `[S11]` **[V]**.
4. **Report it.** A well-evidenced negative result (accuracy up, Elo flat, with intervals) is the most valuable output of Phase 7.

---

## 15. Required quantitative comparisons

### 15.1 By dataset scale

Compute column: one epoch with a 25M-parameter square-token model at 30 TFLOP/s sustained **[C]** (Section 10.3). "Expected" entries are **hypotheses [E]**, anchored to the evidence noted; none is a measured result for your setup, and none should be read as an Elo prediction.

| Scale | Main question | Data quality | Training objectives | Model capacity that data can plausibly use **[E]** | Compute per epoch | Storage (compressed PGN / move arrays) | Likely bottleneck | Expectation and evidence anchor |
|---|---|---|---|---|---|---|---|---|
| **100,000 games** | Can a useful baseline be established? | Hand-selectable, rating-balanced | Human policy; small value head | 1–3M | 0.7 A100-h | 25–35 MB / 16 MB | Overfitting; opening memorization | Yes for pipeline validation and a rating-band baseline. A small model reaches the low-to-mid 50s% accuracy only with far more data (Maia-3's 3M model: 54.8% `[S11]` **[V]**), so expect a lower number here |
| **1 million** | What can a small supervised model learn? | Filterable by rating, time control, clock | Human policy, WDL, optional engine labels (about 2,083 core-hours **[C]**) | 3–10M | 6.7 A100-h | 0.25–0.35 GB / 0.16 GB | Labeling and evaluation CPU | Human-like play; Chess-GPT's "few million games" result suggests 1M is near the low end for char-level LMs `[S8]` **[V]** (the post says "a few million") |
| **10 million** | Do additional games improve generalization? | Multiple months; time-split testing possible | Human policy at multiple rating bins; engine labels on a subset | 10–50M | 67 A100-h | 2.5–3.5 GB / 1.6 GB | Rating balance; engine-label cost for all positions (about 21,000 core-hours for best-move labels on every position **[C]**) | ChessBench used 10M games `[S6]` **[V]**; this is the first scale at which a *searchless strong* model is plausible if labels are strong |
| **100 million** | Does scale justify more infrastructure? | Dominated by middling ratings | As above; rating-resampled | 50–200M | 667 A100-h | 25–35 GB / 16 GB | Data loading and dedup; label cost scales linearly | Human-move prediction is flat beyond several million balanced games `[S11]` **[V]**; infrastructure is justified only for engine-label or self-play objectives |
| **1 billion** | Is it economically and technically justified? | 75B positions; many near-duplicates | As above plus self-play mixture | 200M–1B+ | 6,667 A100-h | 250–350 GB / 160 GB | Compute for training and for labeling, not storage | Not supported by current evidence for human-move targets. For engine-quality targets, the limiting factor is label quality, and positions beyond the 15.3B-label ChessBench scale have no published justification found in this pass |

### 15.2 By model strategy

| Strategy | Input | Target | Data | Compute | Strength evidence | Limits |
|---|---|---|---|---|---|---|
| **Move-sequence Transformer** | Move tokens or characters | Next move | Games only | Low–medium | Over 99% legal moves; 64% win rate against Stockfish level 0 for 16 layers `[S8]` **[V]**; ALLIE 55.7% move-matching `[S11]` **[V]** | Board must be rebuilt internally; no search; weak outside the data distribution **[E]** |
| **Board-state policy/value net** | Position + rule state (+ history) | Human move + result, or engine labels | Games (+ labels) | Low–medium | 57.1% human move-matching with 79M `[S11]` **[V]**; 2374 searchless tournament Elo with engine-distilled 191M `[S11]` **[V]** | Needs a rules engine and encoder |
| **Hybrid policy/value + search** | Same + MCTS | Visit counts, outcomes | Self-play or distilled | High | Over 100 Elo added to full Leela by a better network; wins in TCEC events `[S11]` **[V]** | Search cost; harder to evaluate fairly |
| **Self-play from scratch** | Rules only | Visit counts, outcomes | None | Very high | Superhuman at AlphaZero/Leela scale `[S12]` `[S23]` **[V]** | No verified small-team scratch result found; generation cost dominates |

---

## 16. Recommended project design

### 16.1 Decision-oriented answers

| Question | Answer |
|---|---|
| **Should the project be attempted?** | Yes, staged. It is educational at Option A and publishable at Option B. It is not justified at "billion-game" scale on current evidence. |
| **Smallest experiment that tests the core hypothesis?** | Train a small board-state model and a move-sequence model on 1M rating-balanced Lichess blitz/rapid games with a time-split test; compare accuracy **and** match strength against a Stockfish ladder. |
| **First architecture?** | A square-token transformer or residual CNN (3–10M parameters) with rule-state inputs, policy head, and WDL head. Add a UCI-token decoder as the comparison. |
| **First dataset?** | One or two recent months of Lichess standard games (CC0 `[S1]` **[V]**), filtered to blitz/rapid, both players ≥1500, normal termination, clock above 30 seconds. Use a later month as a temporal test. |
| **How many games for the first meaningful experiment?** | About 1 million. Chess-GPT used "a few million"; Maia used 12 million per rating bin; ChessBench used 10 million `[S8]` `[S9]` `[S6]` **[V]**. Start at 1M, then scale on evidence. |
| **Hardware?** | One 12–24 GB GPU plus 8 CPU cores for Phase 2–4. Rent A100s for Option B runs. |
| **What to measure before scaling?** | Data-scaling curve (100k to 10M) of *strength*, not only accuracy; Elo per A100-hour; calibration; the illegal-sampling rate; label-versus-no-label gain. |
| **When does training from scratch make sense instead of fine-tuning?** | When the research question is about what data teaches, when licensing prevents reusing weights, or when you need a controlled experiment. For an application that only needs a strong player or a human-like opponent, use existing open models (e.g., Maia-family networks and code are public `[S10]` `[S11]` **[V]**) and fine-tune. |
| **When is a conventional engine better?** | For maximum strength, analysis, or tablebase-perfect endgames. Stockfish is free (GPLv3 `[S20]` **[V]**) and its network is trained on Leela's reinforcement-learning data `[S20]` **[V]**. |
| **Most promising path to useful patterns?** | Section 8.5: a rating-conditioned human-behavior model combined with engine evaluation, mining systematic human error regions and high-surprise high-value moves, validated per Section 8.3. Interpretability on an engine-strength model (probing, transcoders) is the second route `[S11]` `[S17]` **[V]**. |

### 16.2 Three implementation options

**Option A: Personal research project (maximize learning, minimize cost)**
- **Budget:** $0–$400. Use an owned GPU, or rent up to about 150 A100-hours ($220–$515 at the Section 10.6 range **[C]**) if needed.
- **Timeline:** the 30-day plan below, about 6–8 weeks part-time.
- **Hardware:** one 12–24 GB GPU, 8 CPU cores, 200 GB disk.
- **Deliverables:** a reproducible pipeline; a 1M-game dataset report; two baseline models; a match harness with Elo intervals; one validated or refuted pattern hypothesis.
- **Uncertainties:** the strength you reach is unknown (I found no verified small-team result to anchor it); throughput assumptions are unmeasured.

**Option B: Serious independent research (reproducible, with pattern discovery)**
- **Budget:** about **$3,000–$25,000** **[E]**. Anchors: a Maia-3-class 23M–79M run costs about $1,000–$4,600 on rented A100s **[C]**; best-move labeling of 10M games is about 20,800 core-hours, roughly $600–$1,000 at an assumed $0.03–$0.05 per vCPU-hour **[C]/[E]**; and failed or repeated runs cost 2–3× **[E]**. Replicating a 270M ChessBench-scale run alone is about $13,000–$25,000 per epoch including full action-value labeling **[C]** and sits at or above this range.
- **Timeline:** 3–6 months part-time.
- **Hardware:** rented 1–8 GPUs, 32–64 CPU cores for labeling and matches, 2–5 TB object storage for several months of raw files.
- **Deliverables:** a 10M-game labeled dataset; scaling curves (strength versus data and parameters); a Maia-style human model and an engine-distilled model; a pattern-discovery study with FDR-controlled results and public code; a write-up with intervals and negative results.
- **Uncertainties:** whether strength scales with human-game count; label costs; whether any discovered pattern survives validation.

**Option C: Large-scale research project (hundreds of millions to billions of games, self-play)**
- **Budget:** **$100,000–$1,000,000+** **[E]**. Anchor: 100M self-play games at 400 simulations with a 50M-parameter network is about $350,000–$810,000 in generation alone **[C]**, a lower bound (Section 10.4); plus labeling, training, evaluation, and staff.
- **Timeline:** 12+ months, with a team.
- **Hardware:** a 32–128 GPU cluster, a distributed self-play farm, a CPU fleet for labeling, tens of terabytes of storage.
- **Deliverables:** a distilled and self-play-improved model competitive in computer-chess tournaments, a full ablation of data, model, and search compute, released datasets with clear licensing.
- **Uncertainties:** Leela's 100+ Elo architectural gain `[S11]` **[V]** shows engine-strength frontiers move by tens of Elo per year and by large engineering efforts; outperforming Leela and Stockfish is uncertain, and I found no verified evidence a new team can do so from PGN data.

### 16.3 A concrete 30-day plan (Option A)

| Days | Work | Deliverable and milestone |
|---|---|---|
| **1–3** | Environment, repo skeleton, tests. Download one Lichess month (and a later month for the test set) with checksum verification `[S1]` **[V]**. Build the Rust or Python parse-validate-dedupe path. | **M1:** reproducible 1M-game shard set, with the dataset report (games per second, invalid and duplicate rates, average plies, storage). |
| **4–7** | Position encoder with unit tests (round-trip FEN checks, known tricky rules: castling, en passant, promotion, repetition). Split by game hash and by month. Implement baselines (uniform legal, opening frequency, *n*-gram). | **M2:** baseline table on the held-out set; leakage unit tests pass. |
| **8–12** | Train the board-state model (3–10M parameters) on 1M games. | **M3:** held-out top-1, top-5, cross-entropy, by phase and rating band. Hypothesis: about 45–50% top-1 on 1500–1900-rated blitz positions, anchored to Maia's 46–52% with 12M games per bin `[S9]` **[V]**; a result under about 38% indicates a bug **[E]**. |
| **13–16** | Train the UCI-token decoder on the same data and compute budget. | **M4:** side-by-side comparison; legal-move rate with and without masking. |
| **17–20** | Build the match harness: UCI wrapper, balanced openings, color-swapped pairs, Stockfish fixed-node ladder, Elo with intervals. Run 400 games per ladder rung (about ±28 Elo at 95% for 30% draws **[C]**). | **M5:** a strength estimate with an interval for each model, with engine settings and hardware recorded. |
| **21–24** | Add the Lichess puzzle benchmark. Optionally add engine best-move labels for a 100k-game subset (about 208 core-hours **[C]**) and fine-tune. | **M6:** puzzle accuracy per model; Elo change from labels. |
| **25–28** | Pattern analysis: embeddings, clustering, linear probes; implement expected-human-loss and surprise scores (Section 8.5) using a pre-trained Maia-style model for comparison. Write one hypothesis in the Section 8.3 template and test it on the time-split data. | **M7:** one hypothesis tested with controls, shuffled-label baseline, and adjusted p-value. |
| **29–30** | Write up the results, including negative results and a scaling recommendation. | **M8:** the Phase-7 decision memo: scale, change objective, or stop, with the numbers behind it. |

**Measurable milestones summary:** M1 dataset report reproducible; M2 baselines and leakage tests; M3 accuracy table by rating and phase; M4 representation comparison; M5 Elo with 95% interval; M6 puzzle accuracy and label effect; M7 one controlled pattern test; M8 scale decision.

---

## 17. Bibliography and source-quality notes

### 17.1 Sources used (dates are publication or retrieval dates)

| Key | Source | Link | Relevance |
|---|---|---|---|
| S1 | Lichess open database (CC0; evaluations updated 2026-10-05; puzzles; variants) | https://database.lichess.org/ | Primary dataset source, license, evaluation and puzzle files |
| S2 | Lichess home page banner, "over 7 billion rated standard games" (snapshot around Aug–Sep 2025) | https://lichess.org/ | Total game count (snapshot) |
| S3 | Hugging Face `Lichess/standard-chess-games` dataset card | https://huggingface.co/datasets/Lichess/standard-chess-games | Per-game byte size, ~6% analysis annotations (quoted from Lichess) |
| S4 | Lichess forum feature request (Sept 2020 file: 68M games, 16.8 GB compressed, 130+ GB uncompressed; Elite DB and Rust reader mentions) | https://lichess.org/forum/lichess-feedback/feature-request-split-lichess-database-files-by-time-control-bullet-blitz-rapid-classical | Bytes-per-game calibration; community tooling |
| S5 | Lichess forum thread on monthly game counts ("100 million per month") | https://lichess.org/forum/redirect/post/92ZB65nQ | Growth rate (community evidence) |
| S6 | Ruoss et al., *Amortized Planning with Large-Scale Transformers: A Case Study on Chess* (arXiv Feb 2024; NeurIPS 2024) | https://arxiv.org/abs/2402.04494 | Engine-label distillation; 2895 Lichess blitz; ChessBench |
| S7 | DeepMind `searchless_chess` repository | https://github.com/google-deepmind/searchless_chess | Code, data, and license notes for S6 |
| S8 | Karvonen, *Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models* (arXiv Mar 2024; COLM 2024) | https://arxiv.org/abs/2403.15498 | PGN-text language model; probes; skill estimation |
| S9 | McIlroy-Young et al., *Aligning Superhuman AI with Human Behavior: Chess as a Model System* (KDD 2020) and Maia project page (24 Aug 2020) | https://arxiv.org/abs/2006.01855 · https://maiachess.com/blog/introducing_maia | Human move-matching baselines (46% to over 52%); 12M games per bin (via University of Toronto news release) |
| S10 | Tang et al., *Maia-2* (arXiv 30 Sep 2024; NeurIPS 2024) | https://arxiv.org/abs/2409.20553 | Unified skill-aware model |
| S11 | Monroe et al., *Chessformer: A Unified Architecture for Chess Modeling* (arXiv 18 May 2026; ICLR 2026) | https://arxiv.org/abs/2605.19091 | Current state of the art for human prediction and searchless strength; compute reported; code at https://github.com/CSSLab/maia3 |
| S12 | Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (arXiv Dec 2017; *Science* Dec 2018) | https://arxiv.org/abs/1712.01815 | AlphaZero method and hardware |
| S13 | Lc0 team, *Mastering Chess with a Transformer Model* (arXiv Sep 2024) | https://arxiv.org/abs/2409.12272 | Transformer for self-play engines; 8× fewer FLOPs than AlphaZero's network |
| S14 | Lc0 blog, *Transformer Progress* (28 Feb 2024), read through a quoting forum thread | https://lczero.org/blog/2024/02/transformer-progress/ | Transformer adoption in Leela (I did not fetch the blog page directly) |
| S15 | Lai, *Giraffe: Using Deep Reinforcement Learning to Play Chess* (arXiv Sep 2015); MIT Technology Review (14 Sep 2015) | https://arxiv.org/abs/1509.01549 | Early learned evaluator at IM level in 72 hours |
| S16 | Zhang et al., *Transcendence: Generative Models Can Outperform the Experts That Train Them* (arXiv 2024; NeurIPS 2024) | https://arxiv.org/abs/2406.11741 | Imitation exceeding the data via low-temperature sampling |
| S17 | Schut et al., *Bridging the Human–AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero* (arXiv Oct 2023; journal version indexed 2025) | https://arxiv.org/abs/2310.16410 | Concept discovery and a grandmaster study |
| S18 | McGrath et al., *Acquisition of chess knowledge in AlphaZero*, PNAS 119(47), 2022 | (cited in S11; not fetched) | Probing human concepts in AlphaZero |
| S19 | Jenner et al., *Evidence of learned look-ahead in a chess-playing neural network*, NeurIPS 2024 | (cited in S11; not fetched) | Learned planning in a Leela network |
| S20 | Stockfish repository and pull requests (#5149 SFNNv9, #4295 Leela T80 data) | https://github.com/official-stockfish/Stockfish | GPLv3; nets trained on Leela data under ODbL; architecture facts |
| S21 | Chess.com Published-Data API documentation (read through a copy) | https://www.chess.com/news/view/published-data-api · https://gist.github.com/andreij/0e3309200c0a6bb26308817a168203f3 | Access model and rate-limit rule; terms of use not verified |
| S22 | GPU pricing: Jarvislabs A100 post (Mar 2026); IntuitionLabs comparison (5 Sep 2026); io.net H100 guide (Apr 2026) | https://docs.jarvislabs.ai/blog/a100-price · https://intuitionlabs.ai/articles/cloud-gpu-rental-prices-comparison · https://io.net/p/h100-gpu-cloud-where-to-rent-complete-pricing-guide-for-2026 | Dated cloud prices (vendor and aggregator blogs; prices move) |
| S23 | Wikipedia, *Leela Chess Zero* (figure "as of June 2024") | https://en.wikipedia.org/wiki/Leela_Chess_Zero | 2.5B+ self-play games; distributed generation |
| S24 | Acolyer, *The Morning Paper* summary of the AlphaZero paper (10 Jan 2018) | https://blog.acolyer.org/2018/01/10/mastering-chess-and-shogi-by-self-play-with-a-general-reinforcement-learning-algorithm/ | Secondary "about 20,000 TPU-hours" estimate |
| S26 | *Neural Networks for Chess* (arXiv 2209.01506) | https://arxiv.org/abs/2209.01506 | Description of Leela's win/draw/loss output |
| S27 | `minimal_lczero` repository | https://github.com/Rocketknight1/minimal_lczero | Illustrates Leela's training losses, including moves-left |

Also cited through S11 and not read in full: Zhang et al., *Human-aligned chess with a bit of search* (ALLIE), ICLR 2025.

### 17.2 Source-quality notes

- **Strongest evidence:** arXiv or conference papers (S6, S8–S13, S16, S17) and official repositories or pages (S1, S7, S20). The most recent and most useful single source is S11 (May 2026), which also reports compute and evaluation protocol.
- **Secondary or weaker:** S5 (forum), S14 (quoted blog), S22 (vendor and aggregator pricing, moves quickly), S23 (encyclopedia), S24 (secondary arithmetic), and a Reddit post about Chess-GPT used only to note a differing community calibration.
- **Conflicts noted rather than resolved:** the Chess-GPT opponent calibration (the paper describes Stockfish level 0 as about 1300 Elo; a community post says about 1500); two Elo scales in S6 and S11.

### 17.3 What I did not verify (read before relying on this report)

1. **Chess.com's terms** on redistribution and machine-learning use. Only the API's technical access rules were read.
2. **License terms** of Leela network weights, `python-chess`, `shakmaty`, TWIC, FICS, and Syzygy tablebases; all marked **[M]**.
3. **DeepChess** and **Chess as a Testbed for Language Model State Tracking** (Toshniwal et al.): I mention them only as background from memory and do not rely on their numbers.
4. **ChessBench training schedule** (steps, batch size, epochs, total GPU-hours). I give a formula-based per-epoch figure and flag it as an extrapolation.
5. **Per-model-size Lichess Elo** of the ChessBench models (2054, 2156, 2895): only 2895 is in the abstract; the other two came from a secondary summary.
6. **Parsing throughput, unique-position counts, and duplicate rates** for Lichess data. These are assumptions to be measured in Phase 1.
7. **Cloud CPU prices** used for labeling cost ($0.03–$0.05 per vCPU-hour) and the **illustrative GPU purchase price** are my assumptions, not retrieved prices.
8. **Current Lichess standard-game total.** The ">7 billion" figure is from a page snapshot around late summer 2025; the 2026 total was not retrieved.
9. **No verified small-team result** of training a strong engine from scratch; Option A and B strength outcomes are therefore open questions the 30-day plan is designed to answer.