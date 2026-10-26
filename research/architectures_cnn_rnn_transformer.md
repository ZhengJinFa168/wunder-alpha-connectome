# Sequence Architectures for Alpha Connectome: 1D CNNs (TCN), GRUs, minGRU, and Transformers

Reference notes translating the literature on 1D CNNs vs. RNNs vs. Transformers (Fawaz et al. 2019; Weytjens & De Weerdt 2021; Bai et al. 2018; Feng et al. 2024) to the exact constraints of Wunder Fund's **Alpha Connectome** competition, together with the implementations added to `pipeline/` for **Sweep 7**.

---

## 1. Literature Benchmarks vs. Alpha Connectome Constraints

Most time-series classification (TSC) and process-mining benchmarks differ from Alpha Connectome in four ways that flip the speed/accuracy trade-off at inference time:

| Dimension | Literature Benchmarks (Fawaz 2019, Weytjens 2021, Bai 2018) | Alpha Connectome |
|---|---|---|
| **Task** | Sequence-level classification after observing the full sequence (or prefix) | Step-by-step regression (`[t0, t1]` at steps 99–19,999 of every sequence) |
| **Sequence length** | $T \sim 10^1\text{–}10^3$ | $T = 20,000$ rows per sequence |
| **Dataset size** | $10^2\text{–}10^5$ sequences (small-data overfitting dominates Transformers) | $10,607$ train sequences $= 2.12 \times 10^8$ rows |
| **Primary bottleneck** | GPU training wall-clock time | **1 vCPU online inference ($\le 91\;\mu\text{s/row}$, 60 min for 39.4M rows)** and 20 MB ZIP |

### Key Consequences
1. **"CNNs are 10× faster than RNNs" applies to GPU training, not 1-row CPU inference.**
   During training, a 1D CNN or TCN processes all $T$ steps in parallel across the sequence dimension on a GPU, whereas a standard GRU must step sequentially through each chunk. During online evaluation, however, `predict(data_point)` receives **one row at a time**, so every architecture executes one step per call.
2. **Our empirical findings already confirm the GRU vs. LSTM literature.**
   In Sweep 3, a 2-layer LSTM (`lstm_h112`, cost 1.05× baseline) scored `0.6764` on validation vs. `0.6784` for an equal-cost 2-layer GRU (`h128`), matching Bai et al. (2018) and Hinkka et al. (2018).
3. **Our diagnostics support finite-memory architectures (TCN / sliding window).**
   In `analysis/error_slices.py`, resetting the GRU hidden state inside a sequence every 1,000 steps cost only `-0.0019` WP (and every 2,000 steps cost `-0.0002`). Nearly all predictive signal lives in the most recent $1,000\text{–}2,000$ rows—well within the receptive field of an 11-layer dilated TCN ($2^{11} = 2,048$ rows) or a 7-layer kernel-3 TCN ($3^7 = 2,187$ rows).
4. **Distillation separates teacher compute from student latency.**
   Our best leaderboard submissions (`sweep6_sw_h128_s1`, **0.65660 LB**) are single 128-unit GRUs distilled from a multi-model teacher (`pipeline/teacher.py`).
   * **Teachers** only run offline on GPU in batched mode (`predict_batch`) to write `teacher.npy`. They never run on the grader's CPU and do not need to meet the $91\;\mu\text{s/row}$ limit.
   * **Students** must run in $\le 1.3\times$ baseline time on 1 CPU core.
   * Averaging more GRUs into the teacher has hit diminishing returns (an 8-GRU teacher gave the same student score as a 4-GRU teacher, and ensembling distilled GRU students added nothing because their errors are highly correlated). Adding genuinely different inductive biases (**causal TCN** and **minGRU**) to the teacher pool is the highest-leverage use of new architectures.

---

## 2. Architecture Comparison & Measured Pipeline Speed

All four model families below are implemented in `pipeline/model.py`, verified for chunk-invariance and exact ONNX export in `pipeline/test_export.py` ($\max |\text{PyTorch} - \text{ONNX}| < 10^{-6}$), and benchmarked row-by-row on 1 CPU thread via `pipeline/speed.py` (Apple M4; grader CPU is ~2× slower in absolute $\mu\text{s}$, so the **ratio to baseline** is the invariant metric):

| Model (`ExpConfig`) | Params | MACs / row | Cost vs. Baseline | M4 ONNX (`speed.py`) | Ratio vs. Baseline | Suitable as Student? | Suitable as Teacher? |
|---|---:|---:|---:|---:|---:|---|---|
| **GRU 2×128** (`arch="gru"`, baseline shape) | 142,850 | 190,720 | **1.00×** | $19.4\;\mu\text{s/row}$ | **0.99×** | **Yes** (proven ~28 min grader) | **Yes** |
| **minGRU 3×192** (`arch="mingru", hidden=192, layers=3`) | 192,386 | 190,848 | **1.00×** | $24.1\;\mu\text{s/row}$ | **1.23×** | **Yes** (~34 min est. grader) | **Yes** (fast long-chunk scan) |
| **TCN 11×64, k=2** (`arch="tcn", hidden=64, layers=11, kernel=2`, RF 2048) | 146,114 | 142,464 | **0.75×** | $84.5\;\mu\text{s/row}$ | **4.31×** | **No** without ring buffer | **Yes** (very fast GPU training) |
| **TCN 7×128, k=3** (`arch="tcn", hidden=128, layers=7, kernel=3`, RF 2187) | 476,802 | 473,344 | **2.48×** | $164.6\;\mu\text{s/row}$ | **8.40×** | **No** (teacher only) | **Yes** |
| **Causal Transformer** (sliding window $W=1024$) | — | $O(W \cdot d)$ | $\gg 5\times$ | — | $\gg 5\times$ | **No** ($O(W)$ KV-cache per step) | **Future teacher candidate** |

### Why TCN Has 0.75× MACs but 4.31× ONNX Step Latency
In `TCNNet.stream_step`, each of the 11 dilated blocks maintains a shift register of its last $(k - 1) \cdot d_l$ normalized inputs (`shape = [1, pad_l, C]`). Across $l = 0 \dots 10$ with $k = 2$, $\sum \text{pad}_l = 2047$ rows $\times 64$ floats $\approx 524\;\text{KB}$ of state. Updating those buffers via `torch.cat([buf, norm(z)], 1)[:, 1:]` in ONNX copies ~524 KB of memory on **every single row**, making the step graph memory-bandwidth bound rather than compute bound.

* **For teacher training and batched evaluation (`predict_batch`):** `TCNNet.forward` runs native `nn.Conv1d` over 2,000-row chunks on GPU and only carries the boundary slice `full[:, -pad:]` between chunks. It is **3–4× faster** than `MinGRUNet` and `GRUNet` on long chunks.
* **If a TCN ever needs to be submitted directly as a student:** see Section 5 for the zero-copy ring-buffer implementation in `solution.py` or shorter-horizon TCN variants.

---

## 3. What Was Implemented in `pipeline/`

### 3.1 `MinGRUNet` (`pipeline/model.py`, `arch="mingru"`)
Based on Feng et al. (2024), *"Were RNNs All We Needed?"*:
* **Recurrence:** Drops the hidden-state dependency from the update gate and candidate:
  $$z_t = \sigma(W_z u_t + b_z), \qquad \tilde{h}_t = g(W_h u_t + b_h), \qquad h_t = (1 - z_t) h_{t-1} + z_t \tilde{h}_t$$
  where $g(x) = \max(x + 0.5, \sigma(x)) > 0$ (continuous, strictly positive candidate so the recurrence can be evaluated in log space).
* **Parallel Log-Space Scan (`MinGRUNet._scan`):** During training and batched evaluation, `forward(x, h, x_prev)` computes $\log a_t = -\text{softplus}(k_t)$ and $\log b_t = -\text{softplus}(-k_t) + \log g(c_t)$ and evaluates the associative scan in blocks of `BLOCK = 128` steps using `cumsum` and `torch.logcumsumexp`.
  * Why `BLOCK = 128`? Over a full 2,000- or 5,000-step chunk, the cumulative sum $a^*_t = \sum_{i \le t} \log a_i$ can drift below $-100$, which underflows `float32` when computing $\log b_t - a^*_t$. Blocking at 128 steps and carrying `h0 = h[:, -1]` across blocks keeps $a^*_t$ well inside `float32` range while remaining fast on GPU/MPS (verified against a `float64` sequential loop over 20,000 steps with extreme gates: max relative error $1.15 \times 10^{-4}$; chunk-invariance error $1.85 \times 10^{-6}$).
* **Chrono Initialization:** Gate biases $b_z$ are initialized so the initial characteristic memory timescales $\tau = 1 / z$ are log-uniformly spaced over $[2, 2000]$ rows ($b_z = -\log(\tau - 1)$), giving the model both fast microstructure units and long-horizon memory units from step 0.
* **Streaming ONNX Export (`stream_step`):** Exports as simple `Linear -> Split -> Sigmoid / Where -> Mul / Add` nodes with one state `[1, H]` per layer (`m{k}_h{l}`). No scan operators appear in the exported ONNX graph.

### 3.2 `TCNNet` (`pipeline/model.py`, `arch="tcn"`)
Based on Bai et al. (2018) and WaveNet residual blocks:
* **Architecture:** Linear input projection to `hidden` channels, followed by `layers` pre-norm residual blocks:
  $$u_l = \text{LayerNorm}(z_l), \qquad y_l = \text{Conv1d}_{k, d_l}(u_l), \qquad z_{l+1} = z_l + W_{\text{mix}, l}\,\text{Dropout}(\text{ReLU}(y_l))$$
  with exponential dilation $d_l = k^l$, followed by a final `LayerNorm` and linear head.
* **Exact Chunk Boundary State:** Between chunks in `trainer.py` and `fast_eval.py`, each block returns its last $\text{pad}_l = (k - 1) d_l$ normalized inputs `full[:, -pad:]` (`shape = [B, pad_l, C]`). Prepending that state to the next chunk guarantees **exact equivalence** between whole-sequence evaluation, chunked evaluation (`chunk=500` or `2000`), and 1-row streaming ONNX inference ($\max |\text{whole} - \text{chunked}| = 1.01 \times 10^{-6}$; $\max |\text{PyTorch} - \text{ONNX}| = 7.15 \times 10^{-7}$).

### 3.3 Unified Checkpoints, Ensembles, and Diversity Reporting
* `build_net(config)` in `pipeline/model.py` reconstructs `GRUNet`, `MinGRUNet`, or `TCNNet` from `ck["config"]` (defaulting to `"gru"` for all legacy checkpoints).
* `EnsembleNet` and `_Step` support arbitrary mixes of `GRUNet`, `MinGRUNet`, and `TCNNet` in a single exported ONNX graph (`test_export.py` verifies `ensemble_mixed`).
* `pipeline/ensemble.py` now prints the **prediction correlation matrix** against the top single model on validation scored rows before running greedy selection, so we can immediately see whether TCN and minGRU make sufficiently different errors from GRUs to boost the teacher ensemble.

---

## 4. Sweep 7 Configuration (`colab/build_notebook.py`)

Sweep 7 trains **13 mask-weighted models** ($\lambda = 0.5$, no teacher, 4 passes) from a single shared data stream, judged on validation half B (reference: Sweep 5 `plain_m50 = 0.6755`):

1. **9 GRU training-dynamics runs (`arch="gru"`, `hidden=128`, `layers=2`):**
   * `b256_c500` (current default reference), `b64_c500`, `b32_c500`
   * `b256_c2000`, `b64_c2000`, `b32_c2000`
   * `b32_c5000`, `b32_c20000` (full-sequence backprop), `b32_c500_lr1e-3`
2. **2 minGRU runs (`arch="mingru"`, `hidden=192`, `layers=3`, cost 1.00× baseline):**
   * `mingru_h192_l3`: paired directly with `b256_c500` (same batch size 256 and chunk 500) for an apples-to-apples comparison with the 128-unit GRU.
   * `mingru_h192_l3_b32_c5000`: paired with `b32_c5000` (`micro_batch=32, chunk=5000`), exploiting the parallel scan on 5,000-step chunks.
3. **2 Causal TCN runs (`arch="tcn"`, `chunk=2000`):**
   * `tcn_c64_l11`: 64 channels, 11 layers, kernel 2, receptive field $2^{11} = 2,048$ rows, cost 0.75× baseline.
   * `tcn_c128_k3_l7`: 128 channels, 7 layers, kernel 3, receptive field $3^7 = 2,187$ rows, cost 2.48× baseline (dedicated high-capacity teacher model).

At the end of Sweep 7, cell 8 automatically runs `pipeline.ensemble` across all 13 checkpoints (reporting prediction correlations with the best model and greedy selection on validation half A → half B) and benchmarks CPU speed via `pipeline.speed`.

---

## 5. Future Implementation Roadmap (Post-Sweep 7)

### A. Building a Multi-Architecture Teacher ("Teacher v3")
If `tcn_c64_l11`, `tcn_c128_k3_l7`, or `mingru_h192_l3*` achieve competitive validation half-B scores ($\ge 0.672$) and show lower prediction correlation with `b256_c500` than GRU seeds have with each other ($r < 0.96$ vs. typical GRU-to-GRU $r \approx 0.985$):
1. Include the Sweep 7 TCN and minGRU checkpoints in `pipeline.ensemble --select-on validA` alongside the Sweep 6 mask-weighted GRUs and the best Sweep 7 GRU recipes.
2. Export the combined teacher predictions with `pipeline.teacher` (works out of the box with mixed architectures via `load_ckpt` $\to$ `build_net`).
3. Distill into a 128-unit GRU student (or a `mingru_h192_l3` student, which runs at 1.23× baseline time on CPU—comfortably inside the 60-minute limit).

### B. Conv1d Microstructure Front-End + GRU Hybrid
As noted in `research/prior_work_gemini.md` (Kieran Malandain's `GAMamba` and CNN-GRU hybrids in Mohammadi Foumani et al. 2024):
* A single causal `Conv1d(kernel_size=4 or 8)` layer before the 2-layer GRU smooths tick-level order-book noise and extracts local slopes before the recurrent gates.
* Because only 1 short buffer of length $k - 1 \le 7$ is needed (`[1, 7, C]`), the ONNX buffer-copy overhead is negligible ($< 1\;\mu\text{s/row}$), making a **Conv1d + GRU** hybrid viable as a **direct student submission** as well as a teacher.

### C. Zero-Copy Ring Buffer If We Ever Want a Standalone TCN Student
If a TCN ever outperforms GRUs strongly enough to submit directly (or in a 2-model student ensemble):
* Instead of shifting `[1, d_l, C]` inside ONNX at every step, keep the ring buffers `buf[l]` of shape `[d_l, C]` in NumPy inside `solution.py` with a step counter `t`.
* Pass only the tapped row `buf[l][t % d_l]` (`shape = [1, C]`) into the ONNX graph, and overwrite `buf[l][t % d_l]` with the new normalized hidden state returned by ONNX.
* That eliminates all 524 KB/row of buffer copies and brings `tcn_c64_l11` from $84.5\;\mu\text{s/row}$ down to $\sim 16\;\mu\text{s/row}$ ($\sim 0.8\times$ baseline).

### D. Sliding-Window Causal Transformer as an Offline Teacher
Why we have **not** added a Transformer to Sweep 7 yet, and when to add one:
* Standard point-wise self-attention over raw time steps is weak without local patching or a convolutional stem (Fawaz et al. 2019; modern PatchTST/iTransformer literature), and full $T = 20,000$ attention is $O(T^2)$.
* If the TCN teacher proves valuable in Sweep 7, the natural next offline-only teacher is a **Conv1d stem (kernel 4) + 4-layer causal Transformer with sliding-window attention ($W = 1,024$)** using `F.scaled_dot_product_attention` with a banded causal mask on 2,000-row chunks. It never needs an ONNX export because `pipeline/teacher.py` calls `predict_batch` directly in PyTorch.
