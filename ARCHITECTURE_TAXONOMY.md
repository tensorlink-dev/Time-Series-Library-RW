# Deep Taxonomy of Time-Series Neural Architectures

## A Structural Analysis Based on Computational Graph Topology

> This taxonomy classifies 40 time-series forecasting architectures based on their **actual
> computational graph (DAG) topology**, not naming conventions or paper-stated categories.
> Every claim is derived from direct source code analysis of the implementations in this
> repository.

---

## Table of Contents

1. [Taxonomy Overview](#1-taxonomy-overview)
2. [Axis 1: Core Computational Primitive](#2-axis-1-core-computational-primitive)
3. [Axis 2: DAG Topology Class](#3-axis-2-dag-topology-class)
4. [Axis 3: Inter-Variate Structure](#4-axis-3-inter-variate-structure)
5. [Axis 4: Temporal Scale Structure](#5-axis-4-temporal-scale-structure)
6. [Axis 5: Domain of Computation](#6-axis-5-domain-of-computation)
7. [Axis 6: Output Generation Strategy](#7-axis-6-output-generation-strategy)
8. [Cross-Cutting Structural Patterns](#8-cross-cutting-structural-patterns)
9. [Detailed Model Profiles](#9-detailed-model-profiles)
10. [Structural Equivalence Classes](#10-structural-equivalence-classes)
11. [Master Classification Matrix](#11-master-classification-matrix)
12. [Phylogenetic Analysis](#12-phylogenetic-analysis)

---

## 1. Taxonomy Overview

Traditional categorizations (e.g., "Transformer-based", "MLP-based") rely on naming and
genealogy rather than computational structure. This taxonomy instead uses six orthogonal
structural axes derived from analyzing what each model's `forward()` method actually computes:

| Axis | What It Captures | Example Values |
|------|-----------------|----------------|
| **Computational Primitive** | The dominant operation type | Attention, Convolution, Recurrence, Linear map, State-space, Graph convolution |
| **DAG Topology** | How computation branches and merges | Sequential chain, Parallel-merge, Encoder-decoder, Recursive tree, Iterative residual |
| **Inter-Variate Structure** | How multiple variables interact | Independent, Mixing-by-embedding, Explicit cross-attention, Learned graph |
| **Temporal Scale** | Single vs. multi-resolution | Single-scale, Pyramid, Wavelet, FFT-period-adaptive, Downsampling cascade |
| **Domain of Computation** | Where core math happens | Time domain, Frequency domain, Legendre basis, Koopman space, 2D spatial |
| **Output Strategy** | How predictions are generated | Direct projection, Autoregressive, Decomposition-additive, Semi-autoregressive |

---

## 2. Axis 1: Core Computational Primitive

### 2.1 Quadratic Attention (Full Dot-Product)

Models where the primary sequence-mixing operation is O(L^2) scaled dot-product attention:

| Model | Attention Variant | Notes |
|-------|------------------|-------|
| **Transformer** | Standard full attention | Baseline encoder-decoder |
| **Nonstationary_Transformer** | De-Stationary attention (full + learned tau/delta modulation) | Attention scores scaled by input-dependent tau and shifted by delta |
| **PatchTST** | Full attention on patches | Reduces effective L via patching; uses BatchNorm instead of LayerNorm |
| **PAttn** | Full attention on patches (1 layer) | Minimalist single-layer patch transformer |
| **Crossformer** | Full attention in Two-Stage (time + cross-dim via router) | Router-mediated indirect cross-variable attention |
| **TimeXer** | Full attention (self + cross with global token) | Dual-pathway: patched endogenous + inverted exogenous |
| **iTransformer** | Full attention on inverted axes | Variates are tokens, time is embedding dim |
| **MultiPatchFormer** | Full attention with causal mask | Multi-scale patches + channel-wise stage |
| **Pyraformer** | Full attention with pyramidal sparse mask | Attention is full softmax but zeroed by structured pyramid mask |
| **MSGNet** | Full attention within period windows | Combined with graph convolution |
| **TemporalFusionTransformer** | Interpretable MHA (shared V, averaged heads) | Combined with LSTM; causal masking |

### 2.2 Efficient/Approximate Attention

| Model | Mechanism | Complexity | Notes |
|-------|-----------|-----------|-------|
| **Informer** | ProbSparse (top-u query selection) | O(L log L) | Selects "important" queries; lazy default for rest |
| **Reformer** | LSH (locality-sensitive hashing) | O(L log L) | Self-attention only; no cross-attention capability |

### 2.3 Frequency-Domain "Attention" Replacements

Models that replace dot-product attention with frequency-domain operations:

| Model | Mechanism | Notes |
|-------|-----------|-------|
| **Autoformer** | AutoCorrelation (FFT cross-correlation + top-k delay aggregation) | Period-based time-delay aggregation via FFT |
| **FEDformer** | FourierBlock (spectral linear transform on selected modes) or MultiWaveletTransform | Learnable complex weights on frequency modes |
| **ETSformer** | Exponential Smoothing (conv1d_fft weighted average, no Q/K) | No query-key matching; pure exponential decay weighting |

### 2.4 Convolution as Primary Primitive

| Model | Conv Type | Notes |
|-------|-----------|-------|
| **TimesNet** | 2D Inception Conv (multi-kernel: 1x1, 3x3, 5x5...) | Applied after 1D-to-2D temporal reshape |
| **MICN** | 1D Conv (downsample-isometric-upsample sandwich) | Multi-scale conv with transposed conv upsampling |
| **SCINet** | 1D Causal Conv with multiplicative coupling | Recursive binary tree of even/odd splitting |
| **KANAD** | 1D Conv (3 layers, kernel=3) | Operates on cosine basis expansion |

### 2.5 State-Space / Recurrent Primitives

| Model | Mechanism | Notes |
|-------|-----------|-------|
| **Mamba** | Selective SSM (mamba_ssm library) | Single block, minimal architecture |
| **MambaSimple** | Selective SSM (from-scratch implementation) | Input-dependent A,B,C,delta; depthwise causal conv + SiLU gating |
| **SegRNN** | GRU on segments | Segment-level recurrence; shared encoder-decoder GRU |
| **FiLM** | HiPPO-LegT (structured SSM on Legendre basis) | Recurrence projects onto Legendre polynomials |

### 2.6 Pure Linear / MLP Primitives

| Model | Structure | Notes |
|-------|-----------|-------|
| **DLinear** | Two nn.Linear (trend + seasonal) | Zero nonlinearities in forecast path |
| **TiDE** | Stacked ResBlocks (Linear + ReLU + skip + LayerNorm) | Dense encoder-decoder MLP |
| **TSMixer** | Alternating temporal-MLP + channel-MLP with residuals | Canonical "mixer" pattern |
| **LightTS** | Dual-sampling IEBlocks (Linear + LeakyReLU) | Continuous vs interval chunk sampling |

### 2.7 Graph Convolution

| Model | Graph Type | Notes |
|-------|------------|-------|
| **MSGNet** | Learned adaptive adjacency + mixprop diffusion GCN | Combined with attention and FFT period detection |
| **TimeFilter** | Learned adjacency via bilinear projection + MoE masking | GCN with structured expert masks (same-variate, same-time, cross) |

### 2.8 Frequency-Domain MLP

| Model | Mechanism | Notes |
|-------|-----------|-------|
| **FreTS** | FFT -> diagonal complex linear -> iFFT | Separate temporal and optional channel FFT paths |
| **WPMixer** | Mixer blocks in wavelet coefficient space | DWT decomposition -> per-level MLP mixing -> inverse DWT |

### 2.9 Operator-Theoretic

| Model | Mechanism | Notes |
|-------|-----------|-------|
| **Koopa** | Koopman operators (learned linear + DMD least-squares) | Dual-stream: time-invariant (learned K) + time-variant (data-driven DMD) |

### 2.10 Foundation Models (Published Architectures)

> **Note on this repository's implementations**: The code in `models/` for these 7 models
> uses simplified HuggingFace proxy wrappers (T5/BERT/GPT-2 with random weights), not the
> actual published architectures. The descriptions below document the **real** architectures
> as published in their respective papers.

#### Chronos (Amazon, 2024)
**Paper**: "Chronos: Learning the Language of Time Series"

- **Backbone**: T5 encoder-decoder transformer (20M to 710M params across Mini/Small/Base/Large)
- **Tokenization**: Continuous values mapped to a **fixed vocabulary of discrete bins** (4096 tokens)
  via quantile or uniform binning with mean-scaling normalization
- **Core mechanism**: Treats time series as a **language modeling problem** -- cross-entropy
  loss on discrete token predictions
- **Decoding**: **Autoregressive** token-by-token generation via the T5 decoder
- **Output**: Categorical distribution over bins at each step; samples converted back to
  continuous values via bin centroids. Inherently **probabilistic**
- **Augmentation**: TSMixup (convex combination of series pairs) + KernelSynth (Gaussian
  process kernel-generated synthetic data)
- **Structural class**: Enc-Dec Transformer | Discrete tokenization | Autoregressive | Probabilistic

#### Chronos-Bolt (Amazon, 2024)

- **Backbone**: T5 **encoder-decoder**, but decoder receives only a single start token
  (NOT autoregressive). Sizes: Tiny (9M), Mini (21M), Small (48M), Base (205M)
- **Input**: Instance normalization -> **patch-based** (non-overlapping patches, NOT discrete
  bins). Patch values + attention mask concatenated, then projected via ResidualBlock to d_model
- **Core change**: Single decoder token cross-attends to all encoder patches, then a
  ResidualBlock projects to the full forecast in one shot. **Non-autoregressive**
- **Output**: Direct **quantile forecasts** (multiple quantiles per timestep)
- **Loss**: Quantile regression loss (not cross-entropy or NLL)
- **Speed**: ~250x faster than original Chronos, 20x less memory
- **Structural class**: Enc-Dec Transformer (single-token decoder) | Patch input | Single-pass | Quantile output

#### Chronos-2 (Amazon, 2025)

- **Backbone**: **Encoder-only** (drops the decoder entirely). T5 encoder with **RoPE**
  replacing T5's relative position embeddings. Sizes: Small (28M), Base (120M)
- **Input**: Robust scaling -> add meta features (time index + observation mask) ->
  non-overlapping patches -> ResidualBlock embedding -> insert REG token (attention sink)
  between context and future patches
- **Group Attention** (the key innovation): Each transformer block alternates between:
  - **Time Attention**: Standard self-attention across patches within a single series
  - **Group Attention**: Attention across all series within a "group" at each patch index.
    No positional encoding (series have no inherent order). Enables **multivariate** and
    **covariate** support
- **Group semantics**: Univariate = each series independent; Multivariate = all variates
  of same entity share group ID; Covariates = targets + known covariates share group
- **Output**: 21 fixed quantiles (0.01 to 0.99) via ResidualBlock projection on future patches
- **Max context**: 8,192 timesteps (vs 2,048 for Bolt)
- **Structural class**: Enc-Only Transformer | Time+Group dual attention | Multivariate-capable | Quantile output

#### Moirai (Salesforce, 2024)
**Paper**: "Unified Training of Universal Time Series Forecasting Transformers"

- **Backbone**: Transformer **masked encoder** (NOT decoder-only). Sizes: Small (14M, 6L/384d),
  Base (91M, 12L/768d), Large (311M, 24L/1024d)
- **Any-Variate Attention (AVA)**: The key innovation. Each variate-timestep pair gets its own
  token. A structured attention mask controls:
  - Temporal attention (same variate across time)
  - Cross-variate attention (different variates at same timestep)
  - Causal masking in forecast direction
  - Handles **arbitrary numbers of variates** at test time without architectural changes
- **Multiple Patch Size Projection (MPSP)**: Multiple patch sizes simultaneously, with
  frequency-aware selection. Each patch size has its own linear projection layer
- **Positional encoding**: RoPE (Rotary Position Embeddings), not sinusoidal
- **Output**: **Mixture distribution heads** -- outputs parameters of Student's t mixture,
  Normal mixture, Negative binomial, or Log-normal (10-20 components). Distribution family
  selected per dataset
- **Loss**: Negative log-likelihood on the mixture distribution
- **Normalization**: RevIN (instance normalization)
- **Structural class**: Masked Encoder | Any-Variate Attention | Multi-patch | Mixture-distribution output

**Moirai-MoE** extends the base with Sparse MoE FFN layers (top-2 of 8-16 experts per token,
up to ~1B total params with ~300M active).

#### Sundial (2025)
**Paper**: "Sundial: A Family of Highly Capable Time Series Foundation Models"

- **Two-component architecture**:
  1. **Context Encoder**: Decoder-only (causal) Transformer. Sizes: Small (37M), Base (198M),
     Large (756M). LLaMA-style: RMSNorm, SwiGLU FFN, RoPE
  2. **Flow Matching Decoder**: Conditional flow matching module for probabilistic generation
- **Input**: Non-overlapping patches -> Linear projection -> RoPE -> Causal Transformer
- **Flow matching mechanism** (the key innovation):
  - Learns a **velocity field** that transforms Gaussian noise into forecast distributions
  - Training: Interpolate between noise and target at random time t; predict velocity
  - Inference: Start from Gaussian noise, integrate learned ODE from t=0 to t=1
  - Each different noise sample produces a different forecast (inherently probabilistic)
- **Output**: Can model **arbitrary distribution shapes** (not limited to parametric families)
- **Univariate focus**: Designed primarily for single-variate time series
- **Structural class**: Causal Transformer + Flow Matching | ODE-based sampling | Non-parametric probabilistic

#### TiRex (2024-2025)
**Paper**: "TiRex: Time-series Representation eXtraction"

- **Backbone**: **sLSTM only** (scalar LSTM variant from the xLSTM family). Single variant,
  **35M parameters**. The paper's ablation study showed that mLSTM *degrades* performance;
  TiRex deliberately excludes mLSTM
- **sLSTM architecture**: Enhanced LSTM with exponential gating and memory mixing.
  Uses **block-diagonal recurrence** (4 heads x 128 dimensions) for efficient
  parallelization. Each block has a SiLU-gated FFN (feed-forward network)
- **Input**: Patch-based tokenization -> Linear projection -> sLSTM blocks
- **Training**: **Contiguous Patch Masking (CPM)** — masks contiguous spans of patches
  during pre-training, more effective than random masking for time-series
- **Complexity**: O(L) in sequence length (like Mamba/SSMs, unlike O(L^2) attention)
- **Output**: **9 quantile outputs** (0.1, 0.2, ..., 0.9) via quantile regression heads.
  Uses **missing-value rollout** for forecasting: pads future horizon with NaN patches
  and processes in a single forward pass (NOT autoregressive patch decoding)
- **Use cases**: Zero-shot forecasting, classification, anomaly detection
- **Structural class**: sLSTM (Recurrent) | Patch-based | Block-diagonal | Quantile output | Linear complexity

#### TimesFM (Google, 2024)
**Paper**: "A Decoder-Only Foundation Model for Time-Series Forecasting"

- **Backbone**: Custom decoder-only Transformer. 200M params (20L, 1280d, 16H)
- **Patched decoding** (the key innovation):
  - **Input**: Fixed patch size of 32 timesteps -> Linear residual block -> d_model
  - **Output**: Multiple output heads for different patch sizes {1, 8, 16, 32, 64, 128}
  - At inference, selects the largest output patch size ≤ remaining horizon
  - **Semi-autoregressive**: Autoregressive across patches, parallel within each patch
- **Frequency token**: Learnable embedding prepended to input indicating temporal granularity
  (hourly, daily, weekly, etc.)
- **Training**: MSE loss on next-patch prediction; trained on 100B time points; masked input
  patching (BERT-style random patch masking)
- **Max context**: 512 patches = 16,384 timesteps
- **TimesFM 2.0**: Added quantile prediction heads for probabilistic output
- **Univariate**: Single-variate processing
- **Structural class**: Decoder-only Transformer | Multi-output-patch-size | Semi-autoregressive | Frequency token

#### TimeMoE (2024)
**Paper**: "TimeMoE: Billion-Scale Time Series Foundation Models with Mixture of Experts"

- **Backbone**: Decoder-only Transformer with **Sparse MoE FFN layers**. LLaMA-style
  (RMSNorm, SwiGLU, RoPE). Causal multi-head self-attention with QKV bias for extrapolation
- **Model sizes**:

  | Variant | Total Params | Active Params | Routed Experts | Shared Expert | Top-k |
  |---------|-------------|---------------|----------------|---------------|-------|
  | Base | ~113M | ~50M | 8 | 1 | 2 |
  | Large | ~453M | ~200M | 8 | 1 | 2 |
  | Ultra | 2.4B | ~1.1B | 8 | 1 | 2 |

- **MoE mechanism**: Each FFN layer replaced by 8 routed SwiGLU expert FFNs + **1 shared
  expert** (always active, sigmoid-gated: `g * shared_FFN(h) + sum(s_i * expert_i(h))`).
  Router uses softmax top-k=2 selection. Load-balancing auxiliary loss (alpha=0.02).
  DeepSeekMoE-style shared expert isolation
- **Input**: **Point-wise tokenization** (each individual timestep is one token, NOT patches).
  SwiGLU embedding maps each scalar to d_model. RoPE positional encoding. Max context: 4096
- **Output**: **Multi-resolution forecasting heads** -- 4 parallel linear heads predicting
  [1, 8, 32, 64] future steps simultaneously during training. At inference, **dynamic head
  scheduling** selects the largest head ≤ remaining horizon and autoregressively chains
  predictions for arbitrary-length forecasting
- **Training**: Huber loss (not MSE, for outlier robustness) + MoE load balancing; trained on
  Time-300B (~300B time points); 128x A100 GPUs
- **Univariate**: Single-variate processing
- **Structural class**: Decoder-only Transformer + Sparse MoE + Shared Expert | Point-wise tokens | Multi-resolution heads | Conditional computation

---

## 3. Axis 2: DAG Topology Class

### 3.1 Sequential Chain (Linear DAG)

Computation flows through a single path with no branching:

```
Input -> [Block]^N -> Output
```

| Model | Chain Description |
|-------|------------------|
| **PatchTST** | Embed -> [SelfAttn + FFN]^N -> Flatten -> Linear |
| **PAttn** | Embed -> [SelfAttn + FFN]^1 -> Flatten -> Linear |
| **iTransformer** | Inverted Embed -> [SelfAttn + FFN]^N -> Linear |
| **Mamba** | Embed -> SSM -> Linear |
| **MambaSimple** | Embed -> [RMSNorm -> SSM + residual]^N -> RMSNorm -> Linear |
| **TSMixer** | [Temporal-MLP + Channel-MLP]^N -> Linear |
| **KANAD** | Basis expand -> [Conv1d]^3 -> Linear |
| **SegRNN** | Segment -> GRU-encode -> GRU-decode -> Linear |
| **Chronos** | Bin-tokenize -> T5 Enc-Dec -> Autoregressive token generation |
| **Chronos-Bolt** | Patch -> T5 Enc-Dec (single-token decoder) -> Quantile projection |
| **Chronos-2** | Patch -> T5 Encoder (Time Attn + Group Attn) -> Quantile projection |
| **TimesFM** | Patch -> Causal Transformer -> Multi-size output patch heads |
| **TimeMoE** | Point-wise SwiGLU embed -> Causal Transformer+MoE -> Multi-resolution heads |
| **TiRex** | Patch -> sLSTM blocks (block-diagonal) -> Quantile output via NaN-rollout |

### 3.2 Parallel Branches Merging (Fan-out / Fan-in)

Computation splits into parallel paths that merge:

```
Input -> Branch_1 -> \
      -> Branch_2 ->  Merge -> Output
      -> Branch_3 -> /
```

| Model | Branches | Merge Strategy |
|-------|----------|---------------|
| **DLinear** | 2 (trend + seasonal after decomp) | Element-wise addition |
| **TimesNet** | k (one per FFT-detected period) | Weighted sum (softmax of FFT amplitudes) |
| **MSGNet** | k (one per FFT-detected period, sequential graph conv) | Weighted sum |
| **LightTS** | 3 (continuous + interval + highway) | Concatenation then addition |
| **FiLM** | 3 (multi-scale: 1x, 2x, 4x lookback) | Learned linear combination |
| **MICN** | 2 main (trend linear + seasonal MIC); within MIC: multi-kernel parallel | Addition (trend+seasonal); Conv2d merge (multi-kernel) |
| **MultiPatchFormer** | 4 patch scales (kernel 8,16,24,32) concatenated in feature dim | Feature concatenation |
| **WPMixer** | level+1 wavelet resolution branches | Inverse DWT reconstruction |

### 3.3 Encoder-Decoder (Two-Stage with Cross-Connection)

```
Input -> Encoder -> [context] -> Decoder -> Output
                       |            ^
                       +------------+  (cross-attention or similar)
```

| Model | Cross-Connection Type | Notes |
|-------|----------------------|-------|
| **Transformer** | Multi-head cross-attention | Standard |
| **Informer** | ProbSparse cross-attention | With distillation pyramid in encoder |
| **Autoformer** | AutoCorrelation cross-attention | Progressive decomposition in both |
| **FEDformer** | Fourier/Wavelet cross-attention | Spectral domain cross-connection |
| **Nonstationary_Transformer** | DS cross-attention (modulated) | tau/delta from parallel Projector MLPs |
| **Crossformer** | Multi-scale cross-attention | Each decoder layer connects to different encoder scale |
| **TimeXer** | Global-token cross-attention with exogenous | Endogenous patches + exogenous inverted embedding |
| **TemporalFusionTransformer** | LSTM seq2seq + interpretable attention | Layer-coupled with static context conditioning |
| **ETSformer** | Layer-by-layer growth/season passing | Each decoder layer receives its corresponding encoder layer's output |
| **TiDE** | Dense MLP encoder-decoder | No attention; ResBlock chain with temporal decoder |
| **Chronos** | T5 encoder-decoder | Autoregressive token decoding with cross-attention |
| **Sundial** | Causal Transformer encoder + Flow Matching decoder | Two-component: context encoder feeds velocity network |

### 3.4 Recursive Tree Structure

```
        Input
       /     \
    even     odd
    / \     / \
  ee  eo  oe  oo
  ...        ...
```

| Model | Tree Structure |
|-------|---------------|
| **SCINet** | Binary tree via even/odd temporal splitting, depth=3 default; multiplicative coupling at each node |

### 3.5 Iterative Residual Refinement

```
Input -> Block_1 -> residual = input - backcast_1
                     -> Block_2 -> residual = residual - backcast_2
                                    -> ...
forecast = sum(forecast_i)
```

| Model | Refinement Strategy |
|-------|-------------------|
| **Koopa** | Time-variant Koopman backcast subtracted from residual each block; forecast accumulated |
| **ETSformer** | Level/Growth/Season refined across layers; trends accumulated |

### 3.6 Two-Phase Sequential (Temporal then Cross-Variable)

```
Input -> [Temporal Processing]^N -> Reshape -> [Cross-Variable Processing]^M -> Output
```

| Model | Phase 1 | Phase 2 |
|-------|---------|---------|
| **MultiPatchFormer** | N layers causal self-attention (channel-independent) | 1 layer channel-wise attention (Conv1d K/V) |
| **Crossformer** | Cross-Time attention per TSA layer | Cross-Dimension attention via router per TSA layer |

---

## 4. Axis 3: Inter-Variate Structure

This axis captures how models handle the relationship between multiple input variables (channels).

### 4.1 Strictly Channel-Independent

Each variate is processed completely separately with zero cross-channel interaction.
Variables are either looped over or folded into the batch dimension:

| Model | Mechanism |
|-------|-----------|
| **PatchTST** | `B*enc_in` batch folding |
| **PAttn** | `B*C` batch folding |
| **DLinear** | Per-channel or shared linear (no mixing) |
| **TiDE** | Python for-loop over channels |
| **KANAD** | `B*D` batch folding |
| **WPMixer** | Shared weights, channel as batch dim |
| **FiLM** | HiPPO per-channel |
| **SegRNN** | `B*C` batch folding for encoding (channel embedding for decoding) |
| **Chronos / Chronos-Bolt** | Univariate by design |
| **Sundial** | Univariate by design |
| **TiRex** | Univariate by design |
| **TimesFM** | Univariate by design |
| **TimeMoE** | Univariate by design |

### 4.2 Channel-Mixing by Embedding

Variables are mixed at the input embedding stage via a shared Conv1d or Linear
that maps all `enc_in` channels to `d_model`, with no explicit cross-channel
mechanism thereafter:

| Model | Embedding Mixer |
|-------|----------------|
| **Transformer** | `Conv1d(enc_in, d_model, k=3)` TokenEmbedding |
| **Informer** | Same Conv1d TokenEmbedding |
| **Autoformer** | Same Conv1d TokenEmbedding |
| **FEDformer** | Same Conv1d TokenEmbedding |
| **Reformer** | Same Conv1d TokenEmbedding |
| **Nonstationary_Transformer** | Same Conv1d TokenEmbedding |
| **Pyraformer** | Same Conv1d TokenEmbedding |
| **ETSformer** | Same Conv1d TokenEmbedding |
| **TimesNet** | Same Conv1d TokenEmbedding |
| **MICN** | Same Conv1d TokenEmbedding (seasonal branch) |
| **Mamba** | Same Conv1d TokenEmbedding |
| **MambaSimple** | Same Conv1d TokenEmbedding |
| **SCINet** | `Conv1d(enc_in, enc_in)` in raw variable space |

### 4.3 Explicit Cross-Variable Attention/Mixing

Models with dedicated mechanisms for cross-variable interaction:

| Model | Mechanism | Type |
|-------|-----------|------|
| **iTransformer** | Full self-attention where tokens = variates | Direct variate-to-variate attention |
| **Crossformer** | Two-Stage Attention: router-mediated cross-dimension attention | Indirect (via learnable router bottleneck) |
| **TimeXer** | Global token cross-attention between endogenous patches and exogenous embedding | Bottleneck (single global token per variate) |
| **MultiPatchFormer** | Channel-wise encoder after temporal encoder | Conv1d-based K/V in channel attention |
| **TSMixer** | `nn.Linear(enc_in, d_model)` -> ReLU -> `nn.Linear(d_model, enc_in)` per block | MLP channel mixer |
| **LightTS** | `nn.Linear(enc_in, enc_in)` in layer_3 (identity-initialized) | Weak channel mixing (identity init) |
| **TemporalFusionTransformer** | VariableSelectionNetwork (learned soft attention over variables) | Gated variable importance weighting |
| **Moirai** | Any-Variate Attention: structured mask enables cross-variate + temporal attention | Handles arbitrary variate counts via attention masking |
| **Chronos-2** | Group Attention: alternates Time Attn (within-series) + Group Attn (cross-series) | Multivariate + covariate support via group IDs |
| **FreTS** | FFT along channel dimension + complex linear (when enabled) | Frequency-domain channel mixing |

### 4.4 Learned Graph Structure

Models that learn an explicit graph topology between variables:

| Model | Graph Construction | Propagation |
|-------|-------------------|-------------|
| **MSGNet** | `softmax(ReLU(nodevec1 @ nodevec2))` -> adaptive adjacency | mixprop K-hop diffusion GCN |
| **TimeFilter** | `GELU(proj_1(x) @ proj_2(x)^T)` + top-k sparsify + MoE structured mask | Multi-head GCN with expert gating |

### 4.5 Dual-Mode (Configurable)

| Model | Modes |
|-------|-------|
| **TimeMixer** | `channel_independence=True`: fully independent; `=False`: cross-channel MLP |
| **Koopa** | Time-invariant stream: independent; Time-variant stream: channels mixed via segment flattening |

---

## 5. Axis 4: Temporal Scale Structure

### 5.1 Single-Scale

Models that process the entire time series at one resolution:

| Model | Notes |
|-------|-------|
| **Transformer**, **Reformer**, **Nonstationary_Transformer** | Full sequence, no downsampling |
| **PatchTST**, **PAttn** | Patches reduce effective length, but single resolution |
| **iTransformer** | Full time series compressed to d_model per variate |
| **Mamba**, **MambaSimple** | Full sequence through SSM |
| **DLinear**, **TiDE**, **TSMixer** | Single temporal linear/MLP |
| **SegRNN** | Fixed-size segments, single granularity |
| **KANAD** | Single resolution convolutions |
| **TemporalFusionTransformer** | LSTM + attention at original resolution |
| **FreTS** | Single-scale frequency domain |
| **TimeFilter** | Single patch size, graph-learned temporal structure |
| **Chronos / Chronos-Bolt / Chronos-2** | Single resolution (per-token / per-patch) |
| **Sundial** | Single resolution (patch-based) |
| **TiRex** | Single resolution (patch-based) |
| **TimeMoE** | Single resolution (point-wise tokens, multi-resolution output heads) |

### 5.2 Downsampling Pyramid (Encoder Side)

| Model | Mechanism | Scales |
|-------|-----------|--------|
| **Informer** | `Conv1d(stride=2) + MaxPool1d(stride=2)` distillation | L -> L/2 -> L/4 -> ... |
| **Pyraformer** | `Conv1d(stride=ws)` bottleneck construction | L -> L/ws -> L/ws^2; with cross-scale attention mask |
| **Crossformer** | SegMerging (concat adjacent segments + Linear) | seg_num -> seg_num/2 -> ... ; U-net skip connections |

### 5.3 Decomposition-Based Multi-Scale

| Model | Decomposition | Notes |
|-------|--------------|-------|
| **TimeMixer** | Downsampling (avg/max/conv pool) at multiple rates | Bottom-up seasonal + top-down trend mixing; multi-predictor fusion |
| **Autoformer** | Progressive moving-average decomposition | Each layer strips trend; not true multi-resolution |
| **FEDformer** | Fourier mode selection or Wavelet multi-resolution | Wavelet version has explicit multi-scale attention |
| **ETSformer** | FourierLayer top-k modes + ExponentialSmoothing | Implicit multi-scale via frequency selection |

### 5.4 FFT-Adaptive Multi-Period

| Model | How Periods Are Found | Processing Per Period |
|-------|----------------------|---------------------|
| **TimesNet** | `torch.fft.rfft` -> topk amplitude -> period = T/freq | 2D Inception Conv on period-reshaped tensor |
| **MSGNet** | Same FFT period detection | Graph Conv + Attention within period windows |

### 5.5 Multi-Scale Input Processing

| Model | Scale Construction | Notes |
|-------|-------------------|-------|
| **FiLM** | Multiple lookback windows: [1x, 2x, 4x] * pred_len | Each processed independently through HiPPO + SpectralConv |
| **MultiPatchFormer** | Four patch sizes (8, 16, 24, 32) | Feature concatenation from all scales |
| **LightTS** | Two chunk sampling strategies (continuous + interval) | Different temporal views of same data |
| **Moirai** | Multiple Patch Size Projection (MPSP) | Frequency-aware patch size selection; each has own projection |
| **TimesFM** | Multiple output patch sizes {1,8,16,32,64,128} | Asymmetric: fixed input patch (32) + variable output patch |

### 5.6 Wavelet Multi-Resolution

| Model | Wavelet | Notes |
|-------|---------|-------|
| **WPMixer** | `DWT1DForward` (configurable wavelet, default db2) | Each coefficient level has independent Mixer branch; reconstructed via inverse DWT |
| **FEDformer** (wavelet mode) | Multi-wavelet with Legendre basis | Wavelet decomposition within attention mechanism |

### 5.7 Recursive Binary Splitting

| Model | Mechanism | Depth |
|-------|-----------|-------|
| **SCINet** | Even/odd index splitting with interleaved conv | Default depth 3 (8x resolution hierarchy) |

---

## 6. Axis 5: Domain of Computation

### 6.1 Purely Time Domain

Models whose core operations stay in the time domain:

**DLinear, LightTS, TiDE, TSMixer, Transformer, Reformer, Nonstationary_Transformer,
PatchTST, PAttn, iTransformer, TimeXer, MultiPatchFormer, Crossformer, Pyraformer,
TemporalFusionTransformer, Mamba, MambaSimple, SegRNN, SCINet, KANAD, TimeFilter,
Chronos, Chronos-Bolt, Chronos-2, Moirai, TimesFM, TimeMoE**

Note: Sundial's flow matching decoder operates in a continuous latent space (noise -> forecast
via ODE integration) but the context encoder is time-domain. TiRex uses sLSTM which is
time-domain recurrent.

### 6.2 Frequency Domain (FFT/DFT)

| Model | Where FFT Is Used | Purpose |
|-------|------------------|---------|
| **Autoformer** | Core attention mechanism | Cross-correlation computation |
| **FEDformer** | Core attention mechanism | Spectral linear transform on selected modes |
| **ETSformer** | FourierLayer in encoder | Top-k mode extraction and sinusoidal extrapolation |
| **TimesNet** | Period detection (preprocessing) | Find dominant periods for 2D reshape |
| **MSGNet** | Period detection (preprocessing) | Find dominant periods for reshape + attention windows |
| **FreTS** | Core computation | FFT along time and optionally channels; complex linear in freq domain |
| **TimeMixer** | Optional DFT_series_decomp | Top-k frequency filtering for seasonal/trend split |
| **Koopa** | FourierFilter (preprocessing) | Split time-invariant vs time-variant components |

### 6.3 Legendre Polynomial Basis

| Model | Mechanism |
|-------|-----------|
| **FiLM** | HiPPO-LegT projects signal onto Legendre basis; SpectralConv in Legendre-coefficient space; reconstruction via eval_matrix |

### 6.4 Wavelet Domain

| Model | Mechanism |
|-------|-----------|
| **WPMixer** | DWT decomposition -> processing in wavelet coefficient space -> inverse DWT |
| **FEDformer** (wavelet mode) | Multi-wavelet decomposition within attention |

### 6.5 Koopman/DMD Space

| Model | Mechanism |
|-------|-----------|
| **Koopa** | MLP encoder maps to Koopman latent space; linear operator (learned or DMD-solved) advances dynamics; MLP decoder maps back |

### 6.6 Reshaped 2D Spatial Domain

| Model | Reshape Strategy |
|-------|-----------------|
| **TimesNet** | `[B, T] -> [B, T/period, period]` -- 2D grid indexed by (inter-period, intra-period) |
| **MSGNet** | Same period-based 2D reshape |

---

## 7. Axis 6: Output Generation Strategy

### 7.1 Direct Linear Projection

The most common strategy: a single linear layer maps from representation to prediction horizon.

| Model | Projection | From What |
|-------|-----------|-----------|
| **DLinear** | `nn.Linear(seq_len, pred_len)` per decomposed component | Raw temporal features |
| **TSMixer** | `nn.Linear(seq_len, pred_len)` | Mixer output |
| **PatchTST** | `nn.Linear(d_model * patch_num, pred_len)` via FlattenHead | Flattened patch representations |
| **PAttn** | `nn.Linear(d_model * patch_num, pred_len)` | Flattened patch representations |
| **iTransformer** | `nn.Linear(d_model, pred_len)` per variate | Encoder output per variate token |
| **TimeXer** | FlattenHead: `nn.Linear(d_model*(patch_num+1), pred_len)` | Flattened patch + global token |
| **MSGNet** | `nn.Linear(seq_len, pred_len)` | ScaleGraphBlock output |
| **Mamba**, **MambaSimple** | `nn.Linear(d_model, c_out)` per timestep, slice last pred_len | SSM hidden states |
| **FreTS** | `nn.Linear(T*D, pred_len)` via 2-layer MLP | Frequency-domain processed features |
| **KANAD** | Conv1d(channels, 1) + Linear | Conv features (anomaly detection only) |

### 7.2 Encoder-Decoder with Cross-Attention

Decoder generates predictions autoregressively or in parallel, conditioned on encoder via cross-attention:

| Model | Decoder Type |
|-------|-------------|
| **Transformer** | Causal self-attention + cross-attention + FFN |
| **Informer** | ProbSparse self + cross-attention |
| **Autoformer** | AutoCorrelation self + cross-attention with trend accumulation |
| **FEDformer** | Fourier/Wavelet self + cross-attention with trend accumulation |
| **Nonstationary_Transformer** | DS self + cross-attention |
| **Crossformer** | TSA self + cross-attention, additive across decoder layers |

### 7.3 Decomposition-Additive Output

Final prediction is an explicit sum of decomposed components:

| Model | Components Summed |
|-------|------------------|
| **Autoformer** | seasonal_projection + accumulated_trend |
| **FEDformer** | seasonal_projection + accumulated_trend |
| **ETSformer** | level + damped_growth + seasonal_extrapolation |
| **DLinear** | seasonal_linear + trend_linear |
| **MICN** | seasonal_MIC + trend_linear |
| **TimeMixer** | sum of per-scale predictions |
| **Koopa** | sum of (time_inv + time_var) forecasts across blocks |

### 7.4 Last-Hidden-State Projection

Using only the final timestep's hidden state to generate the full horizon:

| Model | Mechanism |
|-------|-----------|
| **Pyraformer** | `enc_out[:, -1, :] -> Linear(multi_scale_d, pred_len * enc_in)` |

### 7.5 Autoregressive Token/Patch Decoding

| Model | Mechanism |
|-------|-----------|
| **Chronos** | T5 decoder generates discrete bin tokens one-by-one via cross-entropy; categorical distribution over 4096 bins per step |
| ~~TiRex~~ | *(moved to 7.7 — uses single-pass NaN-rollout, not autoregressive)* |

### 7.6 Semi-Autoregressive Multi-Resolution Decoding

| Model | Mechanism |
|-------|-----------|
| **TimesFM** | Asymmetric patching: input patches of 32, output patches of {1,8,16,32,64,128}. Autoregressive across patches but parallel within each output patch. Selects largest output patch ≤ remaining horizon |
| **TimeMoE** | Point-wise tokenization with 4 multi-resolution output heads [1,8,32,64 steps]. Dynamic head scheduling at inference: select largest head ≤ remaining horizon, chain autoregressively. Huber loss, not MSE |

### 7.7 Single-Pass Distributional Output

| Model | Mechanism |
|-------|-----------|
| **Chronos-Bolt** | T5 enc-dec (single-token decoder cross-attends to encoder) -> ResidualBlock projects to full quantile forecast in one shot |
| **Chronos-2** | T5 encoder-only (Time+Group Attn) -> ResidualBlock projects masked future patches to 21 quantiles |
| **Moirai** | Masked encoder (single pass) -> mixture distribution heads: Student's t / Normal / NegBin / LogNormal with 10-20 components per timestep |
| **TiRex** | sLSTM blocks process context + NaN-padded future patches in a single forward pass (**missing-value rollout**). Output heads produce **9 quantiles** (0.1-0.9) per patch position. No autoregressive decoding |

### 7.8 Flow-Based Generative Output

| Model | Mechanism |
|-------|-----------|
| **Sundial** | Context encoder produces conditioning -> Flow matching decoder: start from Gaussian noise, integrate learned velocity field ODE from t=0 to t=1 -> forecast sample. Multiple samples give prediction intervals. Can model **arbitrary** distribution shapes |

### 7.9 Semi-Autoregressive (Iterative Refinement)

| Model | Mechanism |
|-------|-----------|
| **MultiPatchFormer** | 8 sequential steps, each conditioned on encoding + previous predictions |

### 7.10 Segment-Based Recurrent

| Model | Mechanism |
|-------|-----------|
| **SegRNN** | GRU decodes each output segment with channel+position embeddings |

### 7.11 Basis Reconstruction

| Model | Mechanism |
|-------|-----------|
| **FiLM** | Legendre polynomial evaluation matrix reconstructs time-domain signal from coefficients |
| **WPMixer** | Inverse DWT reconstructs time-domain signal from wavelet coefficients |

---

## 8. Cross-Cutting Structural Patterns

### 8.1 Instance Normalization (RevIN-Style)

Subtract per-instance mean and divide by stdev before processing; reverse after:

**Used by**: Informer (short-term), Reformer (short-term), Nonstationary_Transformer,
PatchTST, iTransformer, TimesNet, TiDE, FiLM, WPMixer, Mamba, MambaSimple, Koopa,
MSGNet, TimeFilter, Chronos (mean scaling), Moirai (RevIN), Sundial, TiRex, TimesFM,
TimeMoE

**Not used by**: Transformer, Autoformer, FEDformer, ETSformer, Pyraformer, Crossformer,
DLinear, TSMixer, LightTS, FreTS, KANAD

### 8.2 Trend-Seasonal Decomposition

Explicit separation of trend and seasonal components:

| Model | Decomposition Method | Where Applied |
|-------|---------------------|--------------|
| **DLinear** | Moving average (AvgPool1d) | Input preprocessing |
| **Autoformer** | Moving average | Every encoder and decoder layer |
| **FEDformer** | Moving average | Every encoder and decoder layer (reuses Autoformer layers) |
| **MICN** | Multi-kernel moving average (averaged) | Input preprocessing + within MIC layers |
| **ETSformer** | Level/Growth/Season components | Structural backbone; FourierLayer + ExponentialSmoothing |
| **TimeMixer** | Moving average or DFT top-k | Every PastDecomposableMixing block |

### 8.3 Patching / Segmentation

Grouping consecutive timesteps into tokens:

| Model | Patch Size | Overlap | Notes |
|-------|-----------|---------|-------|
| **PatchTST** | 16 (default) | Yes (stride=8) | Overlapping patches |
| **PAttn** | 16 (default) | Yes (stride=8) | Same as PatchTST |
| **Crossformer** | seg_len | No | Non-overlapping segments |
| **TimeXer** | patch_len | No | Non-overlapping |
| **MultiPatchFormer** | 8,16,24,32 | Various strides | Multi-scale patches |
| **SegRNN** | seg_len | No | Segments for GRU |
| **WPMixer** | patch_len | Configurable | Patches on wavelet coefficients |
| **TimeFilter** | patch_len | No (stride=patch_len) | Cross-variate patches |
| **Moirai** | Multiple sizes (MPSP) | No | Frequency-aware selection |
| **Sundial** | ~64 (configurable) | No | Non-overlapping |
| **TiRex** | Configurable | No | Non-overlapping |
| **TimesFM** | 32 (input), {1-128} (output) | No | Asymmetric I/O patch sizes |
Note: Original Chronos and TimeMoE use per-timestep tokenization (not patching). Chronos-Bolt and Chronos-2 use patching.

### 8.4 Residual Connection Patterns

| Pattern | Models |
|---------|--------|
| **Pre-norm residual** (norm before, add after) | MambaSimple, Pyraformer |
| **Post-norm residual** (add then norm) | Transformer, Informer, Autoformer, FEDformer, PatchTST, Crossformer, iTransformer, TimeXer, MultiPatchFormer, TimeFilter |
| **ResBlock** (linear skip + MLP + LayerNorm) | TiDE |
| **Per-mixer residual** (temporal + channel separately) | TSMixer |
| **Global highway/skip** (linear input-to-output) | LightTS, TiDE, FreTS |
| **Iterative backcast** (residual -= reconstruction) | Koopa, ETSformer |
| **GLU-gated residual** | TemporalFusionTransformer |
| **None** | DLinear, SegRNN |

### 8.5 Causal vs. Bidirectional

| Causal (masked) | Bidirectional (unmasked) |
|-----------------|------------------------|
| Transformer (decoder), Informer (decoder), Autoformer (decoder), FEDformer (decoder), Nonstationary_Transformer (decoder), MultiPatchFormer (temporal encoder), TemporalFusionTransformer, Pyraformer (pyramid mask) | Transformer (encoder), PatchTST, PAttn, iTransformer, Crossformer, TimeXer (self-attention on patches) |
| Chronos (T5 decoder, causal), Sundial (context encoder, causal), TiRex (sLSTM, causal), TimesFM (decoder-only, causal), TimeMoE (decoder-only+MoE, causal) | Chronos (T5 encoder, bidirectional), Chronos-Bolt (T5 encoder, bidirectional), Chronos-2 (T5 encoder + Group Attn, bidirectional), Moirai (masked encoder, structured AVA mask) |

---

## 9. Detailed Model Profiles

### Quick-Reference Architecture Fingerprints

Each model is characterized by a compact structural fingerprint:

```
Transformer:        Attn-Full | EncDec | ChanMix-Embed | SingleScale | TimeDomain | DecCrossAttn
Informer:           Attn-ProbSparse | EncDec-Pyramid | ChanMix-Embed | MultiScale-Distill | TimeDomain | DecCrossAttn
Autoformer:         Attn-AutoCorr(FFT) | EncDec | ChanMix-Embed | Decomposition | FreqDomain | DecompAdditive
FEDformer:          Attn-Fourier/Wavelet | EncDec | ChanMix-Embed | FreqModes/Wavelet | FreqDomain | DecompAdditive
Reformer:           Attn-LSH | EncOnly | ChanMix-Embed | SingleScale | TimeDomain | DirectProj
Nonstat_Trans:      Attn-DSAttn | EncDec | ChanMix-Embed | SingleScale | TimeDomain | DecCrossAttn
Pyraformer:         Attn-PyramidMask | EncOnly | ChanMix-Embed | MultiScale-Pyramid | TimeDomain | LastHiddenProj
ETSformer:          ExpSmooth+Fourier | EncDec-Layerwise | ChanMix-Embed | FreqModes | FreqDomain | DecompAdditive
Crossformer:        Attn-TwoStage | EncDec-MultiScale | CrossVar-Router | MultiScale-SegMerge | TimeDomain | AdditiveDecLayers
PatchTST:           Attn-Full(patches) | EncOnly | ChanIndep | SingleScale-Patch | TimeDomain | FlattenHead

iTransformer:       Attn-Full(inverted) | EncOnly | CrossVar-Attn | SingleScale | TimeDomain | DirectProj
TimeMixer:          MLP-Mixing | EncOnly+MultiPred | Configurable | MultiScale-Downsample | TimeDomain+OptFFT | DecompAdditive
TimeXer:            Attn-Self+Cross | EncOnly | CrossVar-GlobalToken | SingleScale-Patch | TimeDomain | FlattenHead
PAttn:              Attn-Full(patches,1L) | EncOnly | ChanIndep | SingleScale-Patch | TimeDomain | FlattenHead
MultiPatchFormer:   Attn-Causal+Channel | TwoPhase | ChanIndep->ChanMix | MultiScale-4Patch | TimeDomain | SemiAutoregressive
TFT:                LSTM+InterpAttn | EncDec | VarSelection | SingleScale | TimeDomain | GatedOutput
Chronos:            T5-EncDec | EncDec | Univariate | SingleScale | TimeDomain | AR-TokenDecoding(categorical)
Chronos-Bolt:       T5-EncDec(1tok) | EncDec(single-pass) | Univariate | SingleScale-Patch | TimeDomain | SinglePass-Quantile
Chronos-2:          T5-EncOnly+GroupAttn | EncOnly | CrossVar-GroupAttn | SingleScale-Patch | TimeDomain | SinglePass-Quantile
Moirai:             MaskedEnc+AVA | EncOnly | CrossVar-AVA | MultiPatch-MPSP | TimeDomain | SinglePass-MixtureDistrib
Sundial:            CausalTransformer+FlowMatch | TwoComponent | Univariate | SingleScale-Patch | TimeDomain+Latent | FlowODE-Generative
TiRex:              sLSTM-only(block-diag) | SeqChain | Univariate | SingleScale-Patch | TimeDomain | SinglePass-Quantile(NaNRollout)
TimesFM:            DecOnly-Transformer | SeqChain | Univariate | MultiOutputPatch | TimeDomain | SemiAR-AsymPatch
TimeMoE:            DecOnly-Transformer+SparseMoE+SharedExpert | SeqChain | Univariate | PointWise+MultiResHead | TimeDomain | MultiResHead-DynSchedule

DLinear:            Linear | ParallelMerge | ChanIndep | SingleScale | TimeDomain | DecompAdditive
LightTS:            Linear+LeakyReLU | ParallelMerge+Highway | WeakChanMix | DualSampling | TimeDomain | DirectProj+Skip
TiDE:               MLP-ResBlock | SeqChain+Skip | ChanIndep-Loop | SingleScale | TimeDomain | DirectProj+Skip
TSMixer:            MLP-AltMixing | SeqChain | ChanMix-MLP | SingleScale | TimeDomain | DirectProj
WPMixer:            MLP-Mixer(wavelet) | ParallelMerge | ChanIndep | MultiScale-Wavelet | WaveletDomain | InverseDWT
FiLM:               HiPPO-SSM+SpectralConv | ParallelMerge | ChanIndep | MultiScale-Lookback | LegendreDomain | BasisRecon
FreTS:              FreqMLP(FFT) | SeqChain+Skip | OptChanMix-FFT | SingleScale | FreqDomain | DirectProj

TimesNet:           2DInceptionConv | ParallelMerge(periods) | ChanMix-Embed | MultiScale-FFTPeriod | 2DSpatial | DirectProj
MICN:               1DConv(down-iso-up) | ParallelMerge | ChanMix-Embed | MultiScale-MultiKernel | TimeDomain | DecompAdditive
SCINet:             1DCausalConv | RecursiveTree | ChanMix-RawConv | MultiScale-BinaryTree | TimeDomain | DirectProj
MSGNet:             GraphConv+Attn | ParallelMerge(periods) | ChanMix-Graph | MultiScale-FFTPeriod | 2DSpatial | DirectProj
Mamba:              SSM(library) | SeqChain | ChanMix-Embed | SingleScale | TimeDomain | DirectProj
MambaSimple:        SSM(custom) | SeqChain | ChanMix-Embed | SingleScale | TimeDomain | DirectProj
SegRNN:             GRU(segments) | SeqChain | ChanIndep+Embed | SingleScale-Segment | TimeDomain | SegmentDecode
Koopa:              KoopmanDMD+MLP | DualStream+IterResid | Mixed | DualStream | KoopmanSpace | IterativeAccum
KANAD:              1DConv+CosBasis | SeqChain | ChanIndep | SingleScale | TimeDomain | DirectProj
TimeFilter:         GCN+MoEMask | SeqChain | CrossVar-LearnedGraph | SingleScale-Patch | TimeDomain | DirectProj
```

---

## 10. Structural Equivalence Classes

Models that are structurally near-identical despite different names:

### Class A: Vanilla Patch Transformer (Encoder-Only)
**PatchTST** and **PAttn** are structurally identical except PAttn uses only 1 encoder layer.
Both: Patch embed -> [SelfAttn + FFN]^N -> FlattenHead. Channel-independent. Single-scale.

### Class B: Vanilla Encoder-Decoder Transformer
**Transformer** is the clean baseline. **Nonstationary_Transformer** adds only tau/delta
modulation to the same topology. **Reformer** uses the same topology but replaces attention
with LSH and drops the decoder (encoder-only with appended placeholders).

### Class C: Progressive Decomposition Encoder-Decoder
**Autoformer** and **FEDformer** share the same DAG topology (progressive trend/seasonal
decomposition in encoder and decoder, accumulated trend). They differ only in the attention
replacement mechanism (AutoCorrelation vs. Fourier/Wavelet blocks).

### Class D: Causal Decoder-Only Patch Transformer (Foundation)
**TimesFM** stands alone in this class: patch input -> causal Transformer -> asymmetric
multi-size output patches. Semi-autoregressive (autoregressive across patches, parallel
within). 200M dense parameters.

**TiRex** was previously grouped here but is structurally distinct: it uses **sLSTM-only**
(not a Transformer or mLSTM), outputs **9 quantiles** (not point predictions), and uses
**missing-value NaN-rollout** (single forward pass, not autoregressive). TiRex is better
classified as its own structural class — a recurrent single-pass quantile model.

### Class E: Causal Decoder-Only Point-wise Transformer (Foundation)
**TimeMoE** is unique among foundation models in using **point-wise tokenization** (each
timestep = one token, no patching). It combines Sparse MoE with a shared expert and
multi-resolution output heads [1, 8, 32, 64] with dynamic scheduling.

### Class F: FFT-Period + 2D Processing
**TimesNet** and **MSGNet** share the FFT period detection and 1D-to-2D temporal reshape.
They differ in what happens in 2D space: TimesNet uses Inception Conv, MSGNet uses
graph convolution + attention.

### Class G: Full Self-Attention on Inverted Axes
**iTransformer** is unique in treating variates as tokens. No other model shares this topology.

### Class H: Multi-Scale MLP Mixer
**TimeMixer** is unique in its multi-scale decomposition with bidirectional cross-scale mixing.

---

## 11. Master Classification Matrix

| Model | Primitive | DAG Topology | Inter-Var | Temporal Scale | Compute Domain | Output Strategy |
|-------|-----------|-------------|-----------|---------------|----------------|-----------------|
| Transformer | Full Attn | Enc-Dec | Mix-Embed | Single | Time | Dec-CrossAttn |
| Informer | ProbSparse | Enc-Dec+Pyramid | Mix-Embed | Pyramid | Time | Dec-CrossAttn |
| Autoformer | AutoCorr(FFT) | Enc-Dec | Mix-Embed | Decomp | Freq | Decomp-Add |
| FEDformer | Fourier/Wavelet | Enc-Dec | Mix-Embed | Freq/Wavelet | Freq | Decomp-Add |
| Reformer | LSH | Enc-Only | Mix-Embed | Single | Time | Direct-Proj |
| NS-Transformer | DS-Attn | Enc-Dec | Mix-Embed | Single | Time | Dec-CrossAttn |
| Pyraformer | PyramidMask | Enc-Only | Mix-Embed | Pyramid | Time | LastHidden |
| ETSformer | ExpSmooth+FFT | Enc-Dec(layer) | Mix-Embed | FreqModes | Freq | Decomp-Add |
| Crossformer | TwoStage | Enc-Dec(multiscale) | Router-Attn | SegMerge | Time | Additive-Dec |
| PatchTST | Full Attn | Enc-Only | Independent | Patch | Time | FlattenHead |
| PAttn | Full Attn | Enc-Only(1L) | Independent | Patch | Time | FlattenHead |
| iTransformer | Full Attn(inv) | Enc-Only | Variate-Attn | Single | Time | Direct-Proj |
| TimeMixer | MLP-Mix | Enc+MultiPred | Config | Downsample | Time+optFFT | Decomp-Add |
| TimeXer | Self+Cross | Enc-Only | GlobalToken | Patch | Time | FlattenHead |
| MultiPatchFormer | Causal+Chan | TwoPhase | Indep->Mix | 4-Patch | Time | Semi-AR |
| TFT | LSTM+InterpAttn | Enc-Dec | VarSelect | Single | Time | Gated |
| DLinear | Linear | Parallel-Merge | Independent | Single | Time | Decomp-Add |
| LightTS | Linear+Act | Parallel+Highway | Weak-Mix | DualSample | Time | Direct+Skip |
| TiDE | MLP-ResBlock | SeqChain+Skip | Indep-Loop | Single | Time | Direct+Skip |
| TSMixer | MLP-AltMix | SeqChain | MLP-Mix | Single | Time | Direct-Proj |
| WPMixer | MLP(wavelet) | Parallel-Merge | Independent | Wavelet | Wavelet | InvDWT |
| FiLM | HiPPO+SpectConv | Parallel-Merge | Independent | MultiLookback | Legendre | BasisRecon |
| FreTS | FreqMLP | SeqChain+Skip | OptFFT-Mix | Single | Freq | Direct-Proj |
| TimesNet | 2D-InceptConv | Parallel(periods) | Mix-Embed | FFT-Period | 2D-Spatial | Direct-Proj |
| MICN | 1D-Conv(DUU) | Parallel-Merge | Mix-Embed | MultiKernel | Time | Decomp-Add |
| SCINet | 1D-CausalConv | RecursiveTree | Mix-RawConv | BinaryTree | Time | Direct-Proj |
| MSGNet | GraphConv+Attn | Parallel(periods) | Graph-GCN | FFT-Period | 2D-Spatial | Direct-Proj |
| Mamba | SSM(lib) | SeqChain | Mix-Embed | Single | Time | Direct-Proj |
| MambaSimple | SSM(custom) | SeqChain | Mix-Embed | Single | Time | Direct-Proj |
| SegRNN | GRU(seg) | SeqChain | Indep+Embed | Segment | Time | SegDecode |
| Koopa | Koopman+DMD | DualStream+Iter | Mixed | DualStream | Koopman | IterAccum |
| KANAD | 1D-Conv+Cos | SeqChain | Independent | Single | Time | Direct-Proj |
| TimeFilter | GCN+MoE | SeqChain | Graph-Learned | Single-Patch | Time | Direct-Proj |
| Chronos | T5-EncDec | Enc-Dec | Univariate | Single | Time | AR-TokenDecode(categorical) |
| Chronos-Bolt | T5-EncDec(1tok) | EncDec(single-pass) | Univariate | Patch | Time | SinglePass-Quantile |
| Chronos-2 | T5-Enc+GroupAttn | Enc-Only | GroupAttn-CrossVar | Patch | Time | SinglePass-Quantile |
| Moirai | MaskedEnc+AVA | Enc-Only | AVA-CrossVar | MultiPatch(MPSP) | Time | MixtureDistrib |
| Sundial | CausalTrans+Flow | TwoComponent | Univariate | Patch | Time+Latent | FlowODE-Generative |
| TiRex | sLSTM-only(block-diag) | SeqChain | Univariate | Patch | Time | SinglePass-Quantile(NaNRollout) |
| TimesFM | DecOnly-Trans | SeqChain | Univariate | MultiOutputPatch | Time | SemiAR-AsymPatch |
| TimeMoE | DecOnly+SparseMoE+Shared | SeqChain | Univariate | PointWise+MultiResHead | Time | MultiResHead-DynSchedule |

---

## Key Findings

### 1. Naming vs. Structure Disconnect
Model names frequently misrepresent computational structure. "Transformer" models span at
least 5 distinct attention mechanisms. "MLP-based" models range from zero-nonlinearity linear
maps (DLinear) to sophisticated multi-scale wavelet-domain mixers (WPMixer).

### 2. Channel Handling Is the Primary Structural Divide
The most impactful architectural decision is how inter-variate relationships are modeled:
- **6 foundation models** are strictly univariate by design (Chronos, Chronos-Bolt, Sundial, TiRex, TimesFM, TimeMoE)
- **13 non-foundation models** are strictly channel-independent
- **13 models** mix channels only through the input embedding (Conv1d)
- **10 models** have explicit cross-variable mechanisms (including Moirai's AVA and Chronos-2's Group Attention)
- **2 models** learn graph structure between variables
- **2 models** are configurable

### 3. Multi-Scale Processing Is Diverse
When models do multi-scale processing, they use fundamentally different mechanisms:
downsampling pyramids, FFT-adaptive periods, wavelet decomposition, trend-seasonal decomposition,
multi-patch sizes, recursive binary splitting, or multiple lookback windows. There is no
convergence on a standard multi-scale approach.

### 4. Frequency-Domain Operations Serve Two Distinct Roles
- **Core computation**: Autoformer, FEDformer, FreTS, FiLM -- the main sequence mixing
  happens in frequency/spectral space
- **Preprocessing/auxiliary**: TimesNet, MSGNet, Koopa, TimeMixer -- FFT is used for period
  detection or signal decomposition, but the main computation is in time/spatial domain

### 5. True Encoder-Decoder Architectures Are Becoming Rare
Among task-specific models, encoder-only or encoder-with-direct-projection is the dominant
pattern (PatchTST, iTransformer, TimeMixer, TimeXer, etc.). Full encoder-decoder with
cross-attention is mainly found in older Transformer variants and TFT. Among foundation
models, Chronos (T5 enc-dec) and Sundial (transformer + flow matching) are the exceptions;
the rest are decoder-only or encoder-only.

### 6. Foundation Models Converge on Patch-Autoregressive Design
Despite diverse naming, most foundation models share a causal backbone with autoregressive
multi-step prediction. They diverge primarily in:
- Tokenization: Patch-based (TimesFM, TiRex, Sundial) vs. Point-wise (TimeMoE) vs. Discrete bins (Chronos)
- Backbone choice: Transformer (TimesFM, TimeMoE, Sundial), sLSTM (TiRex), T5 (Chronos family)
- Scaling strategy: Dense (TimesFM, TiRex) vs. Sparse MoE + shared expert (TimeMoE)
- Decoding: Autoregressive (Chronos, TimesFM), single-pass with NaN-rollout (TiRex),
  flow-based generative (Sundial), single-pass quantile (Chronos-Bolt, Chronos-2)
- Output distribution: Point prediction (TimesFM v1), Quantile (Chronos-Bolt, Chronos-2),
  Mixture (Moirai), Flow-based (Sundial), Categorical over bins (Chronos)

### 7. Structural Simplicity Can Match Complexity
The simplest models (DLinear: 2 linear layers; PAttn: 1 attention layer) remain competitive
benchmarks. This suggests that the inductive biases (decomposition, normalization, patching)
matter as much as or more than the core computation primitive.

---

## 12. Phylogenetic Analysis

> A systematic reconstruction of architectural ancestry and divergence, treating each model
> as a taxon and structural features as heritable characters. This analysis identifies
> monophyletic clades (true lineages), convergent evolution (homoplasy), horizontal transfer
> events (trait borrowing across lineages), and vestigial structures.

### 12.1 Methodology

**Outgroup**: Classical statistical methods (ARIMA, ETS, Exponential Smoothing) — no learned
parameters, no neural computation. Used to root the tree and polarize character states.

**Character system**: 12 discrete structural characters derived from the 6-axis taxonomy
(Section 1). Character states are coded as ordered or unordered multi-state. The tree is
constructed via **maximum parsimony** — minimizing the total number of character state changes
needed to explain the observed distribution of traits.

**Reticulation**: Unlike biological evolution, architectural evolution is **reticulate** —
models openly borrow innovations from unrelated lineages. The primary tree represents the
dominant structural ancestry; dashed annotations mark horizontal transfer events.

### 12.2 Character State Matrix

Each model is scored on 12 phylogenetically informative structural characters:

```
Character Key:
  A: Primary Computation      0=Linear  1=Conv  2=Recurrent  3=Attention  4=SSM  5=Graph  6=Operator
  B: Encoder-Decoder          0=No (enc-only/chain)  1=Yes (cross-attn)  2=Two-component
  C: Patching                 0=None  1=Single-scale  2=Multi-scale  3=Asymmetric
  D: Channel Treatment        0=Independent  1=Embed-mix  2=Explicit-cross  3=Graph
  E: Decomposition            0=None  1=Input-only  2=Progressive (every layer)
  F: Frequency-Domain Ops     0=None  1=Auxiliary  2=Core
  G: Recurrent Component      0=None  1=LSTM/GRU  2=SSM  3=sLSTM
  H: Instance Normalization   0=No  1=Yes
  I: Multi-Scale Processing   0=Single  1=Yes
  J: Autoregressive Output    0=No (direct)  1=Full-AR  2=Semi-AR  3=Distributional
  K: Pre-trained Foundation   0=No  1=Yes
  L: Learned Graph Structure  0=No  1=Yes
```

```
                               A  B  C  D  E  F  G  H  I  J  K  L
                              ── ── ── ── ── ── ── ── ── ── ── ──
Transformer                    3  1  0  1  0  0  0  0  0  0  0  0
Informer                       3  1  0  1  0  0  0  1  1  0  0  0
Reformer                       3  0  0  1  0  0  0  1  0  0  0  0
Autoformer                     3  1  0  1  2  2  0  0  1  0  0  0
FEDformer                      3  1  0  1  2  2  0  0  1  0  0  0
ETSformer                      3  1  0  1  2  2  0  0  1  0  0  0
NS_Transformer                 3  1  0  1  0  0  0  1  0  0  0  0
Pyraformer                     3  0  0  1  0  0  0  0  1  0  0  0
Crossformer                    3  1  1  2  0  0  0  0  1  0  0  0
PatchTST                       3  0  1  0  0  0  0  1  0  0  0  0
PAttn                          3  0  1  0  0  0  0  1  0  0  0  0
iTransformer                   3  0  0  2  0  0  0  1  0  0  0  0
TimeXer                        3  0  1  2  0  0  0  1  0  0  0  0
MultiPatchFormer               3  0  2  2  0  0  0  1  1  2  0  0
TimeMixer                      0  0  0  0  2  1  0  1  1  0  0  0
TFT                            3  1  0  2  0  0  1  0  0  0  0  0
DLinear                        0  0  0  0  1  0  0  0  0  0  0  0
LightTS                        0  0  0  0  0  0  0  0  1  0  0  0
TiDE                           0  1  0  0  0  0  0  1  0  0  0  0
TSMixer                        0  0  0  2  0  0  0  0  0  0  0  0
FreTS                          0  0  0  2  0  2  0  0  0  0  0  0
WPMixer                        0  0  1  0  0  0  0  1  1  0  0  0
FiLM                           0  0  0  0  0  2  2  1  1  0  0  0
TimesNet                       1  0  0  1  0  1  0  1  1  0  0  0
MICN                           1  0  0  1  1  0  0  0  1  0  0  0
SCINet                         1  0  0  1  0  0  0  0  1  0  0  0
KANAD                          1  0  0  0  0  0  0  0  0  0  0  0
MSGNet                         5  0  0  3  0  1  0  1  1  0  0  1
TimeFilter                     5  0  1  3  0  0  0  1  0  0  0  1
Mamba                          4  0  0  1  0  0  2  1  0  0  0  0
MambaSimple                    4  0  0  1  0  0  2  1  0  0  0  0
SegRNN                         2  0  0  0  0  0  1  0  0  0  0  0
Koopa                          6  0  0  0  0  1  0  1  1  0  0  0
Chronos                        3  1  0  0  0  0  0  1  0  1  1  0
Chronos-Bolt                   3  1  1  0  0  0  0  1  0  3  1  0
Chronos-2                      3  0  1  2  0  0  0  1  0  3  1  0
Moirai                         3  0  2  2  0  0  0  1  1  3  1  0
Sundial                        3  2  1  0  0  0  0  1  0  3  1  0
TiRex                          2  0  1  0  0  0  3  1  0  3  1  0
TimesFM                        3  0  3  0  0  0  0  1  1  2  1  0
TimeMoE                        3  0  0  0  0  0  0  1  1  2  1  0
```

### 12.3 The Phylogenetic Tree

The following cladogram represents the maximum-parsimony reconstruction of architectural
lineage. Branch lengths are not proportional to time or divergence — this is a topology-only
tree. Annotations at nodes indicate **synapomorphies** (shared derived traits) that define
each clade.

Symbols: `†` = horizontal transfer event, `‡` = convergent trait loss, `§` = vestigial
character, `*` = polyphyletic placement.

```
OUTGROUP: Statistical Methods (ARIMA, ETS)
│
NEURAL SEQUENCE MODELS (stem: learned parameters + gradient descent)
│
├──── RECURRENT CLADE ·································· [Synapomorphy: sequential hidden state]
│     │
│     ├──── Classical RNN/LSTM (stem group, not in repo)
│     │     │
│     │     ├──── SegRNN ······························· segment-level GRU
│     │     │
│     │     └──── TFT ································· LSTM + interpretable attention †Transformer
│     │
│     ├──── Structured SSM (S4/HiPPO lineage)
│     │     │
│     │     ├──── FiLM ································ HiPPO-LegT + spectral conv
│     │     │
│     │     └──── Selective SSM
│     │           ├──── Mamba ·························· library-based selective scan
│     │           └──── MambaSimple ···················· from-scratch selective scan
│     │
│     └──── xLSTM (modernized LSTM)
│           └──── TiRex ······························· sLSTM-only, NaN-rollout, quantile output
│
├──── TRANSFORMER RADIATION ···························· [Synapomorphy: scaled dot-product attention]
│     │
│     ├──── STEM TRANSFORMERS ·························· [Synapomorphy: encoder-decoder + cross-attention]
│     │     │
│     │     ├──── Transformer ·························· ancestral form (vanilla enc-dec)
│     │     │
│     │     ├──── EFFICIENCY SUBCLADE ·················· [Synapomorphy: sub-quadratic attention]
│     │     │     ├──── Informer ······················· ProbSparse + distillation pyramid
│     │     │     ├──── Reformer ······················· LSH attention (decoder lost ‡)
│     │     │     └──── Pyraformer ····················· pyramid sparse mask (decoder lost ‡)
│     │     │
│     │     ├──── DECOMPOSITION SUBCLADE ··············· [Synapomorphy: progressive trend-seasonal in enc+dec]
│     │     │     ├──── Autoformer ····················· FFT cross-correlation replaces attention
│     │     │     ├──── FEDformer ····················· Fourier/Wavelet blocks replace attention
│     │     │     └──── ETSformer ····················· exponential smoothing (attention §vestigial)
│     │     │
│     │     ├──── NS_Transformer ······················ derived: tau/delta de-stationary modulation
│     │     │
│     │     └──── Crossformer ························· derived: two-stage (time+dim) + seg-merging
│     │
│     ├──── CROWN TRANSFORMERS ························ [Synapomorphy: decoder lost, direct projection]
│     │     │
│     │     ├──── PATCH SUBCLADE ······················ [Synapomorphy: patching + channel independence]
│     │     │     ├──── PatchTST ······················ overlapping patches, chan-indep
│     │     │     │     └──── PAttn ··················· reduced to 1 encoder layer
│     │     │     ├──── TimeXer ······················· patches + exogenous cross-attention
│     │     │     └──── MultiPatchFormer ·············· multi-scale patches + channel mixing stage
│     │     │
│     │     └──── INVERTED SUBCLADE
│     │           └──── iTransformer ·················· variates-as-tokens (axis inversion)
│     │
│     └──── FOUNDATION TRANSFORMERS ··················· [Synapomorphy: large-scale pre-training + zero-shot]
│           │
│           ├──── T5 LINEAGE ·························· [Synapomorphy: T5 backbone]
│           │     ├──── Chronos ······················· enc-dec, discrete bin tokenization, full AR
│           │     ├──── Chronos-Bolt ·················· enc-dec (1-token decoder), patches, quantile
│           │     └──── Chronos-2 ····················· enc-only (decoder lost ‡), Group Attention
│           │
│           ├──── MASKED ENCODER
│           │     └──── Moirai ························ Any-Variate Attention, mixture distributions
│           │
│           └──── CAUSAL DECODER-ONLY
│                 ├──── TimesFM ······················· asymmetric patching, semi-AR
│                 ├──── TimeMoE ······················· sparse MoE + shared expert, point-wise tokens
│                 └──── Sundial ······················· + flow matching decoder (two-component)
│
├──── LINEAR/MLP COUNTER-REVOLUTION ··················· [Synapomorphy: no attention mechanism]
│     │
│     ├──── PURE LINEAR
│     │     └──── DLinear ····························· 2 linear layers + decomposition
│     │
│     ├──── MLP CHAIN
│     │     ├──── TiDE ································ ResBlock encoder-decoder
│     │     └──── TSMixer ····························· temporal + channel MLP alternating
│     │
│     ├──── DUAL-SAMPLING
│     │     └──── LightTS ····························· continuous + interval chunk sampling
│     │
│     └──── DOMAIN-SHIFTED MLP
│           ├──── FreTS ······························· FFT + complex linear
│           └──── WPMixer ····························· wavelet domain + mixer
│
├──── CONVOLUTION CLADE ······························· [Synapomorphy: conv as primary primitive]
│     │
│     ├──── 1D TEMPORAL
│     │     ├──── SCINet ······························ recursive binary tree, multiplicative coupling
│     │     ├──── MICN ································ multi-scale down-iso-up sandwich
│     │     └──── KANAD ······························· cosine basis expansion + conv
│     │
│     └──── 2D SPATIAL (FFT-RESHAPED)
│           └──── TimesNet ···························· FFT period detect + 2D Inception
│
├──── GRAPH CLADE ····································· [Synapomorphy: learned graph topology over variates]
│     ├──── MSGNet ···································· FFT periods + GCN diffusion †Attention †FFT
│     └──── TimeFilter ································ bilinear graph + MoE structured masks
│
├──── OPERATOR-THEORETIC
│     └──── Koopa ····································· Koopman operators + DMD least-squares
│
└──── MULTI-SCALE MIXING
      └──── TimeMixer ································· downsample pyramid + bidirectional MLP mixing *
```

`*` TimeMixer is phylogenetically ambiguous — its MLP core places it near the Linear clade,
but its multi-scale decomposition architecture is a derived character unique to this lineage.

### 12.4 Clade Descriptions and Synapomorphies

#### Recurrent Clade
**Defining synapomorphy**: Sequential hidden state passed between timesteps. Information
from the past is compressed into a fixed-dimensional state vector.

Three distinct sub-lineages:
- **Classical RNN/LSTM** → SegRNN (segments the sequence for efficiency), TFT (hybrid
  with attention borrowed from Transformer clade — horizontal transfer)
- **Structured SSM** → FiLM (HiPPO basis projects onto Legendre polynomials),
  Mamba/MambaSimple (selective state space with input-dependent dynamics)
- **xLSTM** → TiRex (uses only sLSTM variant; mLSTM ablated away as deleterious)

The Recurrent clade is **paraphyletic** with respect to the Transformer clade — TFT
bridges both lineages, and the SSM sub-lineage represents a fundamentally different
mathematical framework (continuous-time ODEs discretized) despite sharing the sequential
processing phenotype.

#### Stem Transformers (Encoder-Decoder Retained)
**Defining synapomorphy**: Encoder-decoder architecture with cross-attention linking
encoder context to decoder output generation. This is the **ancestral state** inherited
from the original Transformer (Vaswani et al., 2017).

This clade underwent rapid diversification (2020-2022), producing three major sub-lineages:
- **Efficiency subclade**: Addressed O(L^2) complexity — ProbSparse (Informer), LSH
  (Reformer), pyramid masking (Pyraformer). Both Reformer and Pyraformer convergently
  lost the decoder.
- **Decomposition subclade**: Incorporated trend-seasonal decomposition into every
  encoder and decoder layer. Progressively replaced attention with frequency-domain
  operations, culminating in ETSformer where attention is vestigial (replaced by
  exponential smoothing).
- **Single-innovation offshoots**: NS_Transformer (de-stationary modulation) and
  Crossformer (two-stage attention + segment merging) each add one major derived trait.

#### Crown Transformers (Encoder-Only)
**Defining synapomorphy**: Loss of the decoder. Output is generated by direct projection
from encoder representations, not by autoregressive or cross-attended decoding.

This represents the dominant modern topology. The **Patch subclade** (PatchTST, PAttn,
TimeXer, MultiPatchFormer) combines decoder loss with patching and (initially) channel
independence — this combination proved highly effective and triggered a rapid radiation.
The **Inverted subclade** (iTransformer) is a singleton that independently lost the
decoder while evolving the unique axis-inversion trait.

#### Foundation Transformers
**Defining synapomorphy**: Large-scale pre-training enabling zero-shot forecasting on
unseen datasets. This is a behavioral/training-regime synapomorphy rather than a purely
structural one.

Three structural sub-lineages:
- **T5 lineage** (Chronos → Chronos-Bolt → Chronos-2): A clear anagenetic series.
  Chronos inherited T5's enc-dec; Chronos-Bolt optimized it (single-token decoder);
  Chronos-2 lost the decoder and gained Group Attention (cross-series capability).
- **Masked encoder** (Moirai): Independent origin. Any-Variate Attention is a unique
  derived trait enabling arbitrary variate counts.
- **Causal decoder-only** (TimesFM, TimeMoE, Sundial): Share the decoder-only Transformer
  backbone. Diverge in tokenization (patches vs. point-wise), output (semi-AR vs.
  multi-resolution vs. flow-based), and computation (dense vs. sparse MoE).

Note: **TiRex** is a foundation model by training regime but belongs to the **Recurrent
clade** by structural ancestry (sLSTM backbone). This is analogous to whales being mammals
despite living in the ocean — behavioral convergence does not override structural lineage.

#### Linear/MLP Counter-Revolution
**Defining synapomorphy**: Complete absence of attention mechanism. This clade emerged
as a reaction to Transformer dominance, demonstrating that simpler architectures can be
competitive (DLinear, 2022). The clade subsequently diversified into MLP chains (TiDE,
TSMixer), dual-sampling strategies (LightTS), and domain-shifted variants that perform
MLP operations in frequency (FreTS) or wavelet (WPMixer) space.

#### Convolution Clade
**Defining synapomorphy**: Convolution as the primary sequence-mixing primitive. Splits
into 1D temporal convolution (SCINet, MICN, KANAD) and 2D spatial convolution after
FFT-based temporal reshape (TimesNet). The 2D branch represents a novel derived trait —
no other clade reshapes 1D time series into 2D tensors for processing.

#### Graph Clade
**Defining synapomorphy**: Learned graph topology between variates. MSGNet and TimeFilter
both construct adaptive adjacency matrices and propagate information via graph convolution.
MSGNet additionally borrows FFT period detection (from TimesNet's lineage) and attention
(from the Transformer clade) — a heavily polyphyletic taxon.

### 12.5 Key Evolutionary Innovations

Major innovations that triggered adaptive radiations or opened new ecological niches,
ordered chronologically:

```
2017 ─── Scaled Dot-Product Attention ·········· Enabled the entire Transformer radiation
         (Vaswani et al.)                        40+ architectural descendants in 8 years

2020 ─── Approximate Attention ················· First attempt to break O(L²) barrier
         (Informer, Reformer)                    Spawned efficiency subclade; later superseded

2021 ─── Progressive Decomposition ············· Trend-seasonal split in every layer
         (Autoformer)                            Led to FEDformer, ETSformer variants

2021 ─── HiPPO / Structured SSMs ·············· Continuous-time state space models
         (S4, FiLM)                              Parallel to attention; led to Mamba/SSM clade

2022 ─── Channel Independence ················· Counter-intuitive: ignoring cross-var helps
         (DLinear, PatchTST)                     Adopted across attention, linear, recurrent

2022 ─── Linear Sufficiency ··················· Proved linear maps are competitive baselines
         (DLinear)                                Triggered MLP counter-revolution clade

2023 ─── Patching ····························· Reduce O(L²) to O((L/P)²); captures locality
         (PatchTST)                              Adopted across all major clades; universal

2023 ─── Selective State Spaces ··············· Input-dependent SSM dynamics
         (Mamba)                                  Bridged SSM and attention paradigms

2024 ─── Foundation Pre-training ·············· Large-scale pre-training for zero-shot
         (Chronos, TimesFM, Moirai)              Independent origins in 3+ groups

2024 ─── Any-Variate Attention (AVA) ·········· Handle arbitrary variate count via masking
         (Moirai)                                 Structural solution to variable-count problem

2024 ─── Sparse MoE for Time Series ··········· Conditional computation in TS foundation models
         (TimeMoE, Moirai-MoE)                   Scale parameters without proportional compute

2025 ─── Flow Matching Decoder ················ Non-parametric probabilistic output via ODE
         (Sundial)                                Can model arbitrary distribution shapes

2025 ─── Group Attention ····················· Cross-series attention via group IDs
         (Chronos-2)                              Enables multivariate in an encoder-only model
```

### 12.6 Convergent Evolution (Homoplasy)

The following traits evolved **independently** in multiple unrelated lineages, representing
functional convergence rather than shared ancestry:

#### 1. Loss of the Decoder
Independently evolved at least 6 times:
- Reformer (Efficiency subclade) — replaced with encoder + appended placeholders
- Pyraformer (Efficiency subclade) — last-hidden-state projection
- PatchTST (Crown Transformers) — flatten + linear head
- iTransformer (Crown Transformers) — direct per-variate projection
- Chronos-2 (Foundation T5 lineage) — encoder-only with quantile heads
- Moirai (Foundation masked encoder) — born without a decoder

**Functional explanation**: Encoder-only architectures are simpler, faster, and avoid
error accumulation from autoregressive decoding. The decoder was the ancestral state
inherited from machine translation; time-series forecasting does not require it.

#### 2. Channel Independence
Independently adopted by models from 4+ clades:
- PatchTST, PAttn (Crown Transformers)
- DLinear, TiDE (Linear/MLP)
- SegRNN (Recurrent)
- FiLM (SSM)
- WPMixer (Domain-shifted MLP)
- All univariate foundation models

**Functional explanation**: Cross-variable interactions learned from limited data often
capture noise rather than signal. Channel independence acts as a regularizer.

#### 3. Patching / Segmentation
Independently evolved in:
- PatchTST, TimeXer, MultiPatchFormer (Crown Transformers)
- Crossformer (Stem Transformer)
- WPMixer (Domain-shifted MLP)
- SegRNN (Recurrent)
- All patch-based foundation models (TiRex, TimesFM, Moirai, Sundial, Chronos-Bolt/2)

**Functional explanation**: Patching reduces sequence length (computational savings),
captures local temporal patterns, and creates a natural "token" unit for sequence models.

#### 4. Instance Normalization (RevIN-style)
Adopted across nearly all clades: Transformers, Linear/MLP, SSM, Recurrent, and
Foundation models. Present in 24+ of 40 models. This is the most widespread convergent
trait — a near-universal adaptation.

#### 5. FFT for Period Detection
Used by TimesNet (Convolution clade) and MSGNet (Graph clade) for identical purpose:
detect dominant periods via `torch.fft.rfft`, then reshape or window the time series
accordingly. Clearly independent origins despite identical implementation.

#### 6. Multi-Resolution Output
TimesFM (multiple output patch sizes {1,8,...,128}) and TimeMoE (multi-resolution heads
[1,8,32,64]) independently evolved the same strategy: multiple output heads predicting
different horizon lengths, with dynamic selection at inference.

### 12.7 Reticulate Events (Horizontal Transfer)

Models that borrowed major structural innovations across clade boundaries:

| Model | Home Clade | Borrowed Trait | Source Clade |
|-------|-----------|---------------|-------------|
| **TFT** | Recurrent (LSTM) | Interpretable multi-head attention | Transformer |
| **MSGNet** | Graph | Full attention within period windows | Transformer |
| **MSGNet** | Graph | FFT period detection | Convolution (TimesNet) |
| **MultiPatchFormer** | Crown Transformer | Channel-wise mixing stage | Linear/MLP (TSMixer-like) |
| **Crossformer** | Stem Transformer | Segment merging (downsampling) | Convolution |
| **ETSformer** | Decomposition subclade | Exponential smoothing (replaced attention) | Statistical methods |
| **Koopa** | Operator-theoretic | FFT preprocessing (FourierFilter) | Frequency-domain |
| **Sundial** | Foundation Transformer | Flow matching (generative) | Generative models (external) |
| **TimeMoE** | Foundation Transformer | Sparse MoE architecture | LLM scaling (DeepSeekMoE) |

### 12.8 Vestigial Characters and Functional Loss

In biological evolution, vestigial structures are remnants of ancestral traits that have
lost their original function. Several architectural analogs exist:

| Model | Vestigial Structure | Ancestral Function | Current State |
|-------|--------------------|--------------------|--------------|
| **ETSformer** | Attention mechanism | Primary sequence mixing | Replaced by exponential smoothing; attention scores exist but are not the dominant computation path |
| **Reformer** | Decoder architecture | Autoregressive output generation | Decoder dropped; encoder appends placeholder tokens and projects directly |
| **Chronos-Bolt** | T5 decoder | Autoregressive token generation | Reduced to single-token decoder (cross-attends once, then projects). Functionally non-autoregressive |
| **LightTS** | Channel mixing layer | Cross-variable interaction | Identity-initialized linear (starts as no-op); may or may not learn meaningful mixing |
| **Koopa** | Attention in original paper | Sequence modeling | Completely removed in this implementation; pure Koopman/DMD |

### 12.9 Adaptive Radiations and Extinction Patterns

#### Radiations (rapid diversification events)

**The Transformer Radiation (2020-2022)**: The original Transformer spawned ~10 variants
in 2 years, each modifying a different structural element (attention mechanism, decoder,
decomposition, masking). This is analogous to the Cambrian explosion — a single body plan
rapidly diversifying to fill ecological niches.

**The Patching Radiation (2023-2024)**: After PatchTST demonstrated the effectiveness of
patching + channel independence, rapid diversification produced PAttn, TimeXer,
MultiPatchFormer, and influenced foundation model designs. Patching became a near-universal
trait adopted across clades.

**The Foundation Model Radiation (2024-2025)**: At least 7 independent foundation models
emerged within ~18 months, from 4 different structural sub-lineages (T5-based,
masked-encoder, decoder-only, recurrent). This represents parallel evolution toward the
same ecological niche (zero-shot forecasting) from diverse structural origins.

#### Extinctions (structural patterns that declined)

**Full Encoder-Decoder** (declining): Among post-2023 models, almost none use the
original encoder-decoder + cross-attention topology for task-specific forecasting.
Encoder-only with direct projection has largely replaced it. The enc-dec topology
survives primarily in foundation models (Chronos, Sundial) where it serves different
purposes (autoregressive generation, conditioning).

**Approximate Attention** (largely extinct): The ProbSparse (Informer) and LSH (Reformer)
approaches to sub-quadratic attention have been superseded by patching, which achieves
the same complexity reduction with better performance. No post-2023 model uses these
mechanisms.

**Per-Timestep Processing** (declining): Processing each individual timestep as a
separate token (the ancestral Transformer approach) is being replaced by patch-level
processing. Only TimeMoE (point-wise tokens) and original Chronos (per-bin tokens)
retain per-timestep tokenization among recent models.

### 12.10 Chronostratigraphic Summary

Mapping architectural lineage against publication year reveals waves of innovation:

```
             2017    2019    2020    2021    2022    2023    2024    2025
              │       │       │       │       │       │       │       │
RECURRENT     │       │       │       │  TFT  │       │SegRNN │       │
              │       │       │       │       │ FiLM  │       │       │
SSM           │       │       │       │       │       │ Mamba  │MambaS │
xLSTM         │       │       │       │       │       │       │ TiRex │
              │       │       │       │       │       │       │       │
STEM TRANS  Trans   │     Infmr  │ Autfmr│NS_Trn│Crssfmr│       │       │
              │       │    Rfrmr │ FEDfmr│ETSfmr│       │       │       │
              │       │       │  Pyrafmr│       │       │       │       │
              │       │       │       │       │       │       │       │
CROWN TRANS   │       │       │       │PtchTST│ iTrns │TimeXer│       │
              │       │       │       │       │       │ PAttn │       │
              │       │       │       │       │       │ MPFmr │       │
              │       │       │       │       │       │       │       │
FOUNDATION    │       │       │       │       │       │Chronos│Chrn-2 │
              │       │       │       │       │       │ChrBolt│Sundial│
              │       │       │       │       │       │Moirai │       │
              │       │       │       │       │       │TimesFM│       │
              │       │       │       │       │       │TimeMoE│       │
              │       │       │       │       │       │       │       │
LINEAR/MLP    │       │       │       │       │DLinear│ TiDE  │       │
              │       │       │       │       │LghtTS │TSMixer│       │
              │       │       │       │       │       │ FreTS │WPMxr  │
              │       │       │       │       │       │       │       │
CONV          │       │       │       │       │SCINet │TmsNet │ KANAD │
              │       │       │       │       │ MICN  │       │       │
              │       │       │       │       │       │       │       │
GRAPH         │       │       │       │       │       │MSGNet │TmFltr │
OPERATOR      │       │       │       │       │       │ Koopa │       │
MULTI-SCALE   │       │       │       │       │       │TmMixr │       │
```

**Observed pattern**: The field progresses in waves — Transformer variants dominated
2020-2022, the Linear/MLP counter-revolution peaked 2022-2023, and Foundation models
dominate 2024-2025. Each wave partially displaces the previous one but does not eliminate
it entirely, leaving a diverse ecosystem of coexisting architectures.
