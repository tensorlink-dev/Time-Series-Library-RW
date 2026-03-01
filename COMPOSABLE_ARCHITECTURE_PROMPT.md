# Prompt: Build a Composable, Scalable Time-Series Architecture Library

## Context for the Implementing Agent

You are building a **composable time-series neural architecture library** from the ground up.
This library decomposes 40 published forecasting architectures into their atomic structural
primitives, defines standard interfaces between them, and allows any primitive to be swapped,
scaled, or combined with any other — enabling both faithful reproduction of known architectures
and principled exploration of novel ones.

The design is guided by a deep phylogenetic analysis of all 40 architectures (appended as
reference material). Every design decision below is grounded in that analysis.

---

## 1. Design Philosophy

### 1.1 The Genome Metaphor

Treat each architecture as a **species** defined by a **genome** — a specific combination of
discrete structural genes. The library implements the genes (primitives), the body plans
(DAG topologies), and the regulatory mechanisms (cross-cutting policies). A species is just
a config file that selects one allele per gene.

### 1.2 Core Principles

1. **Primitive-first**: Every computation in every model reduces to one of 8 atomic primitives
   (Section 3). Build these first. Everything else composes them.
2. **Protocol-based composition**: All primitives implement the same tensor protocol
   (Section 4). Any primitive can be dropped into any slot in any topology.
3. **Scaling is a knob, not a rewrite**: Every primitive and every species has a `ScaleConfig`
   that smoothly interpolates from ~1M to ~2B+ parameters by adjusting width, depth, heads,
   and expert count. No architectural changes needed.
4. **Phylogenetic reasoning**: The library's structure mirrors the phylogenetic tree. New
   species are created by recombining traits from existing clades, guided by the character
   state matrix (Section 10).

---

## 2. Scaling Architecture

Every component must be parameterized by a `ScaleConfig` that defines its size. The library
should support named presets and arbitrary interpolation:

```python
@dataclass
class ScaleConfig:
    d_model: int          # Hidden dimension (core width)
    n_layers: int         # Depth (number of stacked blocks)
    n_heads: int          # Attention heads (ignored by non-attention primitives)
    d_ff: int             # FFN intermediate dim (typically 4 * d_model)
    d_state: int          # State dimension (for SSM/recurrent primitives)
    n_experts: int        # MoE expert count (1 = dense, >1 = sparse)
    top_k: int            # MoE top-k routing (ignored if n_experts=1)
    patch_size: int       # Temporal patch size (0 = per-timestep)
    dropout: float        # Dropout rate

# Named presets — every species can use these
SCALE_PRESETS = {
    "tiny":   ScaleConfig(d_model=64,   n_layers=2,  n_heads=2,  d_ff=256,   d_state=16,  n_experts=1, top_k=1, patch_size=16, dropout=0.1),
    "small":  ScaleConfig(d_model=128,  n_layers=4,  n_heads=4,  d_ff=512,   d_state=32,  n_experts=1, top_k=1, patch_size=16, dropout=0.1),
    "base":   ScaleConfig(d_model=256,  n_layers=6,  n_heads=8,  d_ff=1024,  d_state=64,  n_experts=1, top_k=1, patch_size=16, dropout=0.1),
    "large":  ScaleConfig(d_model=512,  n_layers=12, n_heads=16, d_ff=2048,  d_state=128, n_experts=1, top_k=1, patch_size=16, dropout=0.05),
    "xl":     ScaleConfig(d_model=1024, n_layers=24, n_heads=32, d_ff=4096,  d_state=256, n_experts=1, top_k=1, patch_size=32, dropout=0.05),
    "moe-base": ScaleConfig(d_model=256, n_layers=6, n_heads=8, d_ff=1024, d_state=64, n_experts=8, top_k=2, patch_size=16, dropout=0.1),
    "moe-xl":   ScaleConfig(d_model=1024, n_layers=24, n_heads=32, d_ff=4096, d_state=256, n_experts=8, top_k=2, patch_size=32, dropout=0.05),
}
```

**Critical constraint**: Changing `ScaleConfig` must NEVER require changing code — only the
config values. The architecture adapts automatically to any valid config.

---

## 3. Atomic Primitives (The "Genes")

These are the 8 irreducible computation types found across all 40 architectures. Each is a
`nn.Module` that implements the `SequencePrimitive` protocol (Section 4).

### 3.1 Attention Variants

Build a single `Attention` module with a `mode` parameter that selects the variant:

| Variant | Mode String | Key Mechanism | Complexity | Source Species |
|---------|-------------|---------------|------------|----------------|
| Full dot-product | `"full"` | Standard `softmax(QK^T/√d)V` | O(L^2) | Transformer, PatchTST, PAttn, iTransformer |
| ProbSparse | `"prob_sparse"` | Top-u query selection via KL-divergence sampling | O(L log L) | Informer |
| LSH | `"lsh"` | Locality-sensitive hashing buckets | O(L log L) | Reformer |
| AutoCorrelation | `"auto_corr"` | FFT cross-correlation + top-k delay aggregation | O(L log L) | Autoformer |
| Fourier | `"fourier"` | Learnable complex weights on selected frequency modes | O(L log L) | FEDformer |
| Wavelet | `"wavelet"` | Multi-wavelet transform with Legendre basis | O(L) | FEDformer (wavelet mode) |
| Pyramid | `"pyramid"` | Structured sparse mask (parent-child + sibling) | O(L) | Pyraformer |
| DS (De-Stationary) | `"ds"` | Full attention + learnable tau-scaling and delta-shift | O(L^2) | NS_Transformer |
| Two-Stage | `"two_stage"` | Temporal self-attn + router-mediated cross-dim attn | O(L^2) | Crossformer |
| Interpretable | `"interpretable"` | Shared V, averaged heads, causal mask | O(L^2) | TFT |
| Group | `"group"` | Alternating Time Attn + cross-series Group Attn | O(L^2) | Chronos-2 |
| Any-Variate (AVA) | `"ava"` | Structured mask: temporal + cross-variate + causal | O((V*L)^2) | Moirai |

**Masking support**: All variants accept an optional `mask` tensor and a `causal: bool` flag.
When `causal=True`, apply causal (triangular) masking. When a custom mask is provided, apply it
additively before softmax.

**Implementation guidance**:
- All variants share the same `__init__` signature: `(d_model, n_heads, dropout, **variant_kwargs)`
- All variants implement `forward(Q, K, V, mask=None) -> (output, attn_weights)`
- `auto_corr` and `fourier` use `torch.fft.rfft/irfft` internally
- `ava` and `group` construct their structured masks from a `variate_ids` tensor

### 3.2 Convolution Variants

| Variant | Mode String | Key Mechanism | Source Species |
|---------|-------------|---------------|----------------|
| 1D temporal | `"conv1d"` | Standard `nn.Conv1d` (variable kernel) | MICN, KANAD |
| 1D causal | `"causal_conv1d"` | Left-padded Conv1d for strict causality | SCINet, MambaSimple |
| 2D Inception | `"inception2d"` | Parallel multi-kernel (1x1, 3x3, 5x5, ...) + concat | TimesNet |
| Depthwise | `"depthwise"` | Per-channel conv (groups=channels) | MambaSimple (internal) |

**Interface**: `(d_model, kernel_size, stride, dilation, causal, groups) -> forward(x) -> x`

### 3.3 Recurrent / SSM Variants

| Variant | Mode String | Key Mechanism | Source Species |
|---------|-------------|---------------|----------------|
| GRU | `"gru"` | Standard GRU (segment-level) | SegRNN |
| LSTM | `"lstm"` | Standard LSTM | TFT |
| sLSTM | `"slstm"` | Exponential gating + block-diagonal recurrence | TiRex |
| Selective SSM | `"selective_ssm"` | Input-dependent A,B,C,delta + causal conv | Mamba, MambaSimple |
| HiPPO-LegT | `"hippo"` | Structured SSM on Legendre polynomial basis | FiLM |

**Interface**: `(d_model, d_state) -> forward(x) -> (output, final_state)`

For `slstm`: implement block-diagonal recurrence with configurable `n_heads` and `head_dim`.
Each block has independent forget/input/output gates with exponential gating.

For `selective_ssm`: implement the full Mamba selective scan — input-dependent discretization
of continuous-time A/B matrices, causal depthwise conv, and SiLU gating.

### 3.4 Linear / MLP Variants

| Variant | Mode String | Key Mechanism | Source Species |
|---------|-------------|---------------|----------------|
| Pure linear | `"linear"` | `nn.Linear(in, out)`, no activation | DLinear |
| ResBlock | `"res_block"` | Linear -> Act -> Linear + skip + LayerNorm | TiDE |
| Temporal MLP | `"temporal_mlp"` | MLP applied along time axis | TSMixer |
| Channel MLP | `"channel_mlp"` | MLP applied along channel axis | TSMixer |
| SwiGLU | `"swiglu"` | `SiLU(xW1) * (xW2)` gated FFN | Sundial, TimeMoE |
| Complex Linear | `"complex_linear"` | Linear in complex (FFT) domain | FreTS |

### 3.5 Graph Convolution

| Variant | Mode String | Key Mechanism | Source Species |
|---------|-------------|---------------|----------------|
| Diffusion GCN | `"diffusion_gcn"` | K-hop mixprop on learned adjacency | MSGNet |
| MoE-masked GCN | `"moe_gcn"` | Multi-head GCN with expert-gated structured masks | TimeFilter |

**Interface**: `(d_model, n_nodes, k_hops) -> forward(x, adj_matrix) -> x`

The adjacency matrix should be constructible via:
- `"learned"`: `softmax(ReLU(E1 @ E2^T))` with learnable node embeddings
- `"bilinear"`: `GELU(proj1(x) @ proj2(x)^T)` with top-k sparsification

### 3.6 Frequency-Domain Operations

| Operation | Key Mechanism | Source Species |
|-----------|---------------|----------------|
| FFT period detection | `torch.fft.rfft` -> top-k amplitude -> period lengths | TimesNet, MSGNet |
| Spectral conv | Complex multiplication in Legendre basis | FiLM |
| DWT / inverse DWT | Discrete wavelet transform (multi-level) | WPMixer, FEDformer |
| Fourier filter | FFT -> threshold/select modes -> iFFT | Koopa, TimeMixer |

These are **utility modules** rather than full primitives — they are used within or alongside
the main primitives.

### 3.7 Operator-Theoretic

| Variant | Key Mechanism | Source Species |
|---------|---------------|----------------|
| Koopman (learned) | Learnable linear operator K applied in latent space | Koopa (time-invariant) |
| Koopman (DMD) | Dynamic Mode Decomposition via least-squares | Koopa (time-variant) |

### 3.8 Output Distribution Heads

These are the final-layer modules that produce predictions:

| Head Type | Mode String | Output | Source Species |
|-----------|-------------|--------|----------------|
| Point | `"point"` | Single value per timestep | Most task-specific models |
| Quantile | `"quantile"` | N quantile values per timestep | Chronos-Bolt/2, TiRex |
| Categorical | `"categorical"` | Softmax over discrete bins | Chronos |
| Mixture | `"mixture"` | Parameters of mixture distribution | Moirai |
| Flow | `"flow"` | Conditional flow matching (velocity field) | Sundial |
| Multi-resolution | `"multi_res"` | Multiple heads for different horizon lengths | TimesFM, TimeMoE |

---

## 4. The Tensor Protocol (The "Cellular Interface")

**Every** primitive communicates through tensors of this shape:

```
Input:  (batch, seq_len, d_model)    — "BSE" format
Output: (batch, seq_len, d_model)    — "BSE" format
```

This is the universal interface. Primitives that need other shapes handle reshaping internally:
- **iTransformer-style (variates-as-tokens)**: The topology layer transposes before/after
- **2D convolution (TimesNet)**: The primitive reshapes `(B, L, D)` -> `(B, C, period, L/period)` internally
- **Graph convolution**: Expects `(B, N_nodes, L, D)` — the topology layer adds/removes the node dim
- **FFT operations**: Transform internally, return in time domain

**Why BSE?** It's the Transformer standard. Every existing framework (HuggingFace, PyTorch)
uses this convention. Fighting it creates friction.

**Additional protocol rules**:
- Primitives must be **stateless** across forward calls (no persistent hidden state between batches)
  unless explicitly configured for autoregressive generation
- Primitives must accept `ScaleConfig` at init and derive all internal dimensions from it
- Primitives must return the **same seq_len** they received (padding/truncation handled by topology)

---

## 5. DAG Topology Patterns (The "Body Plans")

These are the 7 distinct ways primitives are wired together, identified from the phylogenetic
analysis. Each is a `nn.Module` that accepts a list of primitives and wires them.

### 5.1 Sequential Chain

```
Input -> [Embed] -> [Block]^N -> [Head] -> Output
```

Where `Block = Norm -> Primitive -> Residual -> Norm -> FFN -> Residual`

**Parameters**:
- `primitive`: Which primitive to use (attention, ssm, etc.)
- `n_layers`: Depth from ScaleConfig
- `norm_position`: `"pre"` or `"post"` (pre-norm is default for modern architectures)
- `ffn_type`: `"standard"` | `"swiglu"` | `"none"`

**Species using this**: PatchTST, PAttn, iTransformer, Mamba, MambaSimple, TSMixer, KANAD,
SegRNN, TimesFM, TimeMoE, TiRex, Chronos-2

### 5.2 Encoder-Decoder

```
Input -> [Encoder]^N -> context
                          |
Decoder Input -----> [Decoder]^M ---> Output
                   (cross-connection)
```

**Parameters**:
- `encoder_primitive`, `decoder_primitive`: Which primitives for each
- `cross_connection`: `"cross_attention"` | `"layer_passing"` | `"dense_mlp"` | `"flow_matching"`
- `n_enc_layers`, `n_dec_layers`: Separate depth control

**Species using this**: Transformer, Informer, Autoformer, FEDformer, NS_Transformer,
Crossformer, ETSformer, TFT, TiDE, Chronos, Chronos-Bolt, Sundial

### 5.3 Parallel Branches (Fan-Out / Fan-In)

```
Input -> Branch_1(primitive_1) -> \
      -> Branch_2(primitive_2) ->  Merge(strategy) -> Output
      -> Branch_3(primitive_3) -> /
```

**Parameters**:
- `branches`: List of primitives (one per branch)
- `merge_strategy`: `"add"` | `"concat"` | `"weighted_sum"` | `"learned_linear"` | `"inverse_transform"`
- `branch_constructor`: How each branch gets its input (identity, decompose, FFT-period, wavelet)

**Species using this**: DLinear (2 branches: trend + seasonal), TimesNet (k FFT-period branches),
MSGNet, LightTS, FiLM, MICN, WPMixer, MultiPatchFormer

### 5.4 Recursive Tree

```
        Input
       /     \
    split_0   split_1
    / \       / \
  ...   ...  ...  ...
    \   /       \   /
     merge       merge
       \         /
        \       /
         merge
```

**Parameters**:
- `split_fn`: How to divide input (even/odd indices, halves, etc.)
- `node_primitive`: Primitive applied at each node
- `merge_fn`: How children recombine (add, concat, multiplicative coupling)
- `depth`: Tree depth

**Species using this**: SCINet (even/odd split, multiplicative coupling, depth=3)

### 5.5 Iterative Residual Refinement

```
residual = input
for block in blocks:
    backcast, forecast = block(residual)
    residual = residual - backcast
    output += forecast
```

**Parameters**:
- `n_blocks`: Number of refinement iterations
- `block_primitive`: The primitive inside each block
- `backcast_head`, `forecast_head`: Output projections per block

**Species using this**: Koopa (Koopman backcast), ETSformer (level/growth/season)

### 5.6 Two-Phase Sequential

```
Input -> [Temporal Primitive]^N -> Reshape -> [Cross-Variable Primitive]^M -> Output
```

**Parameters**:
- `temporal_primitive`: Phase 1 primitive (operates per-variate)
- `cross_var_primitive`: Phase 2 primitive (operates cross-variate)
- `n_temporal_layers`, `n_cross_var_layers`: Depth per phase

**Species using this**: MultiPatchFormer (causal attn -> channel attn),
Crossformer (time attn -> dim attn per layer)

### 5.7 Two-Component (Encoder + Generative Decoder)

```
Input -> [Context Encoder] -> conditioning
                                  |
                   Noise -> [Generative Decoder(conditioning)] -> Output
```

**Parameters**:
- `encoder_primitive`: For context encoding
- `decoder_type`: `"flow_matching"` | `"diffusion"` | `"vae"`
- `n_ode_steps`: Integration steps for flow-based decoders

**Species using this**: Sundial (causal Transformer encoder + flow matching decoder)

---

## 6. Cross-Cutting Policies (The "Regulatory Genes")

These are structural decisions that apply orthogonally to the primitive and topology choices.
Each is a boolean or enum in the species config.

### 6.1 Input Normalization

| Policy | Config Value | Mechanism |
|--------|-------------|-----------|
| RevIN | `"revin"` | Subtract mean, divide by std; reverse after model |
| Mean scaling | `"mean_scale"` | Divide by absolute mean |
| RMS norm on input | `"rms"` | RMSNorm on raw input |
| None | `"none"` | Raw input |

**Default**: `"revin"` (used by 24/40 known species)

### 6.2 Decomposition

| Policy | Config Value | Mechanism | Where Applied |
|--------|-------------|-----------|---------------|
| None | `"none"` | No decomposition | Most models |
| Input-only | `"input"` | Moving-average trend/seasonal split before model | DLinear, MICN |
| Progressive | `"progressive"` | Decompose at every encoder and decoder layer | Autoformer, FEDformer, ETSformer |
| DFT-based | `"dft"` | Top-k FFT mode filtering for trend/seasonal | TimeMixer |
| Koopman | `"koopman"` | Split into time-invariant/time-variant via Fourier filter | Koopa |

### 6.3 Patching / Tokenization

| Policy | Config Value | Mechanism |
|--------|-------------|-----------|
| None (per-timestep) | `"none"` | Each timestep is one token | Transformer, TimeMoE, Chronos |
| Fixed patch | `"fixed"` | Non-overlapping patches of `patch_size` | PatchTST, TimeXer, TiRex, Sundial |
| Overlapping patch | `"overlap"` | Patches with `stride < patch_size` | PatchTST (stride=8, patch=16) |
| Multi-scale patch | `"multi_scale"` | Multiple patch sizes simultaneously | MultiPatchFormer, Moirai MPSP |
| Asymmetric I/O patch | `"asymmetric"` | Fixed input patch, variable output patch | TimesFM |
| Segment | `"segment"` | Chunks for RNN processing | SegRNN |
| Discrete bins | `"discrete_bins"` | Quantize values into discrete vocabulary | Chronos |

### 6.4 Channel (Inter-Variate) Policy

| Policy | Config Value | Mechanism |
|--------|-------------|-----------|
| Independent | `"independent"` | Each variate processed separately (fold into batch) |
| Embed-mix | `"embed_mix"` | `Conv1d(n_vars, d_model)` at input; no further mixing |
| Explicit cross-var | `"cross_var"` | Dedicated cross-variable attention/MLP at specific layers |
| Graph | `"graph"` | Learned adjacency + GCN propagation |
| Any-variate (AVA) | `"ava"` | Structured attention mask handling arbitrary variate count |
| Group attention | `"group"` | Alternating temporal + cross-series attention |
| Configurable | `"configurable"` | Runtime switch between independent and mixing |

### 6.5 Positional Encoding

| Policy | Config Value | Mechanism |
|--------|-------------|-----------|
| Sinusoidal | `"sinusoidal"` | Fixed sin/cos embeddings | Vanilla Transformer |
| Learned | `"learned"` | Learnable position embeddings |
| RoPE | `"rope"` | Rotary Position Embeddings | Modern Transformers, Foundation models |
| T5 relative | `"t5_relative"` | T5-style relative position bias | Chronos, Chronos-Bolt |
| None | `"none"` | No positional encoding | Some MLP models, Group Attention dim |
| Temporal (date features) | `"temporal"` | Temporal embeddings (hour, day, month, etc.) |
| Frequency token | `"frequency_token"` | Learnable embedding indicating temporal granularity | TimesFM |

### 6.6 Residual Connection Pattern

| Policy | Config Value |
|--------|-------------|
| Pre-norm | `"pre_norm"` |
| Post-norm | `"post_norm"` |
| ResBlock (skip + MLP + norm) | `"res_block"` |
| GLU-gated | `"glu_gated"` |
| Highway (global skip) | `"highway"` |
| Iterative backcast | `"backcast"` |
| None | `"none"` |

### 6.7 Output Strategy

| Policy | Config Value | Mechanism |
|--------|-------------|-----------|
| Direct projection | `"direct"` | `Linear(d_model, pred_len)` |
| Flatten head | `"flatten"` | `Linear(d_model * n_patches, pred_len)` |
| Decoder cross-attention | `"dec_cross_attn"` | Autoregressive/parallel decoder |
| Decomposition-additive | `"decomp_add"` | Sum of trend + seasonal + ... |
| Autoregressive token | `"ar_token"` | Generate one token at a time |
| Semi-autoregressive | `"semi_ar"` | AR across patches, parallel within |
| Single-pass quantile | `"quantile"` | Quantile regression heads |
| Mixture distribution | `"mixture"` | Mixture model parameters |
| Flow-based generative | `"flow"` | ODE integration from noise |
| Multi-resolution heads | `"multi_res"` | Multiple heads for different horizons |
| NaN-rollout | `"nan_rollout"` | Pad NaN, single forward pass, quantile heads |
| Basis reconstruction | `"basis_recon"` | Inverse transform (wavelet, Legendre) |
| Last-hidden-state | `"last_hidden"` | Use only final position's representation |

---

## 7. Species Definitions (The "Phenotypes")

Each known architecture is defined as a **species config** — a specific selection of
primitives, topology, and policies. Below are all 40 species grouped by phylogenetic clade.

The implementation should accept these configs and automatically build the correct model.

### 7.1 Recurrent Clade

```yaml
SegRNN:
  topology: "sequential_chain"
  primitive: "gru"
  patching: "segment"
  channel: "independent"  # batch-folded, with channel embedding for decoding
  normalization: "none"
  decomposition: "none"
  positional: "none"
  residual: "none"
  output: "segment_decode"

TFT:
  topology: "encoder_decoder"
  encoder_primitive: "lstm"
  decoder_primitive: "lstm"
  cross_connection: "interpretable_attention"  # †horizontal transfer from Transformer
  channel: "cross_var"  # VariableSelectionNetwork
  normalization: "none"
  decomposition: "none"
  output: "glu_gated"

FiLM:
  topology: "parallel_branches"
  primitive: "hippo"  # HiPPO-LegT SSM
  branches: 3  # [1x, 2x, 4x] lookback windows
  merge: "learned_linear"
  channel: "independent"
  decomposition: "none"
  domain: "legendre"
  output: "basis_recon"  # Legendre eval matrix reconstruction

Mamba:
  topology: "sequential_chain"
  primitive: "selective_ssm"
  channel: "embed_mix"
  normalization: "revin"
  residual: "none"  # single block, no stacking
  output: "direct"

MambaSimple:
  topology: "sequential_chain"
  primitive: "selective_ssm"
  channel: "embed_mix"
  normalization: "revin"
  residual: "pre_norm"
  output: "direct"

TiRex:
  topology: "sequential_chain"
  primitive: "slstm"  # sLSTM only, NOT mLSTM
  slstm_config:
    n_heads: 4
    head_dim: 128
    gating: "exponential"
    ffn: "silu_gated"
    recurrence: "block_diagonal"
  patching: "fixed"
  channel: "independent"  # univariate
  normalization: "revin"
  output: "nan_rollout"  # NaN-padded single forward pass
  output_head: "quantile"  # 9 quantiles (0.1-0.9)
  training: "contiguous_patch_masking"
  foundation: true
```

### 7.2 Stem Transformers (Encoder-Decoder)

```yaml
Transformer:
  topology: "encoder_decoder"
  encoder_primitive: "full"  # full attention
  decoder_primitive: "full"
  cross_connection: "cross_attention"
  channel: "embed_mix"
  normalization: "none"
  positional: "sinusoidal"
  residual: "post_norm"
  output: "dec_cross_attn"

Informer:
  topology: "encoder_decoder"
  encoder_primitive: "prob_sparse"
  decoder_primitive: "prob_sparse"
  cross_connection: "cross_attention"
  encoder_extras:
    distillation: true  # Conv1d(stride=2) + MaxPool between layers
  channel: "embed_mix"
  normalization: "revin"
  positional: "sinusoidal"
  residual: "post_norm"
  multi_scale: "pyramid"  # L -> L/2 -> L/4
  output: "dec_cross_attn"

Reformer:
  topology: "sequential_chain"  # decoder lost ‡
  primitive: "lsh"
  channel: "embed_mix"
  normalization: "revin"
  positional: "sinusoidal"
  residual: "post_norm"
  output: "direct"

Autoformer:
  topology: "encoder_decoder"
  encoder_primitive: "auto_corr"
  decoder_primitive: "auto_corr"
  cross_connection: "cross_attention"  # AutoCorrelation-based cross
  channel: "embed_mix"
  decomposition: "progressive"  # trend-seasonal in every layer
  positional: "sinusoidal"
  residual: "post_norm"
  output: "decomp_add"  # seasonal + accumulated trend

FEDformer:
  topology: "encoder_decoder"
  encoder_primitive: "fourier"  # or "wavelet"
  decoder_primitive: "fourier"
  cross_connection: "cross_attention"
  channel: "embed_mix"
  decomposition: "progressive"
  positional: "sinusoidal"
  residual: "post_norm"
  output: "decomp_add"

ETSformer:
  topology: "encoder_decoder"
  encoder_primitive: "full"  # attention is §vestigial
  decoder_primitive: "full"
  encoder_extras:
    exponential_smoothing: true  # primary computation
    fourier_layer: true  # top-k mode extraction
  cross_connection: "layer_passing"
  channel: "embed_mix"
  decomposition: "progressive"  # level/growth/season
  output: "decomp_add"  # level + damped_growth + seasonal

NS_Transformer:
  topology: "encoder_decoder"
  encoder_primitive: "ds"  # de-stationary attention
  decoder_primitive: "ds"
  cross_connection: "cross_attention"
  channel: "embed_mix"
  normalization: "revin"
  positional: "sinusoidal"
  residual: "post_norm"
  output: "dec_cross_attn"

Pyraformer:
  topology: "sequential_chain"  # decoder lost ‡
  primitive: "pyramid"
  channel: "embed_mix"
  multi_scale: "pyramid"
  positional: "sinusoidal"
  residual: "pre_norm"
  output: "last_hidden"

Crossformer:
  topology: "encoder_decoder"
  encoder_primitive: "two_stage"  # time + cross-dim via router
  decoder_primitive: "two_stage"
  cross_connection: "cross_attention"  # multi-scale connections
  channel: "cross_var"  # router-mediated cross-dim
  multi_scale: "segment_merging"  # seg_num -> seg_num/2 -> ...
  patching: "fixed"  # segment-based
  output: "dec_cross_attn"
```

### 7.3 Crown Transformers (Encoder-Only)

```yaml
PatchTST:
  topology: "sequential_chain"
  primitive: "full"
  patching: "overlap"  # stride=8, patch_size=16
  channel: "independent"
  normalization: "revin"
  positional: "learned"
  residual: "post_norm"
  output: "flatten"

PAttn:
  topology: "sequential_chain"
  primitive: "full"
  override_n_layers: 1  # minimal single-layer
  patching: "overlap"
  channel: "independent"
  normalization: "revin"
  positional: "learned"
  residual: "post_norm"
  output: "flatten"

iTransformer:
  topology: "sequential_chain"
  primitive: "full"
  axis_mode: "inverted"  # variates are tokens, time is embedding
  channel: "cross_var"  # full cross-variate attention (inherent)
  normalization: "revin"
  positional: "none"  # no positional encoding on variates
  residual: "post_norm"
  output: "direct"  # per-variate linear projection

TimeXer:
  topology: "sequential_chain"
  primitive: "full"
  patching: "fixed"
  channel: "cross_var"  # global token cross-attention with exogenous
  exogenous_support: true  # dual-pathway: endogenous patches + exogenous inverted
  normalization: "revin"
  positional: "learned"
  residual: "post_norm"
  output: "flatten"

MultiPatchFormer:
  topology: "two_phase"
  temporal_primitive: "full"  # causal masked attention
  cross_var_primitive: "full"  # Conv1d-based K/V channel attention
  temporal_config:
    causal: true
  patching: "multi_scale"  # 4 patch sizes: 8, 16, 24, 32
  channel: "independent_then_mix"  # phase 1 independent, phase 2 channel
  normalization: "revin"
  positional: "learned"
  output: "semi_ar"  # 8 sequential steps
```

### 7.4 Foundation Transformers

```yaml
Chronos:
  topology: "encoder_decoder"
  encoder_primitive: "full"  # T5 encoder
  decoder_primitive: "full"  # T5 decoder (autoregressive)
  cross_connection: "cross_attention"
  patching: "discrete_bins"  # 4096-bin quantization
  channel: "independent"  # univariate
  normalization: "mean_scale"
  positional: "t5_relative"
  output: "ar_token"  # autoregressive categorical token generation
  output_head: "categorical"  # softmax over bins
  foundation: true

Chronos_Bolt:
  topology: "encoder_decoder"
  encoder_primitive: "full"  # T5 encoder (bidirectional)
  decoder_primitive: "full"  # single-token decoder (non-autoregressive)
  decoder_config:
    n_tokens: 1  # single start token cross-attends to encoder
  cross_connection: "cross_attention"
  patching: "fixed"
  channel: "independent"
  normalization: "revin"
  positional: "t5_relative"
  output: "quantile"  # full forecast in one shot via ResidualBlock
  foundation: true

Chronos_2:
  topology: "sequential_chain"  # encoder-only (decoder lost ‡)
  primitive: "group"  # alternating Time Attn + Group Attn
  patching: "fixed"
  channel: "group"  # multivariate via group IDs
  normalization: "robust_scale"
  positional: "rope"
  residual: "pre_norm"
  output: "quantile"  # 21 quantiles via ResidualBlock on future patches
  extras:
    reg_token: true  # attention sink between context and future
    meta_features: true  # time index + observation mask
  foundation: true

Moirai:
  topology: "sequential_chain"
  primitive: "ava"  # Any-Variate Attention
  patching: "multi_scale"  # Multiple Patch Size Projection (MPSP)
  channel: "ava"  # cross-variate via structured attention mask
  normalization: "revin"
  positional: "rope"
  residual: "post_norm"
  output: "mixture"  # Student's t / Normal / NegBin / LogNormal mixture
  foundation: true

Sundial:
  topology: "two_component"
  encoder_primitive: "full"  # LLaMA-style causal Transformer
  decoder_type: "flow_matching"  # conditional flow matching
  encoder_config:
    ffn: "swiglu"
    norm: "rms"
    positional: "rope"
    causal: true
  patching: "fixed"
  channel: "independent"  # univariate
  normalization: "revin"
  output: "flow"  # ODE integration from noise -> forecast
  foundation: true

TimesFM:
  topology: "sequential_chain"
  primitive: "full"  # decoder-only Transformer
  causal: true
  patching: "asymmetric"  # input: 32, output: {1,8,16,32,64,128}
  channel: "independent"  # univariate
  normalization: "revin"
  positional: "rope"
  residual: "post_norm"
  output: "multi_res"  # asymmetric multi-output-patch heads
  extras:
    frequency_token: true  # learnable temporal granularity embedding
  foundation: true

TimeMoE:
  topology: "sequential_chain"
  primitive: "full"  # decoder-only Transformer (causal)
  causal: true
  ffn_type: "moe"  # Sparse MoE FFN
  moe_config:
    n_routed_experts: 8
    n_shared_experts: 1  # always active, sigmoid-gated (DeepSeekMoE-style)
    top_k: 2
    load_balance_alpha: 0.02
    ffn: "swiglu"
  patching: "none"  # point-wise tokenization (each timestep = one token)
  embedding: "swiglu"  # SwiGLU maps scalar -> d_model
  channel: "independent"  # univariate
  normalization: "revin"
  positional: "rope"
  residual: "pre_norm"
  output: "multi_res"  # 4 heads predicting [1, 8, 32, 64] steps
  output_config:
    head_sizes: [1, 8, 32, 64]
    scheduling: "dynamic"  # pick largest head <= remaining horizon
    loss: "huber"  # Huber, not MSE
  foundation: true
```

### 7.5 Linear/MLP Clade

```yaml
DLinear:
  topology: "parallel_branches"
  branches: 2  # trend + seasonal
  branch_constructor: "moving_avg_decompose"
  primitive: "linear"
  merge: "add"
  channel: "independent"
  decomposition: "input"
  output: "decomp_add"

LightTS:
  topology: "parallel_branches"
  branches: 3  # continuous + interval + highway
  primitive: "linear"  # with LeakyReLU
  merge: "add_with_highway"
  channel: "cross_var"  # weak (identity-initialized linear)
  multi_scale: "dual_sampling"
  output: "direct"

TiDE:
  topology: "encoder_decoder"
  encoder_primitive: "res_block"
  decoder_primitive: "res_block"
  cross_connection: "dense_mlp"  # no attention
  channel: "independent"  # loop over channels
  normalization: "revin"
  residual: "res_block"
  output: "direct"

TSMixer:
  topology: "sequential_chain"
  primitive: "temporal_mlp"
  extras:
    channel_mlp: true  # alternating temporal + channel MLP
  channel: "cross_var"
  output: "direct"

FreTS:
  topology: "sequential_chain"
  primitive: "complex_linear"  # FFT -> complex linear -> iFFT
  channel: "cross_var"  # optional FFT along channel dim
  domain: "frequency"
  residual: "highway"
  output: "direct"

WPMixer:
  topology: "parallel_branches"
  branch_constructor: "dwt"  # discrete wavelet transform
  primitive: "temporal_mlp"  # mixer blocks per wavelet level
  merge: "inverse_transform"  # inverse DWT
  patching: "fixed"
  channel: "independent"
  normalization: "revin"
  domain: "wavelet"
  output: "basis_recon"
```

### 7.6 Convolution Clade

```yaml
TimesNet:
  topology: "parallel_branches"
  branch_constructor: "fft_period"  # FFT top-k period detection
  primitive: "inception2d"  # 2D Inception Conv per period
  merge: "weighted_sum"  # softmax of FFT amplitudes
  channel: "embed_mix"
  normalization: "revin"
  domain: "2d_spatial"
  output: "direct"

MICN:
  topology: "parallel_branches"
  branches: 2  # trend (linear) + seasonal (multi-scale conv)
  primitive: "conv1d"
  merge: "add"
  channel: "embed_mix"
  decomposition: "input"
  multi_scale: "multi_kernel"
  output: "decomp_add"

SCINet:
  topology: "recursive_tree"
  split_fn: "even_odd"
  primitive: "causal_conv1d"
  merge_fn: "multiplicative_coupling"
  depth: 3
  channel: "embed_mix"
  multi_scale: "binary_tree"
  output: "direct"

KANAD:
  topology: "sequential_chain"
  primitive: "conv1d"
  extras:
    basis_expansion: "cosine"  # cosine basis expansion at input
  channel: "independent"
  output: "direct"
```

### 7.7 Graph Clade

```yaml
MSGNet:
  topology: "parallel_branches"
  branch_constructor: "fft_period"
  primitive: "diffusion_gcn"  # GCN on learned adjacency
  extras:
    attention: true  # †borrowed from Transformer clade
  merge: "weighted_sum"
  channel: "graph"
  normalization: "revin"
  graph_construction: "learned"  # nodevec1 @ nodevec2^T
  output: "direct"

TimeFilter:
  topology: "sequential_chain"
  primitive: "moe_gcn"
  patching: "fixed"
  channel: "graph"
  normalization: "revin"
  graph_construction: "bilinear"  # proj1(x) @ proj2(x)^T + top-k sparsify
  output: "direct"
```

### 7.8 Other Clades

```yaml
Koopa:
  topology: "iterative_refinement"
  primitive: "koopman_learned"  # time-invariant Koopman
  extras:
    time_variant: "koopman_dmd"  # time-variant DMD branch
  channel: "configurable"
  normalization: "revin"
  domain: "koopman"
  decomposition: "koopman"  # Fourier filter splits inv/var
  output: "decomp_add"

TimeMixer:
  topology: "sequential_chain"
  primitive: "temporal_mlp"
  extras:
    multi_scale_decomp: true  # downsample pyramid
    cross_scale_mixing: "bidirectional"
    multi_predictor_fusion: true
  channel: "configurable"
  normalization: "revin"
  decomposition: "progressive"
  multi_scale: "downsample_pyramid"
  output: "decomp_add"
```

---

## 8. Project Structure

```
ts_genome/
├── primitives/
│   ├── __init__.py              # Registry: PRIMITIVE_REGISTRY[mode_string] -> class
│   ├── attention.py             # All 12 attention variants
│   ├── convolution.py           # All 4 conv variants
│   ├── recurrent.py             # GRU, LSTM, sLSTM, selective SSM, HiPPO
│   ├── linear.py                # Pure linear, ResBlock, temporal/channel MLP, SwiGLU
│   ├── graph.py                 # Diffusion GCN, MoE-masked GCN
│   ├── frequency.py             # FFT period detection, spectral conv, DWT
│   ├── operator.py              # Koopman (learned + DMD)
│   └── output_heads.py          # Point, quantile, categorical, mixture, flow, multi-res
│
├── topologies/
│   ├── __init__.py              # Registry: TOPOLOGY_REGISTRY[name] -> class
│   ├── sequential_chain.py      # [Block]^N
│   ├── encoder_decoder.py       # Enc + Dec with cross-connection
│   ├── parallel_branches.py     # Fan-out / fan-in
│   ├── recursive_tree.py        # Binary tree
│   ├── iterative_refinement.py  # Backcast subtraction
│   ├── two_phase.py             # Temporal then cross-variable
│   └── two_component.py         # Encoder + generative decoder
│
├── policies/
│   ├── __init__.py
│   ├── normalization.py         # RevIN, mean-scale, RMS
│   ├── decomposition.py         # Moving-avg, progressive, DFT, Koopman
│   ├── patching.py              # Fixed, overlap, multi-scale, asymmetric, discrete bins
│   ├── channel.py               # Independent, embed-mix, cross-var, graph, AVA, group
│   ├── positional.py            # Sinusoidal, learned, RoPE, T5 relative
│   └── residual.py              # Pre-norm, post-norm, ResBlock, highway, backcast
│
├── species/
│   ├── __init__.py              # Species registry + builder
│   ├── configs/                 # YAML configs for all 40 known species
│   │   ├── recurrent/           # SegRNN, TFT, FiLM, Mamba, MambaSimple, TiRex
│   │   ├── stem_transformer/    # Transformer, Informer, Autoformer, ...
│   │   ├── crown_transformer/   # PatchTST, PAttn, iTransformer, TimeXer, ...
│   │   ├── foundation/          # Chronos, Moirai, Sundial, TimesFM, TimeMoE, TiRex
│   │   ├── linear_mlp/          # DLinear, LightTS, TiDE, TSMixer, FreTS, WPMixer
│   │   ├── convolution/         # TimesNet, MICN, SCINet, KANAD
│   │   ├── graph/               # MSGNet, TimeFilter
│   │   └── other/               # Koopa, TimeMixer
│   └── builder.py               # build_species(config, scale) -> nn.Module
│
├── scaling/
│   ├── __init__.py
│   ├── config.py                # ScaleConfig dataclass + SCALE_PRESETS
│   └── param_counter.py         # Count params, FLOPs, memory for any species+scale
│
├── evolution/
│   ├── __init__.py
│   ├── character_matrix.py      # 12-character phylogenetic matrix for all species
│   ├── phylogeny.py             # Tree structure, clade membership, distance metrics
│   ├── breeder.py               # Create new species by trait recombination
│   ├── niche_explorer.py        # Find unexplored regions of the character space
│   └── hypothesis_generator.py  # Generate novel architecture hypotheses from phylogeny
│
├── config.py                    # Global config, dtype, device
├── registry.py                  # Unified registry for primitives, topologies, policies
└── model.py                     # Top-level: Model(species_config, scale_config) -> nn.Module
```

---

## 9. The Builder Pattern

The top-level API should work like this:

```python
from ts_genome import Model, ScaleConfig, SCALE_PRESETS

# Build a known species at a specific scale
model = Model.from_species("PatchTST", scale=SCALE_PRESETS["base"])

# Build at a custom scale
model = Model.from_species("Mamba", scale=ScaleConfig(d_model=384, n_layers=8, ...))

# Build from a raw config dict (for custom species)
model = Model.from_config({
    "topology": "sequential_chain",
    "primitive": "selective_ssm",
    "patching": "multi_scale",
    "channel": "cross_var",
    ...
}, scale=SCALE_PRESETS["large"])

# Build a novel species by recombining traits
from ts_genome.evolution import Breeder
novel = Breeder.crossover("PatchTST", "Mamba",
    take_from_a=["topology", "patching", "channel"],
    take_from_b=["primitive"])
# Result: sequential chain + patching + chan-indep + selective SSM
# = "What if PatchTST used Mamba instead of attention?"
model = Model.from_config(novel, scale=SCALE_PRESETS["base"])
```

---

## 10. The Evolution Module (Phylogenetic Explorer)

This is the most novel part of the library. It uses the phylogenetic analysis to
**systematically discover new architectures**.

### 10.1 Character Space Explorer

The 12-character matrix defines a 12-dimensional structural space. Each known species
occupies a point in this space. The explorer finds **empty niches** — valid character
combinations not yet tried:

```python
from ts_genome.evolution import NicheExplorer

explorer = NicheExplorer()

# Find the nearest unoccupied niche to a given species
niches = explorer.find_empty_niches(near="PatchTST", max_distance=2)
# Returns: [
#   {"changes": {"channel": "graph", ...}, "description": "PatchTST with graph structure"},
#   {"changes": {"primitive": "selective_ssm"}, "description": "PatchTST backbone swap to SSM"},
#   ...
# ]

# Find niches that combine traits from two different clades
niches = explorer.cross_clade_niches("Transformer", "Recurrent")
```

### 10.2 Hypothesis Generator

Uses phylogenetic reasoning patterns to propose architecturally novel species:

**Pattern 1: Horizontal Transfer** — Take a trait that succeeded in clade A and graft it
onto a species in clade B.
```
Example: "Group Attention" succeeded in Chronos-2 (Foundation Transformer).
   → Graft onto Mamba (Recurrent clade) = SSM backbone with cross-series attention
   → Graft onto DLinear (Linear clade) = Per-series linear + cross-series MLP
```

**Pattern 2: Convergent Loss Extrapolation** — If a trait was independently lost in N
lineages, it's probably unnecessary. Try removing it from species that still have it.
```
Example: The decoder was independently lost 6 times.
   → Try removing it from: Crossformer, NS_Transformer, ETSformer
   → Hypothesis: encoder-only versions of these may outperform
```

**Pattern 3: Recombinant Offspring** — Take the topology from species A and the primitive
from species B.
```
Example: Topology of TimesNet (parallel FFT-period branches + 2D processing)
   + Primitive of graph convolution (from MSGNet)
   = FFT-period detection → 2D graph convolution per period
   (This is close to MSGNet, but MSGNet doesn't do the 2D reshape — novel)
```

**Pattern 4: Scaling Trait Transfer** — Take a foundation model's scaling innovation and
apply it to a task-specific architecture.
```
Example: TimeMoE's Sparse MoE + shared expert + multi-resolution heads
   → Apply to PatchTST = Patched encoder with MoE FFN + multi-res output
   → Apply to TimesNet = 2D Inception with MoE + multi-res output
```

**Pattern 5: Vestigial Removal** — Identify and remove vestigial traits.
```
Example: ETSformer still has attention weights but barely uses them
   → Remove attention entirely, pure exponential smoothing model
   → Is it better, worse, or equivalent? (Test to find out)
```

**Pattern 6: Missing Combination Mining** — Query the character matrix for combinations
of traits that have never co-occurred:
```python
explorer.never_cooccurred(["graph", "patching"])
# Returns: No existing model combines graph convolution with patching
# → Hypothesis: Patched Graph Transformer with learned adjacency

explorer.never_cooccurred(["hippo", "multi_scale"])
# Returns: No existing model uses HiPPO with multi-scale processing
# → Hypothesis: Multi-scale HiPPO with wavelet decomposition

explorer.never_cooccurred(["flow", "cross_var"])
# Returns: Sundial uses flow matching but is univariate
# → Hypothesis: Multivariate flow matching with AVA encoder
```

### 10.3 Trait Compatibility Matrix

Not all primitive-topology-policy combinations are valid. The evolution module should
maintain a compatibility matrix:

```python
INCOMPATIBLE = [
    ("inception2d", "sequential_chain"),   # 2D conv needs parallel period branches
    ("ar_token", "independent"),           # AR token decoding needs the model to generate tokens
    ("discrete_bins", "direct"),           # Discrete bins require categorical output head
    ("flow", "sequential_chain"),          # Flow matching needs two-component topology
    ("ava", "independent"),               # AVA is inherently cross-variate
]

REQUIRED_PAIRS = [
    ("discrete_bins", "categorical"),      # Bin input requires categorical output
    ("hippo", "basis_recon"),             # HiPPO requires Legendre reconstruction
    ("flow_matching", "two_component"),   # Flow needs separate encoder + decoder
]
```

### 10.4 Guided Exploration Queries

The evolution module should support high-level queries:

```python
# "What would happen if we evolved PatchTST toward the Foundation clade?"
path = explorer.evolutionary_path("PatchTST", target_clade="Foundation")
# Returns: Step 1: Add pre-training capability (scaling up d_model, n_layers)
#          Step 2: Add frequency token (from TimesFM)
#          Step 3: Switch output to quantile heads (from Chronos-2)
#          Step 4: Add Group Attention for multivariate (from Chronos-2)

# "What extinct traits might be worth reviving?"
revivals = explorer.suggest_trait_revivals()
# Returns: "Approximate attention (ProbSparse, LSH) was abandoned in favor of patching.
#           But patching + approximate attention has never been tried together.
#           Hypothesis: ProbSparse attention ON PATCHES could give O(P log P) with P << L."

# "What is the most structurally novel architecture I could build?"
novel = explorer.maximize_novelty(constraint="must_be_scalable")
# Returns: species config that maximizes distance from all known species
#          in the 12-dimensional character space
```

---

## 11. Implementation Order

Build in this order. Each phase should be fully tested before moving on.

### Phase 1: Primitives + Protocol
1. Define `SequencePrimitive` protocol (Section 4)
2. Implement `ScaleConfig` and presets
3. Implement attention module with `"full"` mode first, then add variants
4. Implement `"selective_ssm"`, `"gru"`, `"lstm"`, `"slstm"`
5. Implement `"conv1d"`, `"causal_conv1d"`, `"inception2d"`
6. Implement `"linear"`, `"res_block"`, `"temporal_mlp"`, `"channel_mlp"`, `"swiglu"`
7. Implement output heads: `"point"`, `"quantile"`, `"categorical"`, `"mixture"`
8. **Test**: Every primitive individually with random input, verify BSE protocol

### Phase 2: Topologies
1. Implement `sequential_chain` (covers ~18 species)
2. Implement `encoder_decoder` (covers ~12 species)
3. Implement `parallel_branches` (covers ~8 species)
4. Implement `iterative_refinement`, `recursive_tree`, `two_phase`, `two_component`
5. **Test**: Wire a few primitives into each topology, verify forward pass

### Phase 3: Policies
1. Implement normalization policies (RevIN, mean-scale)
2. Implement patching policies (fixed, overlap, multi-scale, asymmetric)
3. Implement channel policies (independent batch-folding, embed-mix, cross-var)
4. Implement positional encoding (sinusoidal, learned, RoPE, T5 relative)
5. Implement decomposition (moving-avg, progressive)
6. **Test**: Policies compose correctly with topologies

### Phase 4: Species Builder
1. Implement YAML config loader
2. Implement `Model.from_species()` and `Model.from_config()`
3. Define all 40 species configs
4. **Test**: Every species builds and runs forward pass at "tiny" and "base" scales
5. **Validate**: Compare parameter counts against known values from original papers

### Phase 5: Frequency, Graph, Operator Primitives
1. FFT period detection, DWT, spectral conv
2. Graph convolution (diffusion GCN, MoE-masked GCN)
3. Koopman operators (learned + DMD)
4. Flow matching decoder
5. **Test**: Remaining species (TimesNet, MSGNet, TimeFilter, Koopa, Sundial)

### Phase 6: Evolution Module
1. Implement character state matrix
2. Implement phylogenetic tree structure
3. Implement `Breeder.crossover()`
4. Implement `NicheExplorer.find_empty_niches()`
5. Implement hypothesis generator patterns
6. **Test**: Generate and build 10 novel species

---

## 12. Testing Strategy

### Unit Tests (per primitive)
- Forward pass with random input at each scale preset
- Gradient flow (no NaN/Inf gradients)
- Parameter count matches expected formula
- Correct output shape (BSE protocol)

### Integration Tests (per species)
- Forward pass at "tiny" scale
- Training loop (5 steps) doesn't diverge
- Scaling: build at "tiny", "base", "large" — all work
- Output head produces correct distribution type

### Structural Tests (phylogenetic validation)
- Character matrix: each species' coded characters match its config
- Equivalence classes: Class A species (PatchTST, PAttn) produce identical DAGs
  (except depth)
- Compatibility: incompatible combinations raise clear errors
- Novel species: breeder-generated species build and train

---

## 13. Reference Material

The full taxonomy analysis is in `ARCHITECTURE_TAXONOMY.md` in this repository. Key sections:
- **Section 2** (Axis 1): All computational primitives with exact mechanisms
- **Section 3** (Axis 2): All DAG topology patterns with ASCII diagrams
- **Section 4** (Axis 3): Channel handling strategies
- **Section 5** (Axis 4): Multi-scale processing mechanisms
- **Section 7** (Axis 6): Output generation strategies
- **Section 8**: Cross-cutting patterns (RevIN, decomposition, patching, residuals)
- **Section 9**: Architecture fingerprints (compact 6-axis descriptor per model)
- **Section 11**: Master classification matrix (all 40 models x 7 columns)
- **Section 12**: Phylogenetic analysis (tree, clades, innovations, convergent evolution)

The character state matrix (Section 12.2) is the definitive reference for the evolution
module's character encoding.
