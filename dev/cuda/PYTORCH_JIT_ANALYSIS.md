# PyTorch JIT Compilation: State of the Art (2026)

## Overview: Where Are We Now?

PyTorch has made **massive progress** in automatic optimization, but custom CUDA kernels still win for performance-critical operations. Here's the landscape:

```
Performance Gap (relative to custom CUDA = 1.0x):

Eager PyTorch:           ███████░░░░░░░░░░░░░░░░░ ~1.7x slower
torch.compile (default): ████░░░░░░░░░░░░░░░░░░░░ ~1.3x slower
torch.compile (max):     ██░░░░░░░░░░░░░░░░░░░░░░ 1.0x (MATCHES!)
Custom CUDA (optimized): ██░░░░░░░░░░░░░░░░░░░░░░ 1.0x (baseline)
```

**Verdict**: torch.compile (max-autotune) **MATCHES** optimized custom CUDA with zero effort!

**Note**: Results vary by operation. For our temperature+top-nsigma kernel, they tied. For operations with unique algorithmic innovations (like FlashAttention), custom CUDA still wins.

## The Evolution of PyTorch JIT

### 1. **Pre-2017: Eager Mode Only**

```python
# All operations launch separate CUDA kernels
logits = logits / temperature  # Kernel 1
maximum = logits.max(dim=-1)   # Kernel 2
std = logits.std(dim=-1)       # Kernel 3
# ... etc
```

**Problems**:
- Kernel launch overhead (~5 µs each)
- Global memory round-trips
- Python interpreter overhead
- No optimization across operations

### 2. **2017-2020: TorchScript JIT**

```python
@torch.jit.script
def my_function(x):
    y = x * 2
    z = y + 1
    return z
```

**What it does**:
- Removes Python overhead (trace to static graph)
- Type specialization
- Basic operator fusion
- Dead code elimination

**Limitations**:
- Limited kernel fusion (mostly pointwise ops)
- Can't optimize reductions well
- Requires @torch.jit decorators
- Debugging is harder

**Performance gain**: ~10-20% over eager

**Status**: Still used for deployment (TorchScript for production)

### 3. **2022-Present: torch.compile (PyTorch 2.0+)**

```python
@torch.compile
def my_function(x):
    y = x * 2
    z = y + 1
    return z
```

**Revolutionary features**:
- **TorchDynamo**: Captures computation graph at Python bytecode level
- **TorchInductor**: Generates optimized code (Triton kernels for GPU)
- **Automatic kernel fusion**: Fuses ops without manual work
- **AOTAutograd**: Compiles backward pass too
- **Graph breaks**: Handles dynamic Python (fallback when needed)

**What it can do**:

1. **Pointwise fusion**:
   ```python
   # These get fused into ONE kernel
   x = input / temperature
   y = x - mean
   z = torch.where(y > threshold, y, -float('inf'))
   ```

2. **Reduction optimization**:
   ```python
   # Generates efficient reduction code
   max_val = x.max(dim=-1)
   sum_val = x.sum(dim=-1)
   ```

3. **Memory planning**:
   - Reuses buffers
   - Minimizes allocations
   - Optimizes memory layout

4. **Backend selection**:
   - Triton (default for NVIDIA GPUs)
   - OpenAI Triton for custom ops
   - FX graph transformations

**Performance gain**: ~2-3x over eager, ~1.5-2x over TorchScript

**Status**: **Production ready** (PyTorch 2.0+)

## Deep Dive: torch.compile vs Custom CUDA

### What torch.compile Does Well

#### ✅ Pointwise Operation Fusion

**Your code**:
```python
logits = logits / temperature
threshold = maximum - top_nsigma * std
filtered = torch.where(logits >= threshold, logits, float('-inf'))
```

**torch.compile result**: Fuses into 1-2 kernels with Triton

**Generated code** (simplified Triton):
```python
@triton.jit
def fused_kernel(logits_ptr, temp, threshold_ptr, output_ptr, ...):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    logits = tl.load(logits_ptr + idx)

    # Fused operations
    scaled = logits / temp
    thresh = tl.load(threshold_ptr + ...)
    result = tl.where(scaled >= thresh, scaled, float('-inf'))

    tl.store(output_ptr + idx, result)
```

**Performance**: ~80% of hand-written CUDA

#### ✅ Automatic Optimization Selection

```python
@torch.compile(mode="max-autotune")
def my_func(x):
    return x.sum()
```

**What happens**:
- Tries multiple implementations
- Benchmarks each
- Picks the fastest
- Caches the decision

**Example**: For matrix multiply, tries different tile sizes, chooses best

#### ✅ Memory Reuse

**Your code**:
```python
x = input / temperature
y = x.max(dim=-1, keepdim=True)
z = x - y
```

**Naive**: Allocates 3 intermediate tensors

**torch.compile**: Reuses buffers, minimizes allocations

### What torch.compile Struggles With

#### ❌ Complex Reduction Patterns

**Your kernel**: 3 reductions (max, mean, variance) fused with pointwise ops

**torch.compile**:
- May not fuse all reductions
- Reductions often stay separate
- Can't use warp primitives explicitly

**Result**: 2-3 kernels instead of 1

#### ❌ Warp-Level Optimizations

**Custom CUDA**:
```cuda
// Direct warp shuffle - FAST
for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
}
```

**torch.compile**:
- Uses Triton, which abstracts this
- Triton generates warp code, but less control
- Can't hand-tune warp-level patterns

**Performance impact**: 20-30% slower on reduction-heavy code

#### ❌ Custom Memory Layouts

**Custom CUDA**: Full control over shared memory layout

**torch.compile**: Uses heuristics, can't always match hand-tuned

#### ❌ Extreme Specialization

**Custom CUDA**: Optimize for exact vocab_size, block_size, etc.

**torch.compile**: More general, less specialized

#### ⚠️ Parameter Hardcoding (Important!)

torch.compile **hardcodes scalar parameters** into the generated kernel by tracing with specific values:

**Example from our kernel**:

```python
# Original PyTorch call
result = compiled_fn(logits, temperature=1.5, top_nsigma=2.0)
```

**Generated Triton kernel**:
```python
@triton.jit
def triton_kernel(...):
    # Temperature is hardcoded as multiplication constant
    tmp1 = 0.6666666666666666  # Pre-computed 1/1.5
    tmp2 = tmp0 * tmp1

    # top_nsigma is hardcoded
    tmp15 = 2.0  # Hardcoded top_nsigma value
    tmp16 = tmp14 * tmp15
```

**What this means**:

✅ **Pros**:
- Faster execution (no runtime parameter loads)
- Better optimization (compiler knows exact values)
- Can pre-compute constants like `1/temperature`

❌ **Cons**:
- **Recompilation overhead** if you change parameters
- Different parameter values trigger new kernel generation
- Multiple cached kernels if you use many combinations
- Can't dynamically adjust parameters without recompiling

**Comparison with Custom CUDA**:

Our custom kernel takes parameters at **runtime**:
```cuda
void launch_temperature_topnsigma_fused(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size,
    float temperature,      // ✅ Runtime parameter
    float top_nsigma,       // ✅ Runtime parameter
    cudaStream_t stream
)
```

✅ **Pros**:
- No recompilation needed for different values
- One kernel works for all parameter combinations
- Instant parameter changes

❌ **Cons**:
- Parameters loaded at runtime (minor overhead)
- Can't pre-compute `1/temperature` at compile time

**Real-world impact**:

If you frequently change temperature during generation (e.g., adaptive sampling):

```python
# torch.compile: Triggers recompilation for each new temperature
for temp in [0.5, 0.7, 1.0, 1.5, 2.0]:
    result = compiled_fn(logits, temperature=temp, top_nsigma=2.0)
    # First call with new temp: slow (compilation)
    # Subsequent calls: fast (cached)
```

```cuda
// Custom CUDA: Same kernel for all temperatures
for (float temp : {0.5f, 0.7f, 1.0f, 1.5f, 2.0f}) {
    launch_kernel(input, output, ..., temp, top_nsigma);
    // All calls: fast (no recompilation)
}
```

**Recommendation**:
- torch.compile: Best when parameters are fixed or change rarely
- Custom CUDA: Better when parameters change frequently

### Performance Comparison (Actual Results)

For our temperature + top-nsigma operation (batch=8, seq_len=32, vocab_size=156,896):

| Implementation | Time (ms) | Speedup vs Eager | Bandwidth (GB/s) | Kernel Launches |
|----------------|-----------|------------------|------------------|-----------------|
| Eager PyTorch | ~4.43 | 1.0x | - | 5 |
| torch.compile (max-autotune) | **2.62** | 1.69x | 307 | 1 (fused!) |
| Custom CUDA (basic, 7 mem ops) | 4.63 | 0.96x | 173 | 1 |
| Custom CUDA (optimized, 3 mem ops) | **2.63** | 1.68x | 183 | 1 |

**Surprising results**:
- ✅ torch.compile achieves **PARITY** with optimized custom CUDA (2.62ms vs 2.63ms)!
- ✅ torch.compile fuses everything into a **SINGLE kernel** (not 2-3 as expected)
- ✅ torch.compile achieves higher bandwidth (307 GB/s vs 183 GB/s) through better Triton optimizations
- ❌ Naive custom CUDA (7 memory ops) was actually **slower** than eager PyTorch!

**Key insight**: torch.compile with max-autotune mode matches highly-optimized custom CUDA!

## When to Use Each Approach

### Use **Eager PyTorch** when:
- ✅ Prototyping / research
- ✅ Debugging
- ✅ Code clarity is priority
- ✅ Performance is "good enough"

### Use **TorchScript** when:
- ✅ Deploying models (shipping to production)
- ✅ Mobile deployment
- ✅ C++ inference
- ✅ Need scripted model format

### Use **torch.compile** when:
- ✅ **DEFAULT CHOICE** for most cases
- ✅ Want automatic optimization
- ✅ Don't want to write CUDA
- ✅ Good balance of effort vs performance
- ✅ PyTorch 2.0+ available

**Recommendation**: Start here, only go to custom CUDA if needed!

### Use **Custom CUDA** when:
- ✅ Absolute maximum performance needed
- ✅ Operation is a critical bottleneck
- ✅ torch.compile doesn't optimize well enough
- ✅ Need warp-level control
- ✅ Have CUDA expertise
- ✅ Operation is reused frequently

**Example use cases**:
- Attention kernels (FlashAttention, FlashAttention-2)
- Custom sampling algorithms
- Specialized fused operations
- Novel operations not in PyTorch

## Real-World Examples

### FlashAttention

**torch.compile approach**:
```python
# Naive attention
scores = Q @ K.T / sqrt(d)
attn = softmax(scores)
output = attn @ V
```

torch.compile fuses some ops, but:
- Memory usage: O(n²) for attention matrix
- Speed: ~2-3x slower than FlashAttention

**Custom CUDA (FlashAttention)**:
- Memory usage: O(1) - never materializes full attention matrix
- Speed: 2-4x faster with clever tiling and recomputation
- torch.compile **cannot** replicate this without the algorithm

**Verdict**: Custom CUDA wins for algorithmic innovations

### Layer Normalization

**torch.compile approach**:
```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
output = (x - mean) / sqrt(var + eps)
```

torch.compile result: Fuses most ops, pretty good

**Custom CUDA**: ~20% faster with Welford's algorithm

**Verdict**: torch.compile is competitive (20% faster probably not worth effort)

### Our Temperature + Top-NSigma (ACTUAL RESULTS)

**torch.compile (max-autotune mode)**:
- ✅ Fuses **EVERYTHING** into 1 kernel (not 2-3!)
- ✅ Uses Welford's algorithm for variance (numerically stable)
- ✅ 2-pass algorithm: statistics + filter (optimal strategy)
- ✅ Achieves 307 GB/s bandwidth
- ⚡ **2.62ms** - matches optimized custom CUDA!

**Custom CUDA (optimized)**:
- ✅ Everything fused into 1 kernel
- ✅ Uses E[X²]-E[X]² for variance
- ✅ Same 2-pass algorithm
- ⚠️ 183 GB/s bandwidth (lower than torch.compile!)
- ⚡ **2.63ms** - matches torch.compile!

**Both discovered the same optimal algorithm**:
- Pass 1: Load + scale + compute max/variance
- Pass 2: Re-load (from L2 cache) + apply threshold filter
- Total: 2 reads + 1 write (theoretical minimum for this algorithm)

**Verdict**: torch.compile wins on performance **AND** effort! Custom CUDA only needed for:
- Runtime parameter flexibility (no recompilation on param change)
- Algorithmic innovations torch.compile can't discover
- Specific hardware micro-optimizations

## Triton: The Middle Ground

**Triton** is a Python-based kernel language that torch.compile uses internally.

You can also write Triton kernels directly:

```python
import triton
import triton.language as tl

@triton.jit
def temperature_topnsigma_triton(
    logits_ptr, output_ptr,
    temperature, top_nsigma,
    vocab_size,
    BLOCK_SIZE: tl.constexpr
):
    # Similar to CUDA but in Python!
    pid = tl.program_id(0)
    offset = pid * vocab_size

    # Load and scale
    idx = tl.arange(0, BLOCK_SIZE)
    logits = tl.load(logits_ptr + offset + idx)
    scaled = logits / temperature

    # Reductions
    max_val = tl.max(scaled)
    std_val = tl.sqrt(tl.var(scaled))
    threshold = max_val - top_nsigma * std_val

    # Filter
    filtered = tl.where(scaled >= threshold, scaled, float('-inf'))
    tl.store(output_ptr + offset + idx, filtered)
```

**Benefits**:
- Easier than CUDA (Python syntax)
- Portable (works on AMD, NVIDIA)
- Pretty good performance (80-90% of CUDA)
- Less boilerplate than CUDA

**Drawbacks**:
- Less control than raw CUDA
- Can't use all CUDA features (warp intrinsics are abstracted)
- Smaller ecosystem than CUDA

**When to use**: Middle ground between torch.compile and custom CUDA

## The Future: What's Coming?

### 1. **Better Fusion in torch.compile**

PyTorch team is actively improving:
- Multi-kernel fusion passes
- Better reduction optimization
- More aggressive memory planning

**Target**: Close gap to 90% of custom CUDA performance

### 2. **Triton Improvements**

- Better code generation
- More low-level control
- Integration with torch.compile

### 3. **Automatic Warp Optimization**

Compilers are getting smarter:
- Auto-detect reduction patterns
- Generate warp-optimized code
- Match hand-written CUDA

### 4. **Compiler-Driven Specialization**

- Profile-guided optimization (PGO)
- Learn optimal configurations from runs
- Auto-tune for specific hardware

## Bottom Line: How Far Has PyTorch JIT Come?

### 2017: TorchScript
- **Gap to custom CUDA**: ~4x
- **Effort required**: Medium (decorators, debugging)
- **Verdict**: Helped for deployment, not performance

### 2023: torch.compile (PyTorch 2.0)
- **Gap to custom CUDA**: ~1.5-2x
- **Effort required**: Zero (just add @torch.compile)
- **Verdict**: **Game changer** - good enough for 90% of use cases

### 2026: torch.compile (now)
- **Gap to custom CUDA**: ~1.0x (can match with max-autotune!)
- **Improvements**: Excellent fusion, Welford optimizations, smart eviction policies
- **Verdict**: **Excellent** - custom CUDA only needed for:
  - Algorithmic innovations (like FlashAttention)
  - Runtime parameter flexibility
  - Specific hardware micro-optimizations

## Decision Tree

```
Need to optimize PyTorch code?
│
├─ Is it a critical bottleneck? (>10% total time)
│  ├─ No → Keep eager mode, optimize later
│  └─ Yes → Continue
│
├─ Add @torch.compile
│  │
│  ├─ Performance good enough? (within 2x of target)
│  │  └─ Yes → DONE! 🎉
│  │
│  └─ No → Continue
│
├─ Try @torch.compile(mode="max-autotune")
│  │
│  ├─ Performance good enough now?
│  │  └─ Yes → DONE! 🎉
│  │
│  └─ No → Continue
│
├─ Is this a novel algorithm? (like FlashAttention)
│  ├─ Yes → Write custom CUDA kernel
│  └─ No → Continue
│
├─ Try writing Triton kernel
│  │
│  ├─ Performance good enough?
│  │  └─ Yes → DONE! 🎉
│  │
│  └─ No → Write custom CUDA kernel
│
└─ Write custom CUDA kernel (you're here) ⚡
```

## Recommendations

### For 90% of users:
```python
# Just add this line!
@torch.compile
def my_function(x):
    # Your PyTorch code
    return result
```

**Effort**: 1 minute
**Speedup**: 2-3x
**ROI**: Excellent

### For performance-critical operations:
1. Start with torch.compile
2. Profile to find bottleneck
3. Try max-autotune mode
4. If still not fast enough, try Triton
5. Last resort: custom CUDA

### For researchers/engineers:
- **Default**: Use torch.compile everywhere
- **Critical ops**: Custom kernels (like we did)
- **Novel algorithms**: Start with custom kernels

## Conclusion

**PyTorch JIT has come VERY far**:
- torch.compile is production-ready and **excellent**
- Can **MATCH** optimized custom CUDA performance (with max-autotune)
- Automatically discovers optimal algorithms (2-pass, Welford, fusion)
- Zero effort required

**Custom CUDA still has a place**:
- **Algorithmic innovations** (operations torch.compile can't discover)
- **Runtime parameter flexibility** (avoid recompilation overhead)
- **Specific hardware micro-optimizations** (last 5-10% in critical code)
- **Novel operations** not expressible in PyTorch

**Our temperature + top-nsigma kernel (actual measurements)**:
- torch.compile (max-autotune): 2.62ms ⚡
- Custom CUDA (optimized): 2.63ms ⚡
- **They match!** torch.compile discovered the same optimal algorithm
- **Worth writing custom CUDA?** Only if you need runtime parameter flexibility or have algorithmic innovations torch.compile can't discover

**The future is bright**: Compilers will keep improving, but there will always be room for hand-optimized kernels in performance-critical code.
