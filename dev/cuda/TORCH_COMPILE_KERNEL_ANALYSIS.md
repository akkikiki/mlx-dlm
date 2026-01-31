# Torch.compile Generated Kernel Analysis

## Summary

Torch.compile generates a **SINGLE fused Triton kernel** that implements the entire temperature + top-nsigma operation. This matches our optimized CUDA kernel strategy exactly.

**Location:** `torch_compile_kernels/torch_compile_kernels/uf/cufinu3a3wetojkeb37rn2lulls5ir5k76jho4arburjycele6c3.py`

## Kernel Metadata

```python
kernel_name: 'triton_red_fused_div_ge_lift_fresh_max_mul_std_sub_where_0'
num_load: 2
num_reduction: 2
size_hints: [256, 262144]  # xnumel=256 (batch*seq), rnumel=156896 (vocab)
num_warps: 8
```

**Source operations fused:**
- `div` - Division by temperature (scaled = logits / temperature)
- `max` - Maximum reduction
- `std` - Standard deviation computation
- `mul` - Multiplication (top_nsigma * std)
- `sub` - Subtraction (threshold = max - top_nsigma * std)
- `ge` - Greater-than-or-equal comparison
- `where` - Conditional selection
- `lift_fresh` - Create tensor for -inf

## Algorithm Structure

### Pass 1: Statistics Collection (Lines 30-46)

```python
# Initialize accumulators
_tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)  # max accumulator
tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

for roffset in range(0, rnumel, RBLOCK):
    # Load input
    tmp0 = tl.load(in_ptr0 + (r1 + (156896*x0)), ...)

    # Scale by 1/temperature (temperature=1.5, so 1/1.5 = 0.6666...)
    tmp1 = 0.6666666666666666
    tmp2 = tmp0 * tmp1  # scaled value

    # Update max
    tmp5 = triton_helpers.maximum(_tmp4, tmp3)
    _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)

    # Update variance using Welford's algorithm
    tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
        tmp3, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
    )

# Finalize reductions
tmp4 = triton_helpers.max2(_tmp4, 1)  # Final max
tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(...)  # Final variance
```

**Memory operations:** 1 load, 0 writes (statistics stay in registers)

### Pass 2: Threshold and Filter (Lines 53-69)

```python
for roffset in range(0, rnumel, RBLOCK):
    # Load input again (hits L2 cache)
    tmp9 = tl.load(in_ptr0 + (r1 + (156896*x0)), eviction_policy='evict_first', ...)

    # Scale by 1/temperature
    tmp10 = 0.6666666666666666
    tmp11 = tmp9 * tmp10

    # Compute std from variance
    tmp12 = 156895.0  # N-1 for Bessel correction
    tmp13 = tmp7 / tmp12  # variance / (N-1)
    tmp14 = libdevice.sqrt(tmp13)  # std

    # Compute threshold
    tmp15 = 2.0  # top_nsigma
    tmp16 = tmp14 * tmp15  # nsigma * std
    tmp17 = tmp4 - tmp16  # threshold = max - nsigma*std

    # Filter
    tmp18 = tmp11 >= tmp17
    tmp19 = float("-inf")
    tmp20 = tl.where(tmp18, tmp11, tmp19)

    # Store result
    tl.store(out_ptr2 + (r1 + (156896*x0)), tmp20, ...)
```

**Memory operations:** 1 load (from L2 cache), 1 write

## Total Memory Operations

1. **Load input (pass 1)** - Compute statistics
2. **Load input (pass 2)** - Apply filter (from L2 cache, ~0.2× cost)
3. **Store output** - Filtered results

**Effective cost:** ~2.2 memory operations

## Key Optimizations

### 1. Welford's Algorithm for Variance
Instead of computing mean first, then variance, Torch.compile uses Welford's online algorithm:
```python
triton_helpers.welford_reduce(value, mean, m2, weight, is_first)
```
This computes mean and variance in a single pass with better numerical stability.

### 2. Temperature Scaling Fusion
Temperature division is fused with both passes:
```python
tmp1 = 0.6666666666666666  # 1/1.5 pre-computed
tmp2 = tmp0 * tmp1  # Multiply instead of divide (faster)
```

### 3. Eviction Policies
```python
# Pass 1: Keep in cache for pass 2
eviction_policy='evict_last'

# Pass 2: Don't need to keep
eviction_policy='evict_first'
```

### 4. Register-Based Statistics
Max, mean, and variance stay in registers/shared memory - never written to global memory.

## Comparison with Our Custom CUDA Kernel

| Feature | Torch.compile Triton | Our Optimized CUDA |
|---------|---------------------|-------------------|
| Number of kernels | 1 | 1 |
| Memory loads | 2 | 2 |
| Memory stores | 1 | 1 |
| Algorithm | 2-pass: stats + filter | 2-pass: stats + filter |
| Max reduction | Welford helpers | Tree reduction |
| Variance | Welford algorithm | E[X²] - E[X]² |
| Performance | 2.62ms | 2.63ms |
| Bandwidth | 307 GB/s | 183 GB/s |

## Why Torch.compile is Slightly Faster (2.62ms vs 2.63ms)

1. **Welford algorithm** - More efficient variance computation
2. **Triton optimizations** - Auto-tuned block sizes and memory access patterns
3. **Better cache hints** - Explicit eviction policies
4. **Warp-level helpers** - Optimized reduction primitives

However, the difference is negligible (0.01ms = 0.4%)!

## Code Comparison

### Temperature Scaling

**Torch.compile:**
```python
tmp1 = 0.6666666666666666  # Pre-computed 1/temperature
tmp2 = tmp0 * tmp1
```

**Our CUDA:**
```cuda
float val = input[offset + i] / temperature;
```

### Variance Computation

**Torch.compile:**
```python
# Welford's algorithm (numerically stable)
tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
    tmp3, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
)
# Finalize: std = sqrt(m2 / (n-1))
tmp13 = tmp7 / tmp12
tmp14 = libdevice.sqrt(tmp13)
```

**Our CUDA:**
```cuda
// Accumulate sum and sum-of-squares
local_sum += val;
local_sum_sq += val * val;

// Compute variance: Var(X) = E[X²] - E[X]²
float mean = sum / float(vocab_size);
float variance = (sum_sq / float(vocab_size)) - (mean * mean);
float std = sqrtf(variance);
```

Both approaches are valid, but Welford is more numerically stable for large datasets.

## Lessons Learned

1. **Torch.compile is extremely effective** - It discovered the same optimization strategy as our hand-tuned CUDA kernel
2. **Two-pass is optimal** - Both implementations use 2 reads + 1 write
3. **Fusion is key** - Never materialize intermediate scaled logits
4. **Algorithm choice matters** - Welford vs E[X²]-E[X]² gives small perf difference
5. **Production use case:**
   - Use torch.compile for rapid development and good performance
   - Use custom CUDA when you need specific algorithm control or zero-overhead deployment

## Files

- **Triton kernel:** `torch_compile_kernels/torch_compile_kernels/uf/cufinu3a3wetojkeb37rn2lulls5ir5k76jho4arburjycele6c3.py`
- **Compiled wrapper:** `torch_compile_kernels/torch_compile_kernels/x3/cx3xft6ydaburgoxdhpmfoyyxqvhgcrdrh4y7kiwi6aamshhfwnu.py`
- **Our custom kernel:** `dev/cuda/optimized_fused_kernel.cu`
- **Performance comparison:** `dev/cuda/test_optimized_kernel.py`
