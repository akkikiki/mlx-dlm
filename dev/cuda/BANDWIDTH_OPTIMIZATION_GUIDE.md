# CUDA Kernel Bandwidth Optimization Guide

## Goal
Improve our custom CUDA kernel bandwidth from 183 GB/s to match torch.compile's 307 GB/s.

## Current Status

| Kernel | Bandwidth | Performance |
|--------|-----------|-------------|
| torch.compile | 307 GB/s | 2.62ms |
| Our custom CUDA | 183 GB/s | 2.63ms |

**Question**: Why does our kernel match torch.compile's time despite 40% lower bandwidth?
**Answer**: L2 cache! Our second read hits cache (effectively free), so lower measured bandwidth still gives same performance.

## Bandwidth Optimization Techniques

### 1. Memory Coalescing ✅ (Already Optimized)

**What it is**: Threads in a warp access consecutive memory addresses in a single transaction.

**Current code** (already good):
```cuda
for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
    float val = input[offset + i] / temperature;  // Good: consecutive access
    // offset = pos_idx * vocab_size
    // Thread 0 reads input[offset + 0]
    // Thread 1 reads input[offset + 1]
    // Thread 2 reads input[offset + 2]
    // ... (32 threads in warp access consecutive addresses)
}
```

**Why it's good**: 32 threads in a warp load addresses `input[offset + 0..31]` in a single 128-byte memory transaction.

**Bad example** (don't do this):
```cuda
// BAD: Strided access - each thread in different cache line
float val = input[offset + tid * 32];  // Thread 0: offset+0, Thread 1: offset+32, ...
```

### 2. Vectorized Memory Access ⚡ (NEW OPTIMIZATION)

**Problem**: Loading 1 float at a time (4 bytes) doesn't saturate memory bus.

**Solution**: Load multiple floats per thread using `float4`.

**Optimized code**:
```cuda
template<int BLOCK_SIZE>
__global__ void temperature_topnsigma_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int vocab_size,
    const float temperature,
    const float top_nsigma
) {
    const int pos_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int offset = pos_idx * vocab_size;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Load 4 floats at once (16 bytes per load)
    const int vec_size = 4;
    const int vec_vocab_size = vocab_size / vec_size;

    for (int i = tid; i < vec_vocab_size; i += BLOCK_SIZE) {
        // Load 4 floats in one transaction
        float4 vals = reinterpret_cast<const float4*>(input + offset)[i];

        // Process all 4 values
        float v1 = vals.x / temperature;
        float v2 = vals.y / temperature;
        float v3 = vals.z / temperature;
        float v4 = vals.w / temperature;

        local_max = fmaxf(local_max, fmaxf(fmaxf(v1, v2), fmaxf(v3, v4)));
        local_sum += v1 + v2 + v3 + v4;
        local_sum_sq += v1*v1 + v2*v2 + v3*v3 + v4*v4;
    }

    // Handle remainder if vocab_size not divisible by 4
    for (int i = vec_vocab_size * vec_size + tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = input[offset + i] / temperature;
        local_max = fmaxf(local_max, val);
        local_sum += val;
        local_sum_sq += val * val;
    }

    // ... rest of kernel (reductions, filtering)
}
```

**Expected improvement**: 1.3-1.5× bandwidth increase (240-275 GB/s)

### 3. Optimize Block Size ⚡ (NEW OPTIMIZATION)

**Problem**: Our BLOCK_SIZE=256 might not be optimal for memory bandwidth.

**Current code**:
```cuda
const int BLOCK_SIZE = 256;
temperature_topnsigma_fused<256><<<grid, BLOCK_SIZE>>>(...)
```

**Optimal approach**: Auto-tune or use larger blocks for better occupancy.

**Test different block sizes**:
```cuda
// In test_optimized_kernel.py or new benchmark script
std::vector<int> block_sizes = {128, 256, 512, 1024};

for (int bs : block_sizes) {
    // Launch kernel with this block size
    // Measure bandwidth
    // Pick best
}
```

**Why it matters**:
- Larger blocks = more threads to hide memory latency
- But too large = lower occupancy (fewer blocks fit on SM)
- Optimal is usually 256-512 for memory-bound kernels

**Expected improvement**: 5-10% bandwidth increase

### 4. Prefetching / Non-Temporal Loads ⚡ (ADVANCED)

**Problem**: GPU doesn't know we'll read data twice (pass 1 and pass 2).

**Solution for Pass 1**: Use cache-last hint (keep in L2 for pass 2)
```cuda
// Pass 1: Keep in cache
for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
    // Use __ldg() for read-only cached load
    float val = __ldg(&input[offset + i]) / temperature;
    // Or use PTX for cache hint
    local_max = fmaxf(local_max, val);
    local_sum += val;
    local_sum_sq += val * val;
}
```

**Solution for Pass 2**: Streaming store (don't pollute cache)
```cuda
// Pass 2: Write-through to memory
for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
    float val = input[offset + i] / temperature;  // Should hit L2
    float result = (val >= threshold) ? val : -INFINITY;

    // Non-temporal store (don't cache, just write to memory)
    __stwt(&output[offset + i], result);  // Streaming store
}
```

**Note**: `__stwt` is for compute capability 8.0+ (Ampere/Ada/Hopper)

**Expected improvement**: 10-15% bandwidth increase (cache efficiency)

### 5. Increase Occupancy ⚡ (NEW OPTIMIZATION)

**Problem**: Low occupancy = threads waiting for memory.

**Check current occupancy**:
```cuda
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks,
    temperature_topnsigma_fused<256>,
    256,  // threads per block
    0     // dynamic shared memory
);

float occupancy = (numBlocks * 256) / maxThreadsPerSM;
printf("Occupancy: %.2f%%\n", occupancy * 100);
```

**Increase occupancy**:
```cuda
// 1. Reduce register usage
__launch_bounds__(256, 4)  // 256 threads, 4 blocks per SM minimum
template<int BLOCK_SIZE>
__global__ void temperature_topnsigma_fused(...) {
    // Compiler will try to use fewer registers
}

// 2. Reduce shared memory usage (if needed)
// We use 3 * BLOCK_SIZE * 4 bytes = 3KB for BLOCK_SIZE=256
// This is fine, but could optimize if needed
```

**Expected improvement**: 5-10% if occupancy is currently low

### 6. Reduce Shared Memory Bank Conflicts

**Current code** (check for conflicts):
```cuda
shared_max[tid] = local_max;
shared_sum[tid] = local_sum;
shared_sum_sq[tid] = local_sum_sq;
```

**Check**: If BLOCK_SIZE=256, threads access shared memory sequentially - no conflicts! ✅

**Potential optimization** (if we had conflicts):
```cuda
// Pad shared memory to avoid bank conflicts
__shared__ float shared_max[BLOCK_SIZE + 1];  // +1 padding
```

**Current status**: Already optimal ✅

### 7. Overlap Computation with Memory Access

**Problem**: GPU waits for memory loads.

**Solution**: Load next iteration while computing current.

**Optimized code** (software pipelining):
```cuda
// Prefetch first iteration
int i = tid;
float val_next = (i < vocab_size) ? input[offset + i] / temperature : 0.0f;

for (i = tid; i < vocab_size - BLOCK_SIZE; i += BLOCK_SIZE) {
    // Use current value
    float val = val_next;

    // Prefetch next iteration (overlaps with computation)
    val_next = input[offset + i + BLOCK_SIZE] / temperature;

    // Compute with current value
    local_max = fmaxf(local_max, val);
    local_sum += val;
    local_sum_sq += val * val;
}

// Process last iteration
if (i < vocab_size) {
    local_max = fmaxf(local_max, val_next);
    local_sum += val_next;
    local_sum_sq += val_next * val_next;
}
```

**Expected improvement**: 5-10% (hides memory latency)

### 8. Fuse Reductions ⚡ (OPTIMIZATION)

**Current code**: 3 separate reduction loops (max, sum, sum_sq)

**Problem**: Each loop has syncthreads() overhead

**Optimized code**: Fuse into single reduction
```cuda
// CURRENT: 3 separate loops
for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
    if (tid < stride) shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid+stride]);
    __syncthreads();
}
for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
    if (tid < stride) shared_sum[tid] += shared_sum[tid+stride];
    __syncthreads();
}
for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
    if (tid < stride) shared_sum_sq[tid] += shared_sum_sq[tid+stride];
    __syncthreads();
}

// OPTIMIZED: Single fused loop
for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid+stride]);
        shared_sum[tid] += shared_sum[tid+stride];
        shared_sum_sq[tid] += shared_sum_sq[tid+stride];
    }
    __syncthreads();  // Only one syncthreads per iteration
}
```

**Expected improvement**: 5% (reduces sync overhead)

## Implementation Priority

### Tier 1: High Impact, Easy to Implement
1. ✅ **Fuse reductions** (5% improvement, 5 min to implement)
2. ✅ **Vectorized loads (float4)** (30-40% improvement, 30 min to implement)
3. ✅ **Optimize block size** (5-10% improvement, 15 min to benchmark)

### Tier 2: Medium Impact, Moderate Effort
4. **Prefetching / Cache hints** (10-15% improvement, 1 hour)
5. **Occupancy tuning** (5-10% improvement, 30 min)

### Tier 3: Lower Impact / Advanced
6. **Software pipelining** (5-10% improvement, 2 hours)
7. **Non-temporal stores** (requires Ampere+ GPU)

## Expected Bandwidth After Optimizations

| Optimization | Cumulative Bandwidth | Performance |
|--------------|---------------------|-------------|
| Current | 183 GB/s | 2.63ms |
| + Fused reductions | 192 GB/s | 2.50ms |
| + Vectorized loads | 267 GB/s | 1.80ms |
| + Block size tuning | 287 GB/s | 1.67ms |
| + Cache hints | 315 GB/s | 1.52ms |

**Target**: 300+ GB/s to match torch.compile

## Implementation Script

I'll create an optimized version with Tier 1 optimizations that should get us close to 300 GB/s.

## Why torch.compile Gets 307 GB/s

Looking at the Triton kernel we found:
1. ✅ Uses vectorized loads (Triton does this automatically)
2. ✅ Optimal block sizes (auto-tuned with size_hints=[256, 262144])
3. ✅ Cache eviction policies (`eviction_policy='evict_last'` and `'evict_first'`)
4. ✅ Welford's algorithm (slightly more efficient than our variance computation)
5. ✅ Compiler optimizations (instruction scheduling, prefetching)

Our custom kernel can match this with the optimizations above!
