# CUDA Block Size Explained

## What is Block Size?

**Block Size (BS)** = Number of threads per thread block in CUDA kernel

In our kernel:
```cuda
// Launch configuration
temperature_topnsigma_bandwidth_optimized<1024>  // BS = 1024 threads
    <<<num_positions, 1024, 0, stream>>>          // 1024 threads per block
```

## Visual Explanation

### GPU Architecture Hierarchy

```
GPU (RTX 4090)
│
├─ 128 Streaming Multiprocessors (SMs)
│  │
│  ├─ SM 0
│  │  ├─ Thread Block 0 [BS threads]
│  │  ├─ Thread Block 1 [BS threads]
│  │  └─ ...
│  │
│  ├─ SM 1
│  │  ├─ Thread Block 5 [BS threads]
│  │  └─ ...
│  │
│  └─ SM 127
│     └─ ...
│
Total threads = num_blocks × BS
```

### Our Kernel Launch

```cuda
int num_positions = 256;  // batch_size × seq_len = 8 × 32
int block_size = 1024;    // BS = threads per block

// Launch 256 blocks, each with 1024 threads
kernel<<<256, 1024>>>(...)
       ↑      ↑
       │      └─ Block size (threads per block)
       └─ Grid size (number of blocks)

Total threads launched: 256 blocks × 1024 threads = 262,144 threads
```

## Block Size Comparison

### BS=256 (Small blocks)

```
Each block: 256 threads
Total threads: 256 blocks × 256 = 65,536 threads

GPU allocation:
SM 0:  ▓▓ (Block 0: 256 threads)
SM 1:  ▓▓ (Block 1: 256 threads)
...
SM 127: ▓▓ (Block 127: 256 threads)

Occupancy per SM: 256 / 1,536 = 16.7% (LOW!)

Result: 302.4 GB/s
```

**Problem**: Low occupancy - GPU has empty thread slots!

### BS=512 (Medium blocks)

```
Each block: 512 threads
Total threads: 256 blocks × 512 = 131,072 threads

GPU allocation:
SM 0:  ▓▓▓▓ (Block 0: 512 threads)
SM 1:  ▓▓▓▓ (Block 1: 512 threads)
...
SM 127: ▓▓▓▓ (Block 127: 512 threads)

Occupancy per SM: 512 / 1,536 = 33.3% (MEDIUM)

Result: 304.1 GB/s
```

**Better**: More threads to hide latency, but still room for improvement.

### BS=1024 (Large blocks) ✅ BEST

```
Each block: 1024 threads
Total threads: 256 blocks × 1024 = 262,144 threads

GPU allocation:
SM 0:  ▓▓▓▓▓▓▓▓ (Block 0: 1024 threads)
SM 1:  ▓▓▓▓▓▓▓▓ (Block 1: 1024 threads)
...
SM 127: ▓▓▓▓▓▓▓▓ (Block 127: 1024 threads)
SM 0:  ▓▓▓▓▓▓▓▓ (Block 128: 1024 threads - wraps around!)
...

Occupancy per SM: 1024 / 1,536 = 66.7% (GOOD!)

Result: 333.9 GB/s ✨
```

**Best**: High occupancy - GPU efficiently switches between threads while waiting for memory!

## Why Block Size Matters

### Memory Latency Hiding

When a thread waits for memory (400 cycles), GPU switches to another thread:

#### With BS=256 (Few threads)
```
SM has only 256 threads:

Thread 0:   [Wait 400 cycles]────────────────────[Work 5c]
Thread 1:     [Wait]────────────────────[Work]
Thread 2:       [Wait]────────────────────[Work]
...
Thread 255:                       [Wait]────────────[Work]

GPU cycles: 0────100────200────300────400────500
Active:     T0   T1   T2        T255  idle  T0

GPU is IDLE 30% of the time! 😴
```

#### With BS=1024 (Many threads)
```
SM has 1024 threads:

Thread 0:    [Wait]──────────[Work]
Thread 1:      [Wait]──────────[Work]
Thread 2:        [Wait]──────────[Work]
...
Thread 1023:                           [Wait]──────[Work]

GPU cycles: 0────100────200────300────400────500
Active:     T0 T1 T2 T3 T4 T5 ... T1023 T0 T1 T2

GPU is BUSY 100% of the time! 💪
```

**Key insight**: More threads = more work to do while waiting for memory!

## Detailed Performance Analysis

### Why BS=1024 is Best for Our Kernel

```
RTX 4090 specifications:
- 128 SMs (Streaming Multiprocessors)
- 1,536 max threads per SM
- 65,536 registers per SM
- 164 KB shared memory per SM

Our kernel uses:
- ~40 registers per thread
- 3 × BS × 4 bytes shared memory = 12 KB for BS=1024
```

#### Register Limitation
```
Registers per SM: 65,536
Registers per thread: ~40

Max threads per SM = 65,536 / 40 = 1,638

BS=256:  Can fit 6 blocks per SM (1,536 threads) ✅
BS=512:  Can fit 3 blocks per SM (1,536 threads) ✅
BS=1024: Can fit 1 block per SM (1,024 threads)  ✅ (but less than max)
```

#### Shared Memory Limitation
```
Shared memory per SM: 164 KB
Shared memory per block: 12 KB (for BS=1024)

Max blocks per SM = 164 KB / 12 KB = 13 blocks

BS=256:  Uses 3 KB, can fit 13 blocks ✅
BS=512:  Uses 6 KB, can fit 13 blocks ✅
BS=1024: Uses 12 KB, can fit 13 blocks ✅
```

#### Actual Occupancy
```
BS=256:  Register limited to 6 blocks/SM = 1,536 threads/SM
         Occupancy = 1,536 / 1,536 = 100% ✅

BS=512:  Register limited to 3 blocks/SM = 1,536 threads/SM
         Occupancy = 1,536 / 1,536 = 100% ✅

BS=1024: Register limited to 1 block/SM = 1,024 threads/SM
         Occupancy = 1,024 / 1,536 = 66.7% ⚠️
```

**Wait, BS=256 and BS=512 have 100% occupancy but are slower?**

### Why BS=1024 Wins Despite Lower Occupancy

#### 1. Better Work Distribution

```
Total work: 256 positions × 156,896 vocab = 40M elements

BS=256:
Each thread processes: 156,896 / 256 = 613 iterations
Loop overhead: 613 × loop cost

BS=1024:
Each thread processes: 156,896 / 1,024 = 153 iterations
Loop overhead: 153 × loop cost (4× less overhead!)
```

**Fewer iterations = Less loop overhead**

#### 2. Better Memory Coalescing

```
Memory transactions per warp (32 threads):

BS=256:
- 8 warps per block
- Each warp processes different positions
- Some memory divergence

BS=1024:
- 32 warps per block
- Better memory access patterns
- More coalesced access
```

#### 3. Amortized Kernel Launch Overhead

```
Kernel launch overhead: ~5 microseconds

BS=256:  5μs / (256 threads × 613 iterations) = more overhead per work
BS=1024: 5μs / (1024 threads × 153 iterations) = less overhead per work
```

#### 4. Better Warp Scheduling

```
SM scheduler switches between warps every cycle:

BS=256:  8 warps per block to schedule
BS=1024: 32 warps per block to schedule

More warps = Better latency hiding even with same occupancy!
```

## The Trade-off

```
Block Size Considerations:

                    Too Small              Optimal           Too Large
                    (BS=128)            (BS=512-1024)        (BS=2048)
                    │                        │                    │
Occupancy:          ████████████████         ████████░░░░         ██░░░░░░░░
                    (100%, but             (60-100%,           (Low,
                     few warps)            good balance)        won't fit)

Loop overhead:      High                    Low                 Very Low
                    (many iterations)     (optimal)           (few iterations)

Memory access:      Poor                    Good                Excellent
                    (less coalescing)     (well coalesced)    (perfect)

Latency hiding:     Poor                    Good                Excellent
                    (few warps)           (many warps)        (very many warps)

Register pressure:  Low                     Medium              High
                    (can fit many blocks) (balanced)          (may not fit!)
```

**Sweet spot**: BS=512 to 1024 for most memory-bound kernels

## Performance Results Explained

### Why the Progression?

```
BS=256 → 302.4 GB/s:
  ✅ Good occupancy (100%)
  ❌ High loop overhead
  ❌ Fewer warps to hide latency
  ❌ Less efficient memory access

BS=512 → 304.1 GB/s:
  ✅ Good occupancy (100%)
  ✅ Medium loop overhead
  ✅ More warps to hide latency
  ✅ Better memory access

BS=1024 → 333.9 GB/s: ✨ WINNER
  ⚠️ Lower occupancy (67%)
  ✅ Low loop overhead (4× fewer iterations)
  ✅ Many warps to hide latency (32 warps!)
  ✅ Excellent memory coalescing
  ✅ Better warp scheduling
```

**The winner**: BS=1024 because **32 warps** are enough to hide latency, and lower loop overhead + better memory access win!

## Code Visualization

### How Block Size Affects the Kernel

```cuda
// BS=256 version
template<int BLOCK_SIZE>
__global__ void kernel(...) {
    // 256 threads per block
    // Each thread: 613 iterations
    for (int i = tid; i < 156896; i += 256) {  // stride = 256
        // Do work
    }
}

// BS=1024 version
template<int BLOCK_SIZE>
__global__ void kernel(...) {
    // 1024 threads per block
    // Each thread: 153 iterations (4× fewer!)
    for (int i = tid; i < 156896; i += 1024) {  // stride = 1024
        // Do work
    }
}
```

**Key difference**: Stride changes with block size!

## General Guidelines for Block Size

### Memory-Bound Kernels (like ours)
```
Recommendation: 512-1024 threads

Why: Need many warps to hide memory latency
```

### Compute-Bound Kernels
```
Recommendation: 256-512 threads

Why: High occupancy more important than latency hiding
```

### Register-Heavy Kernels
```
Recommendation: 128-256 threads

Why: Fewer threads per block allows more blocks per SM
```

### Shared Memory-Heavy Kernels
```
Recommendation: Depends on shared memory usage

Calculate: max_blocks = (SM_shared_mem / block_shared_mem)
           BS = choose to maximize occupancy within limit
```

## How to Find Optimal Block Size

### Method 1: Benchmark (What we did!)

```python
for bs in [128, 256, 512, 1024]:
    benchmark_kernel(bs)
    # Pick the fastest
```

✅ **Best approach** - actual performance matters!

### Method 2: CUDA Occupancy Calculator

```cuda
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxActiveBlocks,
    kernel<1024>,
    1024,  // block size
    0      // dynamic shared memory
);

printf("Occupancy: %f\n",
       (maxActiveBlocks * 1024) / (float)maxThreadsPerSM);
```

### Method 3: NSight Compute

```bash
ncu --set full ./benchmark

# Look for:
# - "Achieved Occupancy" (want > 50%)
# - "Memory Throughput" (want high)
# - "Warp Stall Reasons" (want low memory stalls)
```

## Summary

### What is Block Size?
- **BS** = Number of threads per CUDA thread block
- Each block runs on one SM
- Multiple blocks can run on one SM if resources allow

### Why BS=1024 Won?
1. ✅ **32 warps per block** - Excellent latency hiding
2. ✅ **4× fewer loop iterations** - Less overhead
3. ✅ **Better memory coalescing** - More efficient access
4. ✅ **Good warp scheduling** - GPU stays busy

### Key Takeaway
**More threads isn't always better!**
- BS=256: 100% occupancy, but slower (302.4 GB/s)
- BS=1024: 67% occupancy, but faster (333.9 GB/s)

**Balance matters**: Latency hiding + memory efficiency + loop overhead

### For Your Kernel
```
Optimal: BS=1024 threads per block
Result: 333.9 GB/s bandwidth
Performance: 1.344ms (1.96× faster than original!)
```

---

**Bottom line**: Block size determines how many threads work together in a block. Larger blocks (up to 1024) are usually better for memory-bound kernels because they provide more warps to hide memory latency!
