# What is Latency Hiding?

## The Restaurant Analogy

Imagine you're at a restaurant:

### Latency = How Long to Get Your Order
```
You: "I'll have the pasta"
     [Wait 20 minutes] ⏰
Chef: "Here's your pasta!"

Latency = 20 minutes (time from order to delivery)
```

### Bandwidth = How Many Orders Per Hour
```
Restaurant serves 60 meals/hour

Bandwidth = 60 meals/hour
```

### The Problem: One Customer at a Time
```
Customer 1: Order → [Wait 20min] → Eat → Leave
Customer 2:          Order → [Wait 20min] → Eat → Leave
Customer 3:                   Order → [Wait 20min] → Eat

Total time for 3 customers: 60+ minutes
Restaurant is IDLE most of the time! 😴
```

### The Solution: Seat Multiple Customers (Latency Hiding!)
```
Customer 1: Order → [Wait 20min] → Eat
Customer 2:   Order → [Wait 20min] → Eat
Customer 3:     Order → [Wait 20min] → Eat

Total time: 20 minutes (same as 1 customer!)
Restaurant is BUSY the whole time! 💪

While Customer 1 waits, kitchen works on Customer 2 & 3!
```

**This is latency hiding**: Do other work while waiting!

---

## GPU Memory: The Same Problem

### Without Latency Hiding (Your Current Kernel)

```cuda
// Thread execution timeline:
for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
    float val = input[offset + i];  // ← Request data from DRAM

    // Thread STALLS waiting for data
    [Wait 400 cycles] ⏰⏰⏰⏰⏰⏰⏰⏰⏰⏰

    // Finally data arrives
    local_sum += val / temperature;  // ← 5 cycles of work

    // Request next data
    // Wait again... 😴
}
```

**Timeline visualization**:
```
Thread 0: [Request]──[Wait 400 cycles]──[Work 5c]──[Request]──[Wait 400c]──[Work 5c]

          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓°°░░ (98% idle, 2% working!)

          ▓ = Waiting for memory
          ° = Computing
          ░ = Requesting next data
```

**Problem**: Thread spends 98% of time WAITING!

---

## With Latency Hiding (Optimized)

### Technique 1: Software Pipelining

```cuda
// Prefetch FIRST value (start the wait early)
int i = tid;
float val_next = input[offset + i];  // ← Request data, but DON'T WAIT yet!

for (i = tid; i < vocab_size - BLOCK_SIZE; i += BLOCK_SIZE) {
    // By now, val_next has probably arrived (we did other work)
    float val_current = val_next;  // ← Use prefetched value

    // While we COMPUTE with current value...
    local_sum += val_current / temperature;  // 5 cycles
    local_max = fmaxf(local_max, val_current);

    // ...GPU FETCHES next value IN PARALLEL!
    val_next = input[offset + i + BLOCK_SIZE];  // ← Overlapped!
}
```

**Timeline visualization**:
```
Thread 0: [Request0]──[Request1]──[Work0 + Fetch1]──[Work1 + Fetch2]──[Work2]

          ░░[Wait]▓▓░░°°°°▓▓°°°°▓▓°°°°▓▓ (50% working!)

          ░ = Request next
          ▓ = Waiting (but doing work in parallel!)
          ° = Computing (while fetching next)
```

**Result**: Thread is 50% busy instead of 2%!

### Technique 2: Multiple Threads (Occupancy)

If one thread is waiting, GPU switches to another thread:

```
SM (Streaming Multiprocessor) with 4 threads:

Thread 0: [Wait]▓▓▓▓▓▓▓▓▓▓▓▓°° [Wait]▓▓▓▓▓▓▓▓▓▓▓▓°°
Thread 1:     [Wait]▓▓▓▓▓▓▓▓▓▓▓▓°° [Wait]▓▓▓▓▓▓▓▓▓▓▓▓°°
Thread 2:         [Wait]▓▓▓▓▓▓▓▓▓▓▓▓°° [Wait]▓▓▓▓▓▓▓▓▓▓▓▓°°
Thread 3:             [Wait]▓▓▓▓▓▓▓▓▓▓▓▓°° [Wait]▓▓▓▓▓▓▓▓▓▓▓▓°°

GPU view:
Time:     0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
Active:   T0 T0 T0 T1 T1 T1 T2 T2 T2 T3 T3 T3 T0 T0 T0 T1
          ░░ °° ░░ °° ░░ °° ░░ °° ░░ °° ░░ °° ░░ °° ░░ °°

GPU is ALWAYS working on something! (100% utilization)
```

**This is why GPUs need thousands of threads!**

---

## Concrete Example: Our Temperature+Top-NSigma Kernel

### Current Code (No Latency Hiding)

```cuda
for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
    float val = input[offset + i] / temperature;
    //     ↑
    //     Request data, WAIT, then divide
    //     400 cycles wait + 5 cycles compute = 405 cycles total

    local_max = fmaxf(local_max, val);
    local_sum += val;
    local_sum_sq += val * val;
}

// For vocab_size = 156,896, BLOCK_SIZE = 256:
// Loop iterations: 156,896 / 256 = 613 iterations per thread
// Total time: 613 iterations × 405 cycles = 248,265 cycles
// At 3 GHz: 248,265 / 3,000,000,000 = 0.08ms per thread
```

But we have 256 threads, and they overlap some, so actual time is ~1.3ms.

**Issue**: 98% of cycles are wasted waiting!

### With Software Pipelining (Latency Hiding)

```cuda
// Prefetch first value
int i = tid;
float val_next = (i < vocab_size) ? input[offset + i] / temperature : 0.0f;
//                                   ↑ Request issued, but continue immediately

// Loop with prefetching
for (i = tid; i < vocab_size - BLOCK_SIZE; i += BLOCK_SIZE) {
    // Use the prefetched value (it's been fetching while we did other work)
    float val = val_next;

    // Start fetching NEXT iteration's value NOW (in parallel with computation)
    val_next = input[offset + i + BLOCK_SIZE] / temperature;
    //         ↑ This fetch happens WHILE we compute below

    // Compute with current value (overlaps with fetch of next)
    local_max = fmaxf(local_max, val);  // These 3 operations
    local_sum += val;                    // take ~5 cycles
    local_sum_sq += val * val;           // while GPU fetches next (400 cycles)

    // By the time we loop back, val_next is ready (or almost ready)!
}

// Handle last value
if (i < vocab_size) {
    local_max = fmaxf(local_max, val_next);
    local_sum += val_next;
    local_sum_sq += val_next * val_next;
}

// New timing:
// Loop iterations: 613
// Per iteration: max(5 cycles compute, 400 cycles fetch) = 400 cycles
//                BUT they overlap! Effective time: ~200 cycles
// Total time: 613 × 200 cycles = 122,650 cycles
// At 3 GHz: 0.04ms per thread (2× faster!)
```

**Result**: 2× speedup by hiding latency!

---

## Visual Timeline Comparison

### Without Latency Hiding (Current)
```
Iteration 1:    [Request mem]────[Wait 400 cycles]────[Compute 5c]──┐
Iteration 2:                [Request mem]────[Wait 400 cycles]────[Compute 5c]──┐
Iteration 3:                            [Request mem]────[Wait 400 cycles]────[Compute 5c]

Total: 405 + 405 + 405 = 1,215 cycles for 3 iterations
```

### With Latency Hiding (Software Pipelining)
```
Iter 1: [Req1]────[Wait 400c]────[Comp1 + Req2]──┐
Iter 2:                           [Comp2 + Req3]──┐ (Req2 finished during Comp1!)
Iter 3:                                      [Comp3]  (Req3 finished during Comp2!)

Total: 400 + 5 + 5 + 5 = 415 cycles for 3 iterations (3× faster!)
```

---

## How GPUs Hide Latency Automatically

GPUs use **occupancy** (many threads) to hide latency:

### Low Occupancy (Bad)
```
1 thread per SM:
Thread 0: [Request]──[Wait 400 cycles IDLE]──[Work 5c]

SM Utilization: 5/405 = 1.2% 😢
```

### High Occupancy (Good)
```
80 threads per SM:
Thread 0:  [Req]──[Wait]──────────────────────────────[Work]
Thread 1:      [Req]──[Wait]──────────────────────────[Work]
Thread 2:          [Req]──[Wait]──────────────────────[Work]
...
Thread 79:                                          [Req]──[Wait]──[Work]

SM switches between threads while they wait!

SM Utilization: ~100% 🎉
```

**This is why your kernel uses 256-1024 threads per block!**

---

## Latency Hiding Techniques Summary

### 1. Software Pipelining (Prefetching)
**What**: Request next data while processing current
**When**: When you have sequential data access
**Speedup**: 2-3×
**Difficulty**: Medium (need to restructure loop)

### 2. Increased Occupancy (More Threads)
**What**: Launch more threads so GPU can switch between them
**When**: Always (basic GPU programming)
**Speedup**: 5-10× (if currently low occupancy)
**Difficulty**: Easy (just use more threads)

### 3. Asynchronous Memory Operations
**What**: Use cudaMemcpyAsync to overlap copy with compute
**When**: Multi-kernel workflows
**Speedup**: 2-5× for multi-kernel workloads
**Difficulty**: Hard (need streams, events)

### 4. Instruction Level Parallelism (ILP)
**What**: Unroll loops so GPU can issue multiple instructions
**When**: Small, tight loops
**Speedup**: 1.2-1.5×
**Difficulty**: Easy (use #pragma unroll)

---

## Code Example: Before vs After

### BEFORE (No Latency Hiding)
```cuda
for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
    // Sequential: Request → Wait → Compute → Request → Wait → Compute
    float val = input[offset + i] / temperature;
    local_sum += val;
}

Performance: 2.63ms (98% waiting)
```

### AFTER (With Latency Hiding)
```cuda
// Prefetch first
float val_next = input[offset + tid] / temperature;

for (int i = tid; i < vocab_size - BLOCK_SIZE; i += BLOCK_SIZE) {
    // Parallel: Request_next + Compute_current (overlapped!)
    float val = val_next;
    val_next = input[offset + i + BLOCK_SIZE] / temperature;  // Fetch while computing

    local_sum += val;  // Compute while fetching
}

Performance: 1.0ms (50% waiting, 50% computing)
Speedup: 2.6×
```

---

## Why This Matters for Your Question

You asked: **"Does high GPU bandwidth mean more optimization potential?"**

The answer involves latency hiding:

### Without Latency Hiding
```
RTX 4090 (1,008 GB/s):  2.63ms
H100 (3,350 GB/s):      2.63ms  ← Same! Because latency-bound, not bandwidth-bound
```

### With Latency Hiding
```
RTX 4090 (1,008 GB/s):  0.6ms
H100 (3,350 GB/s):      0.18ms  ← 3.3× faster! Now bandwidth matters!
```

**Higher bandwidth only helps AFTER you hide latency!**

---

## Practical Takeaway

**Latency hiding** = Do useful work while waiting for memory

Like a chef cooking multiple dishes:
- ❌ **Bad**: Cook dish 1 completely, then start dish 2 (sequential)
- ✅ **Good**: While sauce for dish 1 simmers, chop vegetables for dish 2 (parallel)

In GPUs:
- ❌ **Bad**: One thread waits for memory, doing nothing
- ✅ **Good**: While thread 1 waits, GPU works on thread 2, 3, 4...

**Your kernel**: Currently doing ❌ (2% efficiency)
**Optimized kernel**: Could do ✅ (50-80% efficiency)
**Speedup potential**: 2-40× depending on how well you hide latency!

This is the key to unlocking higher bandwidth on faster GPUs.
