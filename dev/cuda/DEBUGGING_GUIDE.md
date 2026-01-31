# CUDA Kernel Debugging Guide

Complete guide to debugging the temperature + top-nsigma kernel.

## Current Status

**Problem:**
- CUDA kernel produces wrong results (180 infs instead of 580)
- Performance is worse than eager PyTorch (4.63ms vs 4.43ms)
- torch.compile beats custom CUDA by 1.77x!

**Goal:** Find and fix the bugs

## Debugging Strategy

### Phase 1: Verify Basic Kernel Logic

The debug kernel (`debug_kernel.cu`) prints intermediate values:

```bash
cd /cuda
python debug_kernel_test.py
```

**What to look for:**
1. Are the intermediate values (max, mean, std, threshold) correct?
2. Does the threshold filtering work?
3. How many infs are being set?

**Expected output:**
```
Position 0, After temperature scaling:
  output[0] = <value>
  output[1] = <value>
  output[35] = <value>
  Maximum = <value>
  Mean = <value>
  Std dev = <value>
  Threshold = <value>
  Total infs set: <count>
  output[35] after filter = -inf (or value)
```

Compare these values with PyTorch!

### Phase 2: Check for Common CUDA Bugs

#### Bug 1: Variable Shadowing ✅ FIXED
```cuda
// BAD
const int offset = pos_idx * vocab_size;
for (int offset = 16; offset > 0; offset >>= 1) { ... }

// GOOD
const int offset = pos_idx * vocab_size;
for (int delta = 16; delta > 0; delta >>= 1) { ... }
```

#### Bug 2: Warp Broadcast Scope ✅ FIXED
```cuda
// BAD - only broadcasts within warp!
maximum = __shfl_sync(0xffffffff, maximum, 0);

// GOOD - broadcast to all threads via shared memory
if (lane_id == 0) shared_mem[0] = maximum;
__syncthreads();
maximum = shared_mem[0];
```

#### Bug 3: Race Conditions (CHECK THIS!)
```cuda
// Make sure you have __syncthreads() after:
// 1. Writing to shared memory
// 2. Reading from shared memory
// 3. Between different phases (max, mean, variance)
```

#### Bug 4: Reduction Bugs (CHECK THIS!)
```cuda
// Tree reduction must be correct:
for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        shared_mem[tid] = op(shared_mem[tid], shared_mem[tid + stride]);
    }
    __syncthreads();  // CRITICAL!
}
```

#### Bug 5: Shared Memory Reuse (POTENTIAL BUG!)

We reuse `shared_mem` for:
1. Max reduction
2. Sum reduction
3. Variance reduction

**Potential issue:** Are we properly synchronizing between phases?

Let me check the basic kernel more carefully...

### Phase 3: Step-by-Step Validation

Run this to see exactly where divergence happens:

```bash
python debug_kernel_test.py
```

The script will:
1. Print CUDA kernel intermediate values
2. Compare with PyTorch step by step
3. Show exactly where they diverge

### Phase 4: Inspect Actual Kernel Code

Let's check if the basic kernel has issues:

```bash
# Look at the basic kernel (not optimized)
grep -A 100 "temperature_topnsigma_kernel<256>" temperature_topnsigma_kernel.cu
```

## Known Issues to Check

### Issue 1: Input/Output Confusion

**Check:** Are we reading from the right buffer?

```cuda
// After temperature scaling, we write to output
output[offset + i] = input[offset + i] / temperature;

// Then we should read from output for reductions!
local_max = fmaxf(local_max, output[offset + i]);  // ✅ Correct

// NOT from input!
local_max = fmaxf(local_max, input[offset + i]);   // ❌ Bug!
```

### Issue 2: Integer Overflow

**Check:** Is vocab_size causing overflow?

```cuda
const int offset = pos_idx * vocab_size;  // Could overflow if large!
```

For vocab_size = 156896:
- pos_idx = 255 (max)
- offset = 255 * 156896 = 40,008,480 (fits in int32)

Should be OK, but let's be sure.

### Issue 3: Floating Point Precision

**Check:** Are we losing precision in reductions?

```cuda
// Sum of 156896 values might lose precision
// But shouldn't cause inf count to be wrong by 3x
```

### Issue 4: Comparison with -inf

**Check:** Is the comparison correct?

```cuda
// We set values to -inf if below threshold
output[offset + i] = (val >= threshold) ? val : -INFINITY;

// Later, PyTorch counts infs:
torch.isinf(result).sum()

// This should work correctly
```

## Manual Inspection Checklist

Go through the basic kernel line by line:

- [ ] Temperature scaling writes to correct buffer
- [ ] Max reduction reads from correct buffer
- [ ] Max reduction has proper sync
- [ ] Mean calculation reads from correct buffer
- [ ] Mean calculation has proper sync
- [ ] Variance calculation uses correct mean value
- [ ] Variance calculation has proper sync
- [ ] Threshold calculation is correct
- [ ] Filtering uses correct comparison (`>=` not `>`)
- [ ] Filtering writes -INFINITY correctly

## Quick Tests

### Test 1: Verify Temperature Scaling

```python
import torch
import temperature_topnsigma_cuda

x = torch.randn(1, 1, 100, device='cuda')
temp = 2.0

# Just divide
expected = x / temp

# Run through kernel (which should also divide)
result = temperature_topnsigma_cuda.temperature_topnsigma(x, temp, 0.0)
# With top_nsigma=0, no filtering should happen

# They should be very close (not exactly equal due to -inf handling)
```

### Test 2: Verify Max Finding

```python
# Create known max
x = torch.randn(1, 1, 100, device='cuda')
x[0, 0, 50] = 100.0  # Known max

# The max should be ~100.0 / temperature
```

### Test 3: Verify Threshold

```python
# Set all values to 0, except one to 10
x = torch.zeros(1, 1, 100, device='cuda')
x[0, 0, 0] = 10.0

temp = 1.0
nsigma = 2.0

# Max = 10.0
# Mean = 0.1
# Std = ~1.0
# Threshold = 10.0 - 2.0 * 1.0 = 8.0
# All values except [0,0,0] should be -inf
```

## Expected Behavior

For the debug test case (batch=2, seq=4, vocab=100):

**Position [0, 0]:**
- Input: Random values from randn (seed=42)
- After scaling: input / 1.5
- Max: ~max of 100 values
- Mean: ~0 (normal distribution)
- Std: ~std of 100 values
- Threshold: max - 2.0 * std
- Infs: ~30-40 values below threshold (rough estimate)

**With seed=42, position [0,0]:**
- Should see consistent values across runs
- PyTorch and CUDA should match exactly

## Debugging Printf Statements

Add these to the kernel to see what's happening:

```cuda
if (pos_idx == 0 && tid == 0) {
    printf("Max: %.6f, Mean: %.6f, Std: %.6f, Threshold: %.6f\n",
           maximum, mean, std_dev, threshold);
}
```

## Next Steps

1. **Run debug test:**
   ```bash
   python debug_kernel_test.py
   ```

2. **Compare intermediate values:**
   - Does max match?
   - Does mean match?
   - Does std match?
   - Does threshold match?

3. **If intermediates are wrong:**
   - The bug is in the reduction logic
   - Check synchronization
   - Check shared memory usage

4. **If intermediates are correct but output is wrong:**
   - The bug is in the filtering step
   - Check the comparison operator
   - Check -INFINITY is being written

5. **If everything looks correct:**
   - Check if it's a different kernel (basic vs optimized)
   - Check the Python wrapper
   - Check tensor memory layout (contiguous?)

## Tools

### Use CUDA-MEMCHECK

```bash
cuda-memcheck python test_cuda_kernel.py
```

Catches:
- Out of bounds access
- Race conditions
- Synchronization issues

### Use Nsight Compute

```bash
ncu --set full python test_cuda_kernel.py
```

Shows:
- Memory access patterns
- Warp efficiency
- Register usage

### Use Assertions

Add to kernel:

```cuda
if (pos_idx == 0 && tid == 0) {
    assert(maximum > -1000.0f);  // Sanity check
    assert(!isnan(mean));
    assert(std_dev >= 0.0f);
}
```

## Common Gotchas

1. **Block size must match template parameter:**
   ```cuda
   temperature_topnsigma_kernel<256><<<grid, 256>>>(...);  // ✅
   temperature_topnsigma_kernel<256><<<grid, 128>>>(...);  // ❌
   ```

2. **Shared memory must be large enough:**
   ```cuda
   __shared__ float shared_mem[256];  // For 256 threads
   ```

3. **Synchronization after every shared memory use:**
   ```cuda
   shared_mem[tid] = value;
   __syncthreads();  // MUST HAVE THIS!
   ```

4. **All threads must participate in __syncthreads():**
   ```cuda
   // BAD - causes deadlock!
   if (tid == 0) {
       __syncthreads();
   }

   // GOOD - all threads sync
   shared_mem[tid] = value;
   __syncthreads();
   ```
