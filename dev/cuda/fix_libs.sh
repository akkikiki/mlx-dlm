#!/bin/bash
# Fix library path for PyTorch extensions

# Find PyTorch library path
TORCH_LIB=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

echo "PyTorch libraries at: $TORCH_LIB"

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH

echo "Updated LD_LIBRARY_PATH"
echo ""

# Test import
python -c "import temperature_topnsigma_optimized; print('✅ Import successful!')"
