#!/usr/bin/env python3
"""
PyTorch Profiler Example Script

This script demonstrates how to use the PyTorch profiler to analyze
performance of model training and inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def basic_profiling_example():
    """Basic profiling of a single forward pass"""
    print("\n" + "="*60)
    print("BASIC PROFILING EXAMPLE")
    print("="*60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    inputs = torch.randn(32, 784).to(device)

    # Profile the forward pass
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Code to profile
        output = model(inputs)

    # Print results sorted by CPU time
    print("\nTop operations by CPU time:")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10
    ))

    # Print results sorted by memory
    print("\nTop operations by memory usage:")
    print(prof.key_averages().table(
        sort_by="cpu_memory_usage",
        row_limit=10
    ))

    # Export for Chrome trace viewer
    prof.export_chrome_trace("basic_trace.json")
    print("\nTrace exported to: basic_trace.json")
    print("View at: chrome://tracing")


def training_loop_profiling_example():
    """Advanced profiling of a training loop with schedule"""
    print("\n" + "="*60)
    print("TRAINING LOOP PROFILING EXAMPLE")
    print("="*60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create dummy dataset
    num_batches = 10
    batch_size = 32

    # Profile with schedule:
    # - wait=1: skip first batch (initialization overhead)
    # - warmup=1: warmup for 1 batch (not recorded)
    # - active=3: record next 3 batches
    # - repeat=2: repeat this cycle 2 times
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(num_batches):
            # Generate dummy data
            inputs = torch.randn(batch_size, 784).to(device)
            labels = torch.randint(0, 10, (batch_size,)).to(device)

            # Training step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Important: signal the profiler that the step is done
            prof.step()

            print(f"Step {step}: loss={loss.item():.4f}")

    # Print summary
    print("\nProfiling Summary:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=15
    ))

    print("\nTensorBoard logs saved to: ./profiler_logs")
    print("View with: tensorboard --logdir=./profiler_logs")


def context_manager_example():
    """Using profiler as a context manager with custom naming"""
    print("\n" + "="*60)
    print("CONTEXT MANAGER WITH RECORD_FUNCTION")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        # You can name sections of your code
        with torch.profiler.record_function("data_preparation"):
            inputs = torch.randn(64, 784).to(device)

        with torch.profiler.record_function("forward_pass"):
            outputs = model(inputs)

        with torch.profiler.record_function("loss_calculation"):
            loss = outputs.sum()

        with torch.profiler.record_function("backward_pass"):
            loss.backward()

    print(prof.key_averages().table(sort_by="cpu_time_total"))


def inference_profiling_example():
    """Profile inference only"""
    print("\n" + "="*60)
    print("INFERENCE PROFILING EXAMPLE")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    model.eval()  # Set to eval mode

    inputs = torch.randn(100, 784).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(inputs)

    # Profile inference
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                outputs = model(inputs)

    print("\nInference Profile:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Get specific operation stats
    print("\nOperation-level breakdown:")
    for evt in prof.key_averages():
        if evt.key in ["aten::linear", "aten::relu", "aten::addmm"]:
            print(f"{evt.key:20s} | CPU time: {evt.cpu_time_total:10.2f}us | "
                  f"Calls: {evt.count:5d} | Avg: {evt.cpu_time_total/evt.count:8.2f}us")


if __name__ == "__main__":
    print("PyTorch Profiler Examples")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Run all examples
    basic_profiling_example()
    training_loop_profiling_example()
    context_manager_example()
    inference_profiling_example()

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nFiles created:")
    print("- basic_trace.json (view in chrome://tracing)")
    print("- ./profiler_logs/ (view with tensorboard --logdir=./profiler_logs)")
