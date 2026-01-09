# Distributed Inference with HuggingFace and torchrun

This guide explains how to run large models (especially MoE models) across multiple GPUs using tensor parallelism and expert parallelism.

## The Problem

When a model is too large for a single GPU, you have two options:

| Approach | How it works | Speed |
|----------|--------------|-------|
| **Pipeline Parallelism** | Split layers across GPUs, process sequentially | Slow (GPUs wait for each other) |
| **Tensor Parallelism** | Split each layer across GPUs, process in parallel | Fast (GPUs work simultaneously) |

By default, HuggingFace's `device_map="auto"` uses pipeline parallelism:

```python
# Pipeline parallelism (slow for inference)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-235B-A22B",
    device_map="auto",  # Shards layers sequentially across GPUs
)
```

This is slow because GPU 1 must finish before GPU 2 can start.

## The Solution: torchrun + Tensor/Expert Parallelism

### What is torchrun?

`torchrun` is PyTorch's distributed launcher. It:

1. Spawns N processes (one per GPU)
2. Sets environment variables for coordination
3. Enables parallel execution across GPUs

```bash
# Launch 8 processes, one per GPU
torchrun --nproc-per-node=8 my_script.py
```

### Environment Variables Set by torchrun

```bash
RANK=0          # Global rank of this process (0 to WORLD_SIZE-1)
LOCAL_RANK=0    # Local rank on this node (0 to nproc-1)
WORLD_SIZE=8    # Total number of processes
MASTER_ADDR=localhost
MASTER_PORT=29500
```

### How the Pieces Connect

```
┌─────────────────────────────────────────────────────────────────────┐
│  torchrun --nproc-per-node=8 script.py                              │
│      │                                                              │
│      ▼                                                              │
│  Sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT        │
│      │                                                              │
│      ▼                                                              │
│  Your script calls: dist.init_process_group("nccl")                 │
│      │                                                              │
│      ▼                                                              │
│  PyTorch distributed backend initialized                            │
│      │                                                              │
│      ▼                                                              │
│  HuggingFace checks: torch.distributed.is_initialized()             │
│      │                                                              │
│      ▼                                                              │
│  Model sharded with DeviceMesh + DTensor (true parallelism)         │
└─────────────────────────────────────────────────────────────────────┘
```

## Code Example

### Script Setup

```python
import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def main():
    # Initialize distributed if launched with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        if is_main_process():
            print(f"Distributed initialized: world_size={dist.get_world_size()}")

    # Load model with expert parallelism (for MoE models)
    from transformers.distributed import DistributedConfig

    distributed_config = DistributedConfig(enable_expert_parallel=True)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-235B-A22B",
        torch_dtype=torch.bfloat16,
        distributed_config=distributed_config,
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

    # Run inference (all GPUs work in parallel)
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Only print from main process
    if is_main_process():
        print(tokenizer.decode(outputs[0]))

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Launch Command

```bash
torchrun --nproc-per-node=8 my_script.py
```

## Types of Parallelism

### Tensor Parallelism (for dense models)

Splits weight matrices across GPUs. Each GPU computes part of each layer.

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B",
    torch_dtype=torch.bfloat16,
    tp_plan="auto",  # Tensor parallelism
)
```

### Expert Parallelism (for MoE models)

Distributes experts across GPUs. Each GPU holds a subset of experts.

```python
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(enable_expert_parallel=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-235B-A22B",
    torch_dtype=torch.bfloat16,
    distributed_config=distributed_config,
)
```

Note: Expert parallelism automatically enables tensor parallelism for attention layers.

## Supported Models

Models with native tensor parallelism support (as of transformers 4.51+):

- Llama, Llama 2, Llama 3
- Mistral, Mixtral
- Qwen2, Qwen2-MoE (includes Qwen3)
- Gemma, Gemma 2
- Phi, Phi-3
- Cohere, Cohere 2
- And others...

Check: https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi

## Common Patterns

### Print Only from Main Process

```python
def print_main(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
```

### Save Only from Main Process

```python
if is_main_process():
    torch.save(results, "output.pt")
    model.save_pretrained("./saved_model")
```

### Barrier for Synchronization

```python
# Wait for all processes to reach this point
if dist.is_initialized():
    dist.barrier()
```

## Troubleshooting

### "NCCL error"

Usually means GPU communication failed. Check:
- All GPUs are visible: `nvidia-smi`
- NCCL is installed: `python -c "import torch; print(torch.cuda.nccl.version())"`

### Model loads but runs slowly

You might be using pipeline parallelism instead of tensor parallelism. Verify:

```python
# Check if TP is active
print(model._tp_plan)  # Should show partitioning strategy
```

### OOM during loading

Try loading with lower precision:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # or torch.bfloat16
    low_cpu_mem_usage=True,
    distributed_config=distributed_config,
)
```

## References

- [HuggingFace Tensor Parallelism Docs](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi)
- [HuggingFace Expert Parallelism Docs](https://huggingface.co/docs/transformers/main/en/expert_parallelism)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
