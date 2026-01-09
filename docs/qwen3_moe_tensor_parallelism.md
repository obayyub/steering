# Qwen3 MoE + Tensor Parallelism: Known Bug & Workarounds

## Status: TODO - Try TP=4

We got 235B loading and running with pipeline parallelism, but tensor parallelism would be faster.

## The Problem

`tp_plan="auto"` with Qwen3 MoE fails during forward pass with shape mismatch in attention when `TP_size > num_key_value_heads`.

```
RuntimeError: shape '[64, 103, -1, 128]' is invalid for input of size 421888
```

**Root cause**: The TP plan correctly shards `k_proj`/`v_proj` weights column-wise, but the attention code still uses `config.num_key_value_heads` (unsharded value) in the `.view()` reshape. With 8 KV heads and TP=8, each GPU gets 1 KV head but the code tries to reshape assuming 8.

This is the same class of bug that hit Llama2 + DeepSpeed in 2023:
- [DeepSpeed #4016](https://github.com/microsoft/DeepSpeed/issues/4016)
- [DeepSpeed #4027](https://github.com/deepspeedai/DeepSpeed/issues/4027)

## Qwen3 235B-A22B Architecture

| Attribute | Value |
|-----------|-------|
| Total params | 235B |
| Active params | 22B |
| Layers | 94 |
| Q heads | 64 |
| KV heads | 8 |
| Head dim | 128 |
| Hidden dim | 5120 |

## What We Tried

| Approach | Loading | Forward Pass | Notes |
|----------|---------|--------------|-------|
| `device_map="auto"` | Works | Works | Slow (pipeline parallelism) |
| `DistributedConfig(enable_expert_parallel=True)` | OOM at 15% | N/A | Each rank loads full tensors |
| `tp_plan="auto"` with TP=8 | Works (100%) | Fails | Shape mismatch in attention |

## TODO: Try TP=4

Since `TP ≤ num_kv_heads` should work, try TP=4 instead of TP=8:

```bash
# Update script to use tp_plan="auto" for MoE models, then:
uv run torchrun --nproc-per-node=4 compute_logit_diff_all.py --models 235B-A22B --batch-size 64 --train-batch-size 64
```

This would use 4 GPUs with tensor parallelism (each gets 2 KV heads), leaving 4 GPUs idle but potentially faster than pipeline parallelism across 8.

## Performance Comparison

| Approach | Steering Extraction | Per Layer Eval | Notes |
|----------|---------------------|----------------|-------|
| Old (steering-vectors lib) | ~12 min/layer | - | 150 sequential forward passes per layer |
| New (batched, all layers) | ~9 min total | Fast | 2 forward passes for ALL layers |

The batched extraction optimization was a **huge** win regardless of parallelism strategy.

## Code Changes Needed for TP=4

In `compute_logit_diff_all.py`, for MoE models:

```python
if is_moe:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        tp_plan="auto",
        trust_remote_code=True,
    )
```

And initialize NCCL before loading:

```python
if "RANK" in os.environ and not dist.is_initialized():
    dist.init_process_group(backend="nccl")
```

## Alternative: Use vLLM

For pure inference (no hidden state extraction), vLLM handles TP correctly:

```bash
vllm serve Qwen/Qwen3-235B-A22B --tensor-parallel-size 8 --enable-expert-parallel
```

But vLLM doesn't easily expose intermediate layer activations for steering vector extraction.

## References

- [Qwen3 MoE HF docs](https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe)
- [HF Transformers TP tutorial](https://huggingface.co/blog/qgallouedec/tp)
- [HF Expert Parallelism docs](https://huggingface.co/docs/transformers/main/en/expert_parallelism)
- [PR #36335: Memory-efficient TP loading](https://github.com/huggingface/transformers/pull/36335)
