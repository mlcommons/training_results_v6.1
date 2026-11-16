# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
''                                                                    

                                                                         
                                                                        
                                           
   

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_SEED = 20260812
_TRAIN_WARMUP_TIMEOUT_S = float(os.environ.get("MLPERF_WARMUP_TRAIN_TIMEOUT_S", "900"))


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


def _log(msg: str) -> None:
    print(f"[mlperf_warmup] {msg}", flush=True)


def _dry_refit(policy: Any, policy_generation: Any) -> None:
    from nemo_rl.algorithms.grpo import refit_policy_generation

    start = time.perf_counter()
    refit_policy_generation(policy, policy_generation, colocated_inference=False)
    _log(f"dry refit (unchanged init weights): {time.perf_counter() - start:.1f}s")


def _generation_prompt_lengths(
    *,
    max_model_len: int,
    chunk_tokens: int,
    max_tokens: int,
    per_engine: int,
) -> tuple[list[int], int, int]:
    ''                                                                 

         
                                                                              
                         
                                                                          
                                                                              
                                                                                

                                                                           
       
    if max_model_len < 2:
        raise ValueError(f"max_model_len must be at least 2, got {max_model_len}")
    if chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if per_engine < 1:
        raise ValueError(f"per_engine must be positive, got {per_engine}")

    max_tokens = max(1, min(max_tokens, max_model_len - 1))
    max_prompt = max_model_len - max_tokens
    if max_prompt < 1:
        raise ValueError(
            f"no room for a warmup prompt: max_model_len={max_model_len} "
            f"max_tokens={max_tokens}"
        )

                                                                                
                                                                              
                                                                   
    long_tokens = min(2 * chunk_tokens, max_prompt)

                                                                               
                                                                        
                                                                       
    short_cycle = [
        max(1, chunk_tokens // 32),
        max(1, chunk_tokens // 16),
        max(1, chunk_tokens // 8),
        max(1, chunk_tokens // 4),
    ]
                                                                           
                                                                           
    lengths = [long_tokens] + [
        min(short_cycle[(i - 1) % len(short_cycle)], max_prompt)
        for i in range(1, per_engine)
    ]
    return lengths, max_tokens, max_prompt


def _synthetic_prompt_token_ids(
    tokenizer: Any,
    length: int,
    *,
    engine_idx: int,
    request_idx: int,
) -> list[int]:
    ''                                                                   
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    vocab_size = int(vocab_size)
    if vocab_size < 2:
        raise ValueError(f"tokenizer vocabulary is unexpectedly small: {vocab_size}")

                                                                                
                                                                         
                                                                             
    offset = (_SEED + engine_idx * 65537 + request_idx * 257) % vocab_size
    stride = 104729 % vocab_size or 1
    return [(offset + position * stride) % vocab_size for position in range(length)]


def _post_chat_completion(url: str, payload: dict[str, Any], timeout: float) -> int:
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        json.loads(resp.read())
        return resp.status


def _generation_warmup(
    policy_generation: Any,
    tokenizer: Any,
    policy_cfg: dict[str, Any],
    warmup_cfg: dict[str, Any],
) -> None:
    gen_cfg = policy_cfg["generation"]
    backend = gen_cfg.get("backend", "vllm")
                                                                                   
                                                                                  
                                                 
    if backend not in ("vllm", "trtllm"):
        _log(f"generation warmup has no exact-prompt path for backend {backend!r}; skipping")
        return
    urls = [u for u in getattr(policy_generation, "dp_openai_server_base_urls", None) or [] if u]
    if not urls:
        _log("no OpenAI server URLs exposed; skipping generation warmup")
        return
    if backend == "vllm":
        max_model_len = int(gen_cfg["vllm_cfg"]["max_model_len"])
        chunk_tokens = int(gen_cfg["vllm_kwargs"]["max_num_batched_tokens"])
    else:
        max_model_len = int(gen_cfg["trtllm_cfg"]["max_model_len"])
        chunk_tokens = int(gen_cfg["trtllm_cfg"]["max_num_tokens"])
    per_engine = int(
        warmup_cfg.get(
            "generation_requests_per_engine",
            os.environ.get("MLPERF_WARMUP_GEN_REQUESTS_PER_ENGINE", "16"),
        )
    )
    timeout = float(
        warmup_cfg.get(
            "generation_http_timeout_s",
            os.environ.get("MLPERF_WARMUP_GEN_TIMEOUT", "900"),
        )
    )
    max_tokens = min(
        int(
            warmup_cfg.get(
                "generation_max_tokens",
                os.environ.get("MLPERF_WARMUP_GEN_TOKENS", "192"),
            )
        ),
        int(gen_cfg["max_new_tokens"]),
    )
    prompt_lengths, max_tokens, max_prompt = _generation_prompt_lengths(
        max_model_len=max_model_len,
        chunk_tokens=chunk_tokens,
        max_tokens=max_tokens,
        per_engine=per_engine,
    )
    _log(
        "generation warmup shapes: "
        f"max_model_len={max_model_len} chunk={chunk_tokens} "
        f"max_tokens={max_tokens} max_prompt={max_prompt} "
        f"prompt_lengths={prompt_lengths}"
    )
    model_id = policy_cfg["model_name"]
    jobs = []
    for engine_idx, url in enumerate(urls):
        for req_idx, prompt_length in enumerate(prompt_lengths):
            prompt_token_ids = _synthetic_prompt_token_ids(
                tokenizer,
                prompt_length,
                engine_idx=engine_idx,
                request_idx=req_idx,
            )
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": f"MLPerf warmup {engine_idx}-{req_idx}",
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "top_p": 1.0,
                "stream": False,
            }
            if backend == "vllm":
                                                                             
                                                                              
                                                                            
                                                               
                payload["required_prefix_token_ids"] = prompt_token_ids
                                                                       
                payload["ignore_eos"] = True
            else:
                                                                             
                                                                                
                                                                     
                payload["prompt_token_ids"] = prompt_token_ids
            jobs.append((url, payload))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(len(jobs), 512)) as pool:
        futures = [pool.submit(_post_chat_completion, u, p, timeout) for u, p in jobs]
        for future in as_completed(futures):
            future.result()
    _log(
        f"generation warmup: {len(jobs)}/{len(jobs)} requests over "
        f"{len(urls)} engines in {time.perf_counter() - start:.1f}s"
    )
                                                                           
    policy_generation.finish_generation()


def _synthetic_train_batch(tokenizer: Any, gbs: int, seq_len: int) -> BatchedDataDict:
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    generator = torch.Generator(device="cpu").manual_seed(_SEED)
    lengths = torch.tensor(
        [max(1024, seq_len * ((i % 8) + 1) // 8 // 64 * 64) for i in range(gbs)],
        dtype=torch.long,
    )
    input_ids = torch.randint(0, vocab_size, (gbs, seq_len), dtype=torch.long, generator=generator)
    token_mask = torch.zeros((gbs, seq_len), dtype=torch.float32)
    advantages = torch.zeros((gbs, seq_len), dtype=torch.float32)
    noise = torch.randn((gbs, seq_len), dtype=torch.float32, generator=generator)
    for i in range(gbs):
        prompt_len = int(lengths[i]) // 4
        token_mask[i, prompt_len : int(lengths[i])] = 1
        advantages[i, prompt_len : int(lengths[i])] = noise[i, prompt_len : int(lengths[i])]
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(gbs, dtype=torch.float32),
            "advantages": advantages,
        }
    )


def _warmup_batch_size(production_gbs: int, mbs: int, dp_size: int, requested: str | None) -> int:
    ''                                                                        
    gbs = production_gbs if requested is None else int(requested)
    alignment = dp_size * mbs
    if alignment < 1 or gbs < alignment or gbs > production_gbs or gbs % alignment:
        raise ValueError(
            f"warmup GBS={gbs} must be a positive multiple of DP*MBS={alignment} "
            f"and no larger than production GBS={production_gbs}"
        )
    return gbs


def _training_warmup(policy: Any, tokenizer: Any, loss_fn: Any, master_config: Any) -> None:
    worker_group = getattr(policy, "worker_group", None)
    if worker_group is None or not hasattr(policy, "_shard_for_train"):
        _log("policy lacks the split-step worker surface; skipping training warmup")
        return
    pol_cfg = master_config.policy
    production_gbs = int(pol_cfg["train_global_batch_size"])
    mbs = int(pol_cfg["train_micro_batch_size"])
    gbs = _warmup_batch_size(
        production_gbs, mbs, int(policy.data_parallel_size),
        os.environ.get("MLPERF_WARMUP_TRAIN_GBS"),
    )
    seq_len = int(pol_cfg["max_total_sequence_length"])
    _log(f"synthetic training GBS={gbs}; production GBS={production_gbs} unchanged; MBS={mbs}; max sequence={seq_len}")
    steps = int(os.environ.get("MLPERF_WARMUP_TRAIN_STEPS", "1"))
    batch = _synthetic_train_batch(tokenizer, gbs, seq_len)

    from nemo_rl.algorithms.grpo import _resolve_logprob_skip_flags

    skip_prev_logprobs, skip_reference_logprobs = _resolve_logprob_skip_flags(master_config)
    try:
        if skip_prev_logprobs and skip_reference_logprobs:
                                                                                            
            logprobs = torch.zeros_like(batch["token_mask"])
            _log("skipping separate logprob warmup, matching training")
        else:
            start = time.perf_counter()
            policy.prepare_for_lp_inference()
            logprobs = policy.get_logprobs(batch)["logprobs"]
            _log(f"logprob warmup ({gbs}x{seq_len}): {time.perf_counter() - start:.1f}s")

        batch["prev_logprobs"] = logprobs
        batch["generation_logprobs"] = logprobs.clone()
        batch["reference_policy_logprobs"] = logprobs.clone()

        policy.prepare_for_training()
                                                                            
                                                                             
        for step in range(steps):
            start = time.perf_counter()
            ray.get(
                worker_group.run_all_workers_single_data(
                    "begin_train_step", loss_fn=loss_fn, gbs=gbs, mbs=mbs
                )
            )
            try:
                futures = worker_group.run_all_workers_sharded_data(
                    "train_microbatch",
                    data=policy._shard_for_train(batch, gbs),
                    in_sharded_axes=["data_parallel"],
                    replicate_on_axes=[
                        "context_parallel",
                        "tensor_parallel",
                        "pipeline_parallel",
                    ],
                    output_is_replicated=[
                        "context_parallel",
                        "tensor_parallel",
                        "pipeline_parallel",
                    ],
                )
                worker_group.get_all_worker_results(futures)
            finally:
                ray.get(worker_group.run_all_workers_single_data("abort_train_step"))
            _log(
                f"train warmup step {step + 1}/{steps}: {time.perf_counter() - start:.1f}s "
                "(fwd+bwd complete, step aborted, optimizer untouched)"
            )
    except Exception:
                                                                 
        policy.prepare_for_training()
        raise


def _run_warmup_stage(name: str, fn: Callable[[], None]) -> None:
    ''                                                                          
    start = time.perf_counter()
    _log(f"{name} warmup started")
    try:
        fn()
    except Exception as exc:
        _log(f"{name} warmup failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        _log(f"{name} warmup stage finished in {time.perf_counter() - start:.1f}s")


def _run_policy_generation_warmups(
    generation_fn: Callable[[], None],
    training_fn: Callable[[], None],
    *,
    generation_enabled: bool,
    training_enabled: bool,
    concurrent: bool,
) -> None:
    ''                                                                           
    stages = [
        ("generation", generation_enabled, generation_fn),
        ("training", training_enabled, training_fn),
    ]
    enabled_stages = [(name, fn) for name, enabled, fn in stages if enabled]
    for name, enabled, _ in stages:
        if not enabled:
            _log(f"{name} warmup disabled; skipping")

    if concurrent and len(enabled_stages) == 2:
        start = time.perf_counter()
        _log("launching generation and training warmups concurrently")
        pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mlperf-warmup"
        )
        try:
            futures = {
                pool.submit(_run_warmup_stage, name, fn): name
                for name, fn in enabled_stages
            }
            for future in as_completed(futures, timeout=_TRAIN_WARMUP_TIMEOUT_S):
                future.result()
        except FuturesTimeoutError as extra:
            raise TimeoutError(
                f"warmup exceeded {_TRAIN_WARMUP_TIMEOUT_S:.0f}s"
            ) from extra
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        _log(
            "concurrent generation and training warmups joined in "
            f"{time.perf_counter() - start:.1f}s"
        )
        return

    for name, fn in enabled_stages:
        _run_warmup_stage(name, fn)


def maybe_run_mlperf_warmup(
    policy: Any,
    policy_generation: Any,
    tokenizer: Any,
    loss_fn: Any,
    master_config: Any,
    warmup_cfg: dict[str, Any] | None = None,
) -> None:
    ''                                                                          

                                                                             
                                                                             
                                                                                
       
                                                                     
    if not _env_flag("MLPERF_WARMUP", "1"):
        return
    generation_cfg = master_config.policy.get("generation") or {}
    if (generation_cfg.get("colocated") or {}).get("enabled"):
        _log("colocated generation not supported; skipping warmup")
        return
    warmup_cfg = warmup_cfg or {}
    total_start = time.perf_counter()
    _log("starting synthetic warmup (timing is determined by MLLOG run_start)")
    refit_enabled = _env_flag("MLPERF_WARMUP_REFIT") and policy_generation is not None
    if refit_enabled:
        _run_warmup_stage("refit", lambda: _dry_refit(policy, policy_generation))
    else:
        _log("refit warmup disabled; skipping")

    _run_policy_generation_warmups(
        lambda: _generation_warmup(
            policy_generation,
            tokenizer,
            master_config.policy,
            warmup_cfg,
        ),
        lambda: _training_warmup(policy, tokenizer, loss_fn, master_config),
        generation_enabled=(
            _env_flag("MLPERF_WARMUP_GEN") and policy_generation is not None
        ),
        training_enabled=_env_flag("MLPERF_WARMUP_TRAIN"),
        concurrent=_env_flag("MLPERF_WARMUP_CONCURRENT_POLICY_GENERATION"),
    )
    _log(f"warmup done in {time.perf_counter() - total_start:.1f}s")
