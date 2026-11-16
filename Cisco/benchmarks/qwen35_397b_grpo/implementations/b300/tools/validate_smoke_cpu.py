''                                                                             
import json
import math
import shlex

from omegaconf import OmegaConf
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.models.megatron.setup import (
    _merge_model_overrides,
    _validate_model_override_conflicts,
    setup_model_config,
)
from megatron.bridge import AutoBridge
from transformers import AutoConfig

register_omegaconf_resolvers()
recipe = "/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm_mxfp8.yaml"
cfg = load_config(recipe)
import os
for entry in shlex.split(os.environ["EXTRA_ARGS"]):
    name, value = entry.removeprefix("++").split("=", 1)
    OmegaConf.update(cfg, name, OmegaConf.from_dotlist(["value=" + value]).value, force_add=True)
cfg = OmegaConf.to_container(cfg, resolve=True)
config_difference = []
if os.environ.get("QWEN_POLICY_ALLOC_CONF"):
    baseline = load_config(recipe)
    allocator_path = "policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF"
    for entry in shlex.split(os.environ["EXTRA_ARGS"]):
        name, value = entry.removeprefix("++").split("=", 1)
        if name != allocator_path:
            OmegaConf.update(
                baseline, name,
                OmegaConf.from_dotlist(["value=" + value]).value,
                force_add=True,
            )
    baseline = OmegaConf.to_container(baseline, resolve=True)
    def find_differences(before, after, path=""):
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(before.keys() | after.keys()):
                item_path = f"{path}.{key}" if path else key
                if key not in before or key not in after:
                    yield item_path
                else:
                    yield from find_differences(before[key], after[key], item_path)
        elif before != after:
            yield path
    config_difference = list(find_differences(baseline, cfg))
    assert config_difference == [allocator_path], config_difference
p = cfg["policy"]
performance_difference = []
if os.environ.get("QWEN_PERF_CONTROL_EXTRA_ARGS"):
    control = load_config(recipe)
    for entry in shlex.split(os.environ["QWEN_PERF_CONTROL_EXTRA_ARGS"]):
        name, value = entry.removeprefix("++").split("=", 1)
        OmegaConf.update(control, name, OmegaConf.from_dotlist(["value=" + value]).value, force_add=True)
    control = OmegaConf.to_container(control, resolve=True)
    performance_difference = list(find_differences(control, cfg))
    assert performance_difference == ["policy.megatron_cfg.empty_unused_memory_level"], performance_difference
m = p["megatron_cfg"]
assert m.get("empty_unused_memory_level", 1) == int(os.environ.get("QWEN_EMPTY_UNUSED_MEMORY_LEVEL", "1"))
g = cfg["grpo"]
expected_pp = int(os.environ.get("PP", "2"))
assert expected_pp in (2, 4)
assert (m["tensor_model_parallel_size"], m["pipeline_model_parallel_size"],
        m["expert_model_parallel_size"], m["expert_tensor_parallel_size"],
        m["context_parallel_size"]) == (2, expected_pp, 8, 1, 1)
assert p["train_micro_batch_size"] == p["logprob_batch_size"] == 1
assert p["train_global_batch_size"] == 512
assert p["max_total_sequence_length"] == 65536
full_evaluation = os.environ.get("QWEN_FULL_EVAL", "0") == "1"
offline_evaluation = full_evaluation and os.environ.get("OFFLINE_EVAL", "0") == "1"
checkpoint_steps = []
if full_evaluation:
    assert g["max_num_steps"] == (11 if offline_evaluation else 20) and g["val_start_at"] == 10
    assert g["val_period"] == (0 if offline_evaluation else 1)
    assert not g["val_at_start"] and not g["val_at_end"]
    assert g["max_val_samples"] is None and g["val_num_generations_per_prompt"] == 4
    assert g["stop_at_validation_metric"] == "pass_k"
    assert math.isclose(g["stop_at_validation_threshold"], 0.69, abs_tol=1e-12)
    assert math.isclose(p["generation"]["val_temperature"], 0.1, abs_tol=1e-12)
    assert math.isclose(p["generation"]["val_top_p"], 0.95, abs_tol=1e-12)
else:
    assert g["max_num_steps"] == 3
    assert g["val_period"] == 0 and not g["val_at_start"] and not g["val_at_end"]
assert bool(g["deferred_evaluation"]["enabled"]) == offline_evaluation
if offline_evaluation:
    import sys
    import tempfile
    from copy import deepcopy
    from types import SimpleNamespace

    sys.path.insert(0, "/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch")
    from utility.qwen35_mcore_deferred import (
        configure_deferred_evaluation, timestamp_policy_training,
    )
    assert cfg["checkpointing"]["enabled"]
    assert not cfg["checkpointing"]["save_optimizer"]
    assert m["checkpoint"]["async_save"]
    with tempfile.TemporaryDirectory(prefix="qwen-offline-cpu-") as temporary:
        checked = deepcopy(cfg)
        checked["checkpointing"]["checkpoint_dir"] = temporary + "/checkpoint"
        options = configure_deferred_evaluation(checked)
        assert options["threshold"] == 0.69
        assert checked["grpo"]["max_num_steps"] == 11
        assert checked["grpo"]["val_period"] == 0
        checkpoint_steps = [step for step in range(1, 12)
                            if step == 11 or step % checked["checkpointing"]["save_period"] == 0]
        assert checkpoint_steps == [10, 11]
        state = SimpleNamespace()
        policy = SimpleNamespace(train=lambda: 17)
        original_train = policy.train
        with timestamp_policy_training(policy, state):
            assert policy.train() == 17
        assert policy.train is original_train
        assert state.training_step_end_time_ms > 0
                                                                             
        assert json.loads(json.dumps(vars(state)))["training_step_end_time_ms"] > 0
assert m["activation_checkpointing"] and m["recompute_granularity"] == "full"
assert m["freeze_moe_router"] and m["moe_router_dtype"] == "fp32"
assert m["fp8_cfg"]["enabled"] and m["fp8_cfg"]["fp8_recipe"] == "mxfp8"
assert not m["fp8_cfg"]["fp8_param"] and p["precision"] == "bfloat16"
assert m["optimizer"]["exp_avg_dtype"] == m["optimizer"]["exp_avg_sq_dtype"] == "bfloat16"
expected_allocator = os.environ.get("QWEN_POLICY_ALLOC_CONF", "expandable_segments:False")
assert m["env_vars"]["PYTORCH_CUDA_ALLOC_CONF"] == expected_allocator
assert not m.get("use_fused_linear_logprobs", False)
assert math.isclose(m["optimizer"]["lr"], 1e-6 * math.sqrt(2), abs_tol=1e-9)
assert math.isclose(p["max_grad_norm"], .125 / math.sqrt(2), abs_tol=1e-9)
assert not m["moe_shared_expert_overlap"]
overrides = m["model_overrides"]
_validate_model_override_conflicts(m, overrides)
hf_config = AutoConfig.from_pretrained(os.environ["CONTAINER_HF_CKPT_PATH"])
provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)
provider = _merge_model_overrides(provider, overrides)
assert provider.batch_p2p_sync is False
assert provider.overlap_dispatch_backward_with_experts_wgrad is True
assert provider.overlap_moe_expert_parallel_comm is False
assert provider.delay_wgrad_compute is False
assert p["generation"]["backend"] == "trtllm"
assert p["generation"]["trtllm_cfg"]["tensor_parallel_size"] == 4
assert p["generation"]["trtllm_cfg"]["moe_expert_parallel_size"] == 4
import torch
runtime, model = setup_model_config(
    p, rank=0, dtype=torch.bfloat16,
    hf_model_name=os.environ["CONTAINER_HF_CKPT_PATH"],
    pretrained_path="/dev/null", skip_weight_load=True,
)
assert runtime.ddp.bucket_size == 134217728
assert runtime.ddp.align_param_gather is True
assert runtime.ddp.overlap_grad_reduce is True
assert runtime.ddp.overlap_param_gather is True
assert runtime.ddp.fp8_param_gather is False
assert model.pipeline_model_parallel_size == expected_pp
assert model.num_layers == 60 and model.num_layers % expected_pp == 0
assert model.batch_p2p_sync is False
assert model.overlap_dispatch_backward_with_experts_wgrad is True
assert model.moe_router_fusion is True
assert model.moe_flex_dispatcher_num_sms == 32
assert model.moe_hybridep_num_sms_preprocessing == 32
assert model.moe_permute_fusion is True
assert model.apply_rope_fusion is False
                                                                            
                                                                        
from copy import copy
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import (
    apply_rotary_pos_emb_absolute,
)

rope_input = torch.arange(64, dtype=torch.float32).reshape(4, 2, 8)
rope_freqs = torch.zeros((4, 1, 1, 8), dtype=torch.float32)
rope_seqlens = torch.tensor([0, 4], dtype=torch.int32)
rope_output = apply_rotary_pos_emb_absolute(
    rope_input, rope_freqs, model, cu_seqlens=rope_seqlens,
)
torch.testing.assert_close(rope_output, rope_input)
unsupported_rope = copy(model)
unsupported_rope.apply_rope_fusion = True
try:
    apply_rotary_pos_emb_absolute(
        rope_input, rope_freqs, unsupported_rope, cu_seqlens=rope_seqlens,
    )
except AssertionError as error:
    assert "apply_rope_fusion is not supported" in str(error)
else:
    raise AssertionError("Expected the pinned Qwen forward path to reject fused RoPE")
                                                                              
assert m["freeze_moe_router"] and model.moe_router_dtype == "fp32"
assert model.recompute_granularity == "full"
assert not model.use_grouped_gemm_for_shared_expert
assert not model.use_transformer_engine_op_fuser
assert model.moe_mlp_glu_interleave_size is None
assert model.virtual_pipeline_model_parallel_size in (None, 1)
assert model.cuda_graph_impl in (None, "none")
assert model.gradient_accumulation_fusion and model.bias_activation_fusion
assert p["megatron_cfg"]["env_vars"]["TORCH_NCCL_HIGH_PRIORITY"] == "1"
assert p["megatron_cfg"]["env_vars"]["NVTE_NORM_FWD_USE_CUDNN"] == "1"
assert p["megatron_cfg"]["env_vars"]["NVTE_NORM_BWD_USE_CUDNN"] == "1"
if os.environ.get("NCCL_DEBUG") == "INFO":
    assert m["env_vars"]["NCCL_DEBUG"] == "INFO"
    assert m["env_vars"]["NCCL_DEBUG_SUBSYS"] == "INIT,NET"
    assert m["env_vars"]["NCCL_DEBUG_FILE"] == "/results/nccl/%h.%p.log"
                                                                       
from copy import deepcopy
from nemo_rl.models.megatron.setup import _create_megatron_config
from megatron.core.distributed import DistributedDataParallelConfig
base_policy = deepcopy(p)
base_ddp = base_policy["megatron_cfg"]["distributed_data_parallel_config"]
del base_ddp["bucket_size"]
del base_ddp["align_param_gather"]
without_site_keys = _create_megatron_config(
    model, runtime.checkpoint, base_policy,
    os.environ["CONTAINER_HF_CKPT_PATH"], torch.bfloat16, False,
)
canonical = DistributedDataParallelConfig()
assert without_site_keys.ddp.bucket_size == canonical.bucket_size
assert without_site_keys.ddp.align_param_gather == canonical.align_param_gather
if expected_pp == 4:
                                                                           
    from unittest.mock import MagicMock, patch
    import nemo_rl.models.policy.workers.megatron_policy_worker as worker_module

    worker = object.__new__(worker_module.MegatronPolicyWorkerImpl)
    worker.rank = 7
    fake_cuda = MagicMock()
    fake_cuda.is_available.return_value = True
    fake_cuda.current_device.return_value = 0
    fake_cuda.mem_get_info.return_value = (32 * 1024**3, 268 * 1024**3)
    for counter in ("memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"):
        getattr(fake_cuda, counter).return_value = 128 * 1024**3
    fake_cuda.memory_stats.return_value = {"num_alloc_retries": 2, "num_ooms": 0}
    with patch.object(worker_module.torch, "cuda", fake_cuda), patch.object(worker_module, "log") as logger:
        logger.isEnabledFor.return_value = False
        with patch.dict(os.environ, {"PRINT_MEMORY_USAGE": "0"}):
            worker._log_gpu_mem("disabled")
        fake_cuda.is_available.assert_called_once()
        fake_cuda.reset_peak_memory_stats.assert_not_called()
        with patch.dict(os.environ, {"PRINT_MEMORY_USAGE": "1"}):
            worker._log_gpu_mem("test")
        fake_cuda.reset_peak_memory_stats.assert_called_once_with(0)
        assert logger.info.call_args.args[-3:] == (2, 0, 0.0)
    worker._log_gpu_mem = MagicMock()
    @worker_module._b300_memory_phase
    def returns_value(self):
        return 17
    @worker_module._b300_memory_phase
    def raises_error(self):
        raise RuntimeError("original training error")
    with patch.dict(os.environ, {"PRINT_MEMORY_USAGE": "1"}):
        assert returns_value(worker) == 17
        try:
            raises_error(worker)
        except RuntimeError as error:
            assert str(error) == "original training error"
        else:
            raise AssertionError("Memory observations must not suppress training errors")
    assert worker._log_gpu_mem.call_args.args == ("raises_error_exit_error",)
if os.environ.get("QWEN_FAST_TUNING") == "1":
    assert not full_evaluation and g["max_num_steps"] == 3
    assert os.environ["MLPERF_WARMUP_TRAIN"] == "0"
    assert os.environ["MLPERF_WARMUP_REFIT"] == os.environ["MLPERF_WARMUP_GEN"] == "1"
    import importlib.util
    from types import SimpleNamespace
    warmup_path = "/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/mlperf_warmup.py"
    spec = importlib.util.spec_from_file_location("smoke_warmup", warmup_path)
    warmup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmup)
    stages = []
    with patch.object(warmup, "_dry_refit", side_effect=lambda *a: stages.append("refit")), \
         patch.object(warmup, "_generation_warmup", side_effect=lambda *a: stages.append("generation")), \
         patch.object(warmup, "_training_warmup", side_effect=AssertionError("Synthetic training must be skipped")):
        warmup.maybe_run_mlperf_warmup(object(), object(), None, None, SimpleNamespace(policy={"generation": {"colocated": {"enabled": False}}}))
    assert stages == ["refit", "generation"], stages
                                                                                
    worker.model = MagicMock()
    worker.move_model = MagicMock(return_value=worker.model)
    worker.move_optimizer = MagicMock()
    worker.optimizer = object()
    worker.optimizer_self_managed_offload = False
    worker._stage_opd_full_teacher_lm_head_for_training = MagicMock()
    worker.cfg = {"megatron_cfg": {"empty_unused_memory_level": 0}}
    fake_cuda.empty_cache.reset_mock()
    with patch.object(worker_module.torch, "cuda", fake_cuda), patch.object(worker_module, "log"):
        assert worker.prepare_for_training() is None
        fake_cuda.empty_cache.assert_not_called()
        worker.move_optimizer.assert_called_once_with("cuda")
        worker.cfg["megatron_cfg"]["empty_unused_memory_level"] = 1
        assert worker.prepare_for_training() is None
        fake_cuda.empty_cache.assert_called_once_with()
print(json.dumps({"status": "PASS", "policy": f"TP2 PP{expected_pp} EP8 ETP1 CP1 MBS1 MXFP8",
                  "performance_difference_from_14352": performance_difference,
                  "empty_unused_memory_level": m.get("empty_unused_memory_level", 1),
                  "fast_tuning": os.environ.get("QWEN_FAST_TUNING", "0") == "1",
                  "policy_allocator": expected_allocator,
                  "resolved_config_difference": config_difference,
                  "generation": "TRT-LLM TP4 EP4", "gbs": 512,
                  "max_steps": g["max_num_steps"],
                  "evaluation": "full_offline" if offline_evaluation else "full_inline" if full_evaluation else "disabled",
                  "checkpoint_steps": checkpoint_steps,
                  "first_evaluation_step": g["val_start_at"] if full_evaluation else None,
                  "overlap": "P2P sync off; dispatch backward/expert wgrad on",
                  "ddp": {"bucket_size_elements": runtime.ddp.bucket_size,
                          "align_param_gather": runtime.ddp.align_param_gather,
                          "overlap_grad_reduce": runtime.ddp.overlap_grad_reduce,
                          "overlap_param_gather": runtime.ddp.overlap_param_gather},
                  "fusions": "router, permutation, gradients, bias activation; cuDNN norms; native Qwen RoPE",
                  "rope_regression": "CPU packed absolute-RoPE forward passes; unsupported fusion is rejected",
                  "hybridep": {"dispatch_combine_sms": model.moe_flex_dispatcher_num_sms,
                               "preprocessing_sms": model.moe_hybridep_num_sms_preprocessing},
                  "scope": "Constructed CPU model/DDP configs and unchanged-default test; no GPU/accuracy qualification"}))
