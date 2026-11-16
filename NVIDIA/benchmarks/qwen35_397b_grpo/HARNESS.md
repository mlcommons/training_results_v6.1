# OpenHands harness source and deviations

This disclosure applies to all 19 `nemo_rl_*` implementations of `qwen35_397b_grpo` in this directory. Backend-specific parser scope is identified below.

## Source

| Component | Repository and immutable commit |
|---|---|
| OpenHands | [michal2409/nv-OpenHands at `193be2fbae833c5b9bf3e24989451b6a837180f0`](https://github.com/michal2409/nv-OpenHands/tree/193be2fbae833c5b9bf3e24989451b6a837180f0) |
| NeMo Gym | [NVIDIA-NeMo/Gym at `8f2532ed8de619bf7c2ff99bd561f47c41dce776`](https://github.com/NVIDIA-NeMo/Gym/tree/8f2532ed8de619bf7c2ff99bd561f47c41dce776) |
| R2E-Gym | [michal2409/nv-R2E-Gym at `4f5f011afdbe70185d71e3e24dfdb47e42c6f465`](https://github.com/michal2409/nv-R2E-Gym/tree/4f5f011afdbe70185d71e3e24dfdb47e42c6f465) |

OpenHands and R2E-Gym are checked out at the commits explicitly specified in each Dockerfile. NeMo Gym is fetched as the `3rdparty/Gym-workspace/Gym` Git submodule of the Dockerfile's pinned NeMo-RL revision. All three packaged NeMo-RL revisions record Gym commit `8f2532ed8de619bf7c2ff99bd561f47c41dce776`:

| Builds | NeMo-RL revision and Gym gitlink |
|---|---|
| Eight vLLM implementations | [`a6ec8a93df4062e5991e225166f60ea981d170e3`](https://github.com/NVIDIA-NeMo/RL/tree/a6ec8a93df4062e5991e225166f60ea981d170e3/3rdparty/Gym-workspace/Gym) |
| Ten patched TRT-LLM implementations | [`34116c24284b88952ed1f3f27374ab2fb9ee990d`](https://github.com/NVIDIA-NeMo/RL/tree/34116c24284b88952ed1f3f27374ab2fb9ee990d/3rdparty/Gym-workspace/Gym) |
| Hecate 56-GPU TRT-LLM implementation | [`4c823cd8955cd41e09a65399f4abaad6240fe396`](https://github.com/NVIDIA-NeMo/RL/tree/4c823cd8955cd41e09a65399f4abaad6240fe396/3rdparty/Gym-workspace/Gym) |

The comparison baseline is the frozen [NeMo-RL reference at `3fca04c9b313d313302923a5bb6b0c8dc0340ed6`](https://github.com/NVIDIA-NeMo/RL/tree/3fca04c9b313d313302923a5bb6b0c8dc0340ed6/docker/mlperf), including its Dockerfile patches, OpenHands `0d766ad06b2be64a42e6f0175b9ebcc4a06599d9` and NeMo Gym `610a08ab5fe9f8f5fb5fff36b170429ea67f0f92`.

The submitted user/system prompt templates match that reference byte for byte. The implementations retain `CodeActAgent`, 30 agent turns, disabled tool-name diversification, and the reward mapping `1.0 if metrics.resolved else 0.0`.

## Ordered tool schemas

The enabled model-facing tools, in order, are `execute_bash`, `think`, `finish`, `task_tracker`, and `str_replace_editor`. The [tool definitions](https://github.com/michal2409/nv-OpenHands/tree/193be2fbae833c5b9bf3e24989451b6a837180f0/openhands/agenthub/codeact_agent/tools) and [`CodeActAgent._get_tools()` registration](https://github.com/michal2409/nv-OpenHands/blob/193be2fbae833c5b9bf3e24989451b6a837180f0/openhands/agenthub/codeact_agent/codeact_agent.py#L127-L182) identify the exact schema source and ordering.

With the same 60-second command-timeout setting and local-runtime launch settings, the ordered schemas match the reference except that `security_risk` is absent from the bash/editor `required` lists. Its property, description and enum remain present. This optionality is required by the [current benchmark harness rule](https://github.com/mlcommons/training_policies/blob/14bc89fe00691a5181f401bd67dc242dc928b205/training_rules.adoc#L684-L686). The [reference-to-submission OpenHands comparison](https://github.com/michal2409/nv-OpenHands/compare/0d766ad06b2be64a42e6f0175b9ebcc4a06599d9...193be2fbae833c5b9bf3e24989451b6a837180f0) exposes the changes directly.

## Harness deviations and purposes

| Deviation | Purpose |
|---|---|
| Make `security_risk` optional in bash/editor schemas; retain its description and enum. The disabled IPython schema receives the same change. | Match the reference parser's acceptance of omitted risk metadata and the current benchmark schema requirement. |
| Remove LiteLLM-only request fields and redundant tool-message `name` fields. | Satisfy Gym's validated request schema. |
| Apply a default 16 GiB virtual-address-space limit to command shells and descendants. | Contain runaway commands. |
| Replace Gym's inherited 32 GiB virtual-memory limit with a 32 GiB cumulative process-tree RSS watchdog and descendant cleanup. | Bound actual memory consumption without rejecting virtual-memory reservations. |
| Use 60-second command/default-sandbox and evaluation-test timeouts, retaining the 1800-second agent timeout; honor `SANDBOX_TIMEOUT`. | Apply the current benchmark timeout settings consistently. |
| Fail immediately when the local action server has exited. | Avoid retrying a dead process. |
| Convert undecodable text-diff blocks into Git binary literal hunks from recorded blobs during final patch extraction. | Preserve patch bytes without terminal encoding corruption; retain ordinary UTF8 extraction and existing binary-file selection. |
| Extract final patches through fresh static Bash sessions with explicit working directories, sharing the remaining `PATCH_EXTRACTION_TIMEOUT` budget (default 60 seconds, with a one-second minimum per operation). | Avoid blocked agent terminals and long per-operation extraction timeouts. |
| Separate redacted agent-instance data from private evaluator artifacts and hide the private directory with an empty overmount. | Keep gold patches, test patches and grading identifiers outside the agent sandbox. |
| Add PostgreSQL socket setup, Java dependency/network configuration, a Chrome compatibility wrapper and .NET heap limits. | Support task execution inside Apptainer. |
| Lock metrics reads and updates. | Prevent concurrent updates from losing fields. |
| Synchronize setup trees to configured commits, lock shared setup/cache operations and correct prompt argument positioning. | Prevent stale framework execution and incorrectly assigned prompt arguments. |
| Skip installing agent-only `chardet` dependencies during R2E local evaluation; declare `openai==2.53.0` and `docker==7.2.0` dependencies. | Avoid unnecessary evaluation-time installation and make dependencies explicit. |

The wrapper changes are in the pinned [Gym implementation](https://github.com/NVIDIA-NeMo/Gym/blob/8f2532ed8de619bf7c2ff99bd561f47c41dce776/responses_api_agents/swe_agents/app.py). The evaluator changes are in the [R2E-Gym comparison](https://github.com/michal2409/nv-R2E-Gym/compare/6823e64f94ae645f5265c03af0eb2e8523530a0d...4f5f011afdbe70185d71e3e24dfdb47e42c6f465).

## TensorRT-LLM parser scope

For tool parsing, TRT-LLM uses `qwen3_coder` instead of vLLM's `qwen3_xml`; for reasoning parsing, it uses `deepseek-r1` instead of `qwen3` to separate reasoning from assistant content. [NeMo-RL sets `reasoning_at_start` from `enable_thinking`](https://github.com/NVIDIA-NeMo/RL/blob/34116c24284b88952ed1f3f27374ab2fb9ee990d/nemo_rl/models/generation/trtllm/trtllm_http_server.py#L243-L254); with the submitted `enable_thinking: false`, ordinary output without reasoning delimiters remains assistant content, matching [vLLM's non-thinking path](https://github.com/vllm-project/vllm/blob/88d34c6409e9fb3c7b8ca0c04756f061d2099eb1/vllm/reasoning/qwen3_reasoning_parser.py#L115-L158).

`qwen3-coder-arguments.patch` was added for compliance, to align TensorRT-LLM's tool-argument parsing with vLLM semantics. We have not observed differences in training behavior between the vLLM and TRT-LLM backends.

Ten TRT-LLM implementations include this patch. Hecate's 14-node / 56-GPU implementation (`hecate_ngpu56_nemo_rl_trtllm`) predates its application and does not include it; its pinned [TensorRT-LLM parser](https://github.com/NVIDIA/TensorRT-LLM/blob/6c6303bc26ed63a1a5210487ef7c4260b81c2ca6/tensorrt_llm/serve/tool_parser/qwen3_coder_parser.py#L308-L388) remains unpatched. Theia's implementation at the same scale (`theia-jhb_ngpu56_nemo_rl_trtllm`) includes the patch.

The same ten TRT-LLM builds also apply `trtllm-sampler-seed.patch`, replacing the sampler's hardcoded seed `42` with `NEMO_RL_TRTLLM_SEED`. This changes generation RNG initialization, not the OpenHands agent harness or tool schemas; the Hecate 56-GPU build does not include this patch.
