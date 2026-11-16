# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

             

''                                              

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from generative_recommenders.compacted_forward.tensor import (
    CompactedTensor,
    JointCompactedTensor2D,
)
from generative_recommenders.compacted_forward.config import content_feature_pairs
from generative_recommenders.modules.dlrm_hstu import (
    _get_supervision_labels_and_weights,
    DlrmHSTU,
)
from generative_recommenders.modules.preprocessors import (
    get_contextual_input_embeddings,
)
from generative_recommenders.ops.gather_concat import gather_concat
from generative_recommenders.ops.compacted_sum_jagged_concat import (
    compacted_sum_jagged_concat,
)
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from torch.profiler import record_function
from torchrec.sparse.jagged_tensor import JaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

torch.fx.wrap("gather_concat")

ContentFeaturePairs = List[Tuple[str, str]]
PreprocessorOutput = Tuple[
    int,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, torch.Tensor],
]
DlrmHSTUOutput = Tuple[
    torch.Tensor,
    torch.Tensor,
    Dict[str, torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]


def embedding_values(
    dlrm: DlrmHSTU,
    lookup: Dict[str, JaggedTensor],
    feature_pairs: ContentFeaturePairs,
) -> Dict[str, torch.Tensor]:
    ''                                                                            

    feature_names = list(dlrm._hstu_configs.contextual_feature_to_max_length.keys()) + [
        history for history, _ in feature_pairs
    ]
    return {feature: lookup[feature].values() for feature in feature_names}


def preprocessing_forward(
    dlrm: DlrmHSTU,
    batch: Any,
    embeddings: Dict[str, torch.Tensor],
    payloads: Dict[str, torch.Tensor],
    max_uih_len: int,
    uih_lengths: torch.Tensor,
    max_num_candidates: int,
    num_candidates: torch.Tensor,
) -> Tuple[PreprocessorOutput, torch.dtype]:
    ''                                        

    config = dlrm._hstu_configs
    preprocessor = dlrm._hstu_transducer._input_preprocessor

    sequence_feature = config.uih_post_id_feature_name
    sequence_lengths = uih_lengths + num_candidates
    sequence_timestamps = payloads[
        batch.payload_feature_to_merged_key[config.uih_action_time_feature_name]
    ]
    total_targets = batch.total_targets
    total_uih_len = sequence_timestamps.numel() - total_targets

    sequence_values = embeddings[sequence_feature]
    output_dtype = sequence_values.dtype
    sequence_embeddings = CompactedTensor(
        values=sequence_values,
        reverse_indices=batch.reverse_indices,
    )
    additional_embeddings: Optional[JointCompactedTensor2D] = None
    if dlrm._additional_embedding_features:
        row_indices = batch.additional_row_indices
        joint_reverse_indices = batch.additional_reverse_indices
        if row_indices is None or joint_reverse_indices is None:
            raise RuntimeError("additional embeddings require joint routing")
        additional_embeddings = JointCompactedTensor2D(
            tensors=[
                CompactedTensor(
                    values=embeddings[feature],
                    reverse_indices=row_indices[:, index],
                )
                for index, feature in enumerate(dlrm._additional_embedding_features)
            ],
            joint_reverse_indices=joint_reverse_indices,
        )

    if not dlrm._is_inference:
        sequence_embeddings = sequence_embeddings.to(preprocessor._training_dtype)
        if additional_embeddings is not None:
            additional_embeddings = additional_embeddings.to(
                preprocessor._training_dtype
            )

    components: List[CompactedTensor] = [
        CompactedTensor(
            values=preprocessor._content_embedding_mlp(sequence_embeddings.values),
            reverse_indices=sequence_embeddings.reverse_indices,
        )
    ]
    sequence_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(sequence_lengths)

    if additional_embeddings is not None:
        additional = additional_embeddings.to_compacted_tensor(
            kernel=dlrm.hammer_kernel()
        )
        components.append(
            CompactedTensor(
                values=preprocessor._additional_embedding_mlp(additional.values),
                reverse_indices=additional.reverse_indices,
            )
        )

    if preprocessor._action_weights is not None:
        action_encoder = preprocessor._action_encoder
        num_action_classes = 1 << action_encoder._num_action_types
        class_ids = torch.arange(
            num_action_classes,
            dtype=torch.int64,
            device=action_encoder._action_embedding_table.device,
        )
        bit_positions = torch.arange(
            action_encoder._num_action_types,
            dtype=torch.int64,
            device=class_ids.device,
        )
        action_class_mask = (
            torch.bitwise_and(
                class_ids.unsqueeze(1),
                torch.bitwise_left_shift(
                    torch.ones_like(bit_positions), bit_positions
                ).unsqueeze(0),
            )
            > 0
        )
        action_inputs = (
            action_class_mask.unsqueeze(-1)
            * action_encoder._action_embedding_table.unsqueeze(0)
        ).view(num_action_classes, -1)
        candidate_action_input = action_encoder._target_action_embedding_table
        action_row_count = action_inputs.size(0)
        action_values = preprocessor._action_embedding_mlp(
            torch.cat([action_inputs, candidate_action_input], dim=0)
        )
        action_reverse = payloads[
            batch.payload_feature_to_merged_key[preprocessor._action_feature_name]
        ].to(torch.int32).scatter(
            dim=0,
            index=batch.candidate_positions,
            value=action_row_count,
        )
        components.append(
            CompactedTensor(
                values=action_values,
                reverse_indices=action_reverse,
            )
        )

    for feature in config.contextual_feature_to_max_length.keys():
        payloads[feature] = embeddings[feature]
        payloads[feature + "_offsets"] = torch.ops.fbgemm.asynchronous_complete_cumsum(
            batch.compacted_keys[feature].lengths()
        )
    if preprocessor._max_contextual_seq_len > 0:
        contextual_input_embeddings = get_contextual_input_embeddings(
            seq_lengths=sequence_lengths,
            seq_payloads=payloads,
            contextual_feature_to_max_length=(
                preprocessor._contextual_feature_to_max_length
            ),
            contextual_feature_to_min_uih_length=(
                preprocessor._contextual_feature_to_min_uih_length
            ),
            dtype=sequence_embeddings.values.dtype,
        )
        context = (
            torch.baddbmm(
                preprocessor._batched_contextual_linear_bias.view(
                    -1, 1, preprocessor._output_embedding_dim
                ).to(contextual_input_embeddings.dtype),
                contextual_input_embeddings.view(
                    -1,
                    preprocessor._max_contextual_seq_len,
                    preprocessor._input_embedding_dim,
                ).transpose(0, 1),
                preprocessor._batched_contextual_linear_weights.to(
                    contextual_input_embeddings.dtype
                ),
            )
            .transpose(0, 1)
        )
    else:
        context = None

    sequence_embeddings_tensor = compacted_sum_jagged_concat(
        compacted_tensors=components,
        dense=context,
        seq_offsets=sequence_offsets,
        max_seq_len=max_uih_len + max_num_candidates,
        kernel=dlrm.hammer_kernel(),
    )
    sequence_timestamps = concat_2D_jagged(
        max_seq_len=(
            preprocessor._max_contextual_seq_len + max_uih_len + max_num_candidates
        ),
        values_left=torch.zeros(
            (
                sequence_lengths.size(0) * preprocessor._max_contextual_seq_len,
                1,
            ),
            dtype=sequence_timestamps.dtype,
            device=sequence_timestamps.device,
        ),
        values_right=sequence_timestamps.unsqueeze(-1),
        max_len_left=preprocessor._max_contextual_seq_len,
        max_len_right=max_uih_len + max_num_candidates,
        offsets_left=None,
        offsets_right=sequence_offsets,
        kernel=dlrm.hammer_kernel(),
    ).squeeze(-1)
    output_lengths = sequence_lengths + preprocessor._max_contextual_seq_len
    return (
        (
            max_uih_len + max_num_candidates + preprocessor._max_contextual_seq_len,
            total_uih_len
            + preprocessor._max_contextual_seq_len * output_lengths.size(0),
            total_targets,
            output_lengths,
            torch.ops.fbgemm.asynchronous_complete_cumsum(output_lengths),
            sequence_timestamps,
            sequence_embeddings_tensor,
            num_candidates,
            payloads,
        ),
        output_dtype,
    )


def stu_compute_and_postprocessing(
    dlrm: DlrmHSTU,
    max_uih_len: int,
    max_targets: int,
    orig_dtype: torch.dtype,
    preprocessed: PreprocessorOutput,
) -> torch.Tensor:
    ''                                                              

    transducer = dlrm._hstu_transducer
    (
        max_seq_len,
        total_uih_len,
        total_targets,
        seq_lengths,
        seq_offsets,
        seq_timestamps,
        seq_embeddings,
        num_targets,
        seq_payloads,
    ) = preprocessed
    with record_function("hstu_positional_encoder"):
        if transducer._positional_encoder is not None:
            seq_embeddings = transducer._positional_encoder(
                max_seq_len=max_seq_len,
                seq_lengths=seq_lengths,
                seq_offsets=seq_offsets,
                seq_timestamps=seq_timestamps,
                seq_embeddings=seq_embeddings,
                num_targets=(None if transducer._listwise_training else num_targets),
            )
    seq_embeddings = torch.nn.functional.dropout(
        seq_embeddings,
        p=transducer._input_dropout_ratio,
        training=transducer.training,
    )
    encoded_embeddings = transducer._hstu_compute(
        max_seq_len=max_seq_len,
        seq_lengths=seq_lengths,
        seq_offsets=seq_offsets,
        seq_timestamps=seq_timestamps,
        seq_embeddings=seq_embeddings,
        num_targets=num_targets,
    )
    _, encoded_candidate_embeddings = transducer._postprocess(
        max_seq_len=max_seq_len,
        max_uih_len=max_uih_len,
        max_targets=max_targets,
        total_uih_len=total_uih_len,
        total_targets=total_targets,
        seq_lengths=seq_lengths,
        seq_embeddings=encoded_embeddings,
        seq_timestamps=seq_timestamps,
        num_targets=num_targets,
        seq_payloads=seq_payloads,
    )
    if not transducer._is_inference:
        encoded_candidate_embeddings = encoded_candidate_embeddings.to(orig_dtype)
    return encoded_candidate_embeddings


def compacted_dense_forward(
    dlrm: DlrmHSTU,
    batch: Any,
    feature_pairs: ContentFeaturePairs,
    lookup: Dict[str, JaggedTensor],
) -> DlrmHSTUOutput:
    ''                                                                         

    embeddings = embedding_values(dlrm, lookup, feature_pairs)
    payloads: Dict[str, torch.Tensor] = {
        key: batch.payloads_hist_candidate_kjt[key].values()
        for key in batch.payloads_hist_candidate_kjt.keys()
    }
    payloads["uih_offsets"] = batch.uih_offsets
    payloads["candidate_offsets"] = batch.candidate_offsets
    max_uih_len = batch.max_uih_len
    uih_lengths = batch.uih_lengths
    max_num_candidates = batch.max_num_candidates
    num_candidates = batch.num_candidates
    if dlrm._enable_tflops_logging:
        dlrm._last_jagged_flops_per_sample = dlrm._compute_jagged_flops_per_sample(
            uih_seq_lengths=uih_lengths,
            num_candidates=num_candidates,
        )

    with record_function("## item_forward ##"):
        candidate_inputs = gather_concat(
            tensors=[embeddings[history] for history, _ in feature_pairs],
            row_indices=batch.candidate_indices,
            kernel=dlrm.hammer_kernel(),
        )
        candidates_item_embeddings = dlrm._item_embedding_mlp(candidate_inputs)

    with record_function("## user_forward ##"):
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(not dlrm.is_inference) and dlrm._bf16_training,
        ):
            with record_function("hstu_input_preprocessor"):
                preprocessed, output_dtype = preprocessing_forward(
                    dlrm=dlrm,
                    batch=batch,
                    embeddings=embeddings,
                    payloads=payloads,
                    max_uih_len=max_uih_len,
                    uih_lengths=uih_lengths,
                    max_num_candidates=max_num_candidates,
                    num_candidates=num_candidates,
                )
            candidates_user_embeddings = stu_compute_and_postprocessing(
                dlrm=dlrm,
                max_uih_len=max_uih_len,
                max_targets=max_num_candidates,
                orig_dtype=output_dtype,
                preprocessed=preprocessed,
            )

    with record_function("## multitask_module ##"):
        config = dlrm._hstu_configs
        candidate_action_weights = torch.index_select(
            payloads[
                batch.payload_feature_to_merged_key[
                    config.candidates_weight_feature_name
                ]
            ],
            dim=0,
            index=batch.candidate_positions,
        )
        candidate_watchtimes = torch.index_select(
            payloads[
                batch.payload_feature_to_merged_key[
                    config.candidates_watchtime_feature_name
                ]
            ],
            dim=0,
            index=batch.candidate_positions,
        )
        supervision_labels, supervision_weights = _get_supervision_labels_and_weights(
            supervision_bitmasks=candidate_action_weights,
            watchtime_sequence=candidate_watchtimes,
            task_configs=dlrm._multitask_configs,
        )
        mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
            dlrm._multitask_module(
                encoded_user_embeddings=candidates_user_embeddings,
                item_embeddings=candidates_item_embeddings,
                supervision_labels=supervision_labels,
                supervision_weights=supervision_weights,
            )
        )

    aux_losses: Dict[str, torch.Tensor] = {}
    if not dlrm._is_inference and dlrm.training:
        for index, task in enumerate(dlrm._multitask_configs):
            aux_losses[task.task_name] = mt_losses[index]
    return (
        candidates_user_embeddings,
        candidates_item_embeddings,
        aux_losses,
        mt_target_preds,
        mt_target_labels,
        mt_target_weights,
    )


class CompactedDenseForward(torch.nn.Module):
    ''                                                                         

                                                                     
                                                                           
                                                                             
                                                                           
       

    def __init__(self, dlrm: DlrmHSTU, feature_pairs: ContentFeaturePairs) -> None:
        super().__init__()
        object.__setattr__(self, "_dlrm", dlrm)
        self._feature_pairs = feature_pairs

    def forward(self, batch: Any, lookup: Dict[str, JaggedTensor]) -> DlrmHSTUOutput:
        return compacted_dense_forward(
            dlrm=self._dlrm,
            batch=batch,
            feature_pairs=self._feature_pairs,
            lookup=lookup,
        )


class CompactedForwardAdapterModule(torch.nn.Module):
    ''                                                                 

    def __init__(self, dlrm: DlrmHSTU) -> None:
        super().__init__()
        object.__setattr__(self, "_dlrm", dlrm)
                                                                              
                                                           
        self._parameters = dlrm._parameters
        self._buffers = dlrm._buffers
        self._modules = dlrm._modules
        self._non_persistent_buffers_set = dlrm._non_persistent_buffers_set
        self.training = dlrm.training
        self._content_feature_pairs: ContentFeaturePairs = content_feature_pairs(
            dlrm._hstu_configs
        )
                                                                            
                                                                  
        self._compacted_dense_forward = CompactedDenseForward(
            dlrm, self._content_feature_pairs
        )
        logger.info("Compacted preprocessor enabled")

    def train(self, mode: bool = True) -> "CompactedForwardAdapterModule":
        super().train(mode)
        self._dlrm.training = self.training
        return self

    @property
    def _last_jagged_flops_per_sample(self) -> Optional[torch.Tensor]:
        return self._dlrm._last_jagged_flops_per_sample

    @property
    def _last_executed_flops_per_sample(self) -> Optional[torch.Tensor]:
        return self._dlrm._last_executed_flops_per_sample

    @property
    def enable_tflops_logging(self) -> bool:
        return self._dlrm._enable_tflops_logging

    def get_num_flops_per_sample(self) -> float:
        return self._dlrm.get_num_flops_per_sample()

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        ''                                

        return self._dlrm.state_dict(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        ''                              

        return self._dlrm.load_state_dict(state_dict, *args, **kwargs)

    def forward(self, batch: Any) -> DlrmHSTUOutput:
        lookup = self._embedding_collection(batch.compacted_keys)
        return self._compacted_dense_forward(batch, lookup)
