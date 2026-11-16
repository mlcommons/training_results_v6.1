#!/usr/bin/env python3

# pyre-unsafe

"""The M2 dq producer (hstu_dq_e5m2, DLRM_HSTU_FP8_BWD_DQ_E5M2) must stay in
sync with r15's dq-split.

`DqE5m2Sink` runs a fork of `hstu_dq_split._dq_unit`'s main loop. These tests
run the real FlyDSL C8 attention backward twice on the same inputs -- once with
r15's stock dq-split (BF16 dq) and once through the sink -- and require the
sink's e5m2 dz payload [Mp, 2048] and amax to equal the shipped cast of the
stock result byte for byte, and its u|v|k bias column sums to be bitwise (the
q columns are a declared fp32-order change). A change to the ws layout or the
dq main loop on either side fails here instead of silently in training.
Skipped without a gfx950 GPU and FlyDSL.
"""

import os
import unittest
from unittest import mock

import torch

_C8_ENV = {
    "HSTU_ATTN_BWD": "flydsl",
    "DLRM_HSTU_FP8_BWD_QK_MFMA": "fp8",
    "DLRM_HSTU_FP8_BWD_FULL": "1",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT": "1",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED": "ilr",
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT": "sp,r1,wcg",
}


def _flydsl_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    try:
        from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha

        reason = mha.unsupported_hardware_reason()
        if reason:
            return reason
        mha.require_flydsl()
    except Exception as error:  # noqa: BLE001
        return f"FlyDSL unavailable: {error}"
    return None


@unittest.skipIf(_flydsl_reason() is not None, str(_flydsl_reason()))
class DqE5m2SinkSyncTest(unittest.TestCase):
    def _run(self, lengths, sched="ilr"):
        from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        from generative_recommenders.ops.triton import bwd_fp8
        from generative_recommenders.ops.triton.hstu_dq_e5m2 import DqE5m2Sink

        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(7)
        L = torch.tensor(lengths, device=dev)
        offs = torch.cat((torch.zeros(1, dtype=torch.int64, device=dev), L.cumsum(0)))
        T, N, H, D = int(offs[-1]), int(max(lengths)), 4, 128
        s = 0.0173
        vqk = (torch.randn(T, 3 * H * D, device=dev, generator=g) * (0.1 / s)).to(
            torch.float8_e4m3fn
        )
        v, q, k = (x.view(T, H, D) for x in vqk.split(H * D, dim=1))
        dout = (torch.randn(T, H, D, device=dev, generator=g) * 1e-3).to(torch.bfloat16)
        ref = torch.zeros((T, 4 * H * D), device=dev, dtype=torch.bfloat16)
        ref[:, : H * D] = (torch.randn(T, H * D, device=dev, generator=g) * 1e-5).to(
            torch.bfloat16
        )
        got = ref.clone()

        def views(buf):
            return tuple(
                buf[:, c : c + H * D].view(T, H, D) for c in (2 * H * D, 3 * H * D, H * D)
            )

        env = dict(_C8_ENV, DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED=sched)
        if not sched:
            env["DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT"] = ""
        with mock.patch.dict(os.environ, env):
            mha.flydsl_hstu_attention_bwd(
                dout, q, k, v, offs, N, 1.0 / D**0.5, out=views(ref), qkv_scale=s
            )
            amax_q = ref[:, 2 * H * D : 3 * H * D].float().abs().amax()
            inv = (1.0 / (amax_q * (8.0 / 57344.0))).reshape(1).float()
            got[:, 2 * H * D : 3 * H * D] = float("nan")
            sink = DqE5m2Sink(
                rows=T,
                width=4 * H * D,
                q_cols=(2 * H * D, 3 * H * D),
                inv_scale=inv,
                cast_scale=1.0 / inv,
                pad=128,
                grad_dtype=torch.float8_e5m2,
                limit=57344.0,
                device=dev,
            )
            mha.flydsl_hstu_attention_bwd(
                dout, q, k, v, offs, N, 1.0 / D**0.5, out=views(got), qkv_scale=s,
                dq_fp8_sink=sink,
            )
        return bwd_fp8, sink, ref, got, inv, T

    def test_sink_matches_stock_dq_split_and_cast(self) -> None:
        bwd_fp8, sink, ref, got, inv, T = self._run([256, 1000, 37, 1500, 641])
        self.assertTrue(sink.filled, sink.declined)
        # dv / dk came from the same deterministic kernel in both runs.
        self.assertTrue(torch.equal(got[:, 512:1024].view(torch.int16), ref[:, 512:1024].view(torch.int16)))
        self.assertTrue(torch.equal(got[:, 1536:].view(torch.int16), ref[:, 1536:].view(torch.int16)))
        dz8, amax, colsum, m = sink.finish(got)
        r8, ramax, rcol = bwd_fp8.quantize_amax_colsum(ref, inv, pad_to=128)
        self.assertEqual(m, T)
        self.assertEqual(tuple(dz8.shape), tuple(r8.shape))
        self.assertTrue(torch.equal(dz8.view(torch.uint8), r8.view(torch.uint8)))
        self.assertEqual(amax.view(torch.int32).item(), ramax.view(torch.int32).item())
        uvk = torch.cat((torch.arange(0, 1024), torch.arange(1536, 2048))).to(colsum.device)
        self.assertTrue(torch.equal(colsum[uvk].view(torch.int32), rcol[uvk].view(torch.int32)))
        o64 = ref[:, 1024:1536].double().sum(0)
        err = float((colsum[1024:1536].double() - o64).norm() / o64.norm())
        self.assertLess(err, 1e-5)

    def test_unscheduled_dq_split_declines_to_stock(self) -> None:
        _, sink, ref, got, _, _ = self._run([256, 1000, 37], sched="")
        self.assertFalse(sink.filled)
        self.assertIn("sched", str(sink.declined))
        self.assertTrue(torch.equal(got.view(torch.int16), ref.view(torch.int16)))


if __name__ == "__main__":
    unittest.main()
