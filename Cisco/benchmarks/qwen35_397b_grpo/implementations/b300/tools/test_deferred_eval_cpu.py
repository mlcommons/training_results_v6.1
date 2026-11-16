''                                                                            
import ast
import copy
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qwen_deferred_under_test", ROOT / "utility/qwen35_mcore_deferred.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeferredEvaluationTests(unittest.TestCase):
    def test_init_budget_includes_synthetic_warmup(self):
        tree = ast.parse((ROOT / "run_grpo_nemo_gym.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        warmup_if = next(node for node in main.body if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "not is_trajectory_collection")
        text = ast.unparse(warmup_if)
        self.assertLess(text.index("maybe_run_mlperf_warmup("),
                        text.index("log_init_stop_run_start()"))
        self.assertIn("1790", text)
        self.assertIn("init_seconds >= 1790", text)

    def test_synthetic_gbs_override_and_alignment(self):
        tree = ast.parse((ROOT / "mlperf_warmup.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_warmup_batch_size")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     "mlperf_warmup.py", "exec"), namespace)
        choose = namespace["_warmup_batch_size"]
        self.assertEqual(choose(512, 1, 4, None), 512)
        self.assertEqual(choose(512, 1, 4, "128"), 128)
        for invalid in ("0", "-4", "513", "130"):
            with self.assertRaises(ValueError):
                choose(512, 1, 4, invalid)

    def setUp(self):
        self.config = {
            "policy": {"megatron_cfg": {
                "checkpoint": {"async_save": True},
                "distributed_data_parallel_config": {
                    "overlap_param_gather": True, "align_param_gather": True,
                    "overlap_grad_reduce": True, "bucket_size": 40000000,
                },
                "fp8_cfg": {"enabled": True, "fp8_recipe": "mxfp8"},
            }},
            "logger": {}, "grpo": {}, "checkpointing": {},
        }

    def evaluate_config(self, root):
        checkpoint = root / "retained" / "step_10"
        checkpoint.mkdir(parents=True)
        return module.evaluation_config(self.config, checkpoint, root / "evaluation")

    def test_only_eval_ddp_callbacks_disabled(self):
        before = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            evaluated = self.evaluate_config(Path(temporary))
            ddp = evaluated["policy"]["megatron_cfg"]["distributed_data_parallel_config"]
            self.assertFalse(ddp["overlap_param_gather"])
            self.assertFalse(ddp["align_param_gather"])
            self.assertTrue(ddp["overlap_grad_reduce"])
            self.assertEqual(ddp["bucket_size"], 40000000)
            self.assertEqual(evaluated["policy"]["megatron_cfg"]["fp8_cfg"],
                             before["policy"]["megatron_cfg"]["fp8_cfg"])
            self.assertEqual(self.config, before)
            view = Path(evaluated["checkpointing"]["checkpoint_dir"])
            self.assertTrue((view / "step_10").is_symlink())
            self.assertTrue((Path(temporary) / "retained" / "step_10").is_dir())

    def test_exact_pinned_callback_branch(self):
        candidates = [
            Path("/opt/nemo-rl/nemo_rl/models/megatron/setup.py"),
            ROOT.parent / "container_inspections/qwen35_trtllm_20261006/nemo_rl/models/megatron/setup.py",
        ]
        source = next(path for path in candidates if path.is_file())
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "setup_model_and_optimizer")
        branch = next(node for node in function.body if isinstance(node, ast.If)
                      and "megatron_cfg.ddp.align_param_gather" in ast.unparse(node.test))
        code = compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
                       str(source), "exec")
                                                                       
        plain_model = SimpleNamespace()
        training = {"megatron_cfg": SimpleNamespace(ddp=SimpleNamespace(
            overlap_param_gather=True, align_param_gather=True)), "model": [plain_model]}
        with self.assertRaises(AttributeError):
            exec(code, training)
        with tempfile.TemporaryDirectory() as temporary:
            evaluated = self.evaluate_config(Path(temporary))
        ddp = evaluated["policy"]["megatron_cfg"]["distributed_data_parallel_config"]
        evaluation = {"megatron_cfg": SimpleNamespace(ddp=SimpleNamespace(**ddp)),
                      "model": [plain_model], "param_sync_func": None}
        exec(code, evaluation)
        self.assertIsNone(evaluation["param_sync_func"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
