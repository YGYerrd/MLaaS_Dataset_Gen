"""Offline CLIP batch-policy smoke test on fixed synthetic pairs (not a quality benchmark)."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=8)
args = parser.parse_args()
root = Path(args.output).resolve()
root.mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLAAS_PREFETCH_BATCHES"] = "0"
for env, folder in (("MIOPEN_USER_DB_PATH", "db"), ("MIOPEN_CUSTOM_CACHE_DIR", "kernels")):
    path = root / folder
    path.mkdir(exist_ok=True)
    os.environ[env] = str(path)

import numpy as np
import torch
from mlaas_data_generator.models.adapters.hf_core import HFCore
from mlaas_data_generator.models.adapters.hf_task import TextImageRetrievalSpec

torch.manual_seed(42)
np.random.seed(42)
core = HFCore(args.model_path, task_spec=TextImageRetrievalSpec(), device="cuda",
              batch_size=args.batch_size, max_length=77, mixed_precision=True, precision_type="fp16")
config = core.model.config
n = 17
rng = np.random.default_rng(42)
ids = rng.integers(1, config.text_config.vocab_size - 2, size=(n, 77), dtype=np.int64)
ids[:, 0] = config.text_config.bos_token_id
ids[:, -1] = config.text_config.eos_token_id
size = config.vision_config.image_size
xs = {"input_ids": ids, "attention_mask": np.ones_like(ids),
      "pixel_values": rng.normal(size=(n, 3, size, size)).astype(np.float32)}
np.savez(root / "prepared.npz", **xs)
before = core.model.text_projection.weight.detach().cpu().clone()
torch.cuda.reset_peak_memory_stats()
train = core.finetune(xs, None, epochs=args.epochs, lr=1e-6, optimizer="adamw", weight_decay=0.,
                      gradient_accumulation_steps=2, max_train_time_s=180, progress_log_interval=1)
torch.cuda.synchronize()
result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
          "hip": torch.version.hip, "gpu_capacity_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
          "model": args.model_path, "input_shape": list(xs["pixel_values"].shape),
          "text_length": 77, "precision": core.effective_precision_type,
          "gradient_checkpointing": core.gradient_checkpointing_enabled,
          "train": train, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
          "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
          "projection_max_change": float((core.model.text_projection.weight.detach().cpu() - before).abs().max())}
result["evaluation"] = []
for batch in (1, 4):
    core.batch_size = batch
    loss, r1, r5, qos = core.eval(xs, None, progress_log_interval=1)
    result["evaluation"].append({"encoding_batch": batch, "loss": loss, "r1": r1, "r5": r5,
                                 "candidates": qos["eval_loss_candidate_count"]})
result["complete"] = train["successful_optimizer_step_count"] > 0 and result["projection_max_change"] > 0
(root / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
print(json.dumps(result, indent=2, default=str), flush=True)
assert train["successful_optimizer_step_count"] > 0
assert result["projection_max_change"] > 0
assert train["contrastive_candidates_min"] >= 2
print("DIAGNOSTIC_COMPLETE", flush=True)
