"""Controlled, offline segmentation batch diagnostic; writes only under --output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--model-path', default=r'D:\huggingface\hub\models--Intel--dpt-large-ade\snapshots\5ca59e2439c8340340eafccba2377816903f444a')
parser.add_argument('--weight-decay', type=float, default=.05)
parser.add_argument('--find', default='default')
parser.add_argument('--precision', default='fp16', choices=['fp16', 'fp32'])
parser.add_argument('--layers', action='store_true')
parser.add_argument('--eval-only', action='store_true')
parser.add_argument('--contiguous-conv', action='store_true')
parser.add_argument('--runtime-layout', action='store_true', help='Test the production HFCore layout workaround')
parser.add_argument('--verify-layout', action='store_true')
parser.add_argument('--eval-order', default='1,2,2,1')
parser.add_argument('--train-order', default='1,2')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.runtime_layout and (args.contiguous_conv or args.verify_layout):
    parser.error('--runtime-layout cannot be combined with manual layout hooks')
# Preserve the historical slow control after production gains the workaround.
os.environ['MLAAS_ROCM_DPT_CONTIGUOUS_CONV'] = '1' if args.runtime_layout else '0'
os.environ['MLAAS_ROCM_UPERNET_CONTIGUOUS_CONV'] = '1' if args.runtime_layout else '0'
root = Path(args.output).resolve()
root.mkdir(parents=True, exist_ok=True)
for env, folder in [('MIOPEN_USER_DB_PATH', 'db'), ('MIOPEN_CUSTOM_CACHE_DIR', 'kernels')]:
    dest = root / (args.find + '-' + folder)
    dest.mkdir(exist_ok=True)
    os.environ[env] = str(dest)
if args.find == 'default':
    os.environ.pop('MIOPEN_FIND_MODE', None)
else:
    os.environ['MIOPEN_FIND_MODE'] = args.find
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import numpy as np
import torch
import transformers
from datasets import Dataset, Image
from mlaas_data_generator.data.preprocessors.hf_image import _process_split
from mlaas_data_generator.models.adapters.hf_core import HFCore
from mlaas_data_generator.models.adapters.hf_task import ImageSegmentationSpec

model_path = args.model_path
dataset_path = r'D:\huggingface\datasets\EduardoPacheco___food_seg103\default\0.0.0\34e1208e14bc3595d544fc8c3f3c6673253fd9ef\food_seg103-validation.arrow'
prepared = root / 'prepared.npz'
if not prepared.exists():
    ds = Dataset.from_file(dataset_path).select([0, 1])
    ds = ds.cast_column('image', Image(decode=False)).cast_column('label', Image(decode=False))
    processor = transformers.AutoImageProcessor.from_pretrained(model_path, use_fast=False, local_files_only=True)
    x, y, report = _process_split(ds, split_name='diagnostic', image_processor=processor,
        image_column='image', mask_column='label', task_type='segmentation', training=False,
        dataset_id='EduardoPacheco/FoodSeg103', on_decode_error='error')
    np.savez(prepared, pixel_values=x['pixel_values'], labels=y)
    (root / 'preparation.json').write_text(json.dumps({'dataset': dataset_path, 'rows': [0, 1],
        'processor': processor.to_dict(), 'report': report}, default=str, indent=2))
data = np.load(prepared)
cpu_x = torch.from_numpy(data['pixel_values'])
cpu_y = torch.from_numpy(data['labels'])
torch.manual_seed(42)
np.random.seed(42)
core = HFCore(model_path, task_spec=ImageSegmentationSpec(), device='cuda',
    batch_size=2, mixed_precision=args.precision == 'fp16', precision_type=args.precision,
    label_pad_value=255)
core.model = transformers.AutoModelForSemanticSegmentation.from_pretrained(model_path, local_files_only=True).float().cuda()
core._configure_memory_optimizations()
model = core.model
from mlaas_data_generator.models.adapters.singleton_batchnorm import configure_singleton_batchnorm
configure_singleton_batchnorm(model, torch)
def contiguous_input(module, inputs):
    return (inputs[0].contiguous(), *inputs[1:])

def install_contiguous():
    return [module.register_forward_pre_hook(contiguous_input) for module in model.modules()
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]

if args.contiguous_conv:
    install_contiguous()
base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def memory():
    free, total = torch.cuda.mem_get_info()
    return {k: round(v / 2**30, 5) for k, v in dict(allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(), free=free, total=total).items()}

def timed(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    t = time.perf_counter()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, {'wall_ms': (time.perf_counter()-t)*1000, 'gpu_ms': start.elapsed_time(end)}

gpu_x, transfer_x = timed(lambda: cpu_x.cuda())
gpu_y, transfer_y = timed(lambda: cpu_y.cuda())
output = {'find': args.find, 'precision': args.precision, 'contiguous_conv': args.contiguous_conv,
    'runtime_layout': args.runtime_layout,
    'weight_decay': args.weight_decay,
    'pid': os.getpid(), 'torch': torch.__version__,
    'hip': torch.version.hip, 'transformers': transformers.__version__,
    'device': str(torch.cuda.get_device_properties(0)), 'input_shape': list(cpu_x.shape),
    'input_sha256': hashlib.sha256(cpu_x.numpy().tobytes()).hexdigest(),
    'labels_sha256': hashlib.sha256(cpu_y.numpy().tobytes()).hexdigest(),
    'model': model_path, 'gradient_checkpointing': getattr(core, 'gradient_checkpointing_enabled', False),
    'transfer_images': transfer_x, 'transfer_labels': transfer_y, 'initial_memory_gib': memory(),
    'evaluation': [], 'training': [], 'layers': {}}
tag = args.find + '-' + args.precision + ('-contiguous' if args.contiguous_conv else '') + ('-layers' if args.layers else '')
if args.runtime_layout:
    tag += '-runtime'
if args.verify_layout:
    tag += '-verify'
def save():
    (root / (tag + '.json')).write_text(json.dumps(output, indent=2))

def evaluate(batch):
    predictions = []
    with torch.no_grad(), core._make_autocast_context():
        for i in range(0, 2, batch):
            predictions.append(model(pixel_values=gpu_x[i:i+batch]).logits)
    return torch.cat(predictions)

reference = None
for batch in map(int, args.eval_order.split(',')):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pred, cold = timed(lambda: evaluate(batch))
    del pred
    times = []
    for _ in range(args.repeats):
        pred, timing = timed(lambda: evaluate(batch))
        times.append(timing)
        if reference is None:
            reference = pred.detach().cpu()
        comparison = (pred.detach().cpu() - reference).abs()
        agreement = (pred.detach().cpu().argmax(1) == reference.argmax(1)).float().mean().item()
        del pred
    entry = {'batch': batch, 'two_image_times': times, 'cold': cold, 'memory_gib': memory(),
        'logits_max_abs_difference': comparison.max().item(), 'prediction_agreement': agreement}
    output['evaluation'].append(entry)
    print('EVAL', json.dumps(entry), flush=True)
    save()

if args.verify_layout:
    assert not args.contiguous_conv
    original, _ = timed(lambda: evaluate(2))
    handles = install_contiguous()
    revised, timing = timed(lambda: evaluate(2))
    a, b = original.float(), revised.float()
    diff = (a - b).abs()
    output['layout_verification'] = {'batch': 2, 'wall_ms': timing['wall_ms'],
        'both_finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        'max_abs_difference': diff.max().item(), 'mean_abs_difference': diff.mean().item(),
        'relative_l2_difference': ((a-b).norm()/a.norm()).item(),
        'prediction_agreement': (a.argmax(1)==b.argmax(1)).float().mean().item(),
        'baseline_loss': core.task_spec.loss_fn(torch, a, gpu_y, {'ignore_index':255}).item(),
        'contiguous_loss': core.task_spec.loss_fn(torch, b, gpu_y, {'ignore_index':255}).item()}
    print('VERIFY', json.dumps(output['layout_verification']), flush=True)
    for handle in handles:
        handle.remove()
    del original, revised, a, b, diff
    save()

if args.layers:
    for batch in (1, 2):
        records, hooks, active = [], [], {}
        def pre(name):
            def hook(mod, inputs):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                active[name] = (event, time.perf_counter(), list(inputs[0].shape) if inputs and torch.is_tensor(inputs[0]) else None)
            return hook
        def post(name):
            def hook(mod, inputs, result):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                start, t, shape = active.pop(name)
                records.append((name, start, end, (time.perf_counter()-t)*1000, shape))
            return hook
        for name, module in model.named_modules():
            if not list(module.children()):
                hooks.extend([module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))])
        pred, _ = timed(lambda: evaluate(batch))
        del pred
        for hook in hooks:
            hook.remove()
        rows = [{'name': name, 'gpu_ms': start.elapsed_time(end), 'cpu_dispatch_ms': cpu, 'shape': shape}
                for name, start, end, cpu, shape in records]
        output['layers'][str(batch)] = sorted(rows, key=lambda row: row['gpu_ms'], reverse=True)
        print('LAYERS', batch, json.dumps(output['layers'][str(batch)][:12]), flush=True)
        save()

if not args.eval_only:
    for batch in map(int, args.train_order.split(',')):
        model.load_state_dict(base)
        model.train()
        torch.manual_seed(42)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
        applied_updates = [0]
        def count_update(*unused):
            applied_updates[0] += 1
        update_hook = optimizer.register_step_post_hook(count_update)
        def train_step():
            optimizer.zero_grad(set_to_none=True)
            phases, losses = [], []
            for i in range(0, 2, batch):
                def forward():
                    with core._make_autocast_context():
                        logits = model(pixel_values=gpu_x[i:i+batch]).logits
                        loss = core.task_spec.loss_fn(torch, logits, gpu_y[i:i+batch], {'ignore_index':255})
                    return loss
                loss, ft = timed(forward)
                _, bt = timed(lambda: scaler.scale(loss * batch / 2).backward())
                phases.append({'forward': ft, 'backward': bt})
                losses.append(loss.detach().item())
            before_update = applied_updates[0]
            scale_before = scaler.get_scale()
            _, ot = timed(lambda: (scaler.step(optimizer), scaler.update()))
            return {'phases': phases, 'optimizer': ot, 'losses': losses,
                    'optimizer_update_applied': applied_updates[0] > before_update,
                    'scale_before': scale_before, 'scale_after': scaler.get_scale()}
        cold, cold_time = timed(train_step)
        entries = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            details, timing = timed(train_step)
            entries.append({'details': details, 'time': timing, 'memory_gib': memory()})
        entry = {'batch':batch,'effective_batch':2,'accumulation':2//batch,'cold_time':cold_time,'steps':entries}
        output['training'].append(entry)
        print('TRAIN', json.dumps(entry), flush=True)
        optimizer.zero_grad(set_to_none=True)
        update_hook.remove()
        del optimizer, scaler
        torch.cuda.empty_cache()
        save()
save()
print('DIAGNOSTIC_COMPLETE', tag, flush=True)
