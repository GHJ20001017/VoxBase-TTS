"""Compare legacy threshold and cosine codec-token decoding on saved NPZ samples.

Each NPZ is expected to contain target/condition reference codes and texts. The
script writes one compressed NPZ per sample with generated codes plus a JSONL
of decoding diagnostics; it never supplies target codes to the model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))

from model.VoxBase_TTS import VoxBaseTTS
from model.bidirectional_qwen3 import BidirectionalQwen3


def _scalar_text(value):
    return str(value.item()) if isinstance(value, np.ndarray) and value.ndim == 0 else str(value)


def _load_model(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config_data = saved.get('config', {}).get('model', {})
    num_codebooks = int(config_data['num_codebooks'])
    if num_codebooks != 8:
        raise ValueError(f'Inference comparison requires exactly 8 codebooks, got {num_codebooks}')
    config = Qwen3Config.from_json_file(str(checkpoint.parent / 'backbone_config.json'))
    config._attn_implementation = config_data.get('attention_backend', 'sdpa')
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.parent / 'tokenizer', local_files_only=True, trust_remote_code=False)
    model = VoxBaseTTS(
        BidirectionalQwen3(config),
        num_codebooks=int(config_data.get('num_codebooks', 8)),
        codebook_size=int(config_data.get('codebook_size', 2048)),
    )
    model.load_state_dict(saved['model'], strict=True)
    del saved
    gc.collect()
    return model.float().to(device).eval(), tokenizer


def run(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    samples_dir = Path(args.samples_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    codes_dir = output_dir / 'codes'
    samples = [samples_dir / f'sample_{i:03d}.npz' for i in args.sample_indices]
    missing = [str(p) for p in samples if not p.is_file()]
    if missing:
        raise FileNotFoundError(', '.join(missing))
    metrics_path = output_dir / 'metrics.jsonl'
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    codes_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_model(checkpoint, device)
    settings = {
        'checkpoint': str(checkpoint), 'samples_dir': str(samples_dir),
        'sample_indices': args.sample_indices, 'decoding_schedules': ['threshold', 'cosine'],
        'conditions': ['same', 'other', 'none'], 'steps': args.max_inference_steps,
        'threshold': args.confidence_threshold,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (output_dir / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    with metrics_path.open('w', encoding='utf-8') as metrics:
        for sample_path in samples:
            data = np.load(sample_path)
            index = int(sample_path.stem.split('_')[-1])
            payload = {key: data[key] for key in data.files}
            text = _scalar_text(data['target_text'])
            target = torch.from_numpy(data['target'].astype('int64')).to(device)
            if target.ndim != 2 or target.shape[0] != 8 or target.shape[1] == 0:
                raise ValueError(f'{sample_path} target must have shape [8, T]')
            for condition in ('same', 'other'):
                reference_codes = data[f'reference_{condition}']
                if reference_codes.ndim != 2 or reference_codes.shape[0] != 8 or reference_codes.shape[1] == 0:
                    raise ValueError(f'{sample_path} reference_{condition} must have shape [8, R]')
            for schedule in ('threshold', 'cosine'):
                for condition in ('same', 'other', 'none'):
                    reference = None
                    reference_text = None
                    if condition != 'none':
                        reference = torch.from_numpy(
                            data[f'reference_{condition}'].astype('int64')).to(device)
                        reference_text = _scalar_text(data[f'reference_{condition}_text'])
                    start = time.monotonic()
                    context = (torch.autocast('cuda', dtype=torch.float16)
                               if device.type == 'cuda' else torch.no_grad())
                    with torch.inference_mode(), context:
                        result = model.generate(
                            tokenizer, text, int(target.shape[-1]),
                            reference_codes=reference, reference_text=reference_text,
                            decoding_schedule=schedule,
                            max_inference_steps=args.max_inference_steps,
                            confidence_threshold=args.confidence_threshold)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    name = f'generated_{schedule}_{condition}'
                    codes = result['codes'].cpu().numpy().astype('uint16')
                    payload[name] = codes
                    legacy_equal = None
                    if schedule == 'threshold' and name.replace('threshold_', '') in data.files:
                        legacy_equal = bool(np.array_equal(codes, data[name.replace('threshold_', '')]))
                    row = {
                        'sample_index': index, 'condition': condition,
                        'decoding_schedule': schedule,
                        'seconds': time.monotonic() - start,
                        'accuracy_to_target': float((result['codes'] == target).float().mean()),
                        'below_threshold_fraction': float(result['below_threshold'].float().mean()),
                        'forced_fraction': float(result['forced_mask'].float().mean()),
                        'num_steps': int(result['num_steps']),
                        'remaining_history': result['remaining_history'],
                        'remaining_per_codebook_history': result['remaining_per_codebook_history'],
                        'schedule_steps': result['schedule_steps'],
                        'legacy_bitwise_equal': legacy_equal,
                    }
                    metrics.write(json.dumps(row, ensure_ascii=False) + '\n')
                    metrics.flush()
            np.savez_compressed(codes_dir / sample_path.name, **payload)
    (output_dir / 'complete.json').write_text(
        json.dumps({'status': 'complete', 'time': time.time()}, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--samples-dir', required=True, help='Directory containing sample_000.npz files')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sample-indices', nargs='+', type=int, default=[0, 25, 50, 75])
    parser.add_argument('--confidence-threshold', type=float, default=0.8)
    parser.add_argument('--max-inference-steps', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
