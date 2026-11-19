"""Reconstruct WAV files from target/generated codec-token arrays.

The codec decoder is loaded locally through Transformers MimiModel. The input
NPZ directory is normally the ``codes/`` output of evaluate_inference.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import soundfile as sf
from transformers import MimiModel


def run(args):
    source = Path(args.codes_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = MimiModel.from_pretrained(
        args.mimi_model, local_files_only=True, attn_implementation='eager').eval()
    rows = []
    for path in sorted(source.glob('*.npz')):
        data = np.load(path)
        target_length = int(data[args.audio_length_field]) if args.audio_length_field in data else None
        for name in data.files:
            if name != 'target' and not name.startswith(('generated_cosine_', 'generated_threshold_')):
                continue
            codes = torch.from_numpy(data[name].astype('int64'))[None]
            with torch.inference_mode():
                wave = model.decode(codes).audio_values[0, 0].float().numpy()
            if target_length is not None:
                wave = wave[:target_length]
            destination = output / f'{path.stem}_{name}.wav'
            sf.write(destination, np.clip(wave, -1, 1), args.sample_rate, subtype='PCM_16')
            row = {
                'source': str(path), 'name': name, 'path': str(destination),
                'seconds': len(wave) / args.sample_rate,
                'rms': float(np.sqrt(np.mean(wave ** 2))),
                'peak': float(np.max(np.abs(wave))),
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
            rows.append(row)
            print(time.strftime('%H:%M:%S'), row, flush=True)
    (output / 'manifest.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codes-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mimi-model', required=True, help='Local Mimi decoder directory')
    parser.add_argument('--sample-rate', type=int, default=24000)
    parser.add_argument('--audio-length-field', default='audio_num_samples')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
