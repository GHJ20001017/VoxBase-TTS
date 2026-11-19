"""Decode codec-token files or NPZ code collections into WAV files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from transformers import MimiModel


def _decode(model, codes, sample_rate, destination, target_length=None):
    with torch.inference_mode():
        wave = model.decode(codes[None]).audio_values[0, 0].float().numpy()
    if target_length is not None:
        wave = wave[:target_length]
    sf.write(destination, np.clip(wave, -1, 1), sample_rate, subtype='PCM_16')
    return {
        'name': destination.stem,
        'path': str(destination),
        'seconds': len(wave) / sample_rate,
        'rms': float(np.sqrt(np.mean(wave ** 2))),
        'peak': float(np.max(np.abs(wave))),
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def _load_model(mimi_model):
    return MimiModel.from_pretrained(
        mimi_model, local_files_only=True, attn_implementation='eager').eval()


def _decode_pt(args, model):
    source = Path(args.codes_file).expanduser().resolve()
    payload = torch.load(source, map_location='cpu', weights_only=True)
    codes = payload.get('codes') if isinstance(payload, dict) else payload
    if not isinstance(codes, torch.Tensor) or codes.ndim != 2:
        raise ValueError('The .pt file must contain an integer codec tensor [Q, T] or dict key codes')
    if codes.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('Codec tokens must use an integer tensor dtype')
    destination = Path(args.output_wav).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = _decode(model, codes.long(), args.sample_rate, destination)
    print(time.strftime('%H:%M:%S'), row, flush=True)


def _decode_npz(args, model):
    source = Path(args.codes_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(source.glob('*.npz')):
        data = np.load(path)
        target_length = int(data[args.audio_length_field]) if args.audio_length_field in data else None
        for name in data.files:
            if name != 'target' and not name.startswith('generated_cosine_'):
                continue
            codes = torch.from_numpy(data[name].astype('int64'))
            destination = output / f'{path.stem}_{name}.wav'
            row = _decode(model, codes, args.sample_rate, destination, target_length)
            row['source'] = str(path)
            rows.append(row)
            print(time.strftime('%H:%M:%S'), row, flush=True)
    (output / 'manifest.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--codes-file', help='A .pt file produced by infer.py')
    sources.add_argument('--codes-dir', help='Directory containing NPZ codec arrays')
    parser.add_argument('--output-wav', help='Output WAV path for --codes-file')
    parser.add_argument('--output-dir', help='Output directory for --codes-dir')
    parser.add_argument('--mimi-model', required=True, help='Local Mimi decoder directory')
    parser.add_argument('--sample-rate', type=int, default=24000)
    parser.add_argument('--audio-length-field', default='audio_num_samples')
    args = parser.parse_args()
    if args.codes_file and not args.output_wav:
        parser.error('--output-wav is required with --codes-file')
    if args.codes_dir and not args.output_dir:
        parser.error('--output-dir is required with --codes-dir')
    model = _load_model(args.mimi_model)
    if args.codes_file:
        _decode_pt(args, model)
    else:
        _decode_npz(args, model)


if __name__ == '__main__':
    main()
