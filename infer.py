"""Offline codec-token inference; no audio decoder or WAV output is included."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path

import torch


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def run(*, checkpoint, text, num_frames, output, reference_codes=None,
        reference_text=None, confidence_threshold=0.8, max_inference_steps=16,
        device='cpu', precision='fp32'):
    """Generate codec tokens from a trusted training checkpoint, entirely locally.

    The checkpoint is loaded with weights_only=False (pickle can execute code).
    Reference files are restricted to weights_only=True tensor/dict payloads.
    """
    _positive(num_frames, 'num_frames')
    _positive(max_inference_steps, 'max_inference_steps')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be nonempty')
    if (isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1):
        raise ValueError('confidence_threshold must be finite and in [0, 1]')
    if (reference_codes is None) != (reference_text is None):
        raise ValueError('reference_codes and reference_text must be supplied together')
    if reference_text is not None and (not isinstance(reference_text, str) or not reference_text.strip()):
        raise ValueError('reference_text must be nonempty')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and device.index is not None):
        raise ValueError('device must be cpu or cuda[:index]')
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32, fp16 or bf16')
    if device.type != 'cuda' and precision != 'fp32':
        raise ValueError('fp16/bf16 precision requires cuda')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError('CUDA device index is unavailable')
        if precision == 'bf16':
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError('CUDA device does not support bf16')
    output = Path(output).expanduser().absolute()
    if output.suffix != '.pt':
        raise ValueError('output must have a .pt suffix (codec tensors, not WAV)')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite output: {output}')
    if not output.parent.is_dir():
        raise ValueError(f'Output parent directory does not exist: {output.parent}')
    checkpoint = Path(checkpoint).expanduser().resolve()
    config_path = checkpoint.parent / 'backbone_config.json'
    tokenizer_path = checkpoint.parent / 'tokenizer'
    for path in (checkpoint, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    reference = None
    if reference_codes is not None:
        reference_path = Path(reference_codes).expanduser()
        if reference_path.suffix != '.pt':
            raise ValueError('reference_codes must be a .pt file')
        reference = torch.load(reference_path, map_location='cpu', weights_only=True)
        if isinstance(reference, dict):
            reference = reference.get('codes')
        if (not isinstance(reference, torch.Tensor) or reference.ndim != 2
                or reference.shape[0] <= 0 or reference.shape[1] <= 0
                or reference.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError('Reference codes must be an integer tensor [Q, positive R] or dict with codes')
        reference = reference.long()

    # Training checkpoints include Python/NumPy RNG state: trusted pickle only.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get('config'), dict) or not isinstance(saved.get('model'), dict):
        raise ValueError('Expected a training checkpoint with model state_dict and config')
    model_config = saved['config'].get('model')
    if not isinstance(model_config, dict):
        raise ValueError('Checkpoint config must contain model settings')
    q = _positive(model_config.get('num_codebooks'), 'num_codebooks')
    k = _positive(model_config.get('codebook_size'), 'codebook_size')
    if reference is not None and (reference.shape[0] != q or not torch.all((reference >= 0) & (reference < k))):
        raise ValueError(f'Reference codes must have {q} codebooks and values in [0, {k})')
    limits = saved['config'].get('masking', {})
    max_text_tokens = _positive(limits.get('max_text_tokens', 512), 'max_text_tokens')
    max_sequence_length = _positive(limits.get('max_sequence_length', 40960), 'max_sequence_length')

    from transformers import AutoTokenizer, Qwen3Config
    from model.bidirectional_qwen3 import BidirectionalQwen3, TTS_SPECIAL_TOKENS
    from model.VoxBase_TTS import VoxBaseTTS

    config = Qwen3Config.from_json_file(str(config_path))
    backend = model_config.get('attention_backend', 'sdpa')
    if backend not in ('sdpa', 'eager'):
        raise ValueError('Checkpoint attention_backend must be eager or sdpa')
    config._attn_implementation = backend
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    expected = getattr(config, 'tts_token_ids', None)
    vocab = tokenizer.get_vocab()
    actual = {name: vocab.get(token) for name, token in TTS_SPECIAL_TOKENS.items()}
    if (not isinstance(expected, dict) or actual != expected
            or any(type(i) is not int or not 0 <= i < config.vocab_size for i in actual.values())
            or len(set(actual.values())) != len(actual)):
        raise ValueError('Tokenizer TTS token mapping does not equal backbone config.tts_token_ids')
    if any(type(i) is not int or not 0 <= i < config.vocab_size for i in vocab.values()):
        raise ValueError('Tokenizer vocabulary exceeds backbone embedding table')
    target_tokens = tokenizer.encode(text, add_special_tokens=False)
    ref_tokens = tokenizer.encode(reference_text, add_special_tokens=False) if reference is not None else []
    if len(target_tokens) > max_text_tokens or len(ref_tokens) > max_text_tokens:
        raise ValueError(f'Text exceeds max_text_tokens={max_text_tokens}')
    packed_length = len(target_tokens) + num_frames + 4
    if reference is not None:
        packed_length += len(ref_tokens) + reference.shape[1] + 4
    if packed_length > min(max_sequence_length, config.max_position_embeddings):
        raise ValueError('Packed sequence length exceeds configured limit')

    model = VoxBaseTTS(BidirectionalQwen3(config), num_codebooks=q, codebook_size=k)
    model.load_state_dict(saved['model'], strict=True)
    model = model.float().to(device).eval()
    context = (torch.autocast('cuda', dtype={'fp16': torch.float16, 'bf16': torch.bfloat16}[precision])
               if precision != 'fp32' else nullcontext())
    with torch.inference_mode(), context:
        result = model.generate(
            tokenizer, text, num_frames,
            reference_codes=reference.to(device) if reference is not None else None,
            reference_text=reference_text, confidence_threshold=confidence_threshold,
            max_inference_steps=max_inference_steps, max_text_tokens=max_text_tokens,
            max_sequence_length=max_sequence_length)
    payload = {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
               for name, value in result.items()}
    payload.update(text=text, num_frames=num_frames)
    # Exclusive creation also protects against an output appearing during generation.
    with output.open('xb') as stream:
        torch.save(payload, stream)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        'Offline codec-token inference (no WAV). Only load trusted training checkpoints: '
        'torch.load(weights_only=False) can execute arbitrary pickle code. '
        'Requires sibling backbone_config.json and tokenizer/.'))
    parser.add_argument('--checkpoint', required=True, help='Trusted training .pt checkpoint')
    parser.add_argument('--text', required=True)
    parser.add_argument('--num-frames', required=True, type=int)
    parser.add_argument('--output', required=True, help='New .pt codec-token output; never overwritten')
    parser.add_argument('--reference-codes', help='Local .pt tensor [Q,R] or dict with codes; loaded weights_only=True')
    parser.add_argument('--reference-text', help='Aligned transcript; required together with --reference-codes')
    parser.add_argument('--confidence-threshold', type=float, default=0.8)
    parser.add_argument('--max-inference-steps', type=int, default=16)
    parser.add_argument('--device', default='cpu', help='cpu (default) or cuda[:index]')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32', help='fp16/bf16 require CUDA')
    args = parser.parse_args(argv)
    try:
        return run(**vars(args))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
