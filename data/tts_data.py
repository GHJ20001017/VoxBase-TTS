"""Lazy, map-style TTS Parquet dataset (requires torch and pyarrow).

Example::

    from torch.utils.data import DataLoader
    from data.tts_data import TTSDataset, tts_collate_fn

    dataset = TTSDataset('/gpu3/guhj/VoxBase-TTS/dataset/tts_8q/train')
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4,
                        collate_fn=tts_collate_fn, pin_memory=True)
    for batch in loader:
        codes = batch['codes']  # int64 [batch, codebook, time]

For DDP, pass a DistributedSampler instead of shuffle=True, and call
sampler.set_epoch(epoch). Text is returned unchanged for the training tokenizer.
"""

from bisect import bisect_right
from collections import OrderedDict
from operator import index as integer_index
from pathlib import Path
import os
import math
import random
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

__all__ = ['TTSDataset', 'tts_collate_fn']

_REQUIRED_COLUMNS = {'sample_id', 'text', 'codes', 'num_codebooks', 'num_frames'}


class TTSDataset(Dataset):
    """Treat all sample shards below a directory as one indexed dataset.

    Initialization reads Parquet footers and, when reference_prob > 0, only
    pairing metadata columns to build a speaker index (never codec arrays).
    Each worker lazily reads
    row groups and maintains its own bounded LRU cache. ``cache_size`` counts
    row groups, not samples; the memory bound depends on the shard layout.
    The supplied directory is the split root (train/validation/test).
    Parquet files are discovered recursively below it, without appending or
    prioritizing any named subdirectory. Files in indexes/ are excluded.

    Each item preserves the original columns, except ``codes`` becomes a CPU
    int64 tensor [Q, T]. Missing or inconsistent sample fields fail explicitly.
    Valid speakers with another distinct sample in the same release/dataset
    and codebook layout receive a random reference with reference_prob (0.8
    by default). Set it to 0 to disable pairing and metadata scanning. Pairing
    uses Python's RNG, seeded automatically in DataLoader workers; seed random
    explicitly for reproducible num_workers=0 runs. Build datasets separately
    for train/validation: wrapping this dataset in Subset does not restrict its
    reference pool. Missing references have has_reference=False and None fields.
    """

    def __init__(self, data_dir: str | Path, cache_size: int = 2,
                 reference_prob: float = 0.8):
        reference_prob = float(reference_prob)
        if not math.isfinite(reference_prob) or not 0 <= reference_prob <= 1:
            raise ValueError('reference_prob must be finite and between 0 and 1')
        self.reference_prob = reference_prob
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.is_dir():
            raise NotADirectoryError(self.data_dir)
        cache_size = integer_index(cache_size)
        if cache_size < 1:
            raise ValueError('cache_size must be at least 1')
        self.cache_size = cache_size
        root = self.data_dir
        self.shards = tuple(
            p for p in sorted(root.rglob('*.parquet'))
            if 'indexes' not in p.relative_to(root).parts[:-1]
        )
        if not self.shards:
            raise FileNotFoundError(f'No Parquet shards found in {root}')

        self._groups: list[tuple[int, int]] = []
        self._ends: list[int] = []
        total = 0
        for shard_id, path in enumerate(self.shards):
            with pq.ParquetFile(path) as parquet:
                missing = _REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f'{path}: missing columns {sorted(missing)}')
                for group_id in range(parquet.num_row_groups):
                    rows = parquet.metadata.row_group(group_id).num_rows
                    if rows:
                        total += rows
                        self._groups.append((shard_id, group_id))
                        self._ends.append(total)
        if total == 0:
            raise ValueError(f'No samples found in {root}')
        self._cache: OrderedDict[int, pa.Table] = OrderedDict()
        self._pid = os.getpid()
        self._reference_info: list[Any] = []
        self._speaker_samples: dict[tuple, list[int]] = {}
        if self.reference_prob > 0:
            self._build_speaker_index()

    @staticmethod
    def _valid_speaker(value: Any) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        if value.lower() in {'', 'none', 'null', 'unknown', 'nan', 'n/a', 'unk'}:
            return None
        return value

    def _build_speaker_index(self) -> None:
        # Read only pairing metadata, never codec arrays. Pools stay within a
        # release so identically named speakers in unrelated corpora cannot mix.
        for path in self.shards:
            relative = path.relative_to(self.data_dir)
            release = self.data_dir / relative.parts[0] if len(relative.parts) > 1 else self.data_dir
            with pq.ParquetFile(path) as parquet:
                columns = [c for c in ('speaker_id', 'dataset', 'sample_id', 'num_codebooks')
                           if c in parquet.schema_arrow.names]
                for batch in parquet.iter_batches(columns=columns):
                    for row in batch.to_pylist():
                        speaker = self._valid_speaker(row.get('speaker_id'))
                        key = (str(release), row.get('dataset'), speaker, row['num_codebooks'])
                        index = len(self._reference_info)
                        self._reference_info.append((key, row['sample_id']))
                        if speaker is not None:
                            self._speaker_samples.setdefault(key, []).append(index)

    def _with_reference(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        row.update(has_reference=False, reference_codes=None,
                   reference_text=None, reference_sample_id=None)
        # Validate identity and eligibility before consuming a probability draw.
        if self._valid_speaker(row.get('speaker_id')) is None or not self._reference_info:
            return row
        key, sample_id = self._reference_info[index]
        candidates = [i for i in self._speaker_samples.get(key, ())
                      if self._reference_info[i][1] != sample_id]
        if not candidates or random.random() >= self.reference_prob:
            return row
        group, offset = self._locate(random.choice(candidates))
        reference = self._decode(self._read_group(group).slice(offset, 1).to_pylist()[0])
        row.update(has_reference=True, reference_codes=reference['codes'],
                   reference_text=reference['text'], reference_sample_id=reference['sample_id'])
        return row

    def __len__(self) -> int:
        return self._ends[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['_cache'] = OrderedDict()
        state['_pid'] = None
        return state

    def _locate(self, index: int) -> tuple[int, int]:
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group = bisect_right(self._ends, index)
        start = self._ends[group - 1] if group else 0
        return group, index - start

    def _read_group(self, group: int) -> pa.Table:
        # Forked workers must not inherit a populated parent cache.
        if self._pid != os.getpid():
            self._cache = OrderedDict()
            self._pid = os.getpid()
        if group in self._cache:
            self._cache.move_to_end(group)
            return self._cache[group]
        shard_id, group_id = self._groups[group]
        with pq.ParquetFile(self.shards[shard_id]) as parquet:
            table = parquet.read_row_group(group_id, use_threads=False)
        self._cache[group] = table
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return table

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        try:
            codes = torch.tensor(row['codes'], dtype=torch.long)
            q, t = row['num_codebooks'], row['num_frames']
            if q <= 0 or t <= 0 or tuple(codes.shape) != (q, t):
                raise ValueError(f'expected codes shape {(q, t)}, got {tuple(codes.shape)}')
            if not isinstance(row['text'], str):
                raise ValueError('text must be a string')
            if torch.any(codes < 0):
                raise ValueError('codes must be non-negative')
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid sample {row.get('sample_id')!r}: {exc}") from exc
        row['codes'] = codes
        return row

    def __getitem__(self, index: int) -> dict[str, Any]:
        group, offset = self._locate(index)
        row = self._read_group(group).slice(offset, 1).to_pylist()[0]
        start = self._ends[group - 1] if group else 0
        return self._with_reference(self._decode(row), start + offset)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        # DataLoader can request a random batch spanning many shards. Read each
        # touched group once, even when the batch exceeds the cache capacity.
        grouped: dict[int, list[tuple[int, int]]] = {}
        for position, index in enumerate(indices):
            group, offset = self._locate(index)
            grouped.setdefault(group, []).append((position, offset))
        result: list[Any] = [None] * len(indices)
        for group, positions in grouped.items():
            table = self._read_group(group)
            rows = table.take(pa.array([offset for _, offset in positions])).to_pylist()
            for (position, _), row in zip(positions, rows):
                result[position] = self._decode(row)
        return [self._with_reference(row, integer_index(index) % len(self))
                for row, index in zip(result, indices)]


def tts_collate_fn(
    samples: Sequence[dict[str, Any]], codes_pad_value: int = -100,
) -> dict[str, Any]:
    """Pad only the time axis; ``codes_mask=True`` denotes a real frame.

    Returns codes [B,Q,Tmax], codes_lengths [B], codes_mask [B,Tmax],
    text/sample_id/speaker_id lists, and original non-code fields in metadata.
    Reference codes are padded separately to [B,Q,Rmax], with reference_lengths,
    reference_mask, reference_text, reference_sample_id and has_reference [B].
    No-reference items have length 0; an all-unconditioned batch has Rmax=0.
    Models must skip the reference branch for missing conditions rather than
    applying attention to all-padding reference rows.
    The default -100 padding can serve as cross-entropy ignore_index; replace
    masked values before embedding lookup, or select a model-specific pad ID
    via functools.partial(tts_collate_fn, codes_pad_value=...). No BOS/EOS,
    text tokenization, or autoregressive shifting is imposed here.
    """
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    tensors = [sample['codes'] for sample in samples]
    if any(not isinstance(t, torch.Tensor) or t.ndim != 2 for t in tensors):
        raise ValueError('Each codes tensor must have shape [Q, T]')
    q = tensors[0].shape[0]
    if q <= 0 or any(t.shape[0] != q or t.shape[1] <= 0 for t in tensors):
        raise ValueError('Samples must have the same positive codebook count and nonempty time axis')
    lengths = torch.tensor([t.shape[1] for t in tensors], dtype=torch.long)
    max_length = int(lengths.max())
    codes = torch.full((len(samples), q, max_length), codes_pad_value, dtype=torch.long)
    for i, tensor in enumerate(tensors):
        codes[i, :, :tensor.shape[1]] = tensor
    mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
    references = [sample.get('reference_codes') for sample in samples]
    for sample, reference in zip(samples, references):
        if bool(sample.get('has_reference', False)) != (reference is not None):
            raise ValueError('has_reference must agree with reference_codes')
        if reference is not None and (
            not isinstance(reference, torch.Tensor) or reference.ndim != 2
            or reference.shape[0] != q or reference.shape[1] <= 0
        ):
            raise ValueError('Reference codes must have shape [Q, positive T] matching target Q')
    reference_lengths = torch.tensor(
        [r.shape[1] if r is not None else 0 for r in references], dtype=torch.long)
    reference_max = int(reference_lengths.max())
    reference_codes = torch.full((len(samples), q, reference_max), codes_pad_value, dtype=torch.long)
    for i, reference in enumerate(references):
        if reference is not None:
            reference_codes[i, :, :reference.shape[1]] = reference
    reference_mask = torch.arange(reference_max).unsqueeze(0) < reference_lengths.unsqueeze(1)
    return {
        'codes': codes,
        'codes_lengths': lengths,
        'codes_mask': mask,
        'text': [sample['text'] for sample in samples],
        'sample_id': [sample['sample_id'] for sample in samples],
        'speaker_id': [sample.get('speaker_id') for sample in samples],
        'has_reference': reference_lengths > 0,
        'reference_codes': reference_codes,
        'reference_lengths': reference_lengths,
        'reference_mask': reference_mask,
        'reference_text': [sample.get('reference_text') for sample in samples],
        'reference_sample_id': [sample.get('reference_sample_id') for sample in samples],
        'metadata': [{k: v for k, v in sample.items()
                      if k != 'codes' and k != 'has_reference' and not k.startswith('reference_')}
                     for sample in samples],
    }
