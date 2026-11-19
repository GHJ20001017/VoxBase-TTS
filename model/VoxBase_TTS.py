"""Audio embedding/packing and masked-codec reconstruction on BidirectionalQwen3.

Each frame is one Transformer position. Per-codebook embeddings are summed
with sqrt(Q) normalization. A shared RMSNorm/Linear/SiLU feature network
feeds the independent 2048-way classifiers for each codebook.
MASK is local to codec embeddings, never added to the text tokenizer.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from model.bidirectional_qwen3 import BidirectionalQwen3


def mask_codes(codes, lengths, codebook_size, mask_min=0.05, mask_max=1.0):
    """Sample an exact random mask count per item, excluding time padding.

    Noise level t is the sampled mask fraction (linear noise schedule).
    Sampling is without replacement; every item gets at least one masked
    position. Padding -100 is retained and never used for embedding lookup.
    """
    if not 0 <= mask_min <= mask_max <= 1:
        raise ValueError('Expected 0 <= mask_min <= mask_max <= 1')
    if codes.ndim != 3 or lengths.shape != (codes.shape[0],):
        raise ValueError('Expected codes [B,Q,T] and lengths [B]')
    b, q, t = codes.shape
    if b == 0 or q == 0 or not torch.all((lengths > 0) & (lengths <= t)):
        raise ValueError('Each target must have positive length within padded time')
    valid = torch.arange(t, device=codes.device)[None, None, :] < lengths[:, None, None]
    valid = valid.expand_as(codes)
    if not torch.all((codes[valid] >= 0) & (codes[valid] < codebook_size)):
        raise ValueError('Target codes outside codec vocabulary')
    rates = torch.empty(b, device=codes.device).uniform_(mask_min, mask_max)
    selected = torch.zeros_like(codes, dtype=torch.bool)
    for i in range(b):
        n = q * int(lengths[i])
        count = max(1, min(n, round(float(rates[i]) * n)))
        indices = torch.randperm(n, device=codes.device)[:count]
        selected[i, indices // int(lengths[i]), indices % int(lengths[i])] = True
        rates[i] = count / n
    noisy = codes.clone()
    noisy[selected] = codebook_size
    labels = codes.masked_fill(~selected, -100)
    return noisy, labels, selected, rates


class VoxBaseTTS(nn.Module):
    """Trainable audio adapter, timestep MLP and Q prediction heads.

    Input order: optional ref text/audio, then target text/audio, each enclosed
    in the registered boundary tokens. Reference audio is always clean.
    No reference segments are emitted for missing references. Complete text
    and audio are retained; length limits raise rather than silently crop
    audio while retaining an unmatched transcript.
    """

    def __init__(self, backbone: BidirectionalQwen3, num_codebooks=8, codebook_size=2048):
        super().__init__()
        if num_codebooks <= 0 or codebook_size <= 0:
            raise ValueError('Codec dimensions must be positive')
        if not getattr(backbone.config, 'tts_token_ids', None):
            raise ValueError('Register TTS tokenizer tokens before building VoxBaseTTS')
        self.backbone = backbone
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_mask_id = codebook_size
        h = backbone.config.hidden_size
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, h) for _ in range(num_codebooks)])
        self.time_embedding = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h))
        self.shared_head = nn.Sequential(
            nn.RMSNorm(h, eps=backbone.config.rms_norm_eps),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(h, codebook_size) for _ in range(num_codebooks)])

    def _audio_embed(self, codes):
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks:
            raise ValueError(f'Expected audio shape [{self.num_codebooks}, T]')
        if not torch.all((codes >= 0) & (codes <= self.audio_mask_id)):
            raise ValueError('Audio embedding index outside codec/MASK vocabulary')
        return sum(layer(codes[q]) for q, layer in enumerate(self.audio_embeddings)) / math.sqrt(self.num_codebooks)

    def _target_features(self, batch, tokenizer, codes, rates,
                         max_text_tokens, max_sequence_length):
        embed = self.backbone.get_input_embeddings()
        device = embed.weight.device
        if codes.device != device:
            raise ValueError('Move batch tensors to the model device before forward')
        ids = self.backbone.config.tts_token_ids

        def boundary(name):
            return embed(torch.tensor([ids[name]], device=device))

        def text_embed(text):
            # Explicit length validation: do not truncate aligned transcripts.
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_text_tokens:
                raise ValueError(f'Text has {len(tokens)} tokens, limit is {max_text_tokens}')
            return embed(torch.tensor(tokens, dtype=torch.long, device=device))

        sequences, offsets, lengths = [], [], []
        for i in range(codes.shape[0]):
            parts = []
            if bool(batch['has_reference'][i]):
                ref_length = int(batch['reference_lengths'][i])
                if (not 0 < ref_length <= batch['reference_codes'].shape[-1]
                        or batch['reference_text'][i] is None):
                    raise ValueError('A reference requires in-bounds positive audio length and aligned text')
                reference = batch['reference_codes'][i, :, :ref_length]
                if not torch.all((reference >= 0) & (reference < self.codebook_size)):
                    raise ValueError('Reference must contain clean codec indices')
                parts.extend([boundary('REF_TEXT_START'), text_embed(batch['reference_text'][i]),
                              boundary('REF_TEXT_END'), boundary('REF_AUDIO_START'),
                              self._audio_embed(reference), boundary('REF_AUDIO_END')])
            parts.extend([boundary('TEXT_START'), text_embed(batch['text'][i]),
                          boundary('TEXT_END'), boundary('AUDIO_START')])
            offsets.append(sum(part.shape[0] for part in parts))
            length = int(batch['codes_lengths'][i])
            audio = self._audio_embed(codes[i, :, :length])
            time = self.time_embedding(rates[i:i + 1, None].to(self.time_embedding[0].weight.dtype))
            parts.extend([audio + time, boundary('AUDIO_END')])
            sequence = torch.cat(parts)
            if sequence.shape[0] > min(max_sequence_length, self.backbone.config.max_position_embeddings):
                raise ValueError(f"Sample {batch['sample_id'][i]} packed length {sequence.shape[0]} exceeds limit")
            sequences.append(sequence)
            lengths.append(sequence.shape[0])
        max_length = max(lengths)
        # Packed sequence padding uses Qwen AUDIO_PAD; codec MASK stays local.
        pad = boundary('AUDIO_PAD')
        packed = torch.stack([torch.cat([s, pad.expand(max_length - s.shape[0], -1)]) for s in sequences])
        valid = torch.arange(max_length, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
        hidden = self.backbone(inputs_embeds=packed, attention_mask=valid).last_hidden_state
        # Transform target frames once, sharing features across all codebooks.
        target_lengths = batch['codes_lengths'].tolist()
        target_hidden = torch.cat([
            hidden[i, offset:offset + length]
            for i, (offset, length) in enumerate(zip(offsets, target_lengths))
        ])
        return self.shared_head(target_hidden).split(target_lengths)

    @torch.inference_mode()
    def generate(self, tokenizer, text, num_frames, *, reference_codes=None,
                 reference_text=None, confidence_threshold=0.8,
                 max_inference_steps=16, decoding_schedule='threshold',
                 max_text_tokens=512, max_sequence_length=40960):
        """Decode from full MASK with threshold, linear, or cosine filling.

        ``threshold`` preserves the legacy confidence-acceptance behavior.
        ``linear`` and ``cosine`` fill the highest-confidence unresolved
        positions to a per-codebook quota each iteration. Accepted tokens are
        never overwritten; confidence and per-codebook histories are returned
        for diagnostics. Scores are not calibrated correctness probabilities.
        """
        if decoding_schedule not in ('cosine', 'linear', 'threshold'):
            raise ValueError('decoding_schedule must be cosine, linear, or threshold')
        for name, value in [('num_frames', num_frames),
                            ('max_inference_steps', max_inference_steps),
                            ('max_text_tokens', max_text_tokens),
                            ('max_sequence_length', max_sequence_length)]:
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if (isinstance(confidence_threshold, bool)
                or not isinstance(confidence_threshold, (int, float))
                or not math.isfinite(confidence_threshold)
                or not 0 <= confidence_threshold <= 1):
            raise ValueError('confidence_threshold must be finite and in [0, 1]')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('text must be a nonempty string')
        if (reference_codes is None) != (reference_text is None):
            raise ValueError('Provide reference_codes and reference_text together')
        device = self.backbone.get_input_embeddings().weight.device
        has_reference = reference_codes is not None
        if has_reference:
            if not isinstance(reference_text, str) or not reference_text.strip():
                raise ValueError('reference_text must be a nonempty string')
            if (not isinstance(reference_codes, torch.Tensor)
                    or reference_codes.ndim != 2
                    or reference_codes.shape[0] != self.num_codebooks
                    or reference_codes.shape[1] == 0
                    or reference_codes.dtype not in (torch.uint8, torch.int8,
                                                     torch.int16, torch.int32, torch.int64)):
                raise ValueError('reference_codes must be an integer tensor [Q, positive R]')
            if not torch.all((reference_codes >= 0) & (reference_codes < self.codebook_size)):
                raise ValueError('Reference must contain clean codec indices')
            reference_codes = reference_codes.to(device=device, dtype=torch.long)
        text_length = len(tokenizer.encode(text, add_special_tokens=False))
        ref_text_length = (len(tokenizer.encode(reference_text, add_special_tokens=False))
                           if has_reference else 0)
        if text_length > max_text_tokens or ref_text_length > max_text_tokens:
            raise ValueError('Text exceeds max_text_tokens')
        packed_length = text_length + num_frames + 4
        if has_reference:
            packed_length += ref_text_length + reference_codes.shape[1] + 4
        if packed_length > min(max_sequence_length, self.backbone.config.max_position_embeddings):
            raise ValueError('Packed generation length exceeds limit')
        codes = torch.full((1, self.num_codebooks, num_frames), self.audio_mask_id,
                           dtype=torch.long, device=device)
        batch = dict(codes_lengths=torch.tensor([num_frames], device=device),
                     has_reference=[has_reference], reference_text=[reference_text],
                     reference_lengths=[reference_codes.shape[1] if has_reference else 0],
                     reference_codes=reference_codes.unsqueeze(0) if has_reference else None,
                     text=[text], sample_id=['generation'])
        confidence = torch.zeros_like(codes[0], dtype=torch.float32)
        forced_mask = torch.zeros_like(codes[0], dtype=torch.bool)
        history, per_codebook_history, schedule_steps = [], [], []
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            for iteration in range(max_inference_steps):
                selected = codes[0] == self.audio_mask_id
                if decoding_schedule != 'threshold':
                    progress = (iteration + 1) / max_inference_steps
                    fraction = (1 - math.cos(math.pi * 0.5 * progress)
                                if decoding_schedule == 'cosine' else progress)
                    desired = (num_frames if iteration == max_inference_steps - 1
                               else min(num_frames, math.ceil(num_frames * fraction)))
                    quotas = (desired - (~selected).sum(dim=1)).clamp_min(0)
                    if not quotas.any():
                        continue
                rates = selected.float().mean().reshape(1)
                features = self._target_features(
                    batch, tokenizer, codes, rates, max_text_tokens, max_sequence_length)[0]
                if not torch.isfinite(features).all():
                    raise FloatingPointError('Nonfinite generation features')
                for q, head in enumerate(self.heads):
                    mask = selected[q]
                    if not mask.any():
                        continue
                    with torch.autocast(device_type=device.type, enabled=False):
                        logits = F.linear(features[mask].float(), head.weight.float(), head.bias.float())
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError('Nonfinite generation logits')
                        scores, predictions = logits.softmax(-1).max(-1)
                    confidence[q, mask] = scores
                    if decoding_schedule == 'threshold':
                        accepted = scores >= confidence_threshold
                        if iteration == max_inference_steps - 1:
                            forced_mask[q, mask] = ~accepted
                            accepted = torch.ones_like(accepted)
                    else:
                        count = min(int(quotas[q]), scores.numel())
                        accepted = torch.zeros_like(scores, dtype=torch.bool)
                        order = scores.argsort(descending=True, stable=True)
                        accepted[order[:count]] = True
                    codes[0, q, mask] = torch.where(
                        accepted, predictions, self.audio_mask_id)
                remaining = int((codes == self.audio_mask_id).sum())
                history.append(remaining)
                per_codebook_history.append((codes[0] == self.audio_mask_id).sum(1).tolist())
                schedule_steps.append(iteration + 1)
                if remaining == 0:
                    break
        finally:
            for module, training in modes:
                module.training = training
        return dict(codes=codes[0], confidence=confidence,
                    below_threshold=confidence < confidence_threshold,
                    forced_mask=forced_mask, num_steps=len(history),
                    remaining_history=history,
                    remaining_per_codebook_history=per_codebook_history,
                    schedule_steps=schedule_steps, decoding_schedule=decoding_schedule,
                    max_inference_steps=max_inference_steps,
                    confidence_threshold=confidence_threshold)

    def forward(self, batch, tokenizer, mask_min=0.05, mask_max=1.0,
                max_text_tokens=512, max_sequence_length=40960):
        codes = batch['codes']
        if codes.shape[1] != self.num_codebooks:
            raise ValueError(f"Configured {self.num_codebooks} codebooks, batch has {codes.shape[1]}")
        if max_text_tokens <= 0 or max_sequence_length <= 0:
            raise ValueError('Sequence/text limits must be positive')
        noisy, labels, selected, rates = mask_codes(
            codes, batch['codes_lengths'], self.codebook_size, mask_min, mask_max)
        shared_states = self._target_features(
            batch, tokenizer, noisy, rates, max_text_tokens, max_sequence_length)
        target_lengths = batch['codes_lengths'].tolist()
        device = codes.device
        loss_sum = codes.new_zeros((), dtype=torch.float32)
        num_correct = torch.zeros((), device=device, dtype=torch.long)
        # Project only selected positions, avoiding a full [B,Q,T,2048] tensor.
        for q, head in enumerate(self.heads):
            states, targets = [], []
            for i, features in enumerate(shared_states):
                length = target_lengths[i]
                mask = selected[i, q, :length]
                states.append(features[mask])
                targets.append(labels[i, q, :length][mask])
            target = torch.cat(targets)
            if target.numel():
                # Project in FP32 as well as reducing CE in FP32: casting an
                # already-overflowed FP16 projection cannot recover its logits.
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = F.linear(torch.cat(states).float(), head.weight.float(), head.bias.float())
                    loss_sum = loss_sum + F.cross_entropy(logits, target, reduction='sum')
                num_correct = num_correct + (logits.argmax(-1) == target).sum()
        count = int(selected.sum())
        return {'loss': loss_sum / count, 'loss_sum': loss_sum,
                'num_masked': count, 'num_correct': num_correct}
