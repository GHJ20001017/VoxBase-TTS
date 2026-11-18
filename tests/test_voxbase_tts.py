"""Offline CPU coverage of codec masking and tiny Qwen masked TTS training."""

import copy
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config

from data.tts_data import tts_collate_fn
from model.bidirectional_qwen3 import BidirectionalQwen3
from model.VoxBase_TTS import VoxBaseTTS, mask_codes


def make_batch(with_reference=True):
    samples = [
        dict(sample_id='long', text='hello world',
             codes=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [1, 3, 5, 7]])),
        dict(sample_id='short', text='hello',
             codes=torch.tensor([[7, 6], [5, 4], [3, 2]])),
    ]
    if with_reference:
        samples[0].update(has_reference=True, reference_text='reference voice',
                          reference_sample_id='ref',
                          reference_codes=torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]]))
    return tts_collate_fn(samples)


def make_model():
    raw = Tokenizer(WordLevel(
        {'[UNK]': 0, '[PAD]': 1, 'hello': 2, 'world': 3,
         'reference': 4, 'voice': 5, 'changed': 6}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]')
    config = Qwen3Config(
        vocab_size=16, hidden_size=24, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=3, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    backbone = BidirectionalQwen3(config)
    backbone.register_tts_tokens(tokenizer)
    return VoxBaseTTS(backbone, num_codebooks=3, codebook_size=8).cpu(), tokenizer


class MaskCodesTest(unittest.TestCase):
    def setUp(self):
        self.batch = make_batch()

    def test_exact_counts_padding_labels_and_input_unchanged(self):
        codes, lengths = self.batch['codes'], self.batch['codes_lengths']
        original = codes.clone()
        torch.manual_seed(19)
        noisy, labels, selected, rates = mask_codes(codes, lengths, 8, 0.2, 0.8)
        # Replay the sampled rates independently of mask_codes' returned rates.
        torch.manual_seed(19)
        sampled = torch.empty(2).uniform_(0.2, 0.8)
        valid = self.batch['codes_mask'][:, None, :].expand_as(codes)
        for i, length in enumerate(lengths.tolist()):
            total = 3 * length
            expected = max(1, min(total, round(float(sampled[i]) * total)))
            self.assertEqual(int(selected[i].sum()), expected)
            self.assertAlmostEqual(float(rates[i]), expected / total)
        self.assertEqual(selected.dtype, torch.bool)
        self.assertFalse(selected[~valid].any())
        torch.testing.assert_close(noisy[~valid], original[~valid])
        self.assertTrue((labels[~valid] == -100).all())
        self.assertTrue((noisy[selected] == 8).all())
        torch.testing.assert_close(labels[selected], original[selected])
        self.assertTrue((labels[~selected] == -100).all())
        torch.testing.assert_close(noisy[~selected], original[~selected])
        torch.testing.assert_close(codes, original)

    def test_zero_and_one_boundaries(self):
        for rate in (0.0, 1.0):
            with self.subTest(rate=rate):
                _, _, selected, rates = mask_codes(
                    self.batch['codes'], self.batch['codes_lengths'], 8, rate, rate)
                expected = torch.tensor([1, 1] if rate == 0 else [12, 6])
                torch.testing.assert_close(selected.sum((1, 2)), expected)
                torch.testing.assert_close(rates, expected.float() / torch.tensor([12, 6]))
        noisy, labels, selected, rates = mask_codes(
            torch.tensor([[[0]]]), torch.tensor([1]), 1, 0, 0)
        self.assertEqual(noisy.item(), 1)
        self.assertEqual(labels.item(), 0)
        self.assertTrue(selected.item())
        self.assertEqual(rates.item(), 1)

    def test_seeded_repeatability(self):
        torch.manual_seed(101)
        first = mask_codes(self.batch['codes'], self.batch['codes_lengths'], 8)
        torch.manual_seed(101)
        second = mask_codes(self.batch['codes'], self.batch['codes_lengths'], 8)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_ranges_shapes_lengths_and_target_indices(self):
        codes, lengths = self.batch['codes'], self.batch['codes_lengths']
        for lower, upper in [(-0.1, 1), (0, 1.1), (0.8, 0.2), (float('nan'), 1)]:
            with self.subTest(bounds=(lower, upper)), self.assertRaises(ValueError):
                mask_codes(codes, lengths, 8, lower, upper)
        invalid = [
            (codes[0], lengths), (codes, lengths[:1]),
            (codes, torch.tensor([0, 2])), (codes, torch.tensor([-1, 2])),
            (codes, torch.tensor([5, 2])),
            (torch.empty(0, 3, 4, dtype=torch.long), torch.empty(0, dtype=torch.long)),
            (torch.empty(2, 0, 4, dtype=torch.long), lengths),
        ]
        for target, lens in invalid:
            with self.subTest(shape=target.shape, lengths=lens), self.assertRaises(ValueError):
                mask_codes(target, lens, 8)
        for value in (-100, -1, 8):
            bad = codes.clone()
            bad[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'vocabulary'):
                mask_codes(bad, lengths, 8)


class VoxBaseTTSTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model, self.tokenizer = make_model()

    def assert_nonzero_finite_gradient(self, parameter):
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_registered_tokens_and_mixed_and_no_reference_training(self):
        self.assertEqual(len(self.model.backbone.config.tts_token_ids), 9)
        self.assertNotIn('<MASK>', self.tokenizer.get_vocab())
        self.assertEqual(self.model.audio_mask_id, 8)
        for with_reference in (True, False):
            with self.subTest(with_reference=with_reference):
                batch = make_batch(with_reference)
                self.assertEqual(batch['has_reference'].tolist(), [with_reference, False])
                self.assertEqual(batch['reference_lengths'].tolist(), [3 if with_reference else 0, 0])
                self.assertEqual(tuple(batch['reference_codes'].shape),
                                 (2, 3, 3 if with_reference else 0))
                self.model.zero_grad(set_to_none=True)
                # Full masking guarantees every codebook head is exercised.
                result = self.model(batch, self.tokenizer, mask_min=1, mask_max=1)
                self.assertTrue(torch.isfinite(result['loss']))
                self.assertTrue(torch.isfinite(result['loss_sum']))
                self.assertEqual(result['num_masked'], 18)
                self.assertGreaterEqual(int(result['num_correct']), 0)
                self.assertLessEqual(int(result['num_correct']), 18)
                torch.testing.assert_close(result['loss_sum'], result['loss'] * 18)
                result['loss'].backward()
                for parameter in self.model.shared_head.parameters():
                    self.assert_nonzero_finite_gradient(parameter)
                for head in self.model.heads:
                    for parameter in head.parameters():
                        self.assert_nonzero_finite_gradient(parameter)
                for embedding in self.model.audio_embeddings:
                    self.assert_nonzero_finite_gradient(embedding.weight)
                    self.assertGreater(embedding.weight.grad[8].abs().sum().item(), 0)
                for parameter in self.model.time_embedding.parameters():
                    self.assert_nonzero_finite_gradient(parameter)
                self.assert_nonzero_finite_gradient(self.model.backbone.get_input_embeddings().weight)

    def test_shared_features_computed_once_and_private_classifiers(self):
        observed = []
        hook = self.model.shared_head.register_forward_hook(
            lambda module, inputs, output: observed.append((inputs[0].shape, output)))
        try:
            batch = make_batch()
            result = self.model(batch, self.tokenizer, mask_min=1, mask_max=1)
        finally:
            hook.remove()
        self.assertEqual(len(observed), 1)
        self.assertEqual(tuple(observed[0][0]), (6, 24))  # Only real target frames.
        features = observed[0][1]
        features.retain_grad()
        result['loss'].backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertEqual(len({head.weight.data_ptr() for head in self.model.heads}), 3)
        self.assertEqual([tuple(head(features).shape) for head in self.model.heads], [(6, 8)] * 3)
        state = copy.deepcopy(self.model.state_dict())
        restored, tokenizer = make_model()
        restored.load_state_dict(state)
        torch.manual_seed(42)
        expected = self.model(batch, self.tokenizer)['loss']
        torch.manual_seed(42)
        actual = restored(batch, tokenizer)['loss']
        torch.testing.assert_close(actual, expected)

    def test_seeded_forward_repeatability_and_reference_conditioning(self):
        self.model.eval()
        batch = make_batch()

        def run(candidate):
            # Keep masking identical, including randperm and sampled timestep.
            torch.manual_seed(123)
            with torch.no_grad(), patch('model.VoxBase_TTS.mask_codes', wraps=mask_codes) as masking:
                result = self.model(candidate, self.tokenizer, mask_min=0.5, mask_max=0.5)
            self.assertEqual(masking.call_count, 1)
            return result

        first, repeated = run(batch), run(batch)
        for key in first:
            if isinstance(first[key], torch.Tensor):
                torch.testing.assert_close(first[key], repeated[key], rtol=0, atol=0)
            else:
                self.assertEqual(first[key], repeated[key])
        for field in ('reference_codes', 'reference_text'):
            changed = copy.deepcopy(batch)
            if field == 'reference_codes':
                changed[field][0] = (changed[field][0] + 3) % 8
            else:
                changed[field][0] = 'changed voice'
            other = run(changed)
            self.assertEqual(first['num_masked'], other['num_masked'])
            self.assertGreater(abs(first['loss'].item() - other['loss'].item()), 1e-7, field)

    def test_fp16_autocast_prediction_head_is_finite(self):
        with torch.no_grad():
            for head in self.model.heads:
                head.bias.fill_(100000.0)  # Finite FP32, outside FP16 range.
        with torch.autocast('cpu', dtype=torch.float16):
            result = self.model(make_batch(False), self.tokenizer, mask_min=1, mask_max=1)
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertTrue(torch.isfinite(self.model.heads[0].weight.grad).all())

    def test_mismatched_codebook_count(self):
        batch = make_batch()
        batch['codes'] = batch['codes'][:, :2]
        with self.assertRaisesRegex(ValueError, 'Configured 3 codebooks, batch has 2'):
            self.model(batch, self.tokenizer)

    def test_transcript_and_packed_sequence_limits(self):
        for with_reference, field in [(False, 'text'), (True, 'reference_text')]:
            batch = make_batch(with_reference)
            batch[field][0] = 'hello world hello'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Text has 3 tokens, limit is 2'):
                self.model(batch, self.tokenizer, max_text_tokens=2)
        # No-ref sample 0 packs 2 text + 4 boundary + 4 audio = 10 positions.
        batch = make_batch(False)
        self.model(batch, self.tokenizer, max_sequence_length=10)
        with self.assertRaisesRegex(ValueError, 'Sample long packed length 10 exceeds limit'):
            self.model(batch, self.tokenizer, max_sequence_length=9)
        self.model.backbone.config.max_position_embeddings = 9
        with self.assertRaisesRegex(ValueError, 'packed length 10 exceeds limit'):
            self.model(batch, self.tokenizer, max_sequence_length=64)
        for kwargs in ({'max_text_tokens': 0}, {'max_sequence_length': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'must be positive'):
                self.model(batch, self.tokenizer, **kwargs)

    def test_invalid_reference_is_rejected(self):
        for field, value, message in [
            ('reference_lengths', 0, 'positive audio length'),
            ('reference_lengths', 10, 'in-bounds'),
            ('reference_text', None, 'aligned text'),
            ('reference_codes', 8, 'clean codec indices'),
        ]:
            batch = make_batch()
            batch[field][0] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.model(batch, self.tokenizer)


if __name__ == '__main__':
    unittest.main()
