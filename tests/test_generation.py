"""Offline tests of iterative codebook decoding, without pretrained downloads."""
import unittest
from unittest.mock import patch

import torch

from test_voxbase_tts import make_model


class GenerationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model, self.tokenizer = make_model()

    def test_real_forward_early_exit_reference_and_forced_fill(self):
        for reference in (None, torch.ones(3, 2, dtype=torch.long)):
            with self.subTest(reference=reference is not None):
                kwargs = dict(reference_codes=reference,
                              reference_text='reference voice' if reference is not None else None)
                result = self.model.generate(self.tokenizer, 'hello', 3,
                                             confidence_threshold=0, **kwargs)
                self.assertEqual(result['num_steps'], 1)
                self.assertEqual(result['remaining_history'], [0])
                self.assertFalse(result['forced_mask'].any())
                self.assertTrue(self.model.training)
                result = self.model.generate(self.tokenizer, 'hello', 3,
                                             confidence_threshold=1, max_inference_steps=2, **kwargs)
                self.assertEqual(result['remaining_history'], [9, 0])
                self.assertTrue(result['forced_mask'].all())
                self.assertTrue(result['below_threshold'].all())
                self.assertEqual(tuple(result['codes'].shape), (3, 3))
                self.assertTrue(((result['codes'] >= 0) & (result['codes'] < 8)).all())
                self.assertFalse(result['confidence'].requires_grad)

    def test_actual_mask_rate_and_accepted_tokens_preserved(self):
        inputs, rates = [], []

        def features(batch, tokenizer, codes, rate, *limits):
            inputs.append(codes.clone())
            rates.append(rate.item())
            with torch.no_grad():
                for q, head in enumerate(self.model.heads):
                    head.weight.zero_()
                    head.bias.zero_()
                    if q == 0 or len(inputs) > 1:
                        head.bias[2 if len(inputs) == 1 else 4] = 20
            return (torch.zeros(2, 24),)

        with patch.object(self.model, '_target_features', side_effect=features):
            result = self.model.generate(self.tokenizer, 'hello', 2,
                                         confidence_threshold=0.9, max_inference_steps=3)
        self.assertEqual(len(inputs), 2)
        self.assertTrue((inputs[0] == 8).all())
        self.assertTrue((inputs[1][0, 0] == 2).all())
        self.assertTrue((inputs[1][0, 1:] == 8).all())
        self.assertAlmostEqual(rates[0], 1)
        self.assertAlmostEqual(rates[1], 2 / 3, places=6)
        self.assertEqual(result['remaining_history'], [4, 0])
        self.assertTrue((result['codes'][0] == 2).all())
        self.assertTrue((result['codes'][1:] == 4).all())
        self.assertFalse(result['forced_mask'].any())

    def test_validation_and_mode_restoration_on_failure(self):
        for kwargs in [dict(num_frames=0), dict(num_frames=True),
                       dict(max_inference_steps=0), dict(confidence_threshold=float('nan')),
                       dict(confidence_threshold=1.1), dict(text=''),
                       dict(reference_text='voice'),
                       dict(reference_codes=torch.zeros(2, 3, dtype=torch.long), reference_text='voice'),
                       dict(reference_codes=torch.full((3, 2), 8), reference_text='voice'),
                       dict(reference_codes=torch.zeros(3, 2), reference_text='voice'),
                       dict(num_frames=100), dict(max_text_tokens=0)]:
            args = dict(text='hello', num_frames=2)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model.generate(self.tokenizer, **args)
        self.model.train()
        self.model.backbone.eval()
        modes = [module.training for module in self.model.modules()]
        with patch.object(self.model, '_target_features', return_value=(torch.full((2, 24), float('nan')),)):
            with self.assertRaises(FloatingPointError):
                self.model.generate(self.tokenizer, 'hello', 2)
        self.assertEqual(modes, [module.training for module in self.model.modules()])


if __name__ == '__main__':
    unittest.main()
