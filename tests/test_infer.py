"""Entirely offline tiny-checkpoint inference and CLI contract tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config

import infer
from model.bidirectional_qwen3 import BidirectionalQwen3
from model.VoxBase_TTS import VoxBaseTTS


class InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        raw = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, 'hello': 2, 'voice': 3}, unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]')
        config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8,
                             max_position_embeddings=64, attention_dropout=0.0)
        backbone = BidirectionalQwen3(config)
        backbone.register_tts_tokens(tokenizer)
        tokenizer.save_pretrained(self.root / 'tokenizer')
        backbone.config.to_json_file(str(self.root / 'backbone_config.json'))
        model = VoxBaseTTS(backbone, num_codebooks=2, codebook_size=8)
        # Uniform logits guarantee threshold=1 can never be met naturally.
        with torch.no_grad():
            for head in model.heads:
                head.weight.zero_()
                head.bias.zero_()
        self.checkpoint = self.root / 'last.pt'
        torch.save({'model': model.state_dict(), 'config': {
            'model': {'num_codebooks': 2, 'codebook_size': 8, 'attention_backend': 'sdpa',
                      'pretrained_name_or_path': 'DO-NOT-LOAD-REMOTE'},
            'masking': {'max_text_tokens': 8, 'max_sequence_length': 64},
        }}, self.checkpoint)
        self.options = dict(checkpoint=self.checkpoint, text='hello voice', num_frames=3,
                            output=self.root / 'result.pt')

    def test_cli_offline_result_forced_and_reload(self):
        from transformers import AutoTokenizer
        loader = AutoTokenizer.from_pretrained
        with patch.object(AutoTokenizer, 'from_pretrained', wraps=loader) as load, \
                patch.object(BidirectionalQwen3, 'from_pretrained', side_effect=AssertionError('network load')):
            result = infer.main(['--checkpoint', str(self.checkpoint), '--text', 'hello voice',
                                 '--num-frames', '3', '--output', str(self.options['output']),
                                 '--confidence-threshold', '1', '--max-inference-steps', '2'])
        self.assertTrue(load.call_args.kwargs['local_files_only'])
        self.assertFalse(load.call_args.kwargs['trust_remote_code'])
        saved = torch.load(self.options['output'], weights_only=True)
        self.assertEqual(saved['text'], 'hello voice')
        self.assertEqual(saved['num_frames'], 3)
        for key in ('codes', 'confidence', 'below_threshold', 'forced_mask'):
            self.assertEqual(saved[key].shape, (2, 3))
            self.assertEqual(saved[key].device.type, 'cpu')
            torch.testing.assert_close(saved[key], result[key])
        self.assertEqual(saved['codes'].dtype, torch.long)
        self.assertTrue(((saved['codes'] >= 0) & (saved['codes'] < 8)).all())
        self.assertTrue(torch.isfinite(saved['confidence']).all())
        self.assertEqual(saved['below_threshold'].dtype, torch.bool)
        self.assertEqual(saved['forced_mask'].dtype, torch.bool)
        self.assertTrue(saved['below_threshold'].all())
        self.assertTrue(saved['forced_mask'].all())
        self.assertEqual(saved['num_steps'], 2)
        self.assertEqual(saved['remaining_history'][-1], 0)
        before = self.options['output'].read_bytes()
        with patch('infer.torch.load', side_effect=AssertionError('loaded before overwrite check')):
            with self.assertRaises(FileExistsError):
                infer.run(**self.options)
        self.assertEqual(before, self.options['output'].read_bytes())

    def test_reference_tensor_and_dict_weights_only(self):
        for payload in (torch.tensor([[0, 1], [2, 3]]), {'codes': torch.tensor([[0, 1], [2, 3]])}):
            with self.subTest(payload_type=type(payload).__name__):
                ref = self.root / 'reference.pt'
                torch.save(payload, ref)
                options = dict(self.options, output=self.root / f'ref-{type(payload).__name__}.pt',
                               reference_codes=ref, reference_text='voice')
                load = torch.load
                with patch('infer.torch.load', wraps=load) as loads:
                    result = infer.run(**options)
                self.assertEqual(loads.call_args_list[0].kwargs['weights_only'], True)
                self.assertEqual(loads.call_args_list[1].kwargs['weights_only'], False)
                self.assertEqual(result['codes'].shape, (2, 3))

    def test_invalid_arguments_fail_before_checkpoint_load(self):
        cases = [dict(num_frames=0), dict(num_frames=-1), dict(text=' '),
                 dict(max_inference_steps=0), dict(confidence_threshold=float('nan')),
                 dict(confidence_threshold=1.1), dict(confidence_threshold=-0.1),
                 dict(reference_text='voice'), dict(reference_codes=self.root / 'ref.pt'),
                 dict(precision='fp16'), dict(precision='bf16'), dict(device='mps'),
                 dict(output=self.root / 'result.wav'), dict(output=self.root / 'absent' / 'out.pt')]
        for update in cases:
            with self.subTest(update=update), \
                    patch('infer.torch.load', side_effect=AssertionError('heavy load on invalid args')), \
                    self.assertRaises(ValueError):
                infer.run(**dict(self.options, **update))
        with self.assertRaises(SystemExit) as error:
            infer.main(['--checkpoint', str(self.checkpoint)])
        self.assertEqual(error.exception.code, 2)

    def test_mapping_mismatch_rejected_without_remapping(self):
        path = self.root / 'backbone_config.json'
        config = json.loads(path.read_text())
        config['tts_token_ids']['AUDIO_PAD'] = 0
        path.write_text(json.dumps(config))
        with patch.object(BidirectionalQwen3, 'register_tts_tokens', side_effect=AssertionError('remapped')), \
                self.assertRaisesRegex(ValueError, 'mapping'):
            infer.run(**self.options)
        self.assertFalse(self.options['output'].exists())

    def test_strict_checkpoint_load(self):
        saved = torch.load(self.checkpoint, weights_only=False)
        saved['model'].pop(next(iter(saved['model'])))
        torch.save(saved, self.checkpoint)
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            infer.run(**self.options)
        self.assertFalse(self.options['output'].exists())

    def test_invalid_references_and_lengths_before_model_construction(self):
        ref = self.root / 'reference.pt'
        for payload in (torch.zeros(2, 2), torch.zeros(3, 2, dtype=torch.long),
                        torch.full((2, 2), 8), torch.full((2, 2), -1),
                        torch.zeros(2, 0, dtype=torch.long), {'other': torch.zeros(2, 2)}):
            with self.subTest(payload=str(payload)):
                torch.save(payload, ref)
                with patch.object(BidirectionalQwen3, '__init__', side_effect=AssertionError('constructed')), \
                        self.assertRaises(ValueError):
                    infer.run(**dict(self.options, reference_codes=ref, reference_text='voice'))
        for update in (dict(num_frames=64), dict(text=' '.join(['hello'] * 9))):
            with self.subTest(update=update), \
                    patch.object(BidirectionalQwen3, '__init__', side_effect=AssertionError('constructed')), \
                    self.assertRaises(ValueError):
                infer.run(**dict(self.options, **update))


if __name__ == '__main__':
    unittest.main()
