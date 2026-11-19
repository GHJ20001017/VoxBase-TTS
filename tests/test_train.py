"""Offline end-to-end training, deterministic validation and checkpoint replay."""

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
import yaml

import train


class TrainingTest(unittest.TestCase):
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
        pretrained = self.root / 'pretrained'
        raw = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, 'hello': 2, 'voice': 3}, unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]')
        tokenizer.save_pretrained(pretrained)
        config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
                             attention_dropout=0.0)
        Qwen3ForCausalLM(config).save_pretrained(pretrained)
        for split in ['train', 'validation', 'test']:
            directory = self.root / split
            directory.mkdir()
            rows = [dict(sample_id=f'{split}-{i}', text='hello voice', dataset='tiny',
                         speaker_id='a', num_codebooks=2, num_frames=i + 1,
                         codes=[[(i + q) % 8] * (i + 1) for q in range(2)])
                    for i in range(3)]
            pq.write_table(pa.Table.from_pylist(rows), directory / 'part.parquet', row_group_size=2)
        self.config = {
            'model': {'pretrained_name_or_path': str(pretrained), 'num_codebooks': 2,
                      'codebook_size': 8, 'gradient_checkpointing': True},
            'data': {'train_dir': str(self.root / 'train'),
                     'validation_dir': str(self.root / 'validation'),
                     'test_dir': str(self.root / 'test'), 'num_workers': 0,
                     'reference_prob': 0.8},
            'training': {'output_dir': str(self.root / 'run'), 'device': 'cpu',
                         'precision': 'fp32', 'batch_size': 1, 'gradient_accumulation_steps': 2,
                         'epochs': 2, 'warmup_steps': 1, 'save_every': 1, 'log_every': 1},
            'evaluation': {'every_steps': 1, 'batch_size': 2, 'max_batches': 2},
            'masking': {'mask_min': 0.5, 'mask_max': 1, 'max_text_tokens': 8,
                        'max_sequence_length': 64},
        }

    def test_training_checkpoint_resume_and_explicit_test(self):
        c = deepcopy(self.config)
        # Each full epoch includes a partial final accumulation: 2 updates per epoch.
        result = train.run(c)
        self.assertEqual(result['global_step'], 4)
        output = Path(c['training']['output_dir'])
        for name in ['last.pt', 'best.pt', 'backbone_config.json', 'tokenizer']:
            self.assertTrue((output / name).exists())
        baseline = torch.load(output / 'last.pt', weights_only=False)
        self.assertEqual(baseline['global_step'], 4)
        self.assertEqual(baseline['epoch'], 1)
        self.assertEqual(baseline['batch_cursor'], 3)
        self.assertTrue(all(torch.isfinite(p).all() for p in baseline['model'].values()))
        self.assertTrue(np.isfinite(baseline['best_validation_loss']))
        self.assertFalse(list(output.glob('*.tmp')))
        with self.assertRaises(FileExistsError):
            train.run(c)

        save = train._atomic_save

        class Interrupted(Exception):
            pass

        # Step 1 interrupts mid-epoch; step 2 interrupts at the epoch boundary.
        for interrupt_step in (1, 2):
            with self.subTest(interrupt_step=interrupt_step):
                interrupted = deepcopy(c)
                interrupted['training']['output_dir'] = str(self.root / f'resumed-{interrupt_step}')

                def save_then_interrupt(payload, path):
                    save(payload, path)
                    if path.name == 'last.pt' and payload['global_step'] == interrupt_step:
                        raise Interrupted()

                with patch('train._atomic_save', side_effect=save_then_interrupt), \
                        self.assertRaises(Interrupted):
                    train.run(interrupted)
                resumed_path = Path(interrupted['training']['output_dir']) / 'last.pt'
                partial = torch.load(resumed_path, weights_only=False)
                self.assertEqual(partial['epoch'], 0)
                self.assertEqual(partial['batch_cursor'], 2 if interrupt_step == 1 else 3)
                result = train.run(interrupted, str(resumed_path))
                self.assertEqual(result['global_step'], 4)
                resumed = torch.load(resumed_path, weights_only=False)
                self.assertEqual(resumed['epoch'], 1)
                self.assertEqual(resumed['batch_cursor'], 3)
                for name in baseline['model']:
                    torch.testing.assert_close(baseline['model'][name], resumed['model'][name],
                                               rtol=0, atol=0)
                self.assertEqual(baseline['scheduler'], resumed['scheduler'])
                self.assertEqual(baseline['best_validation_loss'], resumed['best_validation_loss'])
                # Resuming a completed epoch budget must not perform another update.
                def unexpected_update(*args, **kwargs):
                    raise AssertionError('completed resume updated weights')

                with patch.object(torch.optim.AdamW, 'step', new=unexpected_update):
                    completed = train.run(interrupted, str(resumed_path))
                self.assertEqual(completed['global_step'], 4)
                self.assertEqual(completed['best_validation_loss'], baseline['best_validation_loss'])
        tested = train.run(interrupted, str(resumed_path), test_only=True)
        self.assertTrue(np.isfinite(tested['test']['loss']))
        self.assertGreater(tested['test']['num_masked'], 0)
        # Training does not need or read the test split.
        other = deepcopy(c)
        other['data']['test_dir'] = str(self.root / 'nonexistent_test')
        other['training']['output_dir'] = str(self.root / 'without_test')
        other['training']['epochs'] = 1
        self.assertEqual(train.run(other)['global_step'], 2)

    def test_amp_overflow_skips_update_without_advancing_scheduler(self):
        from model.VoxBase_TTS import VoxBaseTTS

        class FakeScaler:
            def __init__(self):
                self.overflow = False
                self.skipped = 0
                self.successful = 0
                self.current_scale = 65536.0

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                self.overflow = any(not torch.isfinite(p.grad).all()
                                    for group in optimizer.param_groups for p in group['params']
                                    if p.grad is not None)

            def is_enabled(self):
                return True

            def step(self, optimizer):
                if self.overflow:
                    self.skipped += 1
                else:
                    self.successful += 1
                    optimizer.step()

            def update(self):
                if self.overflow:
                    self.current_scale /= 2

            def get_scale(self):
                return self.current_scale

            def state_dict(self):
                return {}

        original = VoxBaseTTS.forward
        # Overflow the first full group or the final partial accumulation group.
        for overflow_call in (1, 3):
            with self.subTest(overflow_call=overflow_call):
                scaler = FakeScaler()
                c = deepcopy(self.config)
                c['training'].update(epochs=1, save_every=100,
                                     output_dir=str(self.root / f'overflow-{overflow_call}'))
                c['evaluation']['every_steps'] = 100
                calls = 0

                def overflow_once(model, *args, **kwargs):
                    nonlocal calls
                    result = original(model, *args, **kwargs)
                    if model.training:
                        calls += 1
                        if calls == overflow_call:
                            result['loss_sum'].register_hook(lambda gradient: gradient * float('inf'))
                    return result

                with patch('train.torch.amp.GradScaler', return_value=scaler), \
                        patch.object(VoxBaseTTS, 'forward', overflow_once), \
                        patch('train._evaluate', wraps=train._evaluate) as evaluate:
                    result = train.run(c)
                self.assertEqual(calls, 3)
                self.assertEqual(result['global_step'], 1)
                self.assertEqual(scaler.skipped, 1)
                self.assertEqual(scaler.successful, 1)
                self.assertEqual(scaler.get_scale(), 32768.0)
                # No periodic evaluation/save is due: finalization must run even
                # when the last group overflowed and no update followed it.
                evaluate.assert_called_once()
                checkpoint = torch.load(Path(c['training']['output_dir']) / 'last.pt',
                                        weights_only=False)
                self.assertEqual(checkpoint['global_step'], 1)
                self.assertEqual(checkpoint['epoch'], 0)
                self.assertEqual(checkpoint['batch_cursor'], 3)
                self.assertEqual(checkpoint['scheduler']['last_epoch'], 1)
                self.assertTrue(np.isfinite(checkpoint['best_validation_loss']))
                self.assertEqual(checkpoint['best_validation_loss'], result['best_validation_loss'])

    def test_validation_preserves_rng_and_repeats(self):
        from data.tts_data import TTSDataset
        from model.VoxBase_TTS import VoxBaseTTS
        from model.bidirectional_qwen3 import BidirectionalQwen3
        c = train._validate(self.config)
        backbone, tokenizer = BidirectionalQwen3.from_pretrained_with_tokenizer(
            c['model']['pretrained_name_or_path'], attn_implementation='sdpa')
        model = VoxBaseTTS(backbone, 2, 8).train()
        dataset = TTSDataset(c['data']['validation_dir'], reference_prob=0.8)
        train._seed(20)
        before = train._rng()
        first = train._evaluate(model, tokenizer, dataset, c, torch.device('cpu'))
        self.assertTrue(model.training)
        actual = random.random(), float(np.random.rand()), float(torch.rand(()))
        train._restore(before)
        expected = random.random(), float(np.random.rand()), float(torch.rand(()))
        self.assertEqual(actual, expected)
        second = train._evaluate(model, tokenizer, dataset, c, torch.device('cpu'))
        self.assertEqual(first, second)

    def test_config_validation_and_yaml(self):
        path = self.root / 'config.yaml'
        path.write_text(yaml.safe_dump(self.config))
        self.assertEqual(train.load_yaml(path), self.config)
        default_epochs = deepcopy(self.config)
        del default_epochs['training']['epochs']
        self.assertEqual(train._validate(default_epochs)['training']['epochs'], 10)
        for section, key, value in [('training', 'epochs', 0),
                                    ('training', 'epochs', True),
                                    ('training', 'epochs', False),
                                    ('training', 'epochs', 1.0),
                                    ('training', 'epochs', 1.5),
                                    ('training', 'max_steps', 1),
                                    ('training', 'batch_size', 0),
                                    ('training', 'gradient_accumulation_steps', 0),
                                    ('training', 'precision', 'bad'),
                                    ('masking', 'mask_min', 1.5),
                                    ('model', 'attention_backend', 'flash_attention_2'),
                                    ('data', 'train_dir', str(self.root / 'missing'))]:
            c = deepcopy(self.config)
            c[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                train._validate(c)
        c = deepcopy(self.config)
        c['data']['validation_dir'] = c['data']['train_dir']
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            train._validate(c)
        with patch.dict('os.environ', {'WORLD_SIZE': '1'}, clear=False):
            context = train._ddp_context()
        self.assertFalse(context['enabled'])
        self.assertEqual(context['world_size'], 1)


if __name__ == '__main__':
    unittest.main()
