"""Offline TensorBoard event, configuration, and writer-lifecycle contracts."""

import builtins
from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch

import train
import test_train as training_fixture


class TensorBoardTest(unittest.TestCase):
    # Reuse only the tiny offline fixture, not TrainingTest's test methods.
    setUp = training_fixture.TrainingTest.setUp
    setUpClass = classmethod(training_fixture.TrainingTest.setUpClass.__func__)
    tearDownClass = classmethod(training_fixture.TrainingTest.tearDownClass.__func__)

    @staticmethod
    def events(directory):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        return EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()

    @staticmethod
    def missing_tensorboard():
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == 'torch.utils.tensorboard' or name == 'tensorboard' or name.startswith('tensorboard.'):
                raise ImportError('TensorBoard deliberately unavailable in test')
            return original_import(name, *args, **kwargs)

        return patch('builtins.__import__', side_effect=guarded_import)

    def test_default_enabled_and_requires_a_real_boolean(self):
        self.assertIs(train._validate(self.config)['training']['tensorboard'], True)
        for value in (None, 0, 1, 'true', 'false', [], {}):
            with self.subTest(value=value):
                c = deepcopy(self.config)
                c['training']['tensorboard'] = value
                with self.assertRaisesRegex(ValueError, '(?i)tensorboard.*bool'):
                    train._validate(c)
        for value in (False, True):
            c = deepcopy(self.config)
            c['training']['tensorboard'] = value
            self.assertIs(train._validate(c)['training']['tensorboard'], value)

    def test_real_events_have_correct_tags_cadence_and_checkpoint_loss(self):
        c = deepcopy(self.config)
        c['training']['log_every'] = 3
        c['evaluation']['every_steps'] = 3
        result = train.run(c)
        self.assertEqual(result['global_step'], 4)
        output = Path(c['training']['output_dir'])
        events = self.events(output / 'tensorboard')
        expected = {
            'train/loss': [1, 3],
            'train/accuracy': [1, 3],
            'train/grad_norm': [1, 3],
            'train/learning_rate': [1, 3],
            'train/epoch': [1, 3],
            'train/num_masked': [1, 3],
            'validation/loss': [2, 3, 4],
            'validation/accuracy': [2, 3, 4],
            'validation/num_masked': [2, 3, 4],
        }
        self.assertEqual(set(events.Tags()['scalars']), set(expected))
        for tag, steps in expected.items():
            with self.subTest(tag=tag):
                scalars = events.Scalars(tag)
                self.assertEqual([event.step for event in scalars], steps)
                self.assertTrue(all(math.isfinite(event.value) for event in scalars))
                if tag.endswith('accuracy'):
                    self.assertTrue(all(0 <= event.value <= 1 for event in scalars))
                if tag.endswith('num_masked'):
                    self.assertTrue(all(event.value > 0 for event in scalars))
        self.assertEqual([event.value for event in events.Scalars('train/epoch')], [0, 1])
        best_loss = min(event.value for event in events.Scalars('validation/loss'))
        for name in ('last.pt', 'best.pt'):
            checkpoint = torch.load(output / name, weights_only=False)
            self.assertAlmostEqual(best_loss, checkpoint['best_validation_loss'], places=6)
        self.assertAlmostEqual(best_loss, result['best_validation_loss'], places=6)

    def test_disabled_neither_imports_tensorboard_nor_creates_event_directory(self):
        c = deepcopy(self.config)
        c['training'].update(tensorboard=False, epochs=1)
        with self.missing_tensorboard():
            self.assertEqual(train.run(c)['global_step'], 2)
        output = Path(c['training']['output_dir'])
        self.assertFalse((output / 'tensorboard').exists())
        self.assertFalse(list(output.rglob('events.out.tfevents.*')))

    def test_missing_dependency_is_informative_and_fails_before_model_loading(self):
        with self.missing_tensorboard(), patch.object(
                train.BidirectionalQwen3, 'from_pretrained_with_tokenizer') as load:
            with self.assertRaises(ImportError) as raised:
                train.run(self.config)
        message = str(raised.exception).lower()
        self.assertIn('tensorboard', message)
        self.assertIn('false', message)
        load.assert_not_called()

    def test_test_only_neither_imports_tensorboard_nor_writes_events(self):
        c = deepcopy(self.config)
        c['training'].update(tensorboard=False, epochs=1)
        train.run(c)
        output = Path(c['training']['output_dir'])
        c['training']['tensorboard'] = True
        with self.missing_tensorboard():
            result = train.run(c, str(output / 'last.pt'), test_only=True)
        self.assertTrue(math.isfinite(result['test']['loss']))
        self.assertFalse((output / 'tensorboard').exists())
        self.assertFalse(list(output.rglob('events.out.tfevents.*')))

    def test_resume_purges_future_events_but_preserves_committed_step(self):
        from torch.utils.tensorboard import SummaryWriter

        class Interrupted(Exception):
            pass

        original_save = train._atomic_save

        def interrupt_after_checkpoint(payload, path):
            original_save(payload, path)
            if path.name == 'last.pt' and payload['global_step'] == 1:
                raise Interrupted('saved step one')

        with patch('train._atomic_save', side_effect=interrupt_after_checkpoint):
            with self.assertRaises(Interrupted):
                train.run(self.config)
        output = Path(self.config['training']['output_dir'])
        checkpoint = torch.load(output / 'last.pt', weights_only=False)
        self.assertEqual(checkpoint['global_step'], 1)
        directory = output / 'tensorboard'
        with SummaryWriter(str(directory)) as writer:
            writer.add_scalar('train/loss', 123456.0, 999)
            writer.flush()
        before = self.events(directory)
        self.assertIn(999, [event.step for event in before.Scalars('train/loss')])
        self.assertIn(1, [event.step for event in before.Scalars('train/loss')])
        result = train.run(self.config, str(output / 'last.pt'))
        self.assertEqual(result['global_step'], 4)
        after = self.events(directory)
        for tag in ('train/loss', 'validation/loss'):
            scalars = after.Scalars(tag)
            self.assertEqual([event.step for event in scalars], [1, 2, 3, 4])
            self.assertTrue(all(math.isfinite(event.value) for event in scalars))
            self.assertNotIn(123456.0, [event.value for event in scalars])

    def test_metrics_are_flushed_and_writer_closes_after_evaluation_error(self):
        writer = Mock()
        with patch('torch.utils.tensorboard.SummaryWriter', return_value=writer), patch(
                'train._evaluate', side_effect=RuntimeError('evaluation failed')):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                train.run(self.config)
        self.assertIn('train/loss', [call.args[0] for call in writer.add_scalar.call_args_list])
        writer.flush.assert_called()
        writer.close.assert_called_once()

    def test_checkpoint_failure_closes_writer_and_preserves_flushed_metrics(self):
        writer = Mock()
        with patch('torch.utils.tensorboard.SummaryWriter', return_value=writer), patch(
                'train._atomic_save', side_effect=OSError('checkpoint failed')):
            with self.assertRaisesRegex(OSError, 'checkpoint failed'):
                train.run(self.config)
        tags = [call.args[0] for call in writer.add_scalar.call_args_list]
        self.assertIn('train/loss', tags)
        self.assertIn('validation/loss', tags)
        self.assertGreaterEqual(writer.flush.call_count, 2)
        writer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
