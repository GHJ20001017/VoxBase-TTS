"""Offline progress reporting regressions using the shared tiny training fixture."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, call, patch

import torch

import test_train
import train


class ProgressTest(unittest.TestCase):
    # Reuse fixtures without importing or inheriting a collectable TrainingTest.
    setUpClass = classmethod(test_train.TrainingTest.setUpClass.__func__)
    tearDownClass = classmethod(test_train.TrainingTest.tearDownClass.__func__)
    setUp = test_train.TrainingTest.setUp

    def progress_factory(self):
        bars = []

        def create(*args, **kwargs):
            bar = Mock(name=f'epoch_progress_{len(bars)}')
            bars.append(bar)
            return bar

        return Mock(side_effect=create), bars

    def assert_progress(self, factory, bars, initials, epochs):
        self.assertEqual(factory.call_count, len(initials))
        self.assertEqual(len(bars), len(initials))
        for index, (invocation, bar, initial) in enumerate(
                zip(factory.call_args_list, bars, initials)):
            with self.subTest(bar=index):
                self.assertEqual(invocation.kwargs['desc'], f'Epoch {index + 1}/{epochs}')
                self.assertEqual(invocation.kwargs['total'], 3)
                self.assertEqual(invocation.kwargs['initial'], initial)
                self.assertEqual(invocation.kwargs['unit'], 'batch')
                self.assertIs(invocation.kwargs['dynamic_ncols'], True)
                self.assertEqual(invocation.kwargs['mininterval'], 1)
                self.assertEqual(bar.update.call_args_list, [call(1)] * (3 - initial))
                self.assertEqual(sum(c.args[0] for c in bar.update.call_args_list), 3 - initial)
                bar.close.assert_called()

    def test_epoch_bars_postfix_and_validation_summary(self):
        factory, bars = self.progress_factory()
        c = deepcopy(self.config)
        c['training']['tensorboard'] = False
        with patch('train.tqdm', factory), patch('builtins.print') as output:
            result = train.run(c)
        self.assertEqual(result['global_step'], 4)
        self.assert_progress(factory, bars, [0, 0], epochs=2)
        for bar in bars:
            self.assertTrue(bar.set_postfix.called)
            metric_updates = [invocation.kwargs for invocation in bar.set_postfix.call_args_list
                              if 'loss' in invocation.kwargs]
            self.assertTrue(metric_updates)
            for values in metric_updates:
                self.assertRegex(values['loss'], r'^\d+\.\d{4}$')
                self.assertRegex(values['acc'], r'^\d+\.\d{3}%$')
                self.assertRegex(values['lr'], r'^\d\.\d{2}e[+-]\d+$')
                self.assertIsInstance(values['step'], int)
        summaries = [str(c.args[0]) for c in factory.write.call_args_list if c.args]
        self.assertTrue(any('Validation' in summary for summary in summaries))
        self.assertFalse(any('"global_step"' in str(c) for c in output.call_args_list))

    def test_resumed_bar_does_not_count_replayed_batches(self):
        c = deepcopy(self.config)
        c['training'].update(epochs=1, tensorboard=False)
        save = train._atomic_save

        class Interrupted(Exception):
            pass

        def save_then_interrupt(payload, path):
            save(payload, path)
            if path.name == 'last.pt' and payload['global_step'] == 1:
                raise Interrupted('interrupt after first optimizer update')

        interrupted_factory, interrupted_bars = self.progress_factory()
        with patch('train.tqdm', interrupted_factory), \
                patch('train._atomic_save', side_effect=save_then_interrupt), \
                self.assertRaises(Interrupted):
            train.run(c)
        self.assertEqual(len(interrupted_bars), 1)
        self.assertEqual(interrupted_bars[0].update.call_args_list, [call(1), call(1)])
        interrupted_bars[0].close.assert_called()
        checkpoint_path = Path(c['training']['output_dir']) / 'last.pt'
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.assertEqual(checkpoint['global_step'], 1)
        self.assertEqual(checkpoint['batch_cursor'], 2)

        factory, bars = self.progress_factory()
        with patch('train.tqdm', factory):
            result = train.run(c, str(checkpoint_path))
        self.assertEqual(result['global_step'], 2)
        self.assert_progress(factory, bars, [2], epochs=1)

    def test_forward_exception_closes_current_bar(self):
        c = deepcopy(self.config)
        c['training']['tensorboard'] = False
        factory, bars = self.progress_factory()
        original = train.VoxBaseTTS.forward
        training_calls = 0

        def fail_second_microbatch(model, *args, **kwargs):
            nonlocal training_calls
            if model.training:
                training_calls += 1
                if training_calls == 2:
                    raise RuntimeError('injected training failure')
            return original(model, *args, **kwargs)

        with patch('train.tqdm', factory), \
                patch.object(train.VoxBaseTTS, 'forward', fail_second_microbatch), \
                self.assertRaisesRegex(RuntimeError, 'injected training failure'):
            train.run(c)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].update.call_args_list, [call(1)])
        bars[0].close.assert_called()

    def test_amp_overflow_warning_uses_tqdm_write(self):
        c = deepcopy(self.config)
        c['training'].update(epochs=1, tensorboard=False)
        factory, bars = self.progress_factory()
        scaler = Mock()
        scaler.scale.side_effect = lambda loss: loss
        scaler.is_enabled.return_value = True
        scaler.get_scale.return_value = 32768.0
        scaler.state_dict.return_value = {}

        def step_if_finite(optimizer):
            if all(torch.isfinite(p.grad).all()
                   for group in optimizer.param_groups for p in group['params']
                   if p.grad is not None):
                optimizer.step()

        scaler.step.side_effect = step_if_finite
        original = train.VoxBaseTTS.forward
        injected = False

        def overflow_once(model, *args, **kwargs):
            nonlocal injected
            result = original(model, *args, **kwargs)
            if model.training and not injected:
                injected = True
                result['loss_sum'].register_hook(lambda gradient: gradient * float('inf'))
            return result

        with patch('train.tqdm', factory), \
                patch('train.torch.amp.GradScaler', return_value=scaler), \
                patch.object(train.VoxBaseTTS, 'forward', overflow_once):
            result = train.run(c)
        self.assertEqual(result['global_step'], 1)
        self.assert_progress(factory, bars, [0], epochs=1)
        messages = [str(invocation.args[0]) for invocation in factory.write.call_args_list
                    if invocation.args]
        self.assertTrue(any('overflow' in message.lower() and 'AMP' in message
                            for message in messages))

    def test_tensorboard_warmup_and_logging_cadence_unchanged(self):
        for log_every, epochs, expected_steps, expected_rates in (
                (1, 1, [1, 2], [2e-7, 4e-7]),
                (3, 2, [1, 3], [2e-7, 6e-7])):
            with self.subTest(log_every=log_every):
                c = deepcopy(self.config)
                c['training'].update(
                    epochs=epochs, warmup_steps=500, learning_rate=1e-4,
                    log_every=log_every, tensorboard=True,
                    output_dir=str(self.root / f'warmup-{log_every}'))
                original_config = deepcopy(c)
                writer = Mock()
                factory, _ = self.progress_factory()
                with patch('train.tqdm', factory), \
                        patch('torch.utils.tensorboard.SummaryWriter', return_value=writer):
                    result = train.run(c)
                rates = [(invocation.args[2], invocation.args[1])
                         for invocation in writer.add_scalar.call_args_list
                         if invocation.args[0] == 'train/learning_rate']
                self.assertEqual([step for step, _ in rates], expected_steps)
                for (_, actual), expected in zip(rates, expected_rates):
                    self.assertAlmostEqual(actual, expected, places=14)
                self.assertEqual(c, original_config)
                checkpoint = torch.load(Path(c['training']['output_dir']) / 'last.pt',
                                        weights_only=False)
                self.assertEqual(checkpoint['scheduler']['last_epoch'], result['global_step'])
                self.assertAlmostEqual(checkpoint['optimizer']['param_groups'][0]['lr'],
                                       1e-4 * (result['global_step'] + 1) / 500, places=14)
                writer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
