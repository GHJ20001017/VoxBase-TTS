"""Single-process masked TTS trainer.

Resume checkpoints are trusted pickle files (never load untrusted checkpoints).
Resume at optimizer boundaries replays the epoch's skipped batches to reproduce
worker/reference RNG streams. Exact replay assumes unchanged data, worker count,
batch size, software/device and config; CUDA kernels may still be nondeterministic.
The tokenizer and backbone_config.json beside the checkpoint permit offline resume.
No test split is read during training. Run --test-only explicitly to evaluate it.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm
import yaml

from data.tts_data import TTSDataset, tts_collate_fn
from model.bidirectional_qwen3 import BidirectionalQwen3
from model.VoxBase_TTS import VoxBaseTTS


def load_yaml(path):
    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('YAML must contain a mapping')
    return config


def _validate(config, test_only=False):
    c = deepcopy(config)
    defaults = {
        'model': dict(pretrained_name_or_path='Qwen/Qwen3-0.6B', attention_backend='sdpa',
                      num_codebooks=8, codebook_size=2048, freeze_backbone=False,
                      gradient_checkpointing=True),
        'data': dict(train_dir='/gpu3/guhj/VoxBase-TTS/dataset/tts_8q/train',
                     validation_dir='/gpu3/guhj/VoxBase-TTS/dataset/tts_8q/validation',
                     test_dir='/gpu3/guhj/VoxBase-TTS/dataset/tts_8q/test', cache_size=1,
                     reference_prob=0.8, num_workers=2),
        'training': dict(output_dir='outputs/tts', device='cuda', precision='fp16', seed=42,
                         batch_size=128, gradient_accumulation_steps=1, epochs=10,
                         learning_rate=0.0002, weight_decay=0.01, adam_beta1=0.9,
                         adam_beta2=0.95, adam_epsilon=1e-8, warmup_steps=500,
                         max_grad_norm=1.0, log_every=10, save_every=500,
                         tensorboard=True),
        'masking': dict(mask_min=0.05, mask_max=1.0, max_text_tokens=512,
                        max_sequence_length=40960),
        'evaluation': dict(every_steps=500, batch_size=1, max_batches=50, seed=1234),
    }
    if set(c) - set(defaults):
        raise ValueError(f'Unknown config sections: {set(c) - set(defaults)}')
    for section, values in defaults.items():
        provided = c.setdefault(section, {})
        if not isinstance(provided, dict) or set(provided) - set(values):
            raise ValueError(f'Invalid or unknown fields in {section}')
        c[section] = {**values, **provided}
    m, d, t, k, e = (c[s] for s in defaults)
    for section, fields in [(m, ['num_codebooks', 'codebook_size']),
                            (d, ['cache_size']),
                            (t, ['batch_size', 'gradient_accumulation_steps', 'epochs',
                                 'log_every', 'save_every']),
                            (k, ['max_text_tokens', 'max_sequence_length']),
                            (e, ['every_steps', 'batch_size', 'max_batches'])]:
        for key in fields:
            if type(section[key]) is not int or section[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
    for section, key in [(d, 'num_workers'), (t, 'warmup_steps'), (t, 'seed'), (e, 'seed')]:
        if type(section[key]) is not int or section[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ['learning_rate', 'adam_epsilon', 'max_grad_norm']:
        if not isinstance(t[key], (int, float)) or not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(t['weight_decay']) or t['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    if not all(0 <= t[key] < 1 for key in ['adam_beta1', 'adam_beta2']):
        raise ValueError('Adam betas must be in [0, 1)')
    if not 0 <= d['reference_prob'] <= 1:
        raise ValueError('reference_prob must be in [0, 1]')
    if not 0 < k['mask_min'] <= k['mask_max'] <= 1:
        raise ValueError('Require 0 < mask_min <= mask_max <= 1')
    if m['attention_backend'] not in ('eager', 'sdpa'):
        raise ValueError('attention_backend must be eager or sdpa')
    for key in ['freeze_backbone', 'gradient_checkpointing']:
        if type(m[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if not isinstance(m['pretrained_name_or_path'], str) or not m['pretrained_name_or_path']:
        raise ValueError('pretrained_name_or_path must be nonempty')
    if not isinstance(t['output_dir'], str) or not t['output_dir'].strip():
        raise ValueError('output_dir must be nonempty')
    if type(t['tensorboard']) is not bool:
        raise ValueError('tensorboard must be boolean')
    if t['precision'] not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32, fp16 or bf16')
    device = torch.device(t['device'])
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only cpu and cuda devices are supported')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use device: cpu for a smoke test')
    if device.type == 'cuda' and t['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA device does not support bf16')
    for key in (['test_dir'] if test_only else ['train_dir', 'validation_dir']):
        if not isinstance(d[key], str) or not Path(d[key]).is_dir():
            raise ValueError(f'{key} is not an existing directory: {d[key]}')
    if not test_only:
        train_path = Path(d['train_dir']).resolve()
        validation_path = Path(d['validation_dir']).resolve()
        if (train_path == validation_path or train_path in validation_path.parents
                or validation_path in train_path.parents):
            raise ValueError('train_dir and validation_dir must be disjoint split directories')
    return c


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def _worker_seed(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _autocast(device, precision):
    if device.type == 'cpu' or precision == 'fp32':
        return nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def _move(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _evaluate(model, tokenizer, dataset, c, device, ddp=None):
    ddp = ddp or dict(enabled=False, rank=0, main=True)
    state, was_training = _rng(), model.training
    e, t = c['evaluation'], c['training']
    try:
        _seed(e['seed'])
        sampler = (DistributedSampler(dataset, num_replicas=ddp['world_size'], rank=ddp['rank'],
                                      shuffle=False, drop_last=False)
                   if ddp['enabled'] else None)
        loader = DataLoader(dataset, batch_size=e['batch_size'],
                            sampler=sampler, shuffle=False if sampler is not None else False,
                            num_workers=0, collate_fn=tts_collate_fn,
                            generator=torch.Generator().manual_seed(e['seed']))
        model.eval()
        total_loss = total_correct = total_masked = 0
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= e['max_batches']:
                    break
                with _autocast(device, t['precision']):
                    result = model(_move(batch, device), tokenizer, **c['masking'])
                loss = float(result['loss_sum'].detach())
                if not math.isfinite(loss):
                    raise FloatingPointError('Nonfinite evaluation loss')
                total_loss += loss
                total_correct += int(result['num_correct'])
                total_masked += int(result['num_masked'])
        if not total_masked:
            raise ValueError('Evaluation produced no masked tokens')
        metrics = dict(loss=total_loss / total_masked, accuracy=total_correct / total_masked,
                       num_masked=total_masked)
        return _ddp_reduce_metrics(metrics, device, ddp)
    finally:
        _restore(state)
        model.train(was_training)


def _atomic_save(payload, path):
    fd, name = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _print_evaluation(label, metrics, step):
    tqdm.write(f'{label} | step {step:,} | loss {metrics["loss"]:.4f} | '
               f'acc {100 * metrics["accuracy"]:.3f}% | '
               f'masked {metrics["num_masked"]:,}')


def _log_metrics(writer, prefix, metrics, step):
    if writer is not None:
        for name, value in metrics.items():
            writer.add_scalar(f'{prefix}/{name}', value, step)
        writer.flush()


class _NullProgress:
    def update(self, *_args, **_kwargs):
        pass

    def set_postfix(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


def _ddp_context():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return dict(enabled=False, rank=0, local_rank=0, world_size=1, main=True)
    if not dist.is_available():
        raise RuntimeError('DDP requested but torch.distributed is unavailable')
    if not dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    return dict(enabled=True, rank=dist.get_rank(), local_rank=int(os.environ.get('LOCAL_RANK', '0')),
                world_size=dist.get_world_size(), main=dist.get_rank() == 0)


def _ddp_barrier(ddp):
    if ddp['enabled']:
        dist.barrier()


def _ddp_cleanup(ddp):
    if ddp['enabled'] and dist.is_initialized():
        dist.destroy_process_group()

def run(config: dict, resume: str | None = None, *, test_only=False):
    ddp = _ddp_context()
    c = _validate(config, test_only)
    m, d, t = c['model'], c['data'], c['training']
    output = Path(t['output_dir']).expanduser().resolve()
    checkpoint = None
    if resume:
        source = Path(resume).expanduser().resolve()
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        # These fields change replay or model semantics; reject silent divergence.
        old = checkpoint['config']
        if 'epochs' not in old['training']:
            raise ValueError('Resume requires an epoch-based checkpoint; old max_steps checkpoints are unsupported')
        for section in ['model', 'data', 'masking']:
            if c[section] != old[section]:
                raise ValueError(f'Resume requires unchanged {section} configuration')
        for key in ['seed', 'batch_size', 'gradient_accumulation_steps', 'precision', 'device',
                    'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2',
                    'adam_epsilon', 'warmup_steps', 'max_grad_norm', 'epochs']:
            if t[key] != old['training'][key]:
                raise ValueError(f'Resume requires unchanged {key}')
        if output != source.parent:
            raise ValueError('Resume output_dir must be the checkpoint directory')
    elif test_only:
        raise ValueError('--test-only requires --resume')
    elif ddp['main'] and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {output}')
    SummaryWriter = None
    if t['tensorboard'] and not test_only:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError('TensorBoard logging is enabled. Install tensorboard '
                              'or set training.tensorboard: false.') from error
    _seed(t['seed'])
    if checkpoint is None:
        backbone, tokenizer = BidirectionalQwen3.from_pretrained_with_tokenizer(
            m['pretrained_name_or_path'], attn_implementation=m['attention_backend'])
        backbone.register_tts_tokens(tokenizer)
    else:
        from transformers import AutoTokenizer, Qwen3Config
        backbone_config = Qwen3Config.from_json_file(str(output / 'backbone_config.json'))
        backbone_config._attn_implementation = m['attention_backend']
        backbone = BidirectionalQwen3(backbone_config)
        tokenizer = AutoTokenizer.from_pretrained(output / 'tokenizer', local_files_only=True)
        backbone.register_tts_tokens(tokenizer)
    if m['freeze_backbone']:
        backbone.requires_grad_(False)
    if m['gradient_checkpointing']:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = VoxBaseTTS(backbone, num_codebooks=m['num_codebooks'], codebook_size=m['codebook_size'])
    device = (torch.device(f'cuda:{ddp["local_rank"]}') if ddp['enabled']
              else torch.device(t['device']))
    model = model.float().to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    if ddp['enabled']:
        model = DistributedDataParallel(model, device_ids=[ddp['local_rank']],
                                        output_device=ddp['local_rank'])
    def dataset(split):
        result = TTSDataset(d[split + '_dir'], cache_size=d['cache_size'], reference_prob=d['reference_prob'])
        if not len(result):
            raise ValueError(f'{split} dataset is empty')
        return result
    if test_only:
        metrics = _evaluate(model, tokenizer, dataset('test'), c, device, ddp)
        if ddp['main']:
            _print_evaluation('Test', metrics, checkpoint['global_step'])
        result = {'global_step': checkpoint['global_step'], 'test': metrics}
        _ddp_cleanup(ddp)
        return result
    train_data, validation_data = dataset('train'), dataset('validation')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=t['learning_rate'], weight_decay=t['weight_decay'],
                                 betas=(t['adam_beta1'], t['adam_beta2']), eps=t['adam_epsilon'])
    samples_per_rank = math.ceil(len(train_data) / ddp['world_size'])
    batches_per_epoch = math.ceil(samples_per_rank / t['batch_size'])
    updates_per_epoch = math.ceil(batches_per_epoch / t['gradient_accumulation_steps'])
    total_updates = updates_per_epoch * t['epochs']

    def schedule(step):
        if step < t['warmup_steps']:
            return (step + 1) / max(1, t['warmup_steps'])
        progress = (step - t['warmup_steps']) / max(1, total_updates - t['warmup_steps'])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and t['precision'] == 'fp16')
    step = epoch = cursor = 0
    best = float('inf')
    resume_rng = None
    saved_order = None
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        step, epoch, cursor = (checkpoint[k] for k in ['global_step', 'epoch', 'batch_cursor'])
        best, resume_rng = checkpoint['best_validation_loss'], checkpoint['rng']
        saved_order = checkpoint['data_order']
    else:
        if ddp['main']:
            output.mkdir(parents=True, exist_ok=True)
            backbone.config.to_json_file(str(output / 'backbone_config.json'))
            tokenizer.save_pretrained(output / 'tokenizer')
        _ddp_barrier(ddp)
    if epoch >= t['epochs'] or (epoch == t['epochs'] - 1 and cursor == batches_per_epoch):
        result = {'global_step': step, 'best_validation_loss': best}
        _ddp_cleanup(ddp)
        return result
    writer = (SummaryWriter(log_dir=str(output / 'tensorboard'),
                            purge_step=step + 1 if checkpoint is not None else None)
              if SummaryWriter is not None and ddp['main'] else None)

    progress = None
    try:
        def save(order, improved=False):
            if ddp['main']:
                payload = dict(model=(model.module if ddp['enabled'] else model).state_dict(),
                               optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                               scaler=scaler.state_dict(), global_step=step, epoch=epoch,
                               batch_cursor=cursor, config=c, rng=_rng(),
                               epoch_start_rng=epoch_start_rng, data_order=order,
                               best_validation_loss=best)
                _atomic_save(payload, output / 'last.pt')
                if improved:
                    _atomic_save(payload, output / 'best.pt')
                if writer is not None:
                    writer.flush()
            _ddp_barrier(ddp)
        model.train()
        consecutive_overflows = 0
        last_evaluated_step = None
        while epoch < t['epochs']:
            order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(t['seed'] + epoch)).tolist()
            if saved_order is not None:
                if order != saved_order:
                    raise ValueError('Dataset size/order changed since checkpoint')
                saved_order = None
            subset = Subset(train_data, order)
            sampler = (DistributedSampler(subset, num_replicas=ddp['world_size'], rank=ddp['rank'],
                                          shuffle=False, drop_last=False)
                       if ddp['enabled'] else None)
            loader = DataLoader(subset, batch_size=t['batch_size'], sampler=sampler,
                                shuffle=False if sampler is not None else False,
                                num_workers=d['num_workers'], collate_fn=tts_collate_fn,
                                worker_init_fn=_worker_seed, persistent_workers=False,
                                pin_memory=device.type == 'cuda',
                                generator=torch.Generator().manual_seed(t['seed'] + epoch))
            if cursor > len(loader):
                raise ValueError('Checkpoint batch_cursor exceeds epoch length')
            epoch_start_rng = checkpoint['epoch_start_rng'] if resume_rng is not None else _rng()
            if resume_rng is not None:
                # Replay references from the original epoch's main-process RNG.
                _restore(epoch_start_rng)
            iterator = iter(loader)
            for _ in range(cursor):
                next(iterator)
            if resume_rng is not None:
                _restore(resume_rng)
                resume_rng = None
            progress = (tqdm(total=len(loader), initial=cursor,
                              desc=f'Epoch {epoch + 1}/{t["epochs"]}', unit='batch',
                              dynamic_ncols=True, mininterval=1.0)
                       if ddp['main'] else _NullProgress())
            progress.set_postfix(step=step, lr=f'{optimizer.param_groups[0]["lr"]:.2e}',
                                 refresh=False)
            while cursor < len(loader):
                group_size = min(t['gradient_accumulation_steps'], len(loader) - cursor)
                optimizer.zero_grad(set_to_none=True)
                loss_sum = masked = correct = 0
                backward_denominator = None
                # Sum token losses, then normalize gradients once over the complete
                # accumulation group, including a partial final group and variable T.
                for _ in range(group_size):
                    batch = next(iterator)
                    with _autocast(device, t['precision']):
                        result = model(_move(batch, device), tokenizer, **c['masking'])
                    if not torch.isfinite(result['loss_sum']).all():
                        raise FloatingPointError(f'Nonfinite training loss at step {step}')
                    local_masked = int(result['num_masked'])
                    global_masked = local_masked
                    if ddp['enabled']:
                        count = torch.tensor(local_masked, dtype=torch.long, device=device)
                        dist.all_reduce(count, op=dist.ReduceOp.SUM)
                        global_masked = int(count)
                    if backward_denominator is None:
                        backward_denominator = max(1, global_masked) * group_size / ddp['world_size']
                    scaler.scale(result['loss_sum'] / backward_denominator).backward()
                    loss_sum += float(result['loss_sum'].detach())
                    masked += local_masked
                    correct += int(result['num_correct'])
                    cursor += 1
                    progress.update(1)
                if masked <= 0:
                    raise ValueError('Accumulation group has no masked tokens')
                if ddp['enabled']:
                    total = torch.tensor(masked, dtype=torch.long, device=device)
                    dist.all_reduce(total, op=dist.ReduceOp.SUM)
                    masked_total = int(total)
                else:
                    masked_total = masked
                scaler.unscale_(optimizer)
                nonfinite = any(not torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.grad is not None)
                if ddp['enabled']:
                    flag = torch.tensor(int(nonfinite), dtype=torch.long, device=device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    nonfinite = bool(flag)
                if nonfinite:
                    if not scaler.is_enabled():
                        raise FloatingPointError(f'Nonfinite gradients at step {step}')
                    # unscale_ recorded the overflow: step skips this update, and
                    # update reduces the scale. Do not advance scheduler/step.
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_overflows += 1
                    if ddp['main']:
                        tqdm.write(f'AMP overflow | step {step:,} | update skipped | '
                                   f'scale {scaler.get_scale():.0f}')
                    if consecutive_overflows >= 32:
                        raise FloatingPointError('32 consecutive AMP gradient overflows')
                    continue
                consecutive_overflows = 0
                training_metrics = _ddp_reduce_metrics(
                    dict(loss=loss_sum / masked, accuracy=correct / masked, num_masked=masked),
                    device, ddp)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(backward_denominator * ddp['world_size'] / masked_total)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['max_grad_norm'], error_if_nonfinite=True)
                learning_rate = optimizer.param_groups[0]['lr']
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                progress.set_postfix(loss=f'{training_metrics["loss"]:.4f}',
                                     acc=f'{100 * training_metrics["accuracy"]:.3f}%',
                                     lr=f'{learning_rate:.2e}', step=step,
                                     grad=f'{float(norm):.3f}', refresh=False)
                if step % t['log_every'] == 0 or step == 1:
                    training_metrics.update(grad_norm=float(norm), learning_rate=learning_rate,
                                            epoch=epoch)
                    _log_metrics(writer, 'train', training_metrics, step)
                improved = False
                if step % c['evaluation']['every_steps'] == 0:
                    metrics = _evaluate(model, tokenizer, validation_data, c, device, ddp)
                    if ddp['main']:
                        _print_evaluation('Validation', metrics, step)
                    _log_metrics(writer, 'validation', metrics, step)
                    last_evaluated_step = step
                    improved = metrics['loss'] < best
                    best = min(best, metrics['loss'])
                if improved or step % t['save_every'] == 0:
                    save(order, improved)
            # Always evaluate/save completed epochs, including a skipped final AMP update.
            improved = False
            if last_evaluated_step != step:
                metrics = _evaluate(model, tokenizer, validation_data, c, device, ddp)
                if ddp['main']:
                    _print_evaluation('Validation', metrics, step)
                _log_metrics(writer, 'validation', metrics, step)
                last_evaluated_step = step
                improved = metrics['loss'] < best
                best = min(best, metrics['loss'])
            save(order, improved)
            progress.close()
            progress = None
            epoch += 1
            cursor = 0
        result = {'global_step': step, 'best_validation_loss': best}
        _ddp_cleanup(ddp)
        return result
    finally:
        if progress is not None:
            progress.close()
        if writer is not None:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    run(config, args.resume, test_only=args.test_only)


if __name__ == '__main__':
    main()
