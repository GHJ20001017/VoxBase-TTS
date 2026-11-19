"""Run with: python -m unittest discover -s tests -v."""

import os
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, DistributedSampler

from data.tts_data import TTSDataset, tts_collate_fn


def sample(i, length, q=8):
    return {
        'sample_id': str(i), 'text': f'utterance {i}', 'speaker_id': None,
        'num_codebooks': q, 'num_frames': length,
        'codes': [[i + j] * length for j in range(q)],
    }


class TTSDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shards = self.root / 'shards'
        shards.mkdir()
        self.rows = [sample(i, i + 1) for i in range(6)]
        for i in range(2):
            pq.write_table(pa.Table.from_pylist(self.rows[i * 3:(i + 1) * 3]),
                           shards / f'{i}.parquet', row_group_size=2)
        indexes = self.root / 'indexes'
        indexes.mkdir()
        pq.write_table(pa.table({'not_a_sample': [1]}), indexes / 'samples.parquet')

    def test_indexing_and_lazy_cache(self):
        dataset = TTSDataset(self.root, cache_size=1)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(len(dataset.shards), 2)
        self.assertEqual(len(dataset._cache), 0)
        for i in [0, 1, 2, 3, 4, 5, -1, -6]:
            row = dataset[i]
            self.assertEqual(row['sample_id'], str(i % 6))
            self.assertEqual(tuple(row['codes'].shape), (8, i % 6 + 1))
            self.assertEqual(row['codes'].dtype, torch.long)
            self.assertLessEqual(len(dataset._cache), 1)
        for i in [-7, 6]:
            with self.assertRaises(IndexError):
                dataset[i]
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(len(restored._cache), 0)
        self.assertEqual(restored[4]['sample_id'], '4')
        self.assertEqual(len(TTSDataset(self.root / 'shards')), 6)

    def test_split_directories_without_named_subdirectory_priority(self):
        codec, q = 'tts_8q', 8
        for split in ['train', 'validation', 'test']:
            root = self.root / codec / split
            root.mkdir(parents=True)
            rows = [dict(sample(i, 2, q), speaker_id='alice', dataset='corpus')
                    for i in range(2)]
            pq.write_table(pa.Table.from_pylist(rows), root / 'part.parquet')
            dataset = TTSDataset(root, reference_prob=1)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]['reference_sample_id'], '1')
            self.assertEqual(dataset[0]['codes'].shape[0], q)
            self.assertTrue(all(p.parent == root.resolve() for p in dataset.shards))
        # A named subdirectory must never hide directly supplied files.
        root = self.root / 'tts_8q' / 'train'
        nested = root / 'shards'
        nested.mkdir()
        pq.write_table(pa.Table.from_pylist([sample(10, 2, 8)]), nested / 'extra.parquet')
        self.assertEqual(len(TTSDataset(root)), 3)

    def test_bulk_order_duplicates_and_single_group_read(self):
        dataset = TTSDataset(self.root, cache_size=1)
        order = [5, 0, 3, 0, 2, -1, 4]
        with patch.object(dataset, '_read_group', wraps=dataset._read_group) as read:
            rows = dataset.__getitems__(order)
            self.assertEqual(read.call_count, 4)
        self.assertEqual([r['sample_id'] for r in rows], [str(i % 6) for i in order])
        self.assertEqual(dataset.__getitems__([]), [])

    def test_padding(self):
        dataset = TTSDataset(self.root)
        batch = tts_collate_fn([dataset[1], dataset[4]])
        self.assertEqual(tuple(batch['codes'].shape), (2, 8, 5))
        self.assertEqual(batch['codes_lengths'].tolist(), [2, 5])
        self.assertEqual(batch['codes_mask'].tolist(), [[True, True, False, False, False], [True] * 5])
        self.assertTrue(torch.all(batch['codes'][0, :, 2:] == -100))
        self.assertTrue(torch.equal(batch['codes'][1], dataset[4]['codes']))
        self.assertEqual(batch['speaker_id'], [None, None])
        self.assertNotIn('codes', batch['metadata'][0])
        self.assertEqual(tts_collate_fn([dataset[0], dataset[1]], 2048)['codes'][0, 0, 1], 2048)
        with self.assertRaises(ValueError):
            tts_collate_fn([])
        bad = dict(dataset[0], codes=torch.zeros(7, 1))
        with self.assertRaises(ValueError):
            tts_collate_fn([dataset[0], bad])

    def test_workers_shuffle_and_distributed_sampling(self):
        dataset = TTSDataset(self.root, cache_size=1)
        dataset[0]  # Populate parent cache before worker creation.
        for workers in [0, 2]:
            loader = DataLoader(dataset, batch_size=2, shuffle=True,
                                num_workers=workers, collate_fn=tts_collate_fn,
                                generator=torch.Generator().manual_seed(42))
            ids = [sid for batch in loader for sid in batch['sample_id']]
            self.assertEqual(sorted(ids), [str(i) for i in range(6)])
        ranks = [list(DistributedSampler(dataset, num_replicas=2, rank=r, seed=42))
                 for r in range(2)]
        self.assertFalse(set(ranks[0]) & set(ranks[1]))
        self.assertEqual(set(ranks[0]) | set(ranks[1]), set(range(6)))

    def test_invalid_inputs(self):
        with self.assertRaises(NotADirectoryError):
            TTSDataset(self.root / 'missing')
        with self.assertRaises(ValueError):
            TTSDataset(self.root, cache_size=0)
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            TTSDataset(empty)
        pq.write_table(pa.table({'text': ['missing fields']}), empty / 'bad.parquet')
        with self.assertRaisesRegex(ValueError, 'missing columns'):
            TTSDataset(empty)
        row = sample(0, 2)
        row['num_frames'] = 3
        with self.assertRaisesRegex(ValueError, 'Invalid sample'):
            TTSDataset._decode(row)

    @unittest.skipUnless(os.environ.get('TTS_TEST_DATA_DIR'), 'set TTS_TEST_DATA_DIR for real data')
    def test_real_release(self):
        dataset = TTSDataset(os.environ['TTS_TEST_DATA_DIR'])
        self.assertEqual(len(dataset.shards), 18)
        self.assertEqual(len(dataset), 33232)
        indices = [0, 1023, 1024, 1846, 1847, len(dataset) - 1]
        rows = dataset.__getitems__(indices)
        batch = tts_collate_fn(rows)
        self.assertEqual(batch['codes'].shape[:2], (6, 8))
        for row in rows:
            self.assertEqual(tuple(row['codes'].shape), (8, row['num_frames']))
            self.assertTrue(torch.all((row['codes'] >= 0) & (row['codes'] < 2048)))
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2,
                            collate_fn=tts_collate_fn)
        batch = next(iter(loader))
        self.assertEqual(batch['codes'].shape[:2], (4, 8))


class TTSReferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_rows(self, relative_path, rows):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)

    @staticmethod
    def voiced(i, length=2, speaker='speaker', dataset='corpus', q=8):
        return dict(sample(i, length, q), speaker_id=speaker, dataset=dataset)

    def assert_no_reference(self, row):
        self.assertIs(row['has_reference'], False)
        for key in ['reference_codes', 'reference_text', 'reference_sample_id']:
            self.assertIsNone(row[key], key)

    def test_probability_endpoints_and_invalid_values(self):
        self.write_rows('shards/a.parquet', [self.voiced(0), self.voiced(1, 3)])
        for probability in [-0.01, 1.01, float('nan'), float('inf'), -float('inf')]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                TTSDataset(self.root, reference_prob=probability)
        disabled = TTSDataset(self.root, reference_prob=0)
        with patch('data.tts_data.random.random', return_value=0):
            self.assert_no_reference(disabled[0])
        enabled = TTSDataset(self.root, reference_prob=1)
        with patch('data.tts_data.random.random', return_value=0.999999):
            row = enabled[0]
        self.assertIs(row['has_reference'], True)
        self.assertEqual(row['reference_sample_id'], '1')
        self.assertEqual(row['reference_text'], 'utterance 1')
        self.assertTrue(torch.equal(row['reference_codes'], torch.tensor(sample(1, 3)['codes'])))

    def test_default_probability_and_strict_threshold(self):
        self.write_rows('shards/a.parquet', [self.voiced(0), self.voiced(1)])
        dataset = TTSDataset(self.root)
        with patch('data.tts_data.random.random', side_effect=[0.799999, 0.8]):
            self.assertIs(dataset[0]['has_reference'], True)
            self.assert_no_reference(dataset[0])

    def test_invalid_speakers_and_singletons_do_not_draw(self):
        invalid = [None, '', '  ', 'none', ' NULL ', 'Unknown', 'NaN', 'N/A', 'UnK']
        for i, speaker in enumerate(invalid):
            # Separate shards permit both null and string speaker columns.
            self.write_rows(f'shards/{i:02}.parquet',
                            [self.voiced(2 * i, speaker=speaker),
                             self.voiced(2 * i + 1, speaker=speaker)])
        self.write_rows('shards/singleton.parquet', [self.voiced(100, speaker='alone')])
        missing = sample(101, 2)
        del missing['speaker_id']
        self.write_rows('shards/missing.parquet', [missing])
        dataset = TTSDataset(self.root, reference_prob=1)
        with patch('data.tts_data.random.random', side_effect=AssertionError('unexpected draw')):
            for row in dataset.__getitems__(list(range(len(dataset)))):
                self.assert_no_reference(row)
            for i in range(len(dataset)):
                self.assert_no_reference(dataset[i])

    def test_numeric_zero_is_valid_speaker(self):
        self.write_rows('shards/a.parquet',
                        [self.voiced(0, speaker=0), self.voiced(1, speaker=0)])
        dataset = TTSDataset(self.root, reference_prob=1)
        self.assertEqual(dataset[0]['reference_sample_id'], '1')
        self.assertEqual(dataset[1]['reference_sample_id'], '0')

    def test_release_dataset_codebooks_and_speaker_isolation(self):
        allowed = {}
        next_id = 0
        for release in ['release-a', 'release-b']:
            for corpus in ['corpus-a', 'corpus-b']:
                for q in [4, 8]:
                    for speaker in ['alice', 'bob']:
                        first, second = next_id, next_id + 1
                        next_id += 2
                        allowed[str(first)] = str(second)
                        allowed[str(second)] = str(first)
                        # A nested shard tests finding the release ancestor,
                        # not merely the immediate shard directory.
                        for i in [first, second]:
                            self.write_rows(f'{release}/shards/nested-{i}/{i}.parquet',
                                            [self.voiced(i, speaker=speaker, dataset=corpus, q=q)])
        dataset = TTSDataset(self.root, reference_prob=1)
        for row in dataset.__getitems__(list(range(len(dataset)))):
            self.assertEqual(row['reference_sample_id'], allowed[row['sample_id']])
            self.assertEqual(row['reference_codes'].shape[0], row['num_codebooks'])

    def test_release_fallback_uses_shard_parent(self):
        for folder, ids in [('a', [0, 1]), ('b', [2, 3])]:
            for i in ids:
                self.write_rows(f'{folder}/{i}.parquet', [self.voiced(i)])
        dataset = TTSDataset(self.root, reference_prob=1)
        allowed = {'0': '1', '1': '0', '2': '3', '3': '2'}
        for row in dataset.__getitems__(list(range(4))):
            self.assertEqual(row['reference_sample_id'], allowed[row['sample_id']])

    def test_duplicate_sample_ids_are_not_candidates(self):
        self.write_rows('shards/a.parquet',
                        [self.voiced(0), self.voiced(0, 3), self.voiced(1, 4)])
        dataset = TTSDataset(self.root, reference_prob=1)
        for row in dataset.__getitems__([0, 1, 2]):
            self.assertNotEqual(row['sample_id'], row['reference_sample_id'])
        self.assertEqual(dataset[0]['reference_sample_id'], '1')
        self.assertEqual(dataset[1]['reference_sample_id'], '1')
        self.write_rows('duplicates/shards/a.parquet', [self.voiced(2), self.voiced(2)])
        duplicates = TTSDataset(self.root / 'duplicates', reference_prob=1)
        with patch('data.tts_data.random.random', side_effect=AssertionError('unexpected draw')):
            for row in duplicates.__getitems__([0, 1]):
                self.assert_no_reference(row)

    def test_single_and_bulk_pair_afresh_without_recursion(self):
        self.write_rows('shards/a.parquet', [self.voiced(i) for i in range(3)])
        dataset = TTSDataset(self.root, reference_prob=0.5)
        # One probability draw per requested item, none for reference decoding.
        with patch('data.tts_data.random.random', side_effect=[0.1, 0.9, 0.1, 0.9]) as draw, \
                patch('data.tts_data.random.choice', wraps=random.choice) as choose:
            self.assertIs(dataset[0]['has_reference'], True)
            self.assert_no_reference(dataset[0])
            rows = dataset.__getitems__([0, 0])
            self.assertIs(rows[0]['has_reference'], True)
            self.assert_no_reference(rows[1])
            self.assertEqual(draw.call_count, 4)
            self.assertEqual(choose.call_count, 2)
        self.assertIsNot(rows[0], rows[1])
        self.assertEqual(dataset.__getitems__([]), [])

    def test_reference_collation_mixed_and_all_none(self):
        self.write_rows('shards/a.parquet',
                        [self.voiced(0, 2), self.voiced(1, 4), self.voiced(2, 1, speaker=None)])
        dataset = TTSDataset(self.root, reference_prob=1)
        rows = dataset.__getitems__([0, 2, 1])
        batch = tts_collate_fn(rows, codes_pad_value=2048)
        self.assertEqual(tuple(batch['reference_codes'].shape), (3, 8, 4))
        self.assertEqual(batch['reference_codes'].dtype, torch.long)
        self.assertEqual(batch['reference_lengths'].tolist(), [4, 0, 2])
        self.assertEqual(batch['reference_lengths'].dtype, torch.long)
        self.assertEqual(batch['reference_mask'].tolist(),
                         [[True] * 4, [False] * 4, [True, True, False, False]])
        self.assertEqual(batch['reference_mask'].dtype, torch.bool)
        self.assertEqual(batch['has_reference'].tolist(), [True, False, True])
        self.assertEqual(batch['has_reference'].dtype, torch.bool)
        self.assertEqual(batch['reference_text'], ['utterance 1', None, 'utterance 0'])
        self.assertEqual(batch['reference_sample_id'], ['1', None, '0'])
        self.assertTrue(torch.equal(batch['reference_codes'][0], rows[0]['reference_codes']))
        self.assertTrue(torch.all(batch['reference_codes'][1] == 2048))
        self.assertTrue(torch.all(batch['reference_codes'][2, :, 2:] == 2048))
        for metadata in batch['metadata']:
            self.assertFalse(any(key.startswith('reference_') or key == 'has_reference'
                                 for key in metadata))
            self.assertFalse(any(isinstance(value, torch.Tensor) for value in metadata.values()))
        none = tts_collate_fn([rows[1], rows[1]])
        self.assertEqual(tuple(none['reference_codes'].shape), (2, 8, 0))
        self.assertEqual(tuple(none['reference_mask'].shape), (2, 0))
        self.assertEqual(none['reference_lengths'].tolist(), [0, 0])
        self.assertEqual(none['has_reference'].tolist(), [False, False])
        self.assertEqual(none['reference_text'], [None, None])
        self.assertEqual(none['reference_sample_id'], [None, None])

    def test_reference_pairing_survives_pickle_and_workers(self):
        self.write_rows('shards/a.parquet', [self.voiced(i, i + 1) for i in range(4)])
        dataset = TTSDataset(self.root, cache_size=1, reference_prob=1)
        dataset[0]  # Workers must safely handle a populated parent cache.
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(len(restored._cache), 0)
        self.assertIs(restored[0]['has_reference'], True)
        for workers in [0, 2]:
            with self.subTest(workers=workers):
                loader = DataLoader(dataset, batch_size=2, num_workers=workers,
                                    collate_fn=tts_collate_fn,
                                    generator=torch.Generator().manual_seed(42))
                seen = []
                for batch in loader:
                    self.assertEqual(batch['has_reference'].tolist(), [True, True])
                    for i, (sid, reference_id) in enumerate(zip(
                            batch['sample_id'], batch['reference_sample_id'])):
                        seen.append(sid)
                        self.assertNotEqual(sid, reference_id)
                        self.assertEqual(batch['reference_lengths'][i].item(), int(reference_id) + 1)
                        self.assertEqual(batch['reference_text'][i], f'utterance {reference_id}')
                self.assertEqual(seen, ['0', '1', '2', '3'])


if __name__ == '__main__':
    unittest.main()
