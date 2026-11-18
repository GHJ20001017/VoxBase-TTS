"""Small CPU tests: no network or full 0.6B weight allocation required."""

import tempfile
import unittest

import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from model.bidirectional_qwen3 import BidirectionalQwen3, qwen3_06b_config, TTS_SPECIAL_TOKENS


def tiny_config(backend='sdpa'):
    config = Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
    )
    config._attn_implementation = backend
    return config


class BidirectionalQwen3Test(unittest.TestCase):
    def test_official_architecture(self):
        config = qwen3_06b_config()
        self.assertEqual((config.hidden_size, config.num_hidden_layers,
                          config.num_attention_heads, config.num_key_value_heads,
                          config.head_dim), (1024, 28, 16, 8, 128))

    def test_future_visibility_and_padding(self):
        for backend in ['eager', 'sdpa']:
            with self.subTest(backend=backend), torch.no_grad():
                torch.manual_seed(42)
                config = tiny_config(backend)
                model = BidirectionalQwen3(config).eval()
                causal = Qwen3Model(config).eval()
                causal.load_state_dict(model.state_dict())
                original = torch.tensor([[1, 2, 3, 4]])
                changed = torch.tensor([[1, 2, 3, 5]])
                first = model(original).last_hidden_state
                second = model(changed).last_hidden_state
                self.assertGreater((first[:, 0] - second[:, 0]).abs().max().item(), 1e-6)
                torch.testing.assert_close(causal(original).last_hidden_state[:, 0],
                                           causal(changed).last_hidden_state[:, 0])
                for mask in [torch.tensor([[1, 1, 1, 0]]),
                             torch.tensor([[0, 1, 1, 1]])]:
                    other = original.clone()
                    other[mask == 0] = 8
                    a = model(original, attention_mask=mask).last_hidden_state
                    b = model(other, attention_mask=mask).last_hidden_state
                    torch.testing.assert_close(a[mask.bool()], b[mask.bool()])
                torch.testing.assert_close(first, model(inputs_embeds=model.embed_tokens(original)).last_hidden_state)
                self.assertIsNone(model(original).past_key_values)

    def test_gradients_and_invalid_inputs(self):
        model = BidirectionalQwen3(tiny_config())
        embeds = torch.randn(2, 4, 32, requires_grad=True)
        model(inputs_embeds=embeds).last_hidden_state.square().mean().backward()
        self.assertTrue(torch.isfinite(embeds.grad).all())
        self.assertGreater(embeds.grad.abs().sum().item(), 0)
        ids = torch.tensor([[1, 2]])
        for kwargs in [dict(use_cache=True), dict(past_key_values=object()),
                       dict(attention_mask=torch.zeros(1, 2)),
                       dict(attention_mask=torch.ones(1, 3)),
                       dict(attention_mask=torch.tensor([[1.0, 0.5]])),
                       dict(is_causal=True)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(ids, **kwargs)
        with self.assertRaises(ValueError):
            model()
        with self.assertRaises(ValueError):
            model(ids, inputs_embeds=model.embed_tokens(ids))

    def test_tts_vocabulary_registration_and_reload(self):
        for vocab_size in [32, 8]:
            with self.subTest(vocab_size=vocab_size):
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=Tokenizer(WordLevel(
                        {f't{i}': i for i in range(vocab_size)}, unk_token='t0')),
                    unk_token='t0', pad_token='t1', additional_special_tokens=['t2'],
                )
                model = BidirectionalQwen3(tiny_config()).eval()
                original = model.get_input_embeddings().weight.detach().clone()
                ids = model.register_tts_tokens(tokenizer)
                self.assertEqual(len(set(ids.values())), 9)
                self.assertEqual(tokenizer.pad_token, 't1')
                self.assertIn('t2', tokenizer.all_special_tokens)
                self.assertNotIn('<MASK>', tokenizer.get_vocab())
                for name, token in TTS_SPECIAL_TOKENS.items():
                    self.assertEqual(tokenizer.encode(token, add_special_tokens=False), [ids[name]])
                    self.assertIn(token, tokenizer.all_special_tokens)
                    self.assertEqual(tokenizer.decode([ids[name]], skip_special_tokens=True), '')
                torch.testing.assert_close(original, model.get_input_embeddings().weight[:32])
                weight = model.get_input_embeddings().weight.detach().clone()
                size = len(tokenizer)
                self.assertEqual(ids, model.register_tts_tokens(tokenizer))
                self.assertEqual(size, len(tokenizer))
                torch.testing.assert_close(weight, model.get_input_embeddings().weight)
                self.assertEqual(model.config.vocab_size, max(32, size))
                with tempfile.TemporaryDirectory() as path:
                    tokenizer.save_pretrained(path)
                    model.save_pretrained(path)
                    loaded, restored = BidirectionalQwen3.from_pretrained_with_tokenizer(path)
                    self.assertEqual(loaded.config.tts_token_ids, ids)
                    self.assertEqual(restored.get_vocab(), tokenizer.get_vocab())
                    inputs = torch.tensor([list(ids.values())])
                    torch.testing.assert_close(model(inputs).last_hidden_state,
                                               loaded.eval()(inputs).last_hidden_state)

    def test_pretrained_and_round_trip(self):
        with tempfile.TemporaryDirectory() as path, tempfile.TemporaryDirectory() as saved:
            causal = Qwen3ForCausalLM(tiny_config()).eval()
            causal.save_pretrained(path)
            model, info = BidirectionalQwen3.from_pretrained(path, output_loading_info=True)
            self.assertFalse(info['missing_keys'])
            for key, tensor in causal.model.state_dict().items():
                torch.testing.assert_close(tensor, model.state_dict()[key])
            model.eval()
            ids = torch.tensor([[1, 2, 3]])
            expected = model(ids).last_hidden_state
            model.save_pretrained(saved)
            loaded = BidirectionalQwen3.from_pretrained(saved).eval()
            torch.testing.assert_close(expected, loaded(ids).last_hidden_state)
            self.assertTrue(all(not layer.self_attn.is_causal for layer in loaded.layers))


if __name__ == '__main__':
    unittest.main()
