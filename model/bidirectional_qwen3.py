"""Bidirectional Qwen3 backbone for masked audio-token modeling.

Tested with transformers 5.14.1 and torch 2.11.0. Uses Qwen3Model's mask
mapping API and transformers.masking_utils.create_bidirectional_mask.

    model, tokenizer = BidirectionalQwen3.from_pretrained_with_tokenizer('Qwen/Qwen3-0.6B')
    output = model(inputs_embeds=embeddings, attention_mask=valid_positions)
    hidden = output.last_hidden_state  # [B, L, 1024]

Audio embeddings, text/reference packing, timestep conditioning and codec
prediction heads are intentionally left to the training architecture. The
inherited token embedding is Qwen's TEXT vocabulary, not a codec embedding.
"""

from copy import deepcopy

import torch
from transformers import AutoTokenizer, Qwen3Config
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

__all__ = ['BidirectionalQwen3', 'qwen3_06b_config', 'TTS_SPECIAL_TOKENS']

# These belong to Qwen's vocabulary, unlike codec MASK indices.
TTS_SPECIAL_TOKENS = {
    name: f'<{name}>' for name in (
        'AUDIO_PAD', 'REF_AUDIO_START', 'REF_AUDIO_END',
        'AUDIO_START', 'AUDIO_END', 'TEXT_START', 'TEXT_END',
        'REF_TEXT_START', 'REF_TEXT_END',
    )
}


def qwen3_06b_config() -> Qwen3Config:
    """Qwen/Qwen3-0.6B architecture, without downloading any weights.

    Source: https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
    """
    return Qwen3Config(
        vocab_size=151936, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, hidden_act='silu', max_position_embeddings=40960,
        rms_norm_eps=1e-6, rope_theta=1000000.0, attention_bias=False,
        attention_dropout=0.0, tie_word_embeddings=True,
        bos_token_id=151643, eos_token_id=151645,
        use_cache=False, use_sliding_window=False,
    )


class BidirectionalQwen3(Qwen3Model):
    """Qwen3 weights/blocks with full bidirectional, padding-aware attention.

    BidirectionalQwen3() constructs the 0.6B architecture with random weights.
    from_pretrained('Qwen/Qwen3-0.6B') loads the original backbone weights;
    the causal LM head is not used. A smaller Qwen3Config can be supplied for
    tests. Supported attention backends are eager and sdpa. KV caching is
    forbidden: every denoising step must recompute the complete sequence.

    attention_mask must be [B,L] with 1/True for valid positions and 0/False
    for padding. Corrupted MASK tokens are valid positions, NOT padding.
    Hidden states at padded query positions are unspecified; exclude them
    from downstream losses. All-padding sequences are rejected.
    """

    def __init__(self, config: Qwen3Config | None = None):
        config = deepcopy(config) if config is not None else qwen3_06b_config()
        config.use_cache = False
        if any(t != 'full_attention' for t in config.layer_types):
            raise ValueError('BidirectionalQwen3 requires full_attention in every layer')
        super().__init__(config)
        # A bidirectional mask may be None when no padding exists. SDPA must
        # not silently interpret that fast path as causal attention.
        for layer in self.layers:
            layer.self_attn.is_causal = False

    @classmethod
    def from_pretrained_with_tokenizer(cls, pretrained_model_name_or_path,
                                       *, tokenizer_kwargs=None, **model_kwargs):
        """Load both artifacts, register TTS tokens, and return (model, tokenizer).

        Pass tokenizer-specific options via tokenizer_kwargs. Save BOTH with
        save_pretrained to the same directory to preserve the token-ID mapping.
        """
        if model_kwargs.get('output_loading_info'):
            raise ValueError('output_loading_info is not supported by this paired loader')
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **(tokenizer_kwargs or {}))
        model = cls.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
        model.register_tts_tokens(tokenizer)
        return model, tokenizer

    def register_tts_tokens(self, tokenizer) -> dict[str, int]:
        """Idempotently extend a tokenizer and make every new ID embeddable.

        Existing special tokens, text PAD/EOS settings and embedding rows are
        preserved. AUDIO_PAD is a TTS-specific marker, not a replacement for
        the tokenizer's text pad_token. Pass an explicit attention_mask when
        padding a packed sequence. Do this before creating the optimizer.
        """
        existing = getattr(tokenizer, 'extra_special_tokens', None)
        if existing is None:
            existing = tokenizer.special_tokens_map.get('additional_special_tokens', [])
        tokens = list(dict.fromkeys([str(t) for t in existing] + list(TTS_SPECIAL_TOKENS.values())))
        tokenizer.add_special_tokens({'additional_special_tokens': tokens})
        ids = {name: tokenizer.convert_tokens_to_ids(token)
               for name, token in TTS_SPECIAL_TOKENS.items()}
        # Qwen's embedding table can have more rows than its tokenizer has
        # entries. Never shrink it or overwrite its pretrained rows.
        size = max(len(tokenizer), max(ids.values()) + 1,
                   self.get_input_embeddings().num_embeddings)
        if size > self.get_input_embeddings().num_embeddings:
            self.resize_token_embeddings(size, mean_resizing=False)
        self.config.tts_token_ids = ids
        self.config.tts_special_tokens = dict(TTS_SPECIAL_TOKENS)
        return ids

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        if past_key_values is not None or use_cache:
            raise ValueError('Bidirectional denoising does not support KV caching')
        if 'is_causal' in kwargs:
            raise ValueError('is_causal cannot be overridden for BidirectionalQwen3')
        if self.config._attn_implementation not in {'eager', 'sdpa'}:
            raise ValueError('BidirectionalQwen3 currently supports only eager and sdpa attention')
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Provide exactly one of input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] == 0:
            raise ValueError('inputs_embeds must have shape [B, positive L, hidden_size]')
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != inputs_embeds.shape[:2]:
                raise ValueError('attention_mask must be a tensor of shape [B,L]')
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError('attention_mask must contain only 0/1 values')
            if not attention_mask.bool().any(dim=1).all():
                raise ValueError('Every sequence must contain at least one valid token')
        mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # Supplying a mapping bypasses Qwen3Model's causal mask construction.
        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask={'full_attention': mask},
            position_ids=position_ids, use_cache=False, past_key_values=None,
            **kwargs,
        )
