# Modified based on https://github.com/lm-sys/FastChat

import warnings
from typing import Optional, Tuple, List, Union

import torch
from torch import nn
import torch.nn.functional as F
import transformers
from einops import rearrange
from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func
)
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv, rotate_half

from flash_attn.bert_padding import unpad_input, pad_input
import math


group_size_ratio = 1/4

# save memory

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

def forward_mistral_for_causal_lm(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    assert labels is not None

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    torch.cuda.empty_cache()

    hidden_states = outputs[0]
    loss_fct = CrossEntropyLoss(reduction='sum')
    valid_seq_len = input_ids.shape[-1] - 1
    valid_seq_len_slide_win = torch.sum(labels[:, 1:] >= 0).item()
    # print("valid_seq_len_slide_win", valid_seq_len)
    loss = 0.0

    for start_idx in range(0, valid_seq_len, 16384):
        end_idx = min(start_idx + 16384, valid_seq_len)
        shift_logits = self.lm_head(hidden_states[..., start_idx:end_idx, :]).float()
        shift_labels = labels[..., start_idx + 1:end_idx + 1].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss += loss_fct(shift_logits, shift_labels)
        
    loss /= valid_seq_len_slide_win

    return CausalLMOutputWithPast(loss=loss)


def forward_mistral_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    residual = hidden_states.clone()
    batch, seq_len, embed_dim = hidden_states.shape

    for start_idx in range(0, seq_len, 16384):
        end_idx = min(seq_len, start_idx + 16384)
        hidden_states[:, start_idx:end_idx, :] = self.input_layernorm(hidden_states[:, start_idx:end_idx, :])
    # print("LN: A({}) R({}) M({})".format(
    #     torch.cuda.memory_allocated(0) / (1024 ** 3),
    #     torch.cuda.memory_reserved(0) / (1024 ** 3),
    #     torch.cuda.max_memory_reserved(0) / (1024 ** 3),
    # ))
    # torch.cuda.empty_cache()

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states
    # print("At: A({}) R({}) M({})".format(
    #     torch.cuda.memory_allocated(0) / (1024 ** 3),
    #     torch.cuda.memory_reserved(0) / (1024 ** 3),
    #     torch.cuda.max_memory_reserved(0) / (1024 ** 3),
    # ))
    # torch.cuda.empty_cache()

    # Fully Connected
    for start_idx in range(0, seq_len, 16384):
        end_idx = min(seq_len, start_idx + 16384)
        part_hidden_states = hidden_states[:, start_idx:end_idx, :].clone()
        part_hidden_states = self.post_attention_layernorm(part_hidden_states)
        part_hidden_states = self.mlp(part_hidden_states)
        hidden_states[:, start_idx:end_idx, :] += part_hidden_states
    # print("FC: A({}) R({}) M({})".format(
    #     torch.cuda.memory_allocated(0) / (1024 ** 3),
    #     torch.cuda.memory_reserved(0) / (1024 ** 3),
    #     torch.cuda.max_memory_reserved(0) / (1024 ** 3),
    # ) + '\n')
    # torch.cuda.empty_cache()

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def _make_sliding_window_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = torch.full(
        (tgt_len, tgt_len),
        fill_value=1,
        device=device,
    )
    mask = torch.tril(tensor, diagonal=0)
    # make the mask banded to account for sliding window
    mask = torch.triu(mask, diagonal=-sliding_window)
    mask = torch.log(mask).to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
# def _prepare_decoder_attention_mask(
#     self, attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window
# ):
#     # [bsz, seq_len]
#     return attention_mask

def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k

# only support this
def forward_flashattn_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    # print("forward_flashattn_inference mistral")
    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )

    # shape: (b, s, num_heads, head_dim)
    
    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len
    
    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # reuse k, v
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    if attention_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True, window_size=(self.config.sliding_window, self.config.sliding_window)
                                 ).view(
            bsz, q_len, -1
        )
    else:
        raise ValueError("attention_mask should be None")
    
    return self.o_proj(output), None, past_key_value


def forward_flashattn_inference_spliced(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )
    # print("forward_flashattn_inference")
    bsz, q_len, hidden_dim = hidden_states.size()

    cos_sin = self.rotary_emb(hidden_states, seq_len=q_len)

    attn_out = torch.empty_like(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    for head in range(self.num_heads):
        part_q = F.linear(hidden_states, self.q_proj.weight.view(self.num_heads, self.head_dim, hidden_dim)[head]).unsqueeze(2)
        part_k = F.linear(hidden_states, self.k_proj.weight.view(self.num_heads, self.head_dim, hidden_dim)[head]).unsqueeze(2)
        part_v = F.linear(hidden_states, self.v_proj.weight.view(self.num_heads, self.head_dim, hidden_dim)[head]).unsqueeze(2)
        part_q, part_k = apply_rotary_pos_emb_inference(part_q, part_k, cos_sin, position_ids)
        part_o = flash_attn_func(part_q, part_k, part_v, 0.0, softmax_scale=None, causal=True).view(bsz, q_len, self.head_dim)
        attn_out[:, :, head, :] = part_o

    torch.matmul(attn_out.view(bsz, q_len, hidden_dim), self.o_proj.weight.T, out=hidden_states)
    return (hidden_states, None, None)


def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask
    # return None

def replace_mistral_attn(use_flash_attn=True, use_full=False, inference=False,  aggressive_memory=False):
    if use_flash_attn:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        if inference:
            transformers.models.mistral.modeling_mistral.MistralModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
            if aggressive_memory:
                transformers.models.mistral.modeling_mistral.MistralAttention.forward = forward_flashattn_inference_spliced
            else:
                transformers.models.mistral.modeling_mistral.MistralAttention.forward = forward_flashattn_inference
        else:
            raise ValueError("only fit for inference")

    else:
        raise ValueError("only fit for falsh attn")

