"""
Fused Linear Cross-Entropy Loss via quack.

Uses quack's chunked_linear_cross_entropy which avoids materializing
the full [BT, V] logits tensor. Chunked matmul + cross-entropy with
in-place gradient computation and tuned GEMM kernels.

Pads BT to a multiple of 8 when needed (quack's TMA-based GEMM requires it).

Also contains the replacement forward method for Qwen3ForCausalLM that
intercepts hidden states before lm_head to use the fused path.
"""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from quack.linear_cross_entropy import chunked_linear_cross_entropy


_ALIGN = 8


def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias=None,
    ignore_index: int = -100,
    chunk_size: int = 4096,
) -> torch.Tensor:
    if hidden_states.ndim == 3:
        B, T, H = hidden_states.shape
        hidden_states = hidden_states.view(-1, H)
        target = target.view(-1)

    BT, H = hidden_states.shape
    pad_needed = (-BT) % _ALIGN
    if pad_needed:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_needed))
        target = F.pad(target, (0, pad_needed), value=ignore_index)

    if bias is not None:
        hidden_states = F.linear(hidden_states, weight, bias)
        from quack.cross_entropy import cross_entropy
        return cross_entropy(hidden_states, target, ignore_index=ignore_index, reduction="mean")

    return chunked_linear_cross_entropy(
        hidden_states,
        weight,
        target,
        chunk_size=chunk_size,
        ignore_index=ignore_index,
        reduction="mean",
    )


def bastile_lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    return_dict: Optional[bool] = None,
    **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    logits = None
    loss = None

    if self.training and labels is not None:
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        B, T, H = shift_hidden.shape
        shift_hidden = shift_hidden.view(-1, H)
        shift_labels = shift_labels.view(-1)

        loss = fused_linear_cross_entropy(
            shift_hidden,
            self.lm_head.weight,
            shift_labels,
            bias=self.lm_head.bias if hasattr(self.lm_head, 'bias') and self.lm_head.bias is not None else None,
            ignore_index=-100,
        )
    else:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    from transformers.modeling_outputs import CausalLMOutputWithPast
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
