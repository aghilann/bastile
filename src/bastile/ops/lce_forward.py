"""
Replacement forward method for Qwen3ForCausalLM that uses fused linear cross-entropy.

When fused_linear_cross_entropy is enabled, this replaces the standard forward method
to skip logits materialization during training. Instead, hidden states are passed
directly to the fused linear cross-entropy function.

During inference (no labels), falls back to standard logits computation.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from .fused_linear_cross_entropy import fused_linear_cross_entropy


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
    """Qwen3ForCausalLM.forward with fused linear cross-entropy.

    During training (labels provided), skips logits materialization and
    computes loss directly from hidden states using chunked fused linear CE.

    During inference, falls back to standard logits computation.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Run the transformer model
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
        # Training path: use fused linear cross-entropy (skip logits materialization)
        # Shift labels for next-token prediction: predict token n from position n-1
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten to 2D
        B, T, H = shift_hidden.shape
        shift_hidden = shift_hidden.view(-1, H)
        shift_labels = shift_labels.view(-1)

        # Fused linear CE: never materializes full [BT, V] logits
        loss = fused_linear_cross_entropy(
            shift_hidden,
            self.lm_head.weight,
            shift_labels,
            bias=self.lm_head.bias if hasattr(self.lm_head, 'bias') and self.lm_head.bias is not None else None,
            ignore_index=-100,
        )
    else:
        # Inference path: compute logits normally
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            # Eval with labels but not training
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
