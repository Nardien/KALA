import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.bert.modeling_bert import (BertPreTrainedModel,
                                                    BertModel,
                                                    BertEmbeddings,
                                                    BertEncoder,
                                                    BertPooler,
                                                    BertLayer,
                                                    TokenClassifierOutput)
from models.KALA.KFM import KFM

class BertForExtractiveQA(BertPreTrainedModel):
    def __init__(self, config, args, entity_embeddings):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = CustomBertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.kfms = KFM(args, len(args.loc_layer), entity_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        mention_positions=None, # Belows are the additional inputs
        nodes=None,
        edge_index=None,
        edge_attr=None,
        graph_batch=None,
        local_indicator=None,
    ):
        # Build Input for Adaptor
        kfm_inputs = {
            'mention_positions': mention_positions,
            'nodes': nodes,
            'edge_index': edge_index,
            'graph_batch': graph_batch,
            'edge_attr': edge_attr,
            'local_indicator': local_indicator,
        }

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            kfms=self.kfms,
            kfm_inputs=kfm_inputs,
        )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (
            start_logits, 
            end_logits,
            sequence_output
        ) + (outputs[1:]) # Including sequence_output and pooled_output
        return ((total_loss,) + output) if total_loss is not None else output

class BertForNER(BertPreTrainedModel):
    def __init__(self, config, args, entity_embeddings):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = CustomBertModel(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.kfms = KFM(args, len(args.loc_layer), entity_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mention_positions=None, # Belows are the additional inputs
        nodes=None,
        edge_index=None,
        edge_attr=None,
        graph_batch=None,
        local_indicator=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Build Input for Adaptor
        kfm_inputs = {
            'mention_positions': mention_positions,
            'nodes': nodes,
            'edge_index': edge_index,
            'graph_batch': graph_batch,
            'edge_attr': edge_attr,
            'local_indicator': local_indicator,
        }

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            kfms=self.kfms,
            kfm_inputs=kfm_inputs,
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

""" Source of Below Code
https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""
class CustomBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = CustomBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        kfms=None,
        kfm_inputs=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            kfms=kfms,
            kfm_inputs=kfm_inputs,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output) + encoder_outputs[1:]

class CustomBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        kfms=None,
        kfm_inputs=None,
    ):
        all_hidden_states, all_attentions = None, None

        if output_attentions:
            all_attentions = []
        if output_hidden_states:
            all_hidden_states = []

        for i, layer_module in enumerate(self.layer):
            if kfms is not None and i in kfms[0].loc_layer:
                idx = kfms[0].loc_layer.index(i)
                kfm = kfms[idx]
            else:
                kfm = None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                kfm=kfm,
                kfm_inputs=kfm_inputs,
            )
            hidden_states = layer_outputs[0]
            
            if output_hidden_states:
                all_hidden_states.append(layer_outputs[0])
            if output_attentions:
                all_attentions.append(layer_outputs[1])

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

from transformers.models.bert.modeling_bert import (
                                        BertAttention,
                                        BertIntermediate, 
                                        BertOutput,
                                        BertSelfOutput,
                                        )
from transformers.modeling_utils import (apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer
                                        )
class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_scores=False,
        kfm=None,
        kfm_inputs=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # Knowledge-Conditioned Feature Modulation
        if kfm is not None:
            pre_gamma, pre_beta, post_gamma, post_beta = kfm(hidden_states, **kfm_inputs)
            attention_output = (1 + pre_gamma) * attention_output + pre_beta

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        # Knowledge-Conditioned Feature Modulation
        if kfm is not None:
            layer_output = (1 + post_gamma) * layer_output + post_beta

        outputs = (layer_output,) + outputs
        return outputs