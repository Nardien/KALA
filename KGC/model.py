import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import BertPreTrainedModel, BertModel

class BertForDocRED(BertPreTrainedModel):
    def __init__(self, config, num_labels, max_ent_cnt, with_naive_feature=False, entity_structure=False):
        super().__init__(config)
        self.num_labels = num_labels
        self.max_ent_cnt = max_ent_cnt
        self.with_naive_feature = with_naive_feature
        self.reduced_dim = 128
        # self.bert = BertModel(config, with_naive_feature, entity_structure)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dim_reduction = nn.Linear(config.hidden_size, self.reduced_dim)
        # self.feature_size = config.hidden_size
        self.feature_size = self.reduced_dim
        
        self.bili = nn.Bilinear(self.feature_size, self.feature_size, self.num_labels)
        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            ent_pos=None,
            ent_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            label_mask=None,
            feature_indices=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # ner_ids=ent_ner,
            # ent_ids=ent_pos,
            # structure_mask=structure_mask.float(),
        )
        # get sequence outputs
        outputs = outputs[0]

        # projection: dim reduction
        # outputs = torch.relu(self.dim_reduction(outputs))
        ent_rep = torch.matmul(ent_pos, outputs)
        ent_rep_before_reduction = ent_rep
        ent_rep = torch.relu(self.dim_reduction(ent_rep))

        # # prepare entity rep
        ent_rep_h = ent_rep.unsqueeze(2).repeat(1, 1, self.max_ent_cnt, 1)
        ent_rep_t = ent_rep.unsqueeze(1).repeat(1, self.max_ent_cnt, 1, 1)

        # # concate distance feature
        # if self.with_naive_feature:
        #     ent_rep_h = torch.cat([ent_rep_h, self.distance_emb(ent_distance)], dim=-1)
        #     ent_rep_t = torch.cat([ent_rep_t, self.distance_emb(20 - ent_distance)], dim=-1)

        ent_rep_h = self.dropout(ent_rep_h)
        ent_rep_t = self.dropout(ent_rep_t)
        logits = self.bili(ent_rep_h, ent_rep_t)
 
        # preds = (logits * label_mask.unsqueeze(-1)).max(dim=-1).indices
        logits = torch.softmax(logits, dim=-1)

        # loss_all_ent_pair = loss_fct(logits.view(-1, self.num_labels), label.float().view(-1, self.num_labels))
        # # loss_all_ent_pair: [bs, max_ent_cnt, max_ent_cnt]
        # # label_mask: [bs, max_ent_cnt, max_ent_cnt]
        # loss_all_ent_pair = loss_all_ent_pair.view(-1, self.max_ent_cnt, self.max_ent_cnt, self.num_labels)
        # loss_all_ent_pair = torch.mean(loss_all_ent_pair, dim=-1)
        # loss_per_example = torch.sum(loss_all_ent_pair * label_mask, dim=[1, 2]) / torch.sum(label_mask, dim=[1, 2])
        # loss = torch.mean(loss_per_example)

        # logits = torch.sigmoid(logits)
        return logits, ent_rep_before_reduction  # (loss), logits