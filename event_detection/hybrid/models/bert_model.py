from logging import log
import torch
import torch.nn as nn
from arguments import get_model_classes, get_args
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import re
from utils.model_utils import valid_sequence_output

class BertModel(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.label2id = args.label2id
        self.id2label = args.id2label
        self.tokenizer = tokenizer
        self.num_labels = args.num_labels

        self.bert = model_config['model'].from_pretrained(
            args.model_name_or_path
        )

        self.W_1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.W_2 = nn.Linear(self.bert.config.hidden_size * 2, 1)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.bert.config.hidden_size * 2, self.num_labels)

    
    def forward(self, batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_valid_mask, batch_label_ids, mode):
        # In a batch, the first sentence is a combination of all the following sentences
        batch_size = batch_input_ids.size(0)
        sequence_output = torch.tensor([]).cuda()
        all_attention_mask = torch.tensor([]).cuda()
        for i in range(batch_size):
            sent_id = 0 # int
            attention_mask = batch_attention_mask[i][0] # (max_seq_len)

            dialog_input_ids = batch_input_ids[i] # (num_dialog + 1, max_seq_len)
            dialog_token_type_ids = batch_token_type_ids[i]
            dialog_attention_mask = batch_attention_mask[i]
            outputs = self.bert(
                input_ids=dialog_input_ids,
                token_type_ids=dialog_token_type_ids,
                attention_mask=dialog_attention_mask
            )
            last_hidden_state = outputs['last_hidden_state'] # (num_dialog+1, max_seq_len, hidden_size)
            sent_hidden_state = last_hidden_state[0] # (max_seq_len, hidden_size)
            last_hidden_state = last_hidden_state[1:] # (num_dialog, max_seq_len, hidden_size)

            cls_state = last_hidden_state[:,0] # (num_dialog, hidden_size)
            sent_cls = cls_state[sent_id].unsqueeze(0) # (1, hidden_size)
            cls_state_ = self.W_1(cls_state)
            sent_weight = F.softmax(self.tanh(torch.mm(cls_state_, sent_cls.transpose(0, 1))), dim=0) # (num_dialog, 1)
            dialog_state = torch.sum(torch.mul(sent_weight, cls_state), dim=0) # (hidden_size)
            dialog_state_ = dialog_state.repeat(sent_hidden_state.size(0), 1) # (max_seq_len, hidden_size)
            sent_state_ = torch.cat((sent_hidden_state, dialog_state_), dim=-1) # (max_seq_len, hidden_size*2)
            gate_weight = self.sigmoid(self.W_2(sent_state_)) # (max_seq_len, 1)
            gate_weight_ = torch.ones_like(gate_weight) - gate_weight

            gate_state = torch.mul(gate_weight, sent_hidden_state) + torch.mul(gate_weight_, dialog_state_)
            new_sent_state = torch.cat((sent_hidden_state, gate_state), dim=-1) # (max_seq_len, hidden_size*2)

            sequence_output = torch.cat((sequence_output, new_sent_state.unsqueeze(0)), dim=0)
            all_attention_mask = torch.cat((all_attention_mask, attention_mask.unsqueeze(0)), dim=0)

        sequence_output, attention_mask = valid_sequence_output(sequence_output, batch_valid_mask, all_attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits=self.fc(sequence_output)

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
        active_labels = batch_label_ids.contiguous().view(-1)[active_loss]

        loss_fn = CrossEntropyLoss() # ignore_index, default: -100
        loss = loss_fn(active_logits, active_labels)

        if mode == 'train':
            return loss
        else: # test
            return loss, logits, batch_label_ids
        
