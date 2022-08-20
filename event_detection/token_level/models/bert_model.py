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
        self.dropout = nn.Dropout(args.dropout_prob)

        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, valid_mask, label_ids, mode):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits=self.fc(sequence_output)

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
        active_labels = label_ids.contiguous().view(-1)[active_loss]

        loss_fn = CrossEntropyLoss() # ignore_index, default: -100
        loss = loss_fn(active_logits, active_labels)

        if mode == 'train':
            return loss
        else: # test
            return loss, logits, label_ids
