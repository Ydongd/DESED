import pandas as pd
from utils.data_utils import InputExample
from utils.const import TRIGGERS
import os
import json

class DataProcessor:
    def __init__(self, data_path, num_sentence, BIO_tagging=True):
        out_token = 'O'

        all_labels = [out_token]
        for label in TRIGGERS:
            if BIO_tagging:
                all_labels.append('B-{}'.format(label))
                all_labels.append('I-{}'.format(label))
            else:
                all_labels.append(label)

        label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
        idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

        self.data_path = data_path
        self.out_token = out_token
        self.all_labels = all_labels
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.num_sentence = num_sentence
    
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.json'.format(split))
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        for item in data:
            id = item['id']
            sentences = item['sentences'][:self.num_sentence]
            # for ace05
            # sentences = item['tokens_list'][:self.num_sentence]
            events = item['events']
            sent_start = item['sent_start']
            sent_end = item['sent_end']
            example = InputExample(guid=str(id), sentences=sentences, events=events, sent_start=sent_start, sent_end=sent_end)
            examples.append(example)
        return examples
    
