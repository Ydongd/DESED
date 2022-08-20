import logging
from torch.utils.data._utils.collate import default_collate
import torch
from torch.utils.data import TensorDataset, Dataset
import re
import os

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, sentences, events, sent_start, sent_end):
        self.guid = guid
        self.sentences = sentences
        self.events = events
        self.sent_start = sent_start
        self.sent_end = sent_end

class InputFeature(object):
    def __init__(self, guid, input_ids, token_type_ids, attention_mask, valid_mask, label_ids):
        self.guid = guid
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.valid_mask = valid_mask
        self.label_ids = label_ids
        
    @staticmethod
    def collate_fct(batch):
        r'''
        This function is used to collate the input_features.
        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.
        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        '''
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        batch_lens = torch.sum(batch_tuple[2], dim=-1, keepdim=False)
        max_len = batch_lens.max().item()
        results = ()
        for item in batch_tuple:
            if item.dim() >= 2:
                results += (item[:, :max_len],)
            else:
                results += (item,)
        return results

def convert_examples_to_features(
    examples,
    data_processor,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    features = []
    for example in examples:
        guid = example.guid
        # get tokens of each sentence
        tokens = []
        valid_mask = []
        sentences = example.sentences

        for index, sentence in enumerate(sentences):
            words = [sentence[i] for i in range(len(sentence))]
            # for ace05
            # words = sentence
            for word in words:
                word_tokens = tokenizer.tokenize(word)
                # Chinese may have space for separate, use unk_token instead
                if word_tokens == []:
                    word_tokens = [tokenizer.unk_token]
                for i, word_token in enumerate(word_tokens):
                    if i == 0:
                        valid_mask.append(1)
                    else:
                        valid_mask.append(0)
                    tokens.append(word_token)
            # add [SEP]
            if index != len(sentences) - 1:
                valid_mask.append(1)
                tokens.append(sep_token)

        # get label_ids of each sentence
        out_token = data_processor.out_token
        label_ids = [out_token] * len(tokens)
        for event_mention in example.events:
            if event_mention['start'] >= max_seq_length:
                continue
            for i in range(event_mention['start'], min(event_mention['end'], max_seq_length)):
                trigger_type = event_mention['event_type']
                if i == event_mention['start']:
                    if label_ids[i] == out_token:
                        label_ids[i] = 'B-{}'.format(trigger_type)
                else:
                    if label_ids[i] == out_token:
                        label_ids[i] = 'I-{}'.format(trigger_type)
        label_ids = [data_processor.label2idx[i] for i in label_ids]
        # assign pad_token_label_id(-100) to sentences except user review
        sent_start = example.sent_start
        sent_end = example.sent_end
        for i in range(len(label_ids)):
            if i < sent_start or i >= sent_end:
                label_ids[i] = pad_token_label_id
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        valid_mask.append(1)
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            valid_mask.append(1)
        segment_ids = [sequence_a_segment_id] * len(tokens)
        # add cls token
        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            valid_mask.append(1)
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            valid_mask.insert(0, 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            valid_mask = ([0] * padding_length) + valid_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            valid_mask += [0] * padding_length
        while (len(label_ids) < max_seq_length):
            label_ids.append(pad_token_label_id)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid_mask) == max_seq_length
    
        features.append(
            InputFeature( guid=guid,
                          input_ids=input_ids,
                          token_type_ids=segment_ids,
                          attention_mask=input_mask,
                          valid_mask=valid_mask,
                          label_ids=label_ids)
        )
    
    return features

def load_examples(args, data_processor, tokenizer, split):
    logger.info("Loading and converting data from data_utils.py...")
    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = data_processor.get_examples(split=split)
    features = convert_examples_to_features(
        examples,
        data_processor,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_type_id,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_valid_mask, all_label_ids)
    return dataset