import logging
from torch.utils.data._utils.collate import default_collate
import torch
from torch.utils.data import TensorDataset, Dataset
import re
import os

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, sentences, triggers, sent_start, sent_end):
        self.guid = guid
        self.sentences = sentences
        self.triggers = triggers
        self.sent_start = sent_start
        self.sent_end = sent_end


class DialogDataset(Dataset):
    def __init__(self, all_input_ids, all_token_type_ids, all_attention_mask, all_valid_mask, all_label_ids):
        # all_* : (num_data, num_dialog+1, max_seq_len)
        # valid_mask, label_ids: (num_data, max_seq_len)
        # sent_id: (num_data)
        self.all_input_ids = all_input_ids
        self.all_token_type_ids = all_token_type_ids
        self.all_attention_mask = all_attention_mask
        self.all_valid_mask = all_valid_mask
        self.all_label_ids = all_label_ids

    def __getitem__(self, index):
        dialog_input_ids = self.all_input_ids[index] # (num_dialog+1, max_seq_len)
        dialog_token_type_ids = self.all_token_type_ids[index]
        dialog_attention_mask = self.all_attention_mask[index]
        valid_mask = self.all_valid_mask[index] # (max_seq_len)
        label_ids = self.all_label_ids[index]

        return dialog_input_ids, dialog_token_type_ids, dialog_attention_mask, valid_mask, label_ids
    
    def __len__(self):
        return len(self.all_input_ids)

def collate_fn(batch):
    batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_valid_mask, batch_label_ids = [list(item) for item in zip(*batch)]
    batch_size = len(batch)
    max_len = 0
    for dialog_attention_mask in batch_attention_mask:
        attention_lens = torch.sum(dialog_attention_mask, dim=-1, keepdim=False)
        max_len = max(max_len, attention_lens.max())
    for i in range(batch_size):
        batch_input_ids[i] = batch_input_ids[i][:, :max_len]
        batch_token_type_ids[i] = batch_token_type_ids[i][:, :max_len]
        batch_attention_mask[i] = batch_attention_mask[i][:, :max_len]
        batch_valid_mask[i] = batch_valid_mask[i][:max_len]
        batch_label_ids[i] = batch_label_ids[i][:max_len]
    
    batch_input_ids = torch.stack(batch_input_ids, dim=0).cuda()
    batch_token_type_ids = torch.stack(batch_token_type_ids, dim=0).cuda()
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0).cuda()
    batch_valid_mask = torch.stack(batch_valid_mask, dim=0).cuda()
    batch_label_ids = torch.stack(batch_label_ids, dim=0).cuda()
    
    return batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_valid_mask, batch_label_ids


def process_whole_sentence(
    sentences,
    triggers,
    sent_start,
    sent_end,
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
    tokens = []
    valid_mask = []

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
    
    out_token = data_processor.out_token
    label_ids = [out_token] * len(tokens)
    for event_mention in triggers:
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

    return input_ids, segment_ids, input_mask, valid_mask, label_ids


def process_sentence(
    sentence,
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
    tokens = []
    valid_mask = []
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
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]
    tokens += [sep_token]
    valid_mask.append(1)
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        valid_mask.append(1)
    segment_ids = [sequence_a_segment_id] * len(tokens)
    # add cls token
    if cls_token_at_end:
        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]
        valid_mask.append(1)
    else:
        tokens = [cls_token] + tokens
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
        valid_mask = ([0] * padding_length) + valid_mask
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        valid_mask += [0] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(valid_mask) == max_seq_length

    return input_ids, segment_ids, input_mask, valid_mask


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
    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_valid_mask = []
    all_label_ids = []

    for example in examples:
        guid = example.guid
        sentences = example.sentences
        triggers = example.triggers
        sent_start = example.sent_start
        sent_end = example.sent_end
        dialog_input_ids = []
        dialog_token_type_ids = []
        dialog_attention_mask = []

        # process the combination of the whole sentences in a dialog
        input_ids, token_type_ids, attention_mask, valid_mask, label_ids = process_whole_sentence(
            sentences,
            triggers,
            sent_start,
            sent_end,
            data_processor,
            max_seq_length,
            tokenizer,
            cls_token_at_end=cls_token_at_end,
            # xlnet has a cls token at the end
            cls_token=cls_token,
            cls_token_segment_id=cls_token_segment_id,
            sep_token=sep_token,
            sep_token_extra=sep_token_extra,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=pad_on_left,
            # pad on the left for xlnet
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,
        )
        dialog_input_ids.append(input_ids)
        dialog_token_type_ids.append(token_type_ids)
        dialog_attention_mask.append(attention_mask)
        all_valid_mask.append(torch.tensor(valid_mask, dtype=torch.long))
        all_label_ids.append(torch.tensor(label_ids, dtype=torch.long))

        # process sentences in a dialog
        for index, sentence in enumerate(sentences):
            input_ids, token_type_ids, attention_mask, valid_mask_ = process_sentence(
                sentence,
                data_processor,
                max_seq_length,
                tokenizer,
                cls_token_at_end=cls_token_at_end,
                # xlnet has a cls token at the end
                cls_token=cls_token,
                cls_token_segment_id=cls_token_segment_id,
                sep_token=sep_token,
                sep_token_extra=sep_token_extra,
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=pad_on_left,
                # pad on the left for xlnet
                pad_token=pad_token,
                pad_token_segment_id=pad_token_segment_id,
            )
            dialog_input_ids.append(input_ids)
            dialog_token_type_ids.append(token_type_ids)
            dialog_attention_mask.append(attention_mask)
            # get valid_mask and label_ids
        all_input_ids.append(torch.tensor(dialog_input_ids, dtype=torch.long))
        all_token_type_ids.append(torch.tensor(dialog_token_type_ids, dtype=torch.long))
        all_attention_mask.append(torch.tensor(dialog_attention_mask, dtype=torch.long))

    return all_input_ids, all_token_type_ids, all_attention_mask, all_valid_mask, all_label_ids


def load_examples(args, data_processor, tokenizer, split):
    logger.info("Loading data from data_utils.py...")
    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = data_processor.get_examples(split=split)
    all_input_ids, all_token_type_ids, all_attention_mask, all_valid_mask, all_label_ids = convert_examples_to_features(
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

    dataset = DialogDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_valid_mask, all_label_ids)

    return dataset
