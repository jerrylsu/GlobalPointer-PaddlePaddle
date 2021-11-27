import json
import paddle
from paddle.io import Dataset
from paddlenlp.datasets import MapDataset
import numpy as np


MAX_LEN = 1024


class MapDataset(Dataset):
    def __init__(self, data_path, tokenizer, istrain=True):
        super(MapDataset, self).__init__()
        self.categories = set()
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.istrain = istrain

    def __len__(self):
        return len(self.data)

    def load_data(self, filename):
        """加载数据
        单条格式：[text, (start, end, label), (start, end, label), ...]，
                  意味着text[start:end + 1]是类型为label的实体。
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l.strip())
                d = [l[0]]
                for item in l[1:]:
                    self.categories.add(item[2])
                    d.append(tuple(item))
                D.append(d)
        self.categories = list(sorted(self.categories))
        return D

    def encoder(self, item):
        if self.istrain:
            text = item[0]
            # token2char_span_mapping = self.tokenizer(text,
            #                                          return_offsets_mapping=True,
            #                                          max_length=MAX_LEN,
            #                                          truncation=True)["offset_mapping"]
            token2char_span_mapping = self.tokenizer.get_offset_mapping(text)[:MAX_LEN - 2]
            token2char_span_mapping = self.tokenizer.build_offset_mapping_with_special_tokens(offset_mapping_0=token2char_span_mapping)
            start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            encoder_txt = self.tokenizer.encode(text, max_seq_len=MAX_LEN, return_attention_mask=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]

            return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask
        else:
            #TODO 测试
            pass

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)
            labels = np.zeros((len(self.categories), MAX_LEN, MAX_LEN))
            for start, end, label in item[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    label = self.categories.index(label)
                    end = end_mapping[end]
                    labels[label, start, end] = 1
            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
        batch_inputids = paddle.to_tensor(self.sequence_padding(batch_input_ids), dtype='int64')
        batch_segmentids = paddle.to_tensor(self.sequence_padding(batch_segment_ids), dtype='int64')
        batch_attentionmask = paddle.to_tensor(self.sequence_padding(batch_attention_mask), dtype='float32')
        batch_labels = paddle.to_tensor(self.sequence_padding(batch_labels, seq_dims=3), dtype='int64')

        return raw_text_list, batch_inputids, batch_attentionmask, batch_segmentids, batch_labels

    def __getitem__(self, index):
        item = self.data[index]
        return item

