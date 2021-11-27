import paddle
import paddle.nn as nn
import numpy as np


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = paddle.greater_than(y_pred, paddle.to_tensor(0.)).cast('int64')
        return 2. * paddle.sum(y_true * y_pred) / paddle.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = paddle.greater_than(y_pred, paddle.to_tensor(0.)).cast('int64')
        return paddle.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


class GlobalPointer(nn.Layer):
    def __init__(self, encoder, entity_type_num, inner_dim, RoPE=True):
        # encoder: ernie-1.0 as encoder
        # inner_dim: 64
        # entity_type_num: ent_cls_num
        super(GlobalPointer, self).__init__()
        self.encoder = encoder
        self.entity_type_num = entity_type_num
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config['hidden_size']
        self.dense = nn.Linear(self.hidden_size, self.entity_type_num * self.inner_dim * 2)
        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = paddle.arange(0, seq_len, dtype='float32').unsqueeze(-1)

        indices = paddle.arange(0, output_dim // 2, dtype='float32')
        indices = paddle.pow(paddle.to_tensor(10000.), -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = paddle.stack([paddle.sin(embeddings), paddle.cos(embeddings)], axis=-1)
        embeddings = embeddings.tile((batch_size, *([1]*len(embeddings.shape))))
        embeddings = paddle.reshape(embeddings, (batch_size, seq_len, output_dim))
        # embeddings = embeddings.set_device(self.device)
        return embeddings
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # self.device = input_ids.palce()

        context_outputs = self.encoder(input_ids, token_type_ids=token_type_ids)  # , attention_mask=attention_mask)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.shape[0]
        seq_len = last_hidden_state.shape[1]

        # outputs:(batch_size, seq_len, entity_type_num*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = paddle.split(outputs, self.entity_type_num, axis=-1)
        # outputs:(batch_size, seq_len, entity_type_num, inner_dim*2)
        outputs = paddle.stack(outputs, axis=-2)
        # qw,kw:(batch_size, seq_len, entity_type_num, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].reshape((batch_size, seq_len, self.inner_dim // 2, 1)).tile((1, 1, 1, 2)).reshape((batch_size, seq_len, 1, self.inner_dim))
            sin_pos = pos_emb[..., None, ::2].reshape((batch_size, seq_len, self.inner_dim // 2, 1)).tile((1, 1, 1, 2)).reshape((batch_size, seq_len, 1, self.inner_dim))
            qw2 = paddle.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = paddle.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, entity_type_num, seq_len, seq_len)
        logits = paddle.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        pad_mask = paddle.expand(attention_mask, (batch_size, self.entity_type_num, seq_len, seq_len))
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = paddle.tril(paddle.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

