from tqdm import tqdm

import paddle
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieTokenizer, ErnieModel

from data_loader import MapDataset, load_data
from GlobalPointer import GlobalPointer, MetricsCalculator


train_cme_path = './datasets/CMeEE_train.json' # CMeEE 训练集
eval_cme_path = './datasets/CMeEE_dev.json' # CMeEE 测试集
device = paddle.get_device()
BATCH_SIZE = 16

ENT_CLS_NUM = 9
print('1')
# tokenizer
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
print('2')
# train_data and val_data
ner_train = MapDataset(load_data(train_cme_path), tokenizer=tokenizer)
ner_loader_train = DataLoader(ner_train, batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True, num_workers=16)
ner_evl = MapDataset(load_data(eval_cme_path), tokenizer=tokenizer)
ner_loader_evl = DataLoader(ner_evl, batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False, num_workers=16)

# model
encoder = ErnieModel.from_pretrained('ernie-1.0')
print('3')
model = GlobalPointer(encoder, ENT_CLS_NUM, 64).to(device)  # 9个实体类型
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=2e-5)

print('finshed')

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12   # mask the pred outputs of neg classes
    zeros = paddle.zeros_like(y_pred[..., :1])
    y_pred_neg = paddle.concat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = paddle.concat([y_pred_pos, zeros], axis=-1)
    neg_loss = paddle.logsumexp(y_pred_neg, axis=-1)
    pos_loss = paddle.logsumexp(y_pred_pos, axis=-1)
    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
for eo in range(10):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(ner_loader_train):
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
        logits = model(input_ids, attention_mask, segment_ids)
        loss = loss_fun(logits, labels)
        loss.backward()
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/gradient_clip_cn.html
        # clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        optimizer.clear_grad()
        sample_f1 = metrics.get_sample_f1(logits, labels)
        total_loss += loss.item()
        total_f1 += sample_f1.item()

        avg_loss = total_loss / (idx + 1)
        avg_f1 = total_f1 / (idx + 1)
        print("trian_loss:", avg_loss, "\t train_f1:", avg_f1)

    with paddle.no_grad():
        total_f1_, total_precision_, total_recall_ = 0., 0., 0.
        model.eval()
        for batch in tqdm(ner_loader_evl, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids)
            f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r
        avg_f1 = total_f1_ / (len(ner_loader_evl))
        avg_precision = total_precision_ / (len(ner_loader_evl))
        avg_recall = total_recall_ / (len(ner_loader_evl))
        print("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1,avg_precision,avg_recall))

        if avg_f1 > max_f:
            paddle.save(model.state_dict(), './outputs/ent_model.pth'.format(eo))
            max_f = avg_f1
        model.train()
