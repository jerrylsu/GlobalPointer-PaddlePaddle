import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from tqdm import tqdm

import paddle
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieTokenizer, ErnieModel, RobertaTokenizer, RobertaModel, BertModel, BertTokenizer

from data_loader import MapDataset
from global_pointer_model import GlobalPointer, MetricsCalculator


train_data_path = './datasets/train.json'
eval_data_path = './datasets/dev.json'
device = paddle.get_device()

BATCH_SIZE = 6
EPOCHS = 10
LEARNING_RATE = 2e-5


# tokenizer
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
# tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
# tokenizer = BertTokenizer.from_pretrained('bert-wwm-chinese')

# train_data and val_data
train_dataset = MapDataset(train_data_path, tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate, shuffle=True, num_workers=1)
evl_dataset = MapDataset(eval_data_path, tokenizer=tokenizer)
evl_dataloader = DataLoader(evl_dataset, batch_size=BATCH_SIZE, collate_fn=evl_dataset.collate, shuffle=False, num_workers=1)

# model
encoder = ErnieModel.from_pretrained('ernie-1.0')
# encoder = RobertaModel.from_pretrained('roberta-wwm-ext')
# encoder = BertModel.from_pretrained('bert-wwm-chinese')

model = GlobalPointer(encoder, len(train_dataset.categories), 64)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=LEARNING_RATE)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1. - 2. * y_true) * y_pred         # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12          # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1. - y_true) * 1e12   # mask the pred outputs of neg classes
    zeros = paddle.zeros_like(y_pred[..., :1], )
    y_pred_neg = paddle.concat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = paddle.concat([y_pred_pos, zeros], axis=-1)
    neg_loss = paddle.logsumexp(y_pred_neg, axis=-1)
    pos_loss = paddle.logsumexp(y_pred_pos, axis=-1)
    return (neg_loss + pos_loss).mean()


def loss_fun(y_pred, y_true):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape((batch_size * ent_type_size, -1))
    y_pred = y_pred.reshape((batch_size * ent_type_size, -1))
    loss = multilabel_categorical_crossentropy(y_pred, y_true)
    return loss


metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
for epoch in range(EPOCHS):
    total_loss, total_f1 = 0., 0.
    for batch in tqdm(train_dataloader, desc=f"Trianing epoch {epoch}"):
        raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
        # input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
        #     device), segment_ids.to(device), labels.to(device)
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
    avg_loss = round(total_loss / len(train_dataloader), 3)
    avg_f1 = round(total_f1 / len(train_dataloader), 5)
    print(f"Epoch {epoch} Train_loss: {avg_loss} Train_f1: {avg_f1}\n")

    with paddle.no_grad():
        total_f1_, total_precision_, total_recall_ = 0., 0., 0.
        model.eval()
        for batch in tqdm(evl_dataloader, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            # input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
            #     device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids)
            f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r
        avg_f1 = round(total_f1_ / (len(evl_dataloader)), 5)
        avg_precision = round(total_precision_ / (len(evl_dataloader)), 5)
        avg_recall = round(total_recall_ / (len(evl_dataloader)), 5)
        print(f"Epoch {epoch} Eval_f1: {round(avg_f1, 5)} Precision: {round(avg_precision, 5)} Recall: {round(avg_recall, 5)}\n")

        if avg_f1 > max_f:
            paddle.save(model.state_dict(), './outputs/best_model.pth')
            max_f = avg_f1
        model.train()
