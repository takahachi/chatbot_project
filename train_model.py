from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import torch

# === データ読み込み ===
df = pd.read_csv("slack_messages_realistic_10k.csv")

# データ分割
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["message_text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# === トークナイザ準備 ===
MODEL_NAME = "cl-tohoku/bert-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === Dataset クラス定義 ===
class SlackMessageDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SlackMessageDataset(train_texts, train_labels)
val_dataset = SlackMessageDataset(val_texts, val_labels)

# === モデル定義 ===
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# === Trainer API ===
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# === 学習開始 ===
trainer.train()

# === 保存 ===
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")