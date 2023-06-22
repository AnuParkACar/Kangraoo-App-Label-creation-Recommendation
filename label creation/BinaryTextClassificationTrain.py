import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from typing import Dict
from sklearn.model_selection import train_test_split
import pyarrow as pa
from datasets import Dataset


def encode(row) -> Dict:
    text = row['input']
    # ensure that any numerical chars can be processed
    text = ' '.join(str(text).split())
    # tokenize
    encodings = tokenizer(text, padding="max_length",
                          truncation=True, max_length=512)
    # One hot encoding
    label = 1 if row['label'] == 'E' else 0

    encodings['label'] = label
    encodings['text'] = text

    return encodings


def get_processed_data(df: pd) -> list:
    processed_data = [encode(row) for _, row in df.iterrows()]
    return processed_data


def split_data(processed_data: list) -> tuple:
    df = pd.DataFrame(processed_data)
    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )
    return train_df, valid_df


df = pd.read_csv()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

processed_data = get_processed_data(df)

training_data, eval_data = split_data(processed_data)

# to Dataset objects, only object accepted by Trainer class
train_data_set = Dataset(pa.Table.from_pandas(training_data))
eval_data_set = Dataset(pa.Table.from_pandas(eval_data))

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=16,  # batch size per device during training
    weight_decay=0.01,               # strength of weight decay
    logging_steps=500,               # log every 500 steps
    save_strategy="epoch",           # save the model every epoch
    evaluation_strategy="epoch",     # evaluate the model every epoch
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_set,
    eval_dataset=eval_data_set,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./BinaryTextClassificationModel")
tokenizer.save_pretrained("./BinaryTextClassificationTokenizer")

trainer.evaluate()
