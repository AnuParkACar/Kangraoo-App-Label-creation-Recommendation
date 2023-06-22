from transformers import BertForQuestionAnswering, BertTokenizer, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split


def prepare_data(examples):
    inputs = tokenizer([example['question'] for example in examples], [example['context']
                       for example in examples], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    start_positions = torch.tensor(
        [example['context'].index(example['answer']) for example in examples])
    end_positions = torch.tensor([example['context'].index(
        example['answer']) + len(example['answer']) for example in examples])
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs


model = BertForQuestionAnswering.from_pretrained('bert-base-cased')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

train_sent, eval_sent = train_test_split(sentences, test_size=0.2)

train_dataset = prepare_data(train_sent)
eval_dataset = prepare_data(eval_sent)

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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./SQuADModel")
tokenizer.save_pretrained("./SQuADTokenizer")

trainer.evaluate()
