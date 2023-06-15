from transformers import AutoTokenizer, BertForQuestionAnswering
import torch
import pandas as pd
import os
from formatText import FormatText

# download CSV as dataframe and format data in the test test
file_name = os.path.abspath(
    "Kangraoo-App-Label-creation-Recommendation/label creation/Locationsdata.csv")
df = pd.read_csv(rf"{file_name}", header=None)
text = df.to_string().split("\n")
text = text[1:]  # skip the header
ft = FormatText(text)
text = ft.get_text()

# call pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = BertForQuestionAnswering.from_pretrained(
    "deepset/bert-base-cased-squad2")

# run model on the test set
question = "What is the intended work location?"
no_solution = 0
for sentence in text:
    inputs = tokenizer(question, sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0,
                                             answer_start_index: answer_end_index + 1]
    if not tokenizer.decode(predict_answer_tokens, skip_special_tokens=True):
        no_solution += 1
    print(tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))
print("Sentences with no solutions found: " +
      str(no_solution) + "out of a total: " + str(len(text)))
