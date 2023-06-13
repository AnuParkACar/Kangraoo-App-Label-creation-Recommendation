from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import pandas as pd

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")

class Bert:
    def __init__(self) -> None:
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.inputs = df[df.columns[0]].tolist()
        self.features = list()
        """
        Need to first get the encoding for each every single line of input, and then store in a array
        The same for attention mask, and labels
        Luckily, the labels are already in one-hot encoding so that shouldn't be that hard
        """
    
    def createTrainingData(self):
        """
        Creates a list of tuples that each contain the inputID's, the attention masks, and the labels (one-hot encoded)
        """
        for i,input in enumerate(self.inputs):
            data = self.tokenizer.encode_plus(text=input,add_special_tokens=True,return_tensors='pt',return_attention_mask=True,padding=True)
            self.features.append((data["input_ids"],data["attention_mask"],df.iloc[i,1:].values.tolist()))

    

ok = Bert()
ok.createTrainingData()