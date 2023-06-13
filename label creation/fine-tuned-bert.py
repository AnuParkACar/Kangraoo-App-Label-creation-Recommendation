from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AutoModel
import pandas as pd
import torch as pt
import random

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")

class Bert:
    def __init__(self) -> None:
        """
        Need to first get the encoding for each every single line of input, and then store in a array
        The same for attention mask, and labels
        Luckily, the labels are already in one-hot encoding so that shouldn't be that hard
        """
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.inputs = df[df.columns[0]].tolist()
        self.features = list()
        
    
    def createTrainingData(self):
        """
        Creates a list of tuples that each contain the inputID's, the attention masks, and the labels (one-hot encoded)
        """
        for i,input in enumerate(self.inputs):
            data = self.tokenizer.encode_plus(text=input,add_special_tokens=True,return_tensors='pt',return_attention_mask=True,padding=True)
            self.features.append((data["input_ids"],data["attention_mask"],pt.LongTensor(df.iloc[i,1:].values.tolist())))
    
    def gradientDescent(self,data:list):
        """
        Go over the batch, and update the parameters using gradient descent
        """
        optimizer = pt.optim.Adam(self.model.parameters(),lr=0.00005)
        for datum in data:
            inputs, attention_masks, labels = datum
            optimizer.zero_grad()
            outputs = self.model(input_ids=inputs,attention_mask=attention_masks,labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        

    def train(self):
        """
        Train the model by having it go over a batch size of 64, at a learning rate of 5e-5
        """
        self.model.train()
        random.shuffle(self.features)
        batch_size = 64
        epochs = 210

        
        #mini-batch approach
        for i in range(1,epochs):
            i = 0
            while len(self.features) - i < batch_size:
                pass



    

ok = Bert()
ok.createTrainingData()
ok.train()