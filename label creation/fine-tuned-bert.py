from transformers import BertTokenizer
from transformers import BertModel, BertConfig
import pandas as pd
import torch as pt
import random
import torch.nn as nn   

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")

class Bert:
    def __init__(self) -> None:
        """
        Need to first get the encoding for each every single line of input, and then store in a array
        The same for attention mask, and labels
        Luckily, the labels are already in one-hot encoding so that shouldn't be that hard
        """
        self.num_labels = len(df.columns)-1
        self.model = innerBertClassification(num_labels=self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.inputs = df[df.columns[0]].tolist()
        self.features = list()
        
    
    def createTrainingData(self):
        """
        Creates a list of tuples that each contain the inputID's, the attention masks, and the labels (one-hot encoded)
        """
        """
        for i,input in enumerate(self.inputs):
            if i == 0:
                data = self.tokenizer.encode_plus(text=input,add_special_tokens=True,return_tensors='pt',return_attention_mask=True,padding=True)
                self.features.append(data["input_ids"])
                self.features.append(data["attention_mask"])
                self.features.append(pt.LongTensor(df.iloc[i,1:].values.tolist()).unsqueeze(0))
            else:
                data = self.tokenizer.encode_plus(text=input,add_special_tokens=True,return_tensors='pt',return_attention_mask=True,padding=True)
                tensorLabels = pt.LongTensor(df.iloc[i,1:].values.tolist()).unsqueeze(0)
                self.features[0] = pt.cat((self.features[0],data["input_ids"]),dim=0)
                self.features[1] = pt.cat((self.features[1],data["attention_mask"]),dim=0)
                self.features[2] = pt.cat((self.features[2],tensorLabels),dim=0)
        """
        encoding = self.tokenizer(self.inputs,return_tensors="pt",padding=True,add_special_tokens=True)
        self.features.append(encoding["input_ids"])
        self.features.append(encoding["attention_mask"])
        self.features.append(pt.LongTensor(df.iloc[:,1:].values.tolist()))
        print("ok")
    
    def gradientDescent(self,inputs,attention_masks,labels):
        """
        Go over the batch, and update the parameters using gradient descent
        """
        labels = labels.float()
        self.optimizer = pt.optim.Adam(self.model.parameters(),lr=0.00005)
        self.optimizer.zero_grad()
        outputs = self.model(input_ids=inputs,attention_mask=attention_masks,labels=labels)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

        
    def train(self):
        """
        Train the model by having it go over a batch size of 64, at a learning rate of 5e-5
        """
        self.model.train()
        batch_size = 64
        epochs = 210

        
        #mini-batch approach
        for i in range(1,epochs):
            j = 0
            while (j + batch_size + 1)< len(self.inputs):
                if j + batch_size + 1 <= len(self.inputs):
                    inputs = self.features[0][j : (j + batch_size),:]
                    attention_masks = self.features[1][j : (j + batch_size),:]
                    labels = self.features[2][j : (j + batch_size),:]
                    self.gradientDescent(inputs,attention_masks,labels)
                    j+=batch_size + 1
                else:
                    inputs = self.features[0][j : (j+ (len(self.inputs) - j)),:]
                    attention_masks = self.features[1][j : (j+ (len(self.inputs) - j)),:]
                    labels = self.features[2][j : (j+ (len(self.inputs) - j)),:]
                    self.gradientDescent(inputs,attention_masks,labels)
                    j+=len(self.inputs) - j
    
    def saveState(self,directory):
        self.model.saveWeights(directory)

class innerBertClassification(nn.Module):
    def __init__(self,num_labels):
        super(innerBertClassification, self).__init__()
        self.config = BertConfig()
        self.config.num_labels = num_labels
        self.model = BertModel(config=self.config)
        self.dropout =nn.Dropout(self.config.hidden_dropout_prob)
        self.classification = nn.Linear(self.config.hidden_size,self.config.num_labels)

    def forward(self,input_ids,attention_mask,labels=None):
        outputs = self.model(input_ids=input_ids,attention_mask = attention_mask)
        self.poolerOutput = outputs.pooler_output
        self.pooledOutput = self.dropout(self.poolerOutput)
        logits = self.classification(self.pooledOutput)

        if labels is not None:
            lossFunction = nn.BCEWithLogitsLoss()
            loss = lossFunction(logits.view(-1,self.config.num_labels),labels.view(-1,self.config.num_labels))
            return loss
        else:
            return logits
        
    def saveWeights(self,directory):
        self.model.save_pretrained(directory)
    

ok = Bert()
ok.createTrainingData()
ok.train()
ok.saveState("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\bert_weights")