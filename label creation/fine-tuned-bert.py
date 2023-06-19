from transformers import BertTokenizer
from transformers import BertModel, BertConfig
import pandas as pd
import torch as pt
import random
import torch.nn as nn
import torch.nn.functional as F
from generateCSV import GenerateLabels

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")
df = df.sample(frac=1).reset_index(drop=True) #shuffle data

class Bert:
    def __init__(self) -> None:
        """
        Need to first get the encoding for each every single line of input, and then store in a array
        The same for attention mask, and labels
        Luckily, the labels are already in one-hot encoding so that shouldn't be that hard
        """
        self.performanceLoss = float('inf')
        self.trainingSize = 0.8
        self.num_labels = len(df.columns)-1
        self.model = innerBertClassification(num_labels=self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.inputs = df[df.columns[0]].tolist()
        self.validationSet = list()
        self.features = list()
        
    
    def createTrainingData(self):
        """
        Loads the input_ids and the attention_masks for the training and the validation set. 
        """
        encoding = self.tokenizer(self.inputs,return_tensors="pt",padding=True,add_special_tokens=True)
        num_training_set = int(len(self.inputs)  * self.trainingSize)
        self.features.append(encoding["input_ids"][:num_training_set])
        self.features.append(encoding["attention_mask"][:num_training_set])
        self.features.append(pt.LongTensor(df.iloc[:,1:].values.tolist())[:num_training_set])
        self.validationSet.append(encoding["input_ids"][num_training_set:])
        self.validationSet.append(encoding["attention_mask"][num_training_set:])
        self.validationSet.append(pt.LongTensor(df.iloc[:,1:].values.tolist())[num_training_set:])
        print("ok")
    
    def gradientDescent(self,inputs,attention_masks,labels):
        """
        Go over the batch, and update the parameters using gradient descent
        """
        labels = labels.float()
        self.optimizer = pt.optim.Adam(self.model.parameters(),lr=0.00005)
        self.optimizer.zero_grad()
        outputs = self.model(input_ids=inputs,attention_mask=attention_masks,labels=labels)
        loss = outputs['loss']
        loss.backward()
        self.optimizer.step()

        
    def train(self):
        """
        Train the model by having it go over a batch size of 64, at a learning rate of 5e-5, with epochs set to 210.\n
        Will stop training when the performance starts decreasing
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
            
            if self.validateAndSave():
                self.saveState("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\bert_weights")
            else:
                break
            
    def validateAndSave(self) -> bool:
        """
        Checks if the loss on the validation set from the current weights is less than the last epoch.\n
        Saves model if true, and if not true, signals.\n
        Returns: True if saved state, False if performance decreases
        """
        self.model.eval()
        inputs = self.validationSet[0]
        attention_masks = self.validationSet[1]
        labels = self.validationSet[2].float()
        outputs = self.model(input_ids=inputs,attention_mask=attention_masks,labels=labels)
        loss = outputs['loss']

        current_loss = loss.item()
        if current_loss <= self.performanceLoss:
            self.performanceLoss = current_loss
            print("Loss: {}".format(self.performanceLoss))
            return True
        else:
            return False

    def saveState(self,directory):
        """
        Calls innerBertClassification.saveWeights() to save the current state of model
        """
        self.model.saveWeights(directory)

    def evaluate(self,path:str,input:str):
        enoding = self.tokenizer([input],return_tensors="pt",padding=True,add_special_tokens=True)
        self.model.model = BertModel.from_pretrained(path)
        self.model.model.eval()
        outputs = self.model.model(input_ids=enoding["input_ids"],attention_mask = enoding["attention_mask"])
        logits = self.model.applyFinalLayer(outputs)
        predictions = F.sigmoid(logits)
        predictions = predictions.tolist()
        labelGenerator = GenerateLabels()
        predictedLabels = labelGenerator.generateListOfLabels(dataList=predictions[0],confidence_level=0.65)
        return predictedLabels


        

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
        logits = self.applyFinalLayer(outputs)

        if labels is not None:
            lossFunction = nn.BCEWithLogitsLoss()
            loss = lossFunction(logits.view(-1,self.config.num_labels),labels.view(-1,self.config.num_labels))
            return {'loss': loss}
        else:
            return logits
        
    def applyFinalLayer(self,outputs):
        """
        Takes in a BaseModelOutputWithPoolingAndCrossAttentions object, and returns the logits.\n
        Applies dropout and final layer
        """
        poolerOutput = outputs.pooler_output
        pooledOutput = self.dropout(poolerOutput)
        logits = self.classification(pooledOutput)
        return logits
        
    def saveWeights(self,directory):
        """
        Acts as wrapper for BertModel.save_pretrained()
        """
        self.model.save_pretrained(directory)
    

ok = Bert()
#ok.createTrainingData()
#ok.train()

print(ok.evaluate(path="C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\bert_weights",input="Hi there! I'm Olivia Sander, an enthusiastic team player who has a passion for fitness and wellness. As a certified personal trainer, I'm keen on promoting health awareness and maintaining my clients' fitness levels. I believe in living an active lifestyle, frequently indulging in outdoor activities like hiking and sports. I love socializing and getting to know my clients personally to better help them achieve their goals. Currently, I'm searching for a hybrid role that would allow me to interact with my clients both online and in-person."))