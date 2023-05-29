#import spacy
#needed for data preproccessing
import pandas as pd
import csv
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence.random as nar
import nlpaug.augmenter.word.word_embs as wordEmb
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#nlp = spacy.load("en_core_web_sm")

##create more data
df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")
#aug = nas.ContextualWordEmbsForSentenceAug(model_path="distilgpt2")
subAug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',action='substitute',aug_p=0.5)
insertAug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',action='insert',aug_p=0.5)
randomAug = nar.RandomSentAug(mode="left")

i = 0
rows = df.shape[1]


with open("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",mode="a",newline="",encoding='utf-8') as newFile:
    writer = csv.writer(newFile,delimiter=",")
    while i < rows:
        switchedList = randomAug.augment(df.iloc[i][0],n=5)
        for count, value in enumerate(switchedList):
            sub = subAug.augment(value)
            insert = insertAug.augment(value)
            writer.writerow(sub + df.iloc[i][1:].values.tolist())
            writer.writerow(insert + df.iloc[i][1:].values.tolist())
        i+=1
    newFile.close()


#class Tags:
#    def __init__(self, text) -> None:
#        self.nlp = spacy.load("en_core_web_sm")
 #       self.doc = self.nlp(text)

#    def get_tokens(self):
 #       for token in self.doc:
    #        print(token.text)



#import csv and then split into x and y



#remove stop words, tokenize, and then vectorize the data


#create the test, and train split



#Actually train the model



#run prediction, and get loss and accuracy on the test set


