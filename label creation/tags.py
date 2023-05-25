#import spacy
#needed for data preproccessing
import pandas as pd
import csv
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence.random as nar
import nlpaug.augmenter.word.back_translation as nab
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
russianAug = nab.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-ru',to_model_name='Helsinki-NLP/opus-mt-ru-en')
germanAug = nab.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-de',to_model_name='Helsinki-NLP/opus-mt-de-en')
frenchAug = nab.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-fr',to_model_name='Helsinki-NLP/opus-mt-fr-en')
randomAug = nar.RandomSentAug(mode="left")

i = 0
rows = df.shape[1]

while i < rows:
    switchedList = randomAug.augment(df.iloc[i][0],n=5)
    for count, value in enumerate(switchedList):
        with open("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",mode="a",newline="") as newFile:
            writer = csv.writer(newFile,delimiter=",")
            writer.writerow(subAug.augment[value] + df.iloc[i][1:].values.tolist())
            writer.writerow(insertAug.augment[value] + df.iloc[i][1:].values.tolist())
            writer.writerow(russianAug.augment[value] + df.iloc[i][1:].values.tolist())
            writer.writerow(germanAug.augment[value] + df.iloc[i][1:].values.tolist())
            writer.writerow(frenchAug.augment[value] + df.iloc[i][1:].values.tolist())
            newFile.close()
    i+=1





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


