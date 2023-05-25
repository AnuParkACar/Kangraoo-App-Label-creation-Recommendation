#import spacy
import re
import sys
#needed for data preproccessing
import pandas as pd
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word.synonym as naw
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#nlp = spacy.load("en_core_web_sm")

##create more data
df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")
aug = nas.ContextualWordEmbsForSentenceAug(model_path="distilgpt2")
#wordAug = naw.SynonymAug(aug_src='wordnet')
print(aug.augment("I's a hot and windy day here in delaware."))
#print("\n\n")
#print(wordAug.augment(df["Transcriptions"].iloc[0]))


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


