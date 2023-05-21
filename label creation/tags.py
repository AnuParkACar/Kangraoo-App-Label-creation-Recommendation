import spacy

#needed for data preproccessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#nlp = spacy.load("en_core_web_sm")


#class Tags:
#    def __init__(self, text) -> None:
#        self.nlp = spacy.load("en_core_web_sm")
 #       self.doc = self.nlp(text)

#    def get_tokens(self):
 #       for token in self.doc:
    #        print(token.text)



labels = ['In-person only', 'Remote only', 'Hybrid', 'No Preference', 'Full time', 'Part time', 'Internship', 'Acting / Performance', 'Analytics / Data', 'Building with hands', 'Design', 'Environment', 'Fashion', 'Food / Wine', 'Health / Wellness', 'Hospitality', 'Learning', 'Math', 'Meeting new people', 'Outdoors', 'Physical', 'Programming', 'Puzzles', 'Reading', 'Real Estate', 'Research', 'Service-oriented activities', 'Social media', 'Spending time with friends', 'Sports', 'Travel', 'Writing', 'Adaptable', 'Collaborative', 'Comfortable with ambiguity', 'Conflict management', 'Creative / Innovative', 'Critical thinking', 'Detail-oriented', 'Discreet', 'Emotionally intelligent', 'Friendly / Personable' ,'Interpersonal skills', 'Leadership', 'Multitasker', 'Organized', 'Persuasion', 'Problem-solving', 'Self-starter', 'Strong communicator', 'Works well under pressure', 'Accounting', 'Acting / Performings', 'Art / Design', 'Beauty', 'Film production', 'Finance', 'Home improvement', 'Management', 'Marketing', 'Multilingual', 'Photography', 'Presentation creation', 'Research', 'Sales', 'Social media', 'Teaching / Training', 'Tech-savvy', 'Verbal communication', 'Written communication']

#import csv and then split into x and y



#remove stop words, tokenize, and then vectorize the data



#create the test, and train split



#Actually train the model



#run prediction, and get loss and accuracy on the test set


