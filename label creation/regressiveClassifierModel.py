import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as dataPreprocessing
import sklearn.model_selection as modelSelection
import neattext.functions as nfx
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")

featureText = df["Transcriptions"].apply(nfx.remove_stopwords,nfx.remove_non_ascii)


featureTransformer = dataPreprocessing.TfidfVectorizer()
features = featureTransformer.fit_transform(featureText).toarray()
outputs = df[df.columns[1:]].values

x_train,x_test,y_train,y_test = modelSelection.train_test_split(features,outputs,test_size=0.2,random_state=42)
classifier = ClassifierChain(classifier=SVC())
classifier.fit(X=x_train,y=y_train)
prediction = classifier.predict(x_test)
print(prediction.toarray())
print("\nAccuracy: " + str(accuracy_score(y_test,prediction.toarray())))
print("\nLoss: " + str(hamming_loss(y_test,prediction.toarray())))

