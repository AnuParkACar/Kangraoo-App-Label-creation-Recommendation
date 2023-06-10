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
from generateCSV import generateLabels

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")

featureText = df["Transcriptions"].apply(nfx.remove_stopwords,nfx.remove_non_ascii)


featureTransformer = dataPreprocessing.TfidfVectorizer()
fittedVectorizer = featureTransformer.fit(featureText)
#features = featureTransformer.fit_transform(featureText).toarray()
features = fittedVectorizer.transform(featureText).toarray()
outputs = df[df.columns[1:]].values


x_train,x_test,y_train,y_test = modelSelection.train_test_split(features,outputs,test_size=0.2,)

for i in range(y_train.shape[1]):
    unique_values = np.unique(y_train[:, i])
    if len(unique_values) < 2:
        print(f"Output column {i} has less than two unique values: {unique_values}")


print("\n\nLabel: {}".format(df.columns[51]))

classifier = ClassifierChain(classifier=SVC())
classifier.fit(X=x_train,y=y_train)
prediction = classifier.predict(x_test)
print(prediction.toarray())
print("\nAccuracy: " + str(accuracy_score(y_test,prediction.toarray())))
print("\nLoss: " + str(hamming_loss(y_test,prediction.toarray())))


textVector = fittedVectorizer.transform(["Hi, I'm Danielle, the HR manager at Nebula Fashion House. We're looking for a passionate and creative fashion designer to join our innovative team. This is a full-time, in-person role based in our vibrant New York studio. The ideal candidate should have an eye for aesthetics, be comfortable with ambiguity as trends are ever-changing, and should have excellent interpersonal skills to effectively communicate with our team and clients. A background in Art and Design will be highly advantageous for this role"]).toarray()

labels = generateLabels()
predictedLabels = classifier.predict(textVector).toarray()
print(labels.generateListOfLabels(predictedLabels[0]))