import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as dataPreprocessing
import sklearn.model_selection as modelSelection
import neattext.functions as nfx
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
from generateCSV import GenerateLabels


class Classifier:
    def __init__(self) -> None:

        df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv", delimiter=",", encoding="utf-8")

        featureText = df["Transcriptions"].apply(
            nfx.remove_stopwords, nfx.remove_non_ascii)

        featureTransformer = dataPreprocessing.TfidfVectorizer()
        self.fittedVectorizer = featureTransformer.fit(featureText)
        #features = featureTransformer.fit_transform(featureText).toarray()
        features = self.fittedVectorizer.transform(featureText).toarray()
        outputs = df[df.columns[1:].tolist()].values

        x_train, x_test, y_train, y_test = modelSelection.train_test_split(
            features, outputs, test_size=0.2)

        self.classifier = ClassifierChain(classifier=SVC())
        self.classifier.fit(X=features, y=outputs)

    def predict(self, transcript: str) -> str:
        textVector = self.fittedVectorizer.transform([transcript]).toarray()

        labels = GenerateLabels()
        predictedLabels = self.classifier.predict(textVector).toarray()
        ok = predictedLabels[0].tolist()
        return (labels.generateListOfLabels(ok))
