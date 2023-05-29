import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as dataPreprocessing
import neattext.functions as nfx

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")

featureText = df["Transcriptions"].apply(nfx.remove_stopwords,nfx.remove_non_ascii)

featureTransformer = dataPreprocessing.TfidfTransformer()
features = featureTransformer.fit_transform(featureText.values.reshape(-1,1))