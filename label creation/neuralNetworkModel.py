import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as dataPreprocessing
import sklearn.model_selection as modelSelection
import neattext.functions as nfx

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")

featureText = df["Transcriptions"].apply(nfx.remove_stopwords,nfx.remove_non_ascii)


featureTransformer = dataPreprocessing.TfidfVectorizer()
features = featureTransformer.fit_transform(featureText).toarray()
outputs = df[df.columns[1:]].values

x_train,x_test,y_train,y_test = modelSelection.train_test_split(features,outputs,test_size=0.2,random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(70,activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=16,epochs=20)

model.evaluate(x_test,y_test)
