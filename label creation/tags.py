#import spacy
#needed for data preproccessing
import pandas as pd
import csv
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence.random as nar
#import nlpaug.augmenter.word.word_embs as wordEmb
import nlpaug.augmenter.word.back_translation as nas


##create more data
df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")
##aug = nas.ContextualWordEmbsForSentenceAug(model_path="distilgpt2")
subAug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',action='substitute',aug_p=0.5)
insertAug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',action='insert',aug_p=0.5)
randomAug = nar.RandomSentAug(mode="left")

russianAug = nas.BackTranslationAug(from_model_name="Helsinki-NLP/opus-mt-en-ru",to_model_name="Helsinki-NLP/opus-mt-ru-en",device='cuda')
frenchAug = nas.BackTranslationAug(from_model_name="Helsinki-NLP/opus-mt-en-fr",to_model_name="Helsinki-NLP/opus-mt-fr-en",device='cuda')
vietAug = nas.BackTranslationAug(from_model_name="Helsinki-NLP/opus-mt-en-vi",to_model_name="Helsinki-NLP/opus-mt-vi-en",device='cuda')
chineseAug = nas.BackTranslationAug(from_model_name="Helsinki-NLP/opus-mt-en-zh",to_model_name="Helsinki-NLP/opus-mt-zh-en",device='cuda')

i = 0
rows = df.shape[1]


with open("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",mode="a",newline="",encoding='utf-8') as newFile:
    writer = csv.writer(newFile,delimiter=",")
    while i < rows:
        switchedList = randomAug.augment(df.iloc[i][0],n=2)
        for count, value in enumerate(switchedList):
            russian = russianAug.augment(value)
            french = frenchAug.augment(value)
            frenchinsert = insertAug.augment(french)
            viet = vietAug.augment(value)
            chinese = chineseAug.augment(value)
            chineseSub = subAug.augment(chinese)
            writer.writerow(russian + df.iloc[i][1:].values.tolist())
            writer.writerow(french + df.iloc[i][1:].values.tolist())
            writer.writerow(viet + df.iloc[i][1:].values.tolist())
            writer.writerow(chinese + df.iloc[i][1:].values.tolist())
            writer.writerow(frenchinsert + df.iloc[i][1:].values.tolist())
            writer.writerow(chineseSub + df.iloc[i][1:].values.tolist())
        i+=1
    newFile.close()
