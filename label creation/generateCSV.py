import pandas as pd
import numpy as np
import re
import csv

df = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",delimiter=",",encoding="utf-8")
labels = ['In-person only', 'Remote only', 'Hybrid', 'No Preference', 'Full time', 'Part time', 'Internship', 'Acting / Performance', 'Analytics / Data', 'Building with hands', 'Design', 'Environment', 'Fashion', 'Food / Wine', 'Health / Wellness', 'Hospitality', 'Learning', 'Math', 'Meeting new people', 'Outdoors', 'Physical', 'Programming', 'Puzzles', 'Reading', 'Real Estate', 'Research', 'Service-oriented activities', 'Social media', 'Spending time with friends', 'Sports', 'Travel', 'Writing', 'Adaptable', 'Collaborative', 'Comfortable with ambiguity', 'Conflict management', 'Creative / Innovative', 'Critical thinking', 'Detail-oriented', 'Discreet', 'Emotionally intelligent', 'Friendly / Personable' ,'Interpersonal skills', 'Leadership', 'Multitasker', 'Organized', 'Persuasion', 'Problem-solving', 'Self-starter', 'Strong communicator', 'Works well under pressure', 'Accounting', 'Acting / Performings', 'Art / Design', 'Beauty', 'Film production', 'Finance', 'Home improvement', 'Management', 'Marketing', 'Multilingual', 'Photography', 'Presentation creation', 'Research', 'Sales', 'Social media', 'Teaching / Training', 'Tech-savvy', 'Verbal communication', 'Written communication']
newInput = [x.lower() for x in labels]
def getInput():
    key = ""
    usersDict = {}

    while key != "stop":
        print("Enter text:\n")
        text = input()
        if text != "stop":
            labelList = list()
            labelKey = ""
            print("\n\nEnter Labels:\n")
            while labelKey !=  "stop":
                print("\nEnter Label: ")
                label = input()
                while label.lower() not in newInput and label.lower() != "stop":
                    print("\nNot acceptable label, try again: ")
                    label = input()
                if labelKey !=  "stop":
                    labelList.append(label)
                labelKey = label
            usersDict[text] = createLabels(labelList)
        else:
            print("\nStopped")
        key = text
    
    writeToCSV(usersDict)

def createLabels(inputLabel:list[str])->str:
    binaryList = []
    for label in labels:
        if inputLabel.count(label.lower()) > 0:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList

def removeSpace(string:str)->str:
    string = re.sub("[[]|[]]","",string)
    return re.sub("(?<=,)\s","",string)

def writeToCSV(userInfo:dict):
    with open("C:\\Users\\abhin\\OneDrive\\Desktop\\Computing\\Nautical-Internship\\dataPreProcessing\\Kangraoo-App-Label-creation-Recommendation\\transcript_data\\data.csv",'a',newline='') as file:
        for input in list(userInfo.keys()):
            writer = csv.writer(file,delimiter=',')
            writer.writerow([input]+userInfo[input])
        file.close()

getInput()
