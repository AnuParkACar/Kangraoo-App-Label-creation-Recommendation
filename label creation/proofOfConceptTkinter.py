from tkinter import *
import pyaudio
import cv2 as cv
from videoToTranscript import VideoToTranscript
import os
import wave
import ffmpeg
from regressiveClassifierModel import Classifier
import spacy
from transformers import AutoTokenizer, BertForQuestionAnswering
import torch


rootWindow = Tk()
rootWindow.title("KangarooStar App - Label Generation Demo")
mainFrame = Frame(master=rootWindow).grid()


def determine_location_candidates(transcript: str, nlp) -> list:
    doc = nlp(transcript)

    locations = [sentence for sentence in doc.sents if any(
        ent.label_ == "GPE" for ent in sentence.ents)]
    return locations


def generate_location(sentences: list, nlp) -> str:
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    model = BertForQuestionAnswering.from_pretrained(
        "deepset/bert-base-cased-squad2")
    question = "What is the intended job location?"

    for sentence in sentences:
        doc = nlp(sentence)
        inputs = tokenizer(question, sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0,
                                                 answer_start_index: answer_end_index + 1]
        decoded_answer = tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True)
        if decoded_answer and doc.ents and any(ent.label_ == "GPE" for ent in doc.ents):
            return "Desired Location: " + decoded_answer
    return "No desired location found"


def get_location(transcript: str) -> str:
    nlp = spacy.load('en_core_web_sm')
    sentences = determine_location_candidates(transcript, nlp)
    location = generate_location(sentences, nlp)
    return location


def generateLabels(transcript: str) -> str:
    classify = Classifier()
    prediction = classify.predict(transcript)
    # call get_location, will return the desired location label tag
    return prediction


def generateTranscript(filePath: str):
    vt = VideoToTranscript("fakeLink")
    vt.setFileName(filePath)
    vt.mp4_to_mp3()
    vt.mp3_to_text()
    transcript = vt.get_transcript()
    Label(master=mainFrame, text="Generating transcript:\n").grid(row=4, column=3)
    Label(master=mainFrame, text=transcript,
          wraplength=500).grid(row=5, column=3)
    Label(master=mainFrame, text="Generating Labels",).grid(row=6, column=3)

    tags = generateLabels(transcript)
    Label(master=mainFrame, text=tags).grid(row=7, column=3)


def recordVideoCallback():
    fileName = 'output.mp4'
    waveOuputFileName = 'output.wav'
    camera = cv.VideoCapture(0)
    focc = cv.VideoWriter_fourcc(*"mp4v")
    recorder = cv.VideoWriter(fileName, focc, 30.0, (640, 480))
    finalOutput = "finalOutput.mp4"

    chunks = 1300
    format = pyaudio.paInt16
    numSeconds = 200
    rate = 44100
    channels = 2

    pa = pyaudio.PyAudio()

    stream = pa.open(input=True, format=format, channels=channels,
                     rate=rate, frames_per_buffer=chunks)

    frames = []

    while camera.isOpened():
        ret, frame = camera.read()
        frames.append(stream.read(chunks))
        recorder.write(frame)
        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord('q'):
            break

    Label(master=mainFrame, text="Saving Video...").grid(row=1, column=3)
    Label(master=mainFrame, text="Saving Audio...").grid(row=2, column=3)
    stream.stop_stream()
    stream.close()
    recorder.release()
    camera.release()
    cv.destroyAllWindows()
    pa.terminate()

    Label(master=mainFrame, text="Merging Audio and Video...").grid(row=3, column=3)
    wf = wave.open(os.path.join(os.getcwd(), waveOuputFileName), 'wb')
    wf.setnchannels(channels)
    wf.setframerate(rate)
    wf.setsampwidth(pa.get_sample_size(format))
    wf.writeframes(b"".join(frames))
    wf.close()

    inputVid = ffmpeg.input(fileName)
    inputAudio = ffmpeg.input(waveOuputFileName)
    ffmpeg.concat(inputVid, inputAudio, v=1, a=1).output(
        os.path.join(os.getcwd(), finalOutput)).run()

    generateTranscript(os.path.join(os.getcwd(), finalOutput))


rootWindow.geometry(newGeometry="700x700")
recordVideo = Button(master=mainFrame, command=recordVideoCallback,
                     text="Record and Generate Labels").grid(row=0, column=3)


rootWindow.mainloop()
