from tkinter import *
import pyaudio
import cv2 as cv
from videoToTranscript import VideoToTranscript
import os
import wave
import ffmpeg
from regressiveClassifierModel import Classifier

rootWindow = Tk()
rootWindow.title("KangarooStar App - Label Generation Demo")
mainFrame = Frame(master=rootWindow).grid()

def generateLabels(transcript:str)->str:
    classify = Classifier()
    prediction = classify.predict(transcript)
    return prediction

def generateTranscript(filePath : str):
    vt = VideoToTranscript("fakeLink")
    vt.setFileName(filePath)
    vt.mp4_to_mp3()
    vt.mp3_to_text()
    transcript = vt.get_transcript()
    Label(master=mainFrame,text="Generating transcript:\n").grid(row=4,column=3)
    Label(master=mainFrame,text=transcript,wraplength=500).grid(row=5,column=3)
    Label(master=mainFrame,text="Generating Labels",).grid(row=6,column=3)

    tags = generateLabels(transcript)
    Label(master=mainFrame,text=tags).grid(row=7,column=3)


def recordVideoCallback():
    fileName = 'output.mp4'
    waveOuputFileName = 'output.wav'
    camera = cv.VideoCapture(0)
    focc = cv.VideoWriter_fourcc(*"mp4v")
    recorder = cv.VideoWriter(fileName,focc,30.0,(640,480))
    finalOutput = "finalOutput.mp4"
    
    chunks = 1300
    format = pyaudio.paInt16
    numSeconds = 200
    rate = 44100
    channels = 2

    pa = pyaudio.PyAudio()

    stream = pa.open(input=True,format=format,channels=channels,rate=rate,frames_per_buffer=chunks)

    frames = []

    while camera.isOpened():
        ret,frame = camera.read()
        frames.append(stream.read(chunks))
        recorder.write(frame)
        cv.imshow("frame",frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    Label(master=mainFrame,text="Saving Video...").grid(row=1,column=3)
    Label(master=mainFrame,text="Saving Audio...").grid(row=2,column=3)
    stream.stop_stream()
    stream.close()
    recorder.release()
    camera.release()
    cv.destroyAllWindows()
    pa.terminate()


    Label(master=mainFrame,text="Merging Audio and Video...").grid(row=3,column=3)
    wf = wave.open(os.path.join(os.getcwd(),waveOuputFileName),'wb')
    wf.setnchannels(channels)
    wf.setframerate(rate)
    wf.setsampwidth(pa.get_sample_size(format))
    wf.writeframes(b"".join(frames))
    wf.close()

    inputVid = ffmpeg.input(fileName)
    inputAudio = ffmpeg.input(waveOuputFileName)
    ffmpeg.concat(inputVid,inputAudio,v=1,a=1).output(os.path.join(os.getcwd(),finalOutput)).run()

    generateTranscript(os.path.join(os.getcwd(),finalOutput))



rootWindow.geometry(newGeometry="700x700")
recordVideo = Button(master=mainFrame,command=recordVideoCallback,text="Record and Generate Labels").grid(row=0,column=3)



rootWindow.mainloop()