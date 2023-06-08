from tkinter import *
import pyaudio
import cv2 as cv
from videoToTranscript import VideoToTranscript
import os
import wave
import ffmpeg

rootWindow = Tk()
rootWindow.title("KangarooStar App - Label Generation Demo")


def generateTranscript(filePath : str):
    vt = VideoToTranscript("fakeLink")
    vt.setFileName(filePath)
    vt.mp4_to_mp3()
    vt.mp3_to_text()
    transcript = vt.get_transcript()
    Label(master=rootWindow,text="Transcript:\n").grid(row=1,column=3)
    Label(master=rootWindow,text=transcript).grid(row=2,column=3)


def recordVideoCallback():
    fileName = 'output.mp4'
    waveOuputFileName = 'output.wav'
    camera = cv.VideoCapture(0)
    focc = cv.VideoWriter_fourcc(*"mp4v")
    recorder = cv.VideoWriter(fileName,focc,30.0,(640,480))
    finalOutput = "finalOutput.mp4"
    
    chunks = 1024
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
    
    stream.stop_stream()
    stream.close()
    recorder.release()
    camera.release()
    cv.destroyAllWindows()
    pa.terminate()

    wf = wave.open(waveOuputFileName,'wb')
    wf.setnframes(channels)
    wf.setframerate(rate)
    wf.setsampwidth(pa.get_sample_size(format))
    wf.writeframes(b"".join(frames))
    wf.close()

    inputVid = ffmpeg.input(fileName)
    inputAudio = ffmpeg.input(waveOuputFileName)
    ffmpeg.concat(inputVid,inputAudio,v=1,a=1).output(finalOutput)

    generateTranscript(os.path.join(os.getcwd(),finalOutput))



rootWindow.geometry(newGeometry="700x700")
recordVideo = Button(master=rootWindow,command=recordVideoCallback,text="Record and Generate Labels").grid(row=0,column=3)



rootWindow.mainloop()