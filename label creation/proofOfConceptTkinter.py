from tkinter import *
import cv2 as cv
from videoToTranscript import VideoToTranscript
import os

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
    camera = cv.VideoCapture(0)
    focc = cv.VideoWriter_fourcc(*"mp4v")
    recorder = cv.VideoWriter(fileName,focc,30.0,(640,480))

    while camera.isOpened():
        ret,frame = camera.read()
        recorder.write(frame)
        cv.imshow("frame",frame)
        if cv.waitKey(1) == ord('q'):
            break
    recorder.release()
    camera.release()
    cv.destroyAllWindows()
    generateTranscript(os.path.join(os.getcwd(),fileName))



rootWindow.geometry(newGeometry="700x700")
recordVideo = Button(master=rootWindow,command=recordVideoCallback,text="Record and Generate Labels").grid(row=0,column=3)



rootWindow.mainloop()