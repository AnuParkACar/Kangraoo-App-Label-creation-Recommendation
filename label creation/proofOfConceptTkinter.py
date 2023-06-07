from tkinter import *
import cv2 as cv

rootWindow = Tk()



def recordVideoCallback():
    camera = cv.VideoCapture(0)
    focc = cv.VideoWriter_fourcc(*"mp4v")
    recorder = cv.VideoWriter('output.mp4',focc,30.0,(640,480))

    while camera.isOpened():
        ret,frame = camera.read()
        recorder.write(frame)
        cv.imshow("frame",frame)
        if cv.waitKey(1) == ord('q'):
            break
    recorder.release()
    camera.release()
    cv.destroyAllWindows()
    i = 0
    while(i < 5):
        Label(master=rootWindow,text="label:{}".format(i)).pack()
        i+=1


rootWindow.geometry(newGeometry="700x700")
recordVideo = Button(master=rootWindow,command=recordVideoCallback,text="Record and Generate Labels").pack()



rootWindow.mainloop()