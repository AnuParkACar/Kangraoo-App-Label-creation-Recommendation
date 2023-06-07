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

def stopVideoCallback():
    #if camera.isOpened() == False:
        #print("Camera is not running yet")
    pass

rootWindow.geometry(newGeometry="700x700")
recordVideo = Button(master=rootWindow,command=recordVideoCallback,text="record").pack()
stopVideo = Button(master=rootWindow,command=stopVideoCallback,text="stop").pack()
rootWindow.mainloop()