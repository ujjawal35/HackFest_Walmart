import cv2
import numpy as np
import time
cam=cv2.VideoCapture('http://192.168.42.129:8080/video')
pref="magi"
suf=0
while(True):
    _,frame=cam.read()
    frame=cv2.flip(frame,0)
    frame=cv2.flip(frame,1)
    frame2=frame.copy()
    cv2.putText(frame2,pref+str(suf)+".png",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow('HiHaHa',frame2)
    val=cv2.waitKey(1)&0xFF
    if val==ord(' '):
        suf=suf+1
        cv2.imwrite('E:\\images\\'+pref+str(suf)+".png",frame)
        for i in range(100000000000):
            if cv2.waitKey(1)&0xFF==ord(' '):
                break
    elif val==ord('b'):
        suf=suf-1
    elif val==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()