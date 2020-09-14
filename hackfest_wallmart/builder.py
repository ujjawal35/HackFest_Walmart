import integrate
import VideoCapture
import time
import cv2
import numpy
import serial
import pandas as pd
cap = VideoCapture.VideoCapture('http://192.168.42.129:8080/video')
crtwt=0
inv=0
quant =[0 for i in range(5)]
ser=serial.Serial('COM3')
ds=pd.read_csv("product.csv")
ds=ds.as_matrix()
while True:
    frame=cap.read()
    cv2.imshow("jjk",frame)
    cv2.waitKey(1)
    wt=integrate.getWT(ser)
    wtchng=wt-crtwt
    if integrate.getMotion(frame)==1:
        pred1=integrate.getML(frame)
        pred2=integrate.getQR(frame)
        if pred2==-1:
            pred2=pred1
        if pred2==-1:
            continue
        if abs(wtchng)<15 and pred2==-1:
            continue
        if wtchng<15:
            #object is being removed
            time.sleep(0.5)
            nwt = integrate.getWT(ser)
            if abs(nwt - crtwt + ds[pred2][4]) <= 15:
                print(ds[pred2][3]+" removed")
                quant[pred2]=quant[pred2]-1
                crtwt=nwt
        else:
            ctime=time.time()
            #timeout for object addition
            #integrate.initiateTimeout()
            while time.time()<ctime+30:
                nwt=integrate.getWT(ser)
                if nwt-crtwt>15:
                    time.sleep(0.5)
                    nwt=integrate.getWT(ser)
                    if abs(nwt-crtwt-ds[pred2][4])<=15:
                        quant[pred2]+=1
                        #integrate.objectAdded()
                        print(ds[pred2][3]+" added")
                        crtwt=nwt
                        break
    if abs(integrate.getWT(ser)-crtwt)>15 and inv==0:
        print("Cart becomes invalid")
        inv=1
    if inv==1 and abs(integrate.getWT(ser)-crtwt)<15:
        print("Cart becomes valid")
        inv = 0

            #integrate.finishTimeout();






