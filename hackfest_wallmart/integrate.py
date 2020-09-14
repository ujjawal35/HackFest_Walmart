from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import time
import pandas as pd
from keras.models import load_model

cam_detector = load_model('camera_detect.h5')
object_detect = load_model('object_detect.h5')

import serial

def getQR(img):
    ds=pd.read_csv("product.csv")
    ds=ds.as_matrix()

    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decodedObjects = pyzbar.decode(im)
    if len(decodedObjects) > 0:
        str=decodedObjects[0].data.decode('utf-8')
        found=-1
        for i in range(15):
            if ds[i][0]==int(str):
                found=i
                break
        return found
    return -1

def getWT(ser):
    ser.write(str(chr(1)).encode())
    strr=ser.readline()
    strr=strr.decode('utf-8')
    return int(strr)

def process_img(img):
    img = img.resize(img,(244,244))
    img = img/255.0
    img = img[None]
    return img


def getML(img):
    dict={'0':1,'1':14,'2':8,'3':10,'4':12}
    img = process_img(img)
    pred = object_detect.predict(img)[0]
    mx=np.argmax(pred)
    for i in range(5):
        if pred[i]==mx:
            return dict[i]

def getMotion(img):
    img = process_img(img)
    motion = np.argmax(cam_detector.predict(img)[0])
    return  motion


