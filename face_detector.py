import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # Width
cap.set(4,500) # Height
 
while(True):
    ret, img = cap.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor=1.5,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    
    for (x,y,w,h) in faces:
        cv2.square(img,(x,y),(x+w,y+h),(255,255,0),2) # -> Format <- cv2.rectangle(image, start_point, end_point, color, thickness)
       
        roi_gray = grey[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
    
    cv2.imshow('frame', img)
    cv2.imshow('gray', grey)
    
    k = cv2.waitKey(15) & 0xff
    if k == 27: # 'ESC = quit btn'
        break

cap.release()
cv2.destroyAllWindows()