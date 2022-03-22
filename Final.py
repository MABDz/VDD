import numpy as np
import cv2

capture = cv2.VideoCapture(0)

def multi_clahe(img, num):
    for i in range(num):
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4+i*2,4+i*2)).apply(img)
    return img

def image_detection():
    img = cv2.imread('test.jpeg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),2)

    final = multi_clahe(blur, 4)

    cv2.imwrite('image.png',final)
    cv2.imshow('image',final)
  
while(True):
      
    ret, frame = capture.read()
 
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(grayFrame,5)
    
    final = multi_clahe(blur, 4)
 
    cv2.imshow('Project VDD', final)
      
    if cv2.waitKey(1) == ord('q'):
        break
  
capture.release()
cv2.destroyAllWindows()

