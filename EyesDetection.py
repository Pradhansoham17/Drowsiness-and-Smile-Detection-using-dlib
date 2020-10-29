#import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
lefteye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
   ret, img = cap.read()
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.3, 5)
   eye = 0
   openEye = 0
   counter = 0

   for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # regions of interest
        roi_gray = gray[y:y + h, int((x+w)/2):x + w]
        roi_color = img[y:y + h, int((x+w)/2):x + w]

        openEyes = eye_cascade.detectMultiScale(roi_gray)
        AllEyes = lefteye_cascade.detectMultiScale(roi_gray)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 18)
        for (ex, ey, ew, eh) in openEyes:
            openEye += 1
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),2)

        for (ex, ey, ew, eh) in AllEyes:
            eye += 1
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 40),2)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

        #if (openEye != eye):
            #print ('alert')
   font=cv2.FONT_HERSHEY_SIMPLEX
   if (openEye != eye):
       text = 'Eyes: Closed'
       img=cv2.putText(img,text,(10,50),font,1,(0,255,255),2,cv2.LINE_AA)
       cv2.imshow('img', img)
   else:
       text = 'Eyes: Open'
       img = cv2.putText(img, text, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
       cv2.imshow('img', img)

   k = cv2.waitKey(30) & 0xff
   if k == 27:
       break


cap.release()
cv2.destroyAllWindows()
