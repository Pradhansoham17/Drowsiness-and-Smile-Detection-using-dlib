import cv2
import dlib
from scipy.spatial import distance as dist

def smile(mouth):
    #D is the distance between leftmost and rightmost point of the mouth (b/w pts. 48 and 54)
    D = distance.euclidean(mouth[0], mouth[6])
    return D

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        mouth_points=[]

        for n in range(48,68):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	mouth_points.append((x,y))
        	next_point = n+1
        	if n == 67:
        		next_point = 48
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        smile_dist = smile(mouth_points)
        if smile_dist>60:
            cv2.putText(frame,"SMILING",(20,100),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
            print("Smiling")

        print(smile_dist)

    cv2.imshow("Smile", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()