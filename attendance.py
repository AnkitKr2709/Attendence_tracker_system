# from face_recognition.face_recognition.api 
# import load_image_file, face_locations, batch_face_locations, face_landmarks, face_encodings, compare_faces, face_distance
import cv2
import numpy as np
# import face_recognition_models
import face_recognition
import os
from datetime import datetime

path='images'
images=[]
personName=[]
myList = os.listdir(path)
# print(myList)
for cu_img in myList:
    current_Img = cv2.imread(os.path.join(path,cu_img))
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])
# print(personName)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("check")
encodeListKnown = faceEncodings(images)
print("All Encodings Complete!!!")

def attendance(name):
    with open('attendance.csv', 'r') as f:
        myDataList = f.readlines()
        #print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            with open('attendance.csv','a') as f2:
                time_now = datetime.now()
                tstr = time_now.strftime('%H:%M:%S')
                dstr = time_now.strftime('%d/%m/%y')
                print(dstr)
                f2.write(str(name)+','+str(tstr)+','+str(dstr))
                f2.write("\n")
                f2.close()
        f.close()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame  = face_recognition.face_encodings(faces)
    
    #print(facesCurrentFrame)
    #print(encodesCurrentFrame)
    if facesCurrentFrame!=[]:
        #print(facesCurrentFrame)
        y1,x2,y2,x1 = facesCurrentFrame[0]
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,0), 2)
        cv2.rectangle(frame, (x1,y2-35),(x2,y2),(0,0,0), cv2.FILLED)
        matches = face_recognition.compare_faces(encodeListKnown, encodesCurrentFrame[0])
        #print(matches)
        matchIndex = np.argmax(matches)
        #print(matchIndex)

        if matches[matchIndex]==True:
            name = personName[matchIndex].upper()
            print(name)
            cv2.putText(frame, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            attendance(name)
    '''for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        #print(encodeFace,faceLoc)
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.compare_faces(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmax(faceDis)
        print(matchIndex)
        if matches[matchIndex]==True:
            name = personName[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
            cv2.rectangle(frame, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            attendance(name)'''

    cv2.imshow("camera", frame)
    if cv2.waitKey(10) == 13:  # "enter key" ascii (13)
        break
cap.release()
cv2.destroyAllWindows()
