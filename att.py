# from face_recognition.face_recognition.api
# import load_image_file, face_locations, batch_face_locations, face_landmarks, face_encodings, compare_faces, face_distance
import cv2
import numpy as np
# import face_recognition_models
import face_recognition
import os
from datetime import datetime
import smtplib as s

path = 'images'
images = []
personName = []
myList = os.listdir(path)
# print(myList)
for cu_img in myList:
    current_Img = cv2.imread(os.path.join(path, cu_img))
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])
# print(personName)


def mailSender(email):
    ob = s.SMTP('smtp.gmail.com', 587)
    ob.ehlo()
    ob.starttls()
    ob.login('randommister070@gmail.com', 'llkyfhjvwduoogit')
    subject = "Attendance"
    body = "Your attendance for today was marked."
    massage = "subject:{}\n\n{}".format(subject, body)
    listadd = [email]
    ob.sendmail('randommister070@gmail.com', listadd, massage)
    print("send mail")


dict1 = {"SUMIT": "2020ucs0096@iitjammu.ac.in", "AKS": "ankitkumarshah27@gmail.com",
         "MODI": "2020ucs0111@iitjammu.ac.in",
         "VIRAT": "2020ucs0115@iitjammu.ac.in","VINEETH":"2020ucs0115@iitjammu.ac.in",
         "ARPIT": "2020uee0137@iitjammu.ac.in","PRIYANSH":"2020ucs0111@iitjammu.ac.in"}


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
        # print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            with open('attendance.csv', 'a') as f2:
                time_now = datetime.now()
                tstr = time_now.strftime('%H:%M:%S')
                dstr = time_now.strftime('%d/%m/%y')
                print(dstr)
                print(name)
                f2.write(str(name)+','+str(tstr)+','+str(dstr))
                mailSender(dict1[name])
                f2.write("\n")
                f2.close()
        f.close()


cap = cv2.VideoCapture(0)
g=" "
while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces)

    # print(facesCurrentFrame)
    # print(encodesCurrentFrame)
    if facesCurrentFrame != []:
        # print(facesCurrentFrame)
        y1, x2, y2, x1 = facesCurrentFrame[0]
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 0, 0), cv2.FILLED)
        matches = face_recognition.compare_faces(encodeListKnown, encodesCurrentFrame[0])
        # print(matches)
        matchIndex = np.argmax(matches)
        # print(matchIndex)

        if matches[matchIndex] == True:
            name = personName[matchIndex].upper()
            # print(name)
            n1=name
            if(n1!=g):
                # mailSender(dict1[n1])
                g=n1
            
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow("camera", frame)
    if cv2.waitKey(10) == 13:  # "enter key" ascii (13)
        break
cap.release()
cv2.destroyAllWindows()
