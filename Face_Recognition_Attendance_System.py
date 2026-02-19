
import cv2
import os

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_id = input("Enter user ID: ")
name = input("Enter name: ")

dataset_path = f'C:\\Users\\patha\\Downloads\\Face Recognition Attendance System\\Datasets\\Himanshu'
os.makedirs(dataset_path, exist_ok=True)

cam = cv2.VideoCapture(0)
count = 0

print("Capturing faces... Look at the camera")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        cv2.imwrite(f"{dataset_path}/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow('image', img)

    if cv2.waitKey(100) & 0xff == 27:
        break
    elif count >= 20:
        break

cam.release()
cv2.destroyAllWindows()
print("Dataset Created Successfully!")

import cv2
import numpy as np
from PIL import Image
import os
import pickle

project_path = r"C:\\Users\\patha\\Downloads\\Face Recognition Attendance System"

dataset_path = os.path.join(project_path, "Datasets")
trainer_path = os.path.join(project_path, "trainer")

if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    label_dict = {}
    current_id = 0

    for person_name in os.listdir(path):
        person_path = os.path.join(path, person_name)

        if not os.path.isdir(person_path):
            continue

        label_dict[current_id] = person_name

        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)

            try:
                PIL_img = Image.open(img_path).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
            except:
                continue

            faces = detector.detectMultiScale(img_numpy)

            if len(faces) > 0:
                (x,y,w,h) = faces[0]
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(current_id)

        current_id += 1

    return faceSamples, ids, label_dict


print("Loading dataset...")

faces, ids, label_dict = getImagesAndLabels(dataset_path)

if len(faces) == 0:
    print("❌ No faces found!")
    exit()

print("Training model...")

recognizer.train(faces, np.array(ids))

model_path = os.path.join(trainer_path, "trainer.yml")
recognizer.save(model_path)

labels_path = os.path.join(trainer_path, "labels.pickle")

with open(labels_path, 'wb') as f:
    pickle.dump(label_dict, f)

print("✅ Model Trained Successfully!")

import cv2
import numpy as np
import csv
import os
from datetime import datetime
import pickle

def mark_attendance(name):
    file_name = "C:\\Users\\patha\\Downloads\\Face Recognition Attendance System\\Attendance data\\Attendance.csv"
    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y")
    time_string = now.strftime("%H:%M:%S")

    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    with open(file_name, "r", newline="") as f:
        reader = csv.reader(f)
        existing_data = list(reader)

        for row in existing_data:
            if len(row) > 1 and row[0] == name and row[1] == date_string:
                return  

    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_string, time_string])


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

project_path = r"C:\\Users\\patha\\Downloads\\Face Recognition Attendance System"

with open(os.path.join(project_path, "trainer", "labels.pickle"), "rb") as f:
    label_dict = pickle.load(f)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

print("Recognizing...")

recognized = False

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70 and not recognized:
            name = label_dict[id]
            print("Access Granted:", name)
            mark_attendance(name)
            recognized = True 
            break

        else:
            name = "Unknown"

    if recognized:
        break

    cv2.putText(img, str(name), (x+5,y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('camera', img)

    if cv2.waitKey(10) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
