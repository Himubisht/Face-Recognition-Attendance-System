import cv2
import os

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

face_id = input("Enter user ID: ")
name = input("Enter name: ")

project_path = r"Path"

# Folder will be created using ONLY name
dataset_path = os.path.join(project_path, "Datasets", name)

# ❌ If same name already exists → Stop
if os.path.exists(dataset_path):
    print("❌ This name already exists! Dataset not created.")
    exit()

# Create folder
os.makedirs(dataset_path)

cam = cv2.VideoCapture(0)
count = 0

print("Capturing faces... Look at the camera")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        cv2.imwrite(
            os.path.join(dataset_path, f"{count}.jpg"),
            gray[y:y+h, x:x+w]
        )

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow('image', img)

    if cv2.waitKey(100) & 0xff == 27:
        break
    elif count >= 20:
        break

cam.release()
cv2.destroyAllWindows()

print("✅ Dataset Created Successfully!")


###Train model###


import cv2
import numpy as np
from PIL import Image
import os
import pickle

project_path = r"Path"
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

    for person_name in sorted(os.listdir(path)):
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

            for (x,y,w,h) in faces:
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
recognizer.save(os.path.join(trainer_path, "trainer.yml"))

with open(os.path.join(trainer_path, "labels.pickle"), "wb") as f:
    pickle.dump(label_dict, f)

print("✅ Model Trained Successfully!")

### Attendance system ###


import cv2
import numpy as np
import csv
import os
from datetime import datetime
import pickle

project_path = r"Path"

def mark_attendance(name):
    attendance_folder = os.path.join(project_path, "Attendance data")

    # ✅ Create folder if it does not exist
    if not os.path.exists(attendance_folder):
        os.makedirs(attendance_folder)

    file_name = os.path.join(attendance_folder, "Attendance.csv")

    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y")
    time_string = now.strftime("%H:%M:%S")

    # ✅ Create file with header if not exists
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # ✅ Prevent duplicate entry for same day
    with open(file_name, "r", newline="") as f:
        reader = csv.reader(f)
        existing_data = list(reader)

        for row in existing_data:
            if len(row) > 1 and row[0] == name and row[1] == date_string:
                print("⚠ Attendance already marked today.")
                return

    # ✅ Append new attendance
    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_string, time_string])

    print("📁 Saved to:", file_name)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(project_path, "trainer", "trainer.yml"))

with open(os.path.join(project_path, "trainer", "labels.pickle"), "rb") as f:
    label_dict = pickle.load(f)

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

print("🔍 Recognizing...")

recognized_today = set()

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 90:
            name = label_dict[id]

            # Only mark once per run
            if name not in recognized_today:
                print(f"✅ Attendance Marked for {name}")
                mark_attendance(name)
                recognized_today.add(name)

            cv2.putText(img, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(img, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Attendance System", img)

    if cv2.waitKey(10) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
