import imutils
import time
import cv2
import csv
import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths

# File paths and model settings
embeddingModel = "openface_nn4.small2.v1.t7"
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
dataset = 'dataset'
cascade = 'haarcascade_frontalface_default.xml'
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# User input for name and roll number
Name = input("Enter your name: ").strip()
Roll_Number = int(input("Enter your Roll Number: "))

# Dataset path and subfolder for the user
sub_data = Name
path = os.path.join(dataset, sub_data)

# Step 1: Check if 'dataset' folder exists, if not, create it
if not os.path.isdir(dataset):
    os.mkdir(dataset)
    print(f"Parent directory '{dataset}' created.")

# Step 2: Check if user-specific folder exists, if not, create it
if not os.path.isdir(path):
    os.mkdir(path)
    print(f"Folder created for user: {sub_data}")
else:
    print(f"Folder for {sub_data} already exists.")

# Save user information in a CSV file
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)
csvFile.close()

# Step 3: Collect face images from webcam
print("Starting video stream to collect face data...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)
total = 0

while total < 50:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break  # Exit the loop if frame is invalid
    
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300))
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0))
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
            cv2.imwrite(p, frame)
            total += 1

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

# Step 4: Generate face embeddings and train recognizer
print("Generating face embeddings and training model...")
imagePaths = list(paths.list_images(dataset))
knownEmbeddings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96))
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())

print(f"Total embeddings generated: {len(knownEmbeddings)}")
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

# Step 5: Train the recognizer using SVM
print("Training recognizer...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(knownNames)

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(knownEmbeddings, labels)

with open(recognizerFile, "wb") as f:
    f.write(pickle.dumps(recognizer))

with open(labelEncFile, "wb") as f:
    f.write(pickle.dumps(labelEnc))

print("Training completed and model saved.")

# Step 6: Real-time face recognition
print("Starting real-time face recognition...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300))
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0))
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = labelEnc.classes_[j]
            text = "{} : {:.2f}%".format(name, proba * 100)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 'Esc' key to exit
        break

cam.release()
cv2.destroyAllWindows()
