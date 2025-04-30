import os
import cv2
import numpy as np
import pickle
import csv
import imutils
import time
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from collections.abc import Iterable
from imutils import paths

# Function to flatten nested lists
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# Directory setup for dataset
dataset = 'dataset'
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Initialize the detector, embedder, and other resources
print("[INFO] loading face detector...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

# User input for name and roll number
Name = str(input("Enter your name: ")).strip()
Roll_Number = int(input("Enter your Roll Number: "))

# Create dataset folder for the user if it doesn't exist
sub_data = Name
path = os.path.join(dataset, sub_data)

if not os.path.isdir(dataset):
    os.mkdir(dataset)
if not os.path.isdir(path):
    os.mkdir(path)

# Save user information in a CSV file
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a') as csvFile:
    write = csv.writer(csvFile)
    write.writerow(info)
csvFile.close()

# Start capturing images for embeddings
total = 0
while total < 50:
    print("Capturing face image {}/50".format(total + 1))
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    # Prepare the frame for face detection
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Process each detected face
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Prepare face for embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0))
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            # Save embeddings and associated name
            p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
            cv2.imwrite(p, frame)
            total += 1

# Load the embeddings and train the face recognition model
print("[INFO] loading face embeddings...")
imagePaths = list(paths.list_images(dataset))
knownEmbeddings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
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
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0))
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())

# Train the recognizer
print("[INFO] encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(knownNames)
print("[INFO] training face recognizer...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(knownEmbeddings, labels)

# Save the trained model
with open(recognizerFile, "wb") as f:
    f.write(pickle.dumps(recognizer))

# Save the label encoder
with open(labelEncFile, "wb") as f:
    f.write(pickle.dumps(labelEnc))

print("[INFO] model trained and saved.")

# Now we perform live face recognition
print("[INFO] starting video stream...")
while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
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

            # Check student data from the CSV file
            Roll_Number = ""
            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    if name in row:
                        Roll_Number = row[1]

            text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    # Exit on pressing the 'Esc' key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()
