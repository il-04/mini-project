from collections.abc import Iterable  # Fix for DeprecationWarning
import numpy as np
import imutils
import pickle
import time
import cv2
import csv

# Function to flatten nested lists
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# File paths for embeddings, recognizer, and label encodings
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"  # Corrected variable name
labelEncFile = "output/le.pickle"
conf = 0.5

# Initialize face recognizer
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())  # Fixed typo here
le = pickle.loads(open(labelEncFile, "rb").read())
Roll_Number = ""
box = []

# Initialize face detector (Caffe model)
print("[INFO] loading face detector...")
prototxt = "deploy.prototxt"  # Path to your deploy.prototxt file
model = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to your res10_300x300_ssd_iter_140000.caffemodel file
detector = cv2.dnn.readNetFromCaffe(prototxt, model)  # Initialize the detector

# Initialize video stream
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    # Prepare image for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    # Detect faces
    detector.setInput(imageBlob)  # Ensure that `detector` is correctly initialized
    detections = detector.forward()
    
    # Process each detected face
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf:
            # Get the bounding box coordinates for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            
            # Skip small faces
            if fW < 20 or fH < 20:
                continue
            
            # Prepare face for embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0))
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            # Predict the identity of the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            # Check against student database (CSV file)
            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    box = np.append(box, row)
                    name = str(name)
                    if name in row:
                        person = str(row)
                        listString = str(box)
                        if name in listString:
                            singleList = list(flatten(box))
                            listlen = len(singleList)
                            Index = singleList.index(name)
                            name = singleList[Index]
                            Roll_Number = singleList[Index + 1]
            
            # Display name and roll number on frame
            text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow("Frame", frame)
    
    # Exit on pressing the 'Esc' key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
