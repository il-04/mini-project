import numpy as np
import imutils
import pickle
import time
import cv2

# Define the paths for the models and files
embeddingModel = "openface_nn4.small2.v1.t7"
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5

# Load the face detector model
print("Loading face detector...")
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Load the face recognizer model
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Start the video stream
print("Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    # Read the frame from the video capture
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Prepare the frame for face detection
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300))
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Loop through the detected faces and recognize them
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf:
            # Calculate bounding box for the detected face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # If face dimensions are too small, skip this detection
            if fW < 20 or fH < 20:
                continue

            # Prepare the face for recognition
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0))
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Predict the class (person) and probability
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            text = "{} : {:.2f}%".format(name, proba * 100)

            # Position for text to be displayed on the frame
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            
            # Corrected line: Display text on the frame
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Show the frame with the recognized face and text
    cv2.imshow("Frame", frame)

    # Wait for key press; break if 'Esc' key (27) is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 'Esc' key
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
