import os
import cv2
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Paths for the dataset and where to save the files
dataset_path = 'dataset'  # Folder containing subfolders of faces
output_dir = 'output'     # Folder to save the generated files
embedding_model = "openface_nn4.small2.v1.t7"

# Load the face detector and embedder
detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
embedder = cv2.dnn.readNetFromTorch(embedding_model)

# Initialize the list of faces and labels
embeddings = []
labels = []

# Get the paths of all the images in the dataset
image_paths = list(paths.list_images(dataset_path))

# Process each image in the dataset
for image_path in image_paths:
    # Get the name of the person (folder name)
    label = image_path.split(os.path.sep)[-2]
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Detect face
    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300))
    detector.setInput(image_blob)
    detections = detector.forward()

    # Loop over detections and extract face embeddings
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]

            # Prepare the face for embedding
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0))
            embedder.setInput(face_blob)
            vec = embedder.forward()

            # Append the embeddings and the corresponding label
            embeddings.append(vec.flatten())
            labels.append(label)

# Convert labels to numbers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Train the face recognizer
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(embeddings, labels)

# Save the model and label encoder
print("Saving recognizer and label encoder...")
with open(os.path.join(output_dir, "recognizer.pickle"), "wb") as f:
    pickle.dump(recognizer, f)
with open(os.path.join(output_dir, "le.pickle"), "wb") as f:
    pickle.dump(le, f)

print("Training complete!")
