import imutils
import time
import cv2
import csv
import os

# Load the pre-trained classifier for face detection
cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)

# User input for name and roll number
Name = str(input("Enter your name: ")).strip()  # Remove extra spaces
Roll_Number = int(input("Enter your Roll Number: "))

# Set the path for the 'dataset' folder
dataset = 'dataset'
sub_data = Name  # Use the Name entered by the user

# Combine dataset folder path with user-specific folder name
path = os.path.join(dataset, sub_data)

# Step 1: Check if the 'dataset' folder exists. If not, create it.
if not os.path.isdir(dataset):
    os.mkdir(dataset)
    print(f"Parent directory '{dataset}' created.")

# Step 2: Check if the subfolder (user's folder) exists. If not, create it.
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

# Start the video stream
print("Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

total = 0
while total < 50:
    print(total)
    ret, frame = cam.read()  # Capture frame
    
    if not ret:
        print("Failed to grab frame.")
        break  # Exit the loop if the frame is invalid
    
    img = imutils.resize(frame, width=400)  # Resize the image to speed up processing

    # Detect faces in the image
    rects = detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and save the images
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, img)
        total += 1

    # Show the frame
    cv2.imshow("Frame", frame)
    
    # Wait for the 'q' key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
