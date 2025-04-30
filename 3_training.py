import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np

# File paths
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

# Loading face embeddings
print("Loading face embeddings...")
data = pickle.loads(open(embeddingFile, "rb").read())

# Debugging: Check unique names and embeddings shape
print("Unique names in dataset:", set(data["names"]))
print("Total number of names:", len(data["names"]))
print("Embeddings shape:", np.array(data["embeddings"]).shape)

# Encoding labels
print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

# Debugging: Check the number of unique labels
print("Number of unique labels:", len(set(labels)))

# Check if we have more than 1 class for training
if len(set(labels)) > 1:
    print("Training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    
    # Save the trained model
    with open(recognizerFile, "wb") as f:
        f.write(pickle.dumps(recognizer))
    
    # Save the label encoder
    with open(labelEncFile, "wb") as f:
        f.write(pickle.dumps(labelEnc))
    
    print("Training completed and model saved.")
else:
    print("Error: Not enough classes for training. Please add more data.")
