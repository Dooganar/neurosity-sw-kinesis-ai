# -----------------------------------

data_path = "/home/reuben/Documents/remote-repos/neurosity-sw-kinesis-ai/data/eeg/output-balanced-classes-10000-or-more.json"
model_path = "/home/reuben/Documents/remote-repos/neurosity-sw-kinesis-ai/data/models/random_forest_model.joblib"

# -----------------------------------



dictionary_encoded = {
  34: 0, 7: 1, 2: 2, 8: 3,
        4: 4, 22: 5, 0: 6, 6: 7
}
reversed_dictionary_encoded = {v: k for k, v in dictionary_encoded.items()}
dictionary = {
  "rest": 0,
  "artifactDetector": 1,
  "leftArm": 2,
  "rightArm": 3,
  "leftHandPinch": 4,
  "rightHandPinch": 5,
  "tongue": 6,
  "jumpingJacks": 7,
  "leftFoot": 8,
  "rightFoot": 9,
  "leftThumbFinger": 10,
  "leftIndexFinger": 11,
  "leftMiddleFinger": 12,
  "leftRingFinger": 13,
  "leftPinkyFinger": 14,
  "rightThumbFinger": 15,
  "rightIndexFinger": 16,
  "rightMiddleFinger": 17,
  "rightRingFinger": 18,
  "rightPinkyFinger": 19,
  "mentalMath": 20,
  "bitingALemon": 21,
  "push": 22,
  "pull": 23,
  "lift": 24,
  "drop": 25,
  "moveLeft": 26,
  "moveRight": 27,
  "moveForward": 28,
  "moveBackward": 29,
  "rotateLeft": 30,
  "rotateRight": 31,
  "rotateClockwise": 32,
  "rotateCounterClockwise": 33,
  "disappear": 34
}
reversed_dict = {v: k for k, v in dictionary.items()}

dictionary = {
  34: 0, 7: 1, 2: 2, 8: 3,
        4: 4, 22: 5, 0: 6, 6: 7
}

import json
import numpy as np
import pickle
import pandas as pd

# Assuming x contains your features and y contains your classes


def parse_json(json_data):
    X = []
    y = []

    for key, value in json_data.items():
        encoded_array = value['x']
        decoded_array = pickle.loads(encoded_array.encode('latin1'))
        X.append(decoded_array)
        y.append(value['y'])

    X = np.array(X)
    y = np.array(y)

    return X, y

# # Example usage:
with open(data_path, 'r') as f:
    json_data = json.load(f)

x, y = parse_json(json_data)
x = np.reshape(x, (x.shape[0], -1))  # Reshape X to (10, 64)

# Combine x and y into a single DataFrame
df = pd.DataFrame(x)
df['y'] = y

# Calculate the counts of each class (y)
class_counts = df['y'].value_counts()

# Filter out classes with fewer than 10,000 occurrences
valid_classes = class_counts[class_counts >= 10000].index

# Remove rows where y has fewer than 10,000 occurrences
df_filtered = df[df['y'].isin(valid_classes)]
df_filtered.dropna(inplace=True)

# Separate x and y again
x = df_filtered.drop(columns=['y']).values
y = df_filtered['y'].values
y = np.vectorize(dictionary_encoded.get)(y)


# --------------------------------------------------------

import cupy as cp
from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, pickle

# Assuming X contains your 8x8 matrices and y contains the corresponding classes
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Normalizing the data
scaler = StandardScaler()
# with open('standard_scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Convert numpy arrays to cupy arrays for GPU acceleration
# X_train_gpu = cp.array(X_train)
X_test_gpu = cp.array(X_test)
# y_train_gpu = cp.array(y_train)
y_test_gpu = cp.array(y_test)

# Initializing and training the Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train_gpu, y_train_gpu)
clf = joblib.load(model_path)

# Evaluate the model
test_accuracy = clf.score(X_test_gpu, y_test_gpu)
print("Testing Accuracy:", test_accuracy)

# rfc_test_predictions = clf.predict(X_test_gpu)

# Obtain predictions on the test data
# rfc_test_predictions = clf.predict(X_test_gpu)


# from cuml import ForestInference
# 
# # Load the model using FIL directly (not joblib)
# clf = ForestInference.load(model_path, output_class=True)
# 
# def predict_in_batches(model, X_test, batch_size=100):
#     predictions = []
#     for i in range(0, X_test.shape[0], batch_size):
#         print(f"batch {i}")
#         batch = cp.array(X_test[i:i + batch_size])
#         preds = model.predict(batch)
#         predictions.append(cp.asnumpy(preds))  # keep memory use low
#     return np.concatenate(predictions)
# 
# rfc_test_predictions = predict_in_batches(clf, X_test, batch_size=100)
# 


