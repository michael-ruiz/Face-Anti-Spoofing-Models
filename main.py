import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data/anti-spoofing.csv')
print(data.head())

# Create a new column for labels
def assign_label(row):
    # Assuming the following logic for labeling:
    # live_selfie and live_video are genuine (label 0)
    # printouts and replay are spoof (label 1)
    if 'live_selfie' in row['live_selfie'] or 'live_video' in row['live_video']:
        return 0  # Genuine
    elif 'printouts' in row['printouts'] or 'replay' in row['replay']:
        return 1  # Spoof
    else:
        return None  # Unknown

# Apply the function to create a new 'label' column
data['label'] = data.apply(assign_label, axis=1)

# Check the new DataFrame and label distribution
print(data.head())
print(data['label'].value_counts())

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Example usage
# Extract frames from the first video in the 'live_selfie' column
video_frames = extract_frames(data['live_selfie'][0])

print(data.columns)

# Load MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the data for training
# Extract frames for all videos in the 'live_selfie' column
X = np.array([extract_frames(video) for video in data['live_selfie']])
y = data['label'].values  # Use the newly created label column

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Compile the model again
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Continue training
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)