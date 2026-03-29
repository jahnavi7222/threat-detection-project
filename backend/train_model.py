import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Dataset path
DATASET_PATH = "C:/Users/Jahnavi/OneDrive/Desktop/projects/mini_project/Threat_Detection_Project/dataset"

# Labels
labels = ["Normal_Videos", "Chasing", "Robbery", "Fighting", "Assault"]
label_map = {label: i for i, label in enumerate(labels)}

X = []
y = []

IMG_SIZE = 64

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)

        if len(frames) >= 20:  # limit frames per video
            break

    cap.release()
    return frames


# Load dataset
for label in labels:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        video_path = os.path.join(folder, file)

        frames = extract_frames(video_path)

        for frame in frames:
            X.append(frame)
            y.append(label_map[label])

# Convert to numpy
X = np.array(X) / 255.0
y = to_categorical(y, num_classes=len(labels))

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X, y, epochs=5, batch_size=32)

# Save model
model.save("model.h5")

print("✅ Model trained and saved!")