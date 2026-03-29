import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")

labels = ["Normal", "Suspicious", "Suspicious", "Dangerous", "Dangerous"]

# Mapping based on training folders
class_names = ["Normal_Videos", "Chasing", "Robbery", "Fighting", "Assault"]

IMG_SIZE = 64

def detect_threat(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    frame = np.reshape(frame, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(frame, verbose=0)
    class_index = np.argmax(prediction)

    return labels[class_index]