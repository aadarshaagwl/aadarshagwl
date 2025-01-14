import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import cvzone

# Load the trained helmet detection model
model = load_model("new_helmet_detection_cnn.h5")

# Load Haar Cascade for face detection(eyes,nose,etc)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define class names
class_names = ["With Helmet", "Without Helmet","Not Sure"]

# Create a directory to save cropped faces for debugging
debug_dir = "debug_cropped_faces"
os.makedirs(debug_dir, exist_ok=True)

# Input source: Webcam 
# 1. Webcam 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

# Parameters
IMG_SIZE = model.input_shape[1]  # Input size for the model
CONFIDENCE_THRESHOLD = 0.5
PADDING = 30  # Padding to expand bounding boxes

while True:
    success, frame = cap.read()
    if not success:
        print("Error reading frames")
        break

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces(finds and return bounding box cord)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for idx, (x, y, w, h) in enumerate(faces):
        # Expand the bounding box to include the helmet
        x_start = max(0, x - PADDING)
        y_start = max(0, y - PADDING)
        x_end = min(resized_frame.shape[1], x + w + PADDING)
        y_end = min(resized_frame.shape[0], y + h + PADDING)

        # Crop the expanded region
        face_roi = resized_frame[y_start:y_end, x_start:x_end]
        if face_roi.size == 0:
            continue

        # Save cropped face for debugging
        debug_path = os.path.join(debug_dir, f"expanded_face_{idx}.jpg")
        cv2.imwrite(debug_path, face_roi)

        # Prepare face for prediction
        resized_face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_array = np.expand_dims(resized_face / 255.0, axis=0)

        # Predict using the model
        predictions = model.predict(face_array, verbose=0)
        confidence = np.max(predictions)
        class_index = np.argmax(predictions)

        # Log prediction details
        print(f"Face {idx}: Predictions: {predictions}, Confidence: {confidence:.2f}, Class Index: {class_index}")

        if confidence > CONFIDENCE_THRESHOLD:
            # Draw bounding box and label
            color = (0, 255, 0) if class_index == 0 else (0, 0, 255)
            label = f"{class_names[class_index]} ({confidence * 100:.2f}%)"
            cvzone.cornerRect(resized_frame, (x_start, y_start, x_end - x_start, y_end - y_start), l=9, rt=2, colorR=color)
            cvzone.putTextRect(resized_frame, label, (x_start, y_start - 10), scale=1, thickness=1)

    # Display FPS(frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(resized_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Helmet Detection", resized_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
