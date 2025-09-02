import cv2
import matplotlib.pyplot as plt
from simple_facerec import SimpleFacerec
import os
from datetime import datetime
import time
import numpy as np

# Directory to save unknown faces
unknown_faces_dir = "unknown"

# Cooldown for capturing faces (in seconds)
cooldown_time = 10
last_capture_time = 0

# Log file path
log_file_path = "detection_log.txt"

# Set to keep track of logged entries
logged_entries = set()

# Function to save unknown face
def save_unknown_face(frame):
    global last_capture_time
    current_time = time.time()
    if current_time - last_capture_time >= cooldown_time:
        last_capture_time = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.join(unknown_faces_dir, f"{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"Unknown face captured and saved as {filename}")
        log_entry = f"Unknown person found at {timestamp}"
        if log_entry not in logged_entries:
            logged_entries.add(log_entry)
            with open(log_file_path, "a") as log_file:
                log_file.write(log_entry + "\n")

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load YOLOv3 for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
layer_ids = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in layer_ids]

# Load Camera
cap = cv2.VideoCapture(0)

# Create figure for plotting
fig, ax = plt.subplots()

# Initialize counters for logging
unknown_persons_count = 0
unknown_persons_with_bottles_count = 0
known_faces_count = 0
known_faces_with_bottles_count = 0

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        if name == "Unknown":
            unknown_persons_count += 1
        else:
            known_faces_count += 1

    # Detect Objects (Keys)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 39:  # Class ID for "bottle"
                # Get coordinates for the rectangle
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.putText(frame, "Bottle", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

                if name == "Unknown":
                    unknown_persons_with_bottles_count += 1
                else:
                    known_faces_with_bottles_count += 1

    # Detect Unknown Faces and Save
    if "Unknown" in face_names:
        save_unknown_face(frame)

    # Log entries
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Unknown persons found: {unknown_persons_count}\n")
        log_file.write(f"Unknown persons found with bottles: {unknown_persons_with_bottles_count}\n")
        log_file.write(f"Known faces found: {known_faces_count}\n")
        log_file.write(f"Known faces found with bottles: {known_faces_with_bottles_count}\n")

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Update the image in the plot
    ax.imshow(frame)

    # Clear the previous plot
    plt.pause(0.1)
    ax.clear()

plt.close()
cap.release()

