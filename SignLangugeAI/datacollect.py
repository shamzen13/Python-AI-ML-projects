import os
import cv2

# Define the directory for storing data
DATA_DIR = './data'

# Check if the directory exists, if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes and dataset size
number_of_classes = 3
dataset_size = 100

# Initialize the camera capture
cap = cv2.VideoCapture(0)

# Loop through each class to collect data
for class_id in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    
    # Create a directory for the class if it doesn't exist
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Collecting data for class {class_id}")

    # Wait for the user to press 'Q' to start capturing frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display a message prompting the user
        cv2.putText(frame, 'Press "Q" to capture', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture frames for the class
    for counter in range(dataset_size):
        ret, frame = cap.read()
        if not ret:
            break

        # Show the captured frame
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

        # Save the captured frame to the respective class directory
        img_filename = os.path.join(class_dir, f"{counter}.jpg")
        cv2.imwrite(img_filename, frame)

    print(f"Finished collecting data for class {class_id}")

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
