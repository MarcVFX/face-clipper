import cv2
import dlib
import face_recognition
import datetime
import numpy as np
import os
from video_slicer import slice_video  # Import the slice_video function

if dlib.DLIB_USE_CUDA:
    print("dlib is using CUDA")
else:
    print("dlib is not using CUDA")
    exit()

# Load the reference images and get their face encodings
reference_images = []
refs_directory = "src\\refs"
video_path = 'D:\\Windows\\Jdownloader\\01.mp4'

for filename in os.listdir(refs_directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        reference_images.append(os.path.join(refs_directory, filename))
reference_encodings = []
print("Reference images loaded:", reference_images)

# Set the start and end times in milliseconds
start_time_ms = (3*60*60 + 11*60 + 30) * 1000
end_time_ms = (6*60*60 + 15*60 + 00) * 1000   # 01:30:00
seconds_to_skip = 10  # Skip 10 seconds after detecting a face

for image_path in reference_images:
    reference_image = face_recognition.load_image_file(image_path)
    reference_encoding = face_recognition.face_encodings(reference_image)[0]
    reference_encodings.append(reference_encoding)

# Initialize dlib's CNN-based face detector (requires a pre-trained model file)
cnn_face_detector = dlib.cnn_face_detection_model_v1('src/mmod_human_face_detector.dat')
shape_predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("src/dlib_face_recognition_resnet_model_v1.dat")

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Jump to the start time
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

# List to store timestamps
timestamps = []

# Function to write timestamps to a file
def write_timestamp(timestamp):
    with open("timestamps.txt", "a") as file:
        file.write(timestamp + "\n")

# Get the video frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
frames_to_skip_on_detection = int(fps * seconds_to_skip)  # Number of frames to skip for exactly 10 seconds
# Process the video frame by frame
frame_count = int(start_time_ms * fps / 1000)
frame_skip = int(fps)  # Process every 1 second

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the current time on the screen
    current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    current_time_str = str(datetime.timedelta(milliseconds=current_time_ms))
    cv2.putText(frame, f"Time: {current_time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    if end_time_ms > 0:
        if current_time_ms > end_time_ms:
            print("The selected timestamps have been reached. Exiting...")
            break

    frame_count += 1

    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        continue

    # Resize the frame to a smaller size for faster processing
    # Get the aspect ratio of the video
    height, width = frame.shape[:2]
    aspect_ratio = width / height

    # Resize the frame while maintaining the aspect ratio
    if aspect_ratio > 1:
        frame = cv2.resize(frame, (640, 360))
    else:
        # Vertical video
        frame = cv2.resize(frame, (360, 640))

    # Convert the frame to RGB (dlib expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame using the CNN-based detector
    faces = cnn_face_detector(rgb_frame, 1)

    # Loop through each face found in the frame to see if it matches the reference face
    for face in faces:
        shape = shape_predictor(rgb_frame, face.rect)
        face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame, shape)
        face_encoding = np.array(face_encoding)  # Convert to NumPy array
        matches = face_recognition.compare_faces(reference_encodings, face_encoding)
        if True in matches:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamp = str(datetime.timedelta(seconds=int(current_time))) + ":00"
            print("Specific face detected at timestamp:", timestamp)
            timestamps.append(timestamp)
            write_timestamp(timestamp)  # Write the timestamp to the file
            slice_video(timestamp, video_path)  # Call the slice_video function to create a clip
            # Pause the video
            # print("Pausing video. Press any key to continue...")
            # cv2.imshow('Video', frame)
            # cv2.waitKey(0)  # Wait for a key press to continue

            # Jump 60 frames ahead
            new_frame_position = frame_count + frames_to_skip_on_detection
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_position)
            frame_count = new_frame_position

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Timestamps have been exported to timestamps.txt")