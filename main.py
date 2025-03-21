import cv2
import face_recognition
import datetime
import os

# Load the reference image and get its face encoding
reference_image = face_recognition.load_image_file("src/ref.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Open the video file
video_path = 'src/video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# List to store timestamps
timestamps = []

# Process the video frame by frame
frame_count = 0
frame_skip = 5  # Process every 5th frame
frames_to_skip_on_detection = 60  # Number of frames to skip when the face is detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        continue

    # Resize the frame to a smaller size for faster processing
    frame = cv2.resize(frame, (360, 640))

    # Convert the frame to RGB (face_recognition expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame to see if it matches the reference face
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if True in matches:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamp = str(datetime.timedelta(seconds=current_time))
            print("Specific face detected at timestamp:", timestamp)
            timestamps.append(timestamp)
            # Pause the video
            print("Pausing video. Press any key to continue...")
            cv2.imshow('Video', frame)
            cv2.waitKey(0)  # Wait for a key press to continue

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

# Print the timestamps
print("Timestamps of specific face appearances:")
for timestamp in timestamps:
    with open('timestamps.txt', 'a') as f:
        f.write(timestamp + '\n')
    print(timestamp)
    
# Open the timestamps.txt file with Windows Notepad
os.system('notepad.exe timestamps.txt')