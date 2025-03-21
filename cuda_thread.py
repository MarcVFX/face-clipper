import cv2
import dlib
import face_recognition
import datetime
import numpy as np
import os
import threading
import queue
from video_slicer import slice_video  # Import the slice_video function

cnn_face_detector = dlib.cnn_face_detection_model_v1('src/mmod_human_face_detector.dat')
shape_predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("src/dlib_face_recognition_resnet_model_v1.dat")

seconds_to_skip_when_detect = 60
seconds_to_skip_each_frame = 10


if dlib.DLIB_USE_CUDA:
    print("dlib is using CUDA")
else:
    print("dlib is not using CUDA")
    exit()


reference_images = []
refs_directory = "src\\refs"
video_paths = [
    # {'path': 'D:\\Windows\\Jdownloader\\BLAST SIX INVITATIONAL BOSTON - PLAYOFFS - DIA 1 (1080p_60fps_H264-128kbit_AAC).mp4', 'start_time_ms': (9*60*60 + 50*60 + 00) * 1000, 'end_time_ms': (11*60*60 + 38*60 + 00) * 1000},
    {'path': 'D:\\Windows\\Jdownloader\\SIX INVITATIONAL BOSTON 2025 - FASE DE GRUPOS - DIA 5 (1080p_60fps_H264-128kbit_AAC).mp4', 'start_time_ms': (11*60*60 + 4*60 + 00) * 1000, 'end_time_ms': (11*60*60 + 54*60 + 58) * 1000},
]

for filename in os.listdir(refs_directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        reference_images.append(os.path.join(refs_directory, filename))
reference_encodings = []
print("Reference images loaded:", reference_images)



for image_path in reference_images:
    reference_image = face_recognition.load_image_file(image_path)
    reference_encoding = face_recognition.face_encodings(reference_image)[0]
    reference_encodings.append(reference_encoding)

class VideoProcessor:
    def __init__(self, video_path, start_time_ms, end_time_ms):
        self.video_path = video_path
        self.start_time_ms = start_time_ms
        self.end_time_ms = end_time_ms
        self.timestamps = []

    def write_timestamp(self, timestamp):
        with open("timestamps.txt", "a") as file:
            file.write(timestamp + "\n")

    def process_video(self, reference_encodings, cnn_face_detector, shape_predictor, face_recognition_model, q):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}.")
            return

        cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time_ms)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_to_skip_on_detection = int(fps * seconds_to_skip_when_detect)
        frame_skip = int(fps * seconds_to_skip_each_frame)
        frame_count = int(self.start_time_ms * fps / 1000)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_time_str = str(datetime.timedelta(milliseconds=current_time_ms))
            cv2.putText(frame, f"Time: {current_time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if self.end_time_ms > 0 and current_time_ms > self.end_time_ms:
                print(f"The selected timestamps for {self.video_path} have been reached. Exiting...")
                break

            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            height, width = frame.shape[:2]
            aspect_ratio = width / height
            frame = cv2.resize(frame, (640, 360)) if aspect_ratio > 1 else cv2.resize(frame, (360, 640))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = cnn_face_detector(rgb_frame, 1)

            for face in faces:
                shape = shape_predictor(rgb_frame, face.rect)
                face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame, shape)
                face_encoding = np.array(face_encoding)
                matches = face_recognition.compare_faces(reference_encodings, face_encoding)
                if True in matches:
                    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    timestamp = str(datetime.timedelta(seconds=int(current_time))) + ":00"
                    print(f"Specific face detected at timestamp: {timestamp} in video {self.video_path}")
                    self.timestamps.append(timestamp)
                    self.write_timestamp(timestamp)
                    q.put((timestamp, self.video_path))
                    new_frame_position = frame_count + frames_to_skip_on_detection
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_position)
                    frame_count = new_frame_position

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------

def process_queue(q):
    while True:
        item = q.get()
        if item is None:
            break
        timestamp, video_path = item
        slice_video(timestamp, video_path)
        q.task_done()

q = queue.Queue()
thread = threading.Thread(target=process_queue, args=(q,))
thread.start()

for video_info in video_paths:
    processor = VideoProcessor(video_info['path'], video_info['start_time_ms'], video_info['end_time_ms'])
    processor.process_video(reference_encodings, cnn_face_detector, shape_predictor, face_recognition_model, q)

q.put(None)
thread.join()

print("Timestamps have been exported to timestamps.txt")