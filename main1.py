from ultralytics import YOLO
from pyfirmata import Arduino, SERVO
from time import sleep
import cv2
import tkinter as tk
from tkinter import filedialog
import subprocess
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv

# Initialize Arduino
port = 'COM4'
pin = 10
board = Arduino(port)
board.digital[pin].mode = SERVO


def rotateservo(pin, angle):
    board.digital[pin].write(angle)
    sleep(0.015)


# Load YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Initialize SORT tracker
mot_tracker = Sort()


class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        # Create widgets
        self.label = tk.Label(root, text="Video Player")
        self.label.pack(pady=10)

        self.btn_upload = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.btn_upload.pack(pady=10)

        self.video_path = None

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
        if self.video_path:
            self.process_video()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_nmr = -1

        # Results dictionary
        results = {}

        # List of vehicle classes
        vehicles = [2, 3, 5, 7]

        # Read frames
        ret = True
        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                results[frame_nmr] = {}
                # Detect vehicles
                detections = coco_model(frame)[0]
                detections_ = []
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                # Track vehicles
                track_ids = mot_tracker.update(np.asarray(detections_))

                # Detect license plates
                license_plates = license_plate_detector(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    # Assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    if car_id != -1:
                        # Crop license plate
                        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                        # Process license plate
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                                      cv2.THRESH_BINARY_INV)

                        # Read license plate number
                        license_plate_text, license_plate_text_score = read_license_plate(
                            license_plate_crop_thresh)

                        if license_plate_text is not None:
                            results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                          'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}}

        # Write results to CSV
        write_csv(results, './test.csv')

        # Open the CSV file
        csv_file_path = './test.csv'
        subprocess.Popen(['start', 'excel', csv_file_path], shell=True)

        # Perform action if specific license plate is detected
        texts = [car['license_plate']['text'] for frame in results.values() for car in frame.values()]

        for text in texts:
            if "AP05JEO" in text:
                rotateservo(pin, 90)
                sleep(3)
                rotateservo(pin, 0)  # Release the servo to its default position
                sleep(1)  # Add a delay for the servo to settle
                break

        # Release video capture
        cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayerApp(root)
    root.mainloop()

