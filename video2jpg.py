import os
from datetime import datetime

import cv2
import numpy as np

count = 0
vid_files = [name for name in os.listdir("./data/video_data/") if name[0] != '.']

for file in vid_files:
    cap = cv2.VideoCapture("./data/video_data/" + file)
    if not cap.isOpened():
        print("Error opening video file")

    while cap.isOpened():
        st_time = datetime.now()
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            modelFile = "./opencv_face_detector_uint8.pb"
            configFile = "./opencv_face_detector.pbtxt"

            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         [np.mean(frame[:, :, 0]), np.mean(frame[:, :, 1]), np.mean(frame[:, :, 2])],
                                         False, False)
            net.setInput(blob)
            detections = net.forward()

            i = np.argmax(detections[0, 0, :, 2])
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])

            frame = frame[y1:y1 + 10 + int((y2 - y1) / 2), x1:x2]
            frame = cv2.resize(frame, (150, 100))

            cv2.imwrite("./data/img_data/" + file.split('_')[0] + "/" + str(count) + ".jpg", frame)
            print(file + " " + str(count))
            count += 1
            print(datetime.now() - st_time)
        else:
            break

    cap.release()

cv2.destroyAllWindows()
