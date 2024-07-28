import cv2
import numpy as np
import time
from time import time

from KalmanFilter import KalmanFilter
kf = KalmanFilter()
# Loading the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

def find_circle(frame, gray_eye):
    # Check if the input gray_eye is not None and has content
    if gray_eye is None or gray_eye.size == 0:
        # The eye region is empty (likely out of bounds), so skip processing
        return frame
    # Preprocess the eye region for circle detection
    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    rows = gray_eye.shape[0]

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(gray_eye, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=50, param2=30,  # Increased param2 for stricter detection
                               minRadius=7, maxRadius=21)  # Adjusted radius values

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Additional filtering based on properties such as size and relative position
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # Assuming that the largest circle within a reasonable range is the pupil
            if radius >= 7 and radius <= 21:  # Adjust these values as needed
                # Draw the circle center
                cv2.circle(frame, center, 1, (0, 100, 100), 3)
                # Draw the circle outline
                cv2.circle(frame, center, radius, (255, 0, 255), 3)
                return center, frame

    return None,frame

# Defining a function that will do the detections
def detect_and_track(gray, frame):
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for face_index,(x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
    

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]
            center, eye_color = find_circle(eye_color, eye_gray)
            if center is not None:
                measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
                kf.correct(measurement)
                prediction = kf.predict()
                predicted_center = (int(prediction[0, 0]), int(prediction[1, 0]))
                cv2.circle(frame, predicted_center, 20, (0, 255, 0), 2)  # Draw predicted center
                return frame

    return frame