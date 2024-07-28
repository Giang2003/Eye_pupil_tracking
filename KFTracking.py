import cv2
from Detection import detect_and_track
from Detection import find_circle
from time import time
video_capture = cv2.VideoCapture(1)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect_and_track(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()