## Project Description

This project involves the development of a real-time eye pupil detection and tracking system. The system leverages computer vision techniques to detect the eye region and track the center of the pupil using a Kalman filter. It uses OpenCV for image processing, NumPy for numerical computations, and the Kalman filter for predictive tracking.

## Features

- Real-time eye detection using Haar cascades.
- Precise pupil localization using Hough Circle Transform.
- Predictive tracking of the pupil center using a Kalman filter.
- Parameter tuning of the Kalman filter to optimize tracking accuracy.
- Calculation of tracking error and iterative refinement of the filter parameters.

## Tools & Technologies

- Python
- OpenCV
- NumPy
- Kalman Filter

## Setup Instructions

Clone the repository to your local machine:
    ```bash
    git clone https://github.com/Giang2003/Eye_pupil_tracking.git
    ```


## Usage

1. Ensure your webcam is connected.
2. Run the tracking script:
    ```bash
    python KFTracking.py
    ```
3. The system will open a window displaying the video feed with the detected eye region and tracked pupil center.

## Project Structure

- `KalmanFilter.py`: Contains the implementation of the Kalman filter used for tracking the pupil.
- `KFTracking.py`: Captures video feed from the webcam and applies the detection and tracking algorithms.
- `Detection.py`: Contains the functions for detecting the eye region and localizing the pupil using Hough Circle Transform.
