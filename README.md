```markdown
# Hand Gesture Recognition using OpenCV and MediaPipe

## Overview

This Python project utilizes OpenCV and Google's MediaPipe library to recognize hand gestures in real-time through webcam input. It demonstrates the ability to detect specific gestures such as "thumbs up" and the "peace sign" by analyzing the position and orientation of fingers using MediaPipe's hand tracking capabilities.

## Features

- **Real-Time Hand Gesture Recognition:** Detects specific hand gestures in real-time using a webcam.
- **Dynamic Gesture Analysis:** Utilizes geometric properties and relationships between hand landmarks to identify gestures.
- **Visual Feedback:** Provides visual feedback by drawing hand landmarks and annotations on the video feed.

## Dependencies

- Python 3.x
- OpenCV (cv2): Open Source Computer Vision Library, licensed under BSD 3-Clause License.
- MediaPipe: An open source cross-platform, customizable ML solutions for live and streaming media, licensed under Apache License 2.0.
- NumPy: A fundamental package for scientific computing with Python, licensed under BSD 3-Clause License.
- Math: Part of Python's Standard Library, not requiring external licensing.

## Setup

1. **Install Python:** Ensure Python 3.x is installed on your system.

2. **Install Required Libraries:**
   Use pip to install the required Python libraries:

   ```
   pip install opencv-python mediapipe numpy
   ```

3. **Run the Script:**
   Execute the script in your Python environment:

   ```
   python hand_gesture_recognition.py
   ```

## Usage Instructions

- **Start the Program:** Run the script to start gesture recognition. The webcam feed will appear in a window.
- **Gesture Recognition:** Perform gestures such as "thumbs up" or "peace sign" in view of the webcam.
- **Toggle Skeleton View:** Press 's' to toggle the visibility of the hand skeleton overlay on the video feed.
- **Drawing Mode:** Press 'd' to toggle drawing mode. In drawing mode, move your index finger to draw on the video feed.
- **Clear Canvas:** Press 'c' to clear the canvas in drawing mode.
- **Exit:** Press 'q' to quit the program.

## Function Descriptions

- `is_thumbsUp(hand_landmarks)`: Determines if a "thumbs up" gesture is made.
- `calculate_angle_between_points(P1, P2)`: Calculates the angle between two points.
- `two_point_box_angle(...)`: Determines the angle between joints to aid in gesture recognition.
- `is_peaceSign(hand_landmarks)`: Identifies if a "peace sign" gesture is present.
- `calculate_distance(P1, P2)`: Computes the distance between two points.

## Contribution

Contributions to this project are welcome. You can enhance gesture recognition algorithms, add new gestures, or improve the user interface.

## License

This project itself is open-sourced under the MIT License. Please note that the use of dependencies like OpenCV, MediaPipe, and NumPy are subject to their respective licenses.
```
