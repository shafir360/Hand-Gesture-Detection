import cv2
import mediapipe as mp
import math
import sys
from collections import deque
import numpy as np
print("Start")
#const

show_skeleton = False

def is_thumbsUp(hand_landmarks):
    
    for i in range(4):
        start = (i+1)*4 + 1
        if two_point_box_angle(frame, hand_landmarks.landmark[start], hand_landmarks.landmark[start+1],hand_landmarks.landmark[start+3], width, height) == 1: 
            return False
    
    if two_point_box_angle(frame, hand_landmarks.landmark[2], hand_landmarks.landmark[3],hand_landmarks.landmark[4], width, height,limit = True, tol = 20) == 1:
        return True
    else:
        False

    

def calculate_angle_between_points(P1, P2):
    # Extract coordinates
    x1, y1 = P1
    x2, y2 = P2
    
    x = x2 - x1
    y = y1 - y2
    
    # Calculate the angle in radians
    angle_rad = math.atan2(y, x)
    
    # Convert the angle to degrees for easier interpretation
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def two_point_box_angle(frame,landmark1, landmark2,landmark3, w, h,tol = 15, limit = True ):
    # Convert normalized coordinates to pixel coordinates
    x1, y1 = int(landmark1.x * w), int(landmark1.y * h)
    x2, y2 = int(landmark2.x * w), int(landmark2.y * h)
    x3, y3 = int(landmark3.x * w), int(landmark3.y * h)

    angle1 = calculate_angle_between_points( (x1,y1), (x2,y2) )
    angle2 = calculate_angle_between_points( (x2,y2), (x3,y3) )
   
    diff_angle = abs(angle1-angle2)
    
    if limit:
        if angle1 < 0:
            return 0
    
    if  diff_angle < tol:
        #cv2.putText(frame,"yes " + str(diff_angle), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        return 1
        
    else:
        #cv2.putText(frame,"no " + str(diff_angle), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        return 0
        

def is_peaceSign(hand_landmarks):
    
    if two_point_box_angle(frame, hand_landmarks.landmark[5], hand_landmarks.landmark[6],hand_landmarks.landmark[8], width, height) == 0:
        return False
    if two_point_box_angle(frame, hand_landmarks.landmark[9], hand_landmarks.landmark[10],hand_landmarks.landmark[12], width, height) == 0:
        return False
    if two_point_box_angle(frame, hand_landmarks.landmark[13], hand_landmarks.landmark[14],hand_landmarks.landmark[16], width, height) == 1:
        return False
    if two_point_box_angle(frame, hand_landmarks.landmark[17], hand_landmarks.landmark[18],hand_landmarks.landmark[20], width, height) == 1:
        return False
    
    return True

def calculate_distance(P1, P2):
    x1, y1 = P1  # Unpacking the first coordinate
    x2, y2 = P2  # Unpacking the second coordinate

    # Calculating the Euclidean distance
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

custom_fps = 60

cap = cv2.VideoCapture(0)

# Get the resolution of the video capture
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FPS, custom_fps)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Initialize MediaPipe drawing utilities for drawing hand landmarks on the image.
mp_drawing = mp.solutions.drawing_utils

# Initialize a blank canvas for drawing
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Initialize drawing mode and previous landmark point
drawing_mode = False
prev_point = None
prev_point_2 = None
closed_hand_both = False
stop_0 = False
stop_1 = False


while cap.isOpened():
    count = 0
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if not ret:
        continue

    # Convert the frame color from BGR to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands.
    results = hands.process(frame_rgb)

    
    hand_array = [False,False,False,False]

    if results.multi_hand_landmarks:
        for idx_hand, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if show_skeleton:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # If in drawing mode, trace one of the hand landmarks to draw
            if drawing_mode:
                
                    
                # Choose a landmark to trace, e.g., the tip of the index finger
                landmark_idx = mp_hands.HandLandmark.INDEX_FINGER_TIP
                landmark = hand_landmarks.landmark[landmark_idx]

                # Convert normalized coordinates to pixel coordinates
                x, y = int(landmark.x * width), int(landmark.y * height)
                curr_point = (x, y)

                # Draw line segment if the previous point is available
                
                if prev_point and idx_hand == 0:
                    if stop_0:
                        prev_point = None
                    else:
                        cv2.line(canvas, prev_point, curr_point, (255, 255, 255), 2)
                elif prev_point_2 and idx_hand == 1 :
                    if stop_1:
                        prev_point_2 = None
                    else:
                        cv2.line(canvas, prev_point_2, curr_point, (255, 255, 255), 2)
                
                # Update the previous point
                if idx_hand == 0:
                    prev_point = curr_point
                else:
                    prev_point_2 = curr_point
            
            for i in range(4):
                start = (i+1)*4 + 1
                count += two_point_box_angle(frame, hand_landmarks.landmark[start], hand_landmarks.landmark[start+1],hand_landmarks.landmark[start+3], width, height)
            
            count += two_point_box_angle(frame, hand_landmarks.landmark[2], hand_landmarks.landmark[3],hand_landmarks.landmark[4], width, height,limit = False, tol = 20)
            
            if  idx_hand == 0:
                stop_0 = is_peaceSign(hand_landmarks) 
            elif  idx_hand ==1:
                stop_1 = is_peaceSign(hand_landmarks)
    
            
            
            
            is_thumbs = is_thumbsUp(hand_landmarks)
            hand_array[idx_hand] = is_thumbs
            
            # Annotate each landmark with its index
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Convert normalized landmark position to relative pixel position
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
                if landmark_px and show_skeleton:  # Check if conversion is successful
                    
                    cv2.putText(frame, str(idx), landmark_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            #cv2.putText(frame,str(hand_landmarks.landmark[5]), (20,height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
            #cv2.putText(frame,str(hand_landmarks.landmark[6]), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
            #two_point_box(frame,hand_landmarks.landmark[5],hand_landmarks.landmark[5],width,height)
            #cv2.putText(frame,two_point_box(hand_landmarks.landmark[5],hand_landmarks.landmark[5]), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        
        #cv2.putText(frame,str(hand_array), (20,height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        cv2.putText(frame,str(hand_landmarks.landmark[8]), (20,height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        cv2.putText(frame,str(count), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        
    # Overlay the canvas on the frame
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)

    # Display the frame
    cv2.imshow('MediaPipe Hands', frame)

    # Capture key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        show_skeleton = not show_skeleton
    elif key == ord('d'):
        # Toggle drawing mode
        drawing_mode = not drawing_mode
        if drawing_mode:
            prev_point = None  # Reset previous point
            prev_point_2 = None
    elif key == ord('c'):
        # Clear the canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("end")