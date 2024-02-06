import cv2
import mediapipe as mp
import math
import sys
from collections import deque
import numpy as np
print("Start")
#const

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
        

def two_point_box_gradient(frame,lifo, landmark1, landmark2,landmark3, w, h,tol = 0.4 ):
    # Convert normalized coordinates to pixel coordinates
    x1, y1 = int(landmark1.x * w), int(landmark1.y * h)
    x2, y2 = int(landmark2.x * w), int(landmark2.y * h)
    x3, y3 = int(landmark3.x * w), int(landmark3.y * h)

    
    
    
    if x2 - x1 == 0 or x3 - x1 == 0:
        grad1 = sys.maxsize
        
    else:
        grad1 = (y2-y1) / (x2 - x1)
    
    
    if x3-x1 == 0:
        grad2 = sys.maxsize
    else:
        grad2 =  (y3-y1) / (x3 - x1)
    
    
    cv2.putText(frame,str(grad1-grad2), (20,height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
    
    grad_diff_abs = abs(grad1-grad2)
    lifo.append(grad_diff_abs)
    
    data_sorted = sorted(list(lifo))
    Q1 = np.percentile(data_sorted, 25)
    Q3 = np.percentile(data_sorted, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = [x for x in data_sorted if lower_bound <= x <= upper_bound]

    
    average = sum(filtered_data) / len(filtered_data) if filtered_data else None

    
    if  average < tol:
        cv2.putText(frame,"yes", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
    else:
        cv2.putText(frame,"no", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        
   


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

lifo_stack = deque(maxlen=30)


while cap.isOpened():
    count = 0
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame color from BGR to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands.
    results = hands.process(frame_rgb)

    

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Example usage of two_point_box function
            # Make sure to pass the frame, two landmarks, and the frame's width and height
            
            #two_point_box_gradient(frame,lifo_stack, hand_landmarks.landmark[5], hand_landmarks.landmark[6],hand_landmarks.landmark[8], width, height)
            
            for i in range(4):
                start = (i+1)*4 + 1
                count += two_point_box_angle(frame, hand_landmarks.landmark[start], hand_landmarks.landmark[start+1],hand_landmarks.landmark[start+3], width, height)
            
            count += two_point_box_angle(frame, hand_landmarks.landmark[2], hand_landmarks.landmark[3],hand_landmarks.landmark[4], width, height,limit = False, tol = 20)
            
            # Annotate each landmark with its index
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Convert normalized landmark position to relative pixel position
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
                if landmark_px:  # Check if conversion is successful
                    
                    cv2.putText(frame, str(idx), landmark_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            #cv2.putText(frame,str(hand_landmarks.landmark[5]), (20,height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
            #cv2.putText(frame,str(hand_landmarks.landmark[6]), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
            #two_point_box(frame,hand_landmarks.landmark[5],hand_landmarks.landmark[5],width,height)
            #cv2.putText(frame,two_point_box(hand_landmarks.landmark[5],hand_landmarks.landmark[5]), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        
        cv2.putText(frame,str(count), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
    # Display the frame.
    cv2.imshow('MediaPipe Hands', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the webcam and close all OpenCV windows

cap.release()
cv2.destroyAllWindows()
print("end")