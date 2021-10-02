"""
Code Sources
- cv2 part of taking the image, producing a img variable, and the coordinates of the face in the image:
        https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
"""

import cv2
import time
import os

# CONSTANTS
LOW_FOCUS_SCORE_THRESHOLD = .7
FOCUS_COUNTER_THRESHOLD = 5

# LOW_FOCUS_SCORE IMPLEMENTATION
def low_focus_score (valence, arousal, dominance):
    return 0

# TAKES A PICTURE EVERY SECOND, DETECTS FACE, CALLS ML MODEL, USES PREDICTION TO CALCULATE 
# LOW_FOCUS_SCORE, AND IF SCORE IS ABOVE THRESHOLD WE NOTIFY USER WITH SCREEN FLASH
# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# To capture video from webcam. 
cap = cv2.VideoCapture(0)

# focus_counter keeps track of how many consecutive seconds you are distraught
focus_counter = 0

# Infinite loop performs all the functionality
while True:
    # Adjust time as desired
    time.sleep(1)

    # Read the frame
    _, img = cap.read()

    # Convert to grayscale for face box model
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces, get box coordinates of first face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
         continue

    x, y, w, h = faces[0]
    x1, y1, x2, y2 = x, y, x + w, y + h
    print(f'{x1} {y1} {x2} {y2}')

    # To visualize rectangle around face on screen
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # Display the image
    cv2.imshow('img', img)

    # Resize img to (224,224) for faster processing by model
    img = cv2.resize(img, (224, 224))
    # Enable the below line instead of the above cv2.imshow line to see resized image
    # cv2.imshow('img', img)

    # Call model to make the inference on the face's valence, arousal, dominance
    valence, arousal, dominance = ML_MODEL_CALL_WITH_IMG_AND_COORDINATES

    # Calculate the low_focus_score of the current image's face
    low_focus_score = low_focus_score(valence, arousal, dominance)

    # If low_focus_score is above the LOW_FOCUS_SCORE_THRESHOLD for more than 5 seconds,
    # make a sound and notify the user to get back to work. Seconds counter resets 
    # whenever user gets focused again within that 5 second span. 
    if low_focus_score > LOW_FOCUS_SCORE_THRESHOLD:
        focus_counter += 1
        if focus_counter > FOCUS_COUNTER_THRESHOLD:
            os.system("afplay /System/Library/Sounds/Morse.aiff -v 2")
            print("You're distracted. Get back to work!")
            focus_counter = 0
    else:
        focus_counter = 0
        
    # End program if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break


# Release the VideoCapture object
cap.release()