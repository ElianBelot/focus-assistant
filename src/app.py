"""
Code Sources
- cv2 part of taking the image, producing a img variable, and the coordinates of the face in the image:
        https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
"""

# =====[ IMPORTS ]=====
import cv2
import time
import os
from inference import *
from emotic import Emotic


# =====[ CONSTANTS ]=====
LOW_FOCUS_SCORE_THRESHOLD = .7
FOCUS_COUNTER_THRESHOLD = 5
DEBUG = True


# =====[ FOCUS SCORE ]=====
def low_focus_score(valence, arousal, dominance):
    return arousal


# =====[ FACE CAPTURE ]=====
# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam.
cap = cv2.VideoCapture(0)

# focus_counter keeps track of how many consecutive seconds you are distraught
focus_counter = 0


# =====[ CAPTURE LOOP ]=====
while True:
    # Adjust time as desired
    time.sleep(1)

    # Read the frame
    _, img = cap.read()

    # Convert to grayscale for face box model
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces, get box coordinates of first face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Skip frame if no face detected
    if len(faces) == 0:
        continue

    # Get bounding box
    x, y, w, h = faces[0]
    x1, y1, x2, y2 = x, y, x + w, y + h

    # Draw box and display image
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow('img', img)

    # Call model to make the inference on the face's valence, arousal, dominance
    results = predict(img, x1, y1, x2, y2)
    valence, arousal, dominance = results['valence'], results['arousal'], results['dominance']

    # Print results to screen (debugging)
    if DEBUG:
        print()
        for k, v in results.items():
            if k != 'emotions':
                print(f'{k.upper()}: \t {v:.3f}')

    # Calculate the low_focus_score of the current image's face
    score = low_focus_score(valence, arousal, dominance)

    # If low_focus_score is above the LOW_FOCUS_SCORE_THRESHOLD for more than 5 seconds,
    # make a sound and notify the user to get back to work. Seconds counter resets
    # whenever user gets focused again within that 5 second span.
    if score > LOW_FOCUS_SCORE_THRESHOLD:
        focus_counter += 1
        if focus_counter > FOCUS_COUNTER_THRESHOLD:
            if DEBUG:
                # os.system("afplay /System/Library/Sounds/Morse.aiff -v 2")
                print("DISTRACTION DETECTED")
            focus_counter = 0
    else:
        focus_counter = 0

    # End program if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# =====[ END CAPTURE ]=====
cap.release()
