import sys
sys.stdout.reconfigure(encoding='utf-8')  # Set UTF-8 encoding to avoid Unicode errors



import cv2
import time
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Track time and emotions
start_time = time.time()
emotion_changes = 0
prev_emotion = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            if isinstance(result, list) and len(result) > 0:
                emotion = result[0]['dominant_emotion']
            else:
                continue  # Skip if no emotion detected

            # Check if emotion has changed
            if prev_emotion and prev_emotion != emotion:
                emotion_changes += 1
                print(f"Emotion changed! ({prev_emotion} -> {emotion})")

            prev_emotion = emotion

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error detecting emotion: {e}")
            continue  # Skip this frame if DeepFace fails

    # Display the resulting frame
    cv2.imshow('10-sec Emotion Detection', frame)

    # Stop after 10 seconds or if user presses 'q'
    if time.time() - start_time > 10 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Check for bipolar-like symptoms
if emotion_changes >= 7:
    print("⚠️ ALERT: Frequent emotion changes detected! Possible bipolar symptom.")
else:
    print("✅ Normal emotional range detected.")
