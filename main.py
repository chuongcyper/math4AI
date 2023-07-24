import cv2
from keras.models import model_from_json
import numpy as np

# Load pre-trained emotion detection model
model_architecture_path = 'model\emotion_model.json'
model_weights_path = 'model\emotion_model.h5'

with open(model_architecture_path, 'r') as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(model_weights_path)

# Define the emotions labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Access the camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    # Preprocess the input frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Perform emotion detection
        emotion_prediction = emotion_model.predict(face_roi)
        max_index = np.argmax(emotion_prediction[0])
        emotion_label = emotions[max_index]

        # Display the emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
camera.release()
cv2.destroyAllWindows()
