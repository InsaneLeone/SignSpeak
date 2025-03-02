import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # For text-to-speech announcement
import threading
import time  # Import time library

# Load the pre-trained model
model_data = pickle.load(open('./model.p', 'rb'))
sign_language_model = model_data['model']

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # Ensure you have the correct camera index (0, 1, 2, etc.)

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Labels for the predictions (A-Z)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

# Initialize pyttsx3 engine
text_to_speech_engine = pyttsx3.init()

# Initialize variables to keep track of the last predicted character and its timestamp
last_announced_character = None
last_announced_time = 0

# Flag to prevent multiple speech threads
is_speech_thread_running = False

# Function to handle speech in a separate thread
def speak(text):
    global is_speech_thread_running
    if is_speech_thread_running:
        return  # Prevent starting a new speech thread if one is already running
    is_speech_thread_running = True
    text_to_speech_engine.say(text)
    text_to_speech_engine.runAndWait()  # Block until speech is completed
    is_speech_thread_running = False

# Thread function to run speak only when needed
def announce_character(predicted_character):
    global last_announced_character, last_announced_time
    current_time = time.time()
    # Check if the predicted character is different from the last one or if it has been on screen for 2 seconds
    if predicted_character != last_announced_character:
        last_announced_character = predicted_character
        last_announced_time = current_time
    elif current_time - last_announced_time >= 2:  # 2 seconds
        # Run the speak function in a separate thread to avoid blocking
        threading.Thread(target=speak, args=(predicted_character,)).start()
        last_announced_time = current_time  # Reset the timestamp

while True:
    hand_landmarks_data = []
    x_coordinates = []
    y_coordinates = []

    ret, frame = video_capture.read()

    if not ret:  # Check if the frame was successfully captured
        print("Failed to grab frame. Exiting...")
        break

    frame_height, frame_width, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands_detector.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # landmark color
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # connection color
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_coordinates.append(x)
                y_coordinates.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                hand_landmarks_data.append(x - min(x_coordinates))
                hand_landmarks_data.append(y - min(y_coordinates))

        x1 = int(min(x_coordinates) * frame_width) - 10
        y1 = int(min(y_coordinates) * frame_height) - 10

        x2 = int(max(x_coordinates) * frame_width) - 10
        y2 = int(max(y_coordinates) * frame_height) - 10

        # Ensure the number of features matches the model's expected input size
        if len(hand_landmarks_data) == sign_language_model.n_features_in_:
            prediction = sign_language_model.predict([np.asarray(hand_landmarks_data)])
            predicted_character = labels_dict[int(prediction[0])]

            # Announce the letter if it is different from the last one or has been on screen for 2 seconds
            announce_character(predicted_character)

            # Draw bounding box and predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, 
                        cv2.LINE_AA)

        else:
            print(f"Feature size mismatch: expected {sign_language_model.n_features_in_}, but got {len(hand_landmarks_data)}")

    # Display the frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
        break

video_capture.release()
cv2.destroyAllWindows()
