# ==========================================================
# SMART GESTURE CONTROLLED DASHBOARD
# Predict Gesture (Part 1)
# Random Forest + MediaPipe + Arduino Serial
# ==========================================================

import cv2
import mediapipe as mp
import numpy as np
import joblib
import serial
import serial.tools.list_ports
import time

# ==========================================================
# CONFIGURATION
# =========================================


MODEL_PATH = "models/gesture_rf.pkl"
CONFIDENCE_THRESHOLD = 0.90
SERIAL_BAUDRATE = 9600
CAMERA_INDEX = 0

PREDICTION_INTERVAL = 3
frame_count = 0
last_prediction = "NO HAND"
last_confidence = 0.0


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading Random Forest model...")
model = joblib.load(MODEL_PATH)
print("Model Loaded Successfully")


# ==========================================================
# AUTO DETECT ARDUINO PORT
# ==========================================================

def connect_arduino():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        try:
            board = serial.Serial(
                port.device,
                SERIAL_BAUDRATE,
                timeout=1
            )
            time.sleep(2)
            print(f"Connected to Arduino on {port.device}")
            return board
        except Exception:
            pass

    print("Arduino not found.")
    print("Running in Recognition Only Mode.")
    return None


try:
    arduino = serial.Serial("COM13", 9600, timeout=1)
    time.sleep(2)
    print("Arduino Connected")
except Exception:
    arduino = None
    print("Running Camera Only")


# ==========================================================
# MEDIAPIPE SETUP
# ==========================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.80,
    min_tracking_confidence=0.80
)
drawer = mp.solutions.drawing_utils


# ==========================================================
# FEATURE EXTRACTION
# (63 Features)
# ==========================================================

def extract_features(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    wx = wrist.x
    wy = wrist.y
    wz = wrist.z

    middle = hand_landmarks.landmark[9]
    scale = np.sqrt(
        (middle.x - wx) ** 2 +
        (middle.y - wy) ** 2 +
        (middle.z - wz) ** 2
    )

    if scale == 0:
        scale = 1

    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([
            (landmark.x - wx) / scale,
            (landmark.y - wy) / scale,
            (landmark.z - wz) / scale
        ])

    return np.array(features).reshape(1, -1)


# ==========================================================
# GESTURE CONFIRMATION
# ==========================================================

REQUIRED_CONFIRMATIONS = 5
candidate_gesture = None
candidate_count = 0
confirmed_gesture = "NO HAND"

def confirm_gesture(prediction):
    global candidate_gesture
    global candidate_count
    global confirmed_gesture

    # New candidate gesture
    if prediction != candidate_gesture:
        candidate_gesture = prediction
        candidate_count = 1
        return confirmed_gesture

    # Same prediction continues
    candidate_count += 1

    # Confirm only after several consecutive predictions
    if candidate_count >= REQUIRED_CONFIRMATIONS:
        confirmed_gesture = candidate_gesture

    return confirmed_gesture


hand_detected = False
hand_start_time = 0
HAND_WARMUP_TIME = 1.0


# ==========================================================
# SEND TO ARDUINO
# ==========================================================

last_sent = ""
gesture_locked = False

def send_to_arduino(gesture):
    global last_sent, gesture_locked

    if arduino is None:
        return

    gesture_commands = {
        "OPEN_PALM": "1",
        "THUMBS_UP": "2",
        "THUMBS_DOWN": "3",
        "FIST": "4",
        "VICTORY": "5"
    }

    if gesture not in gesture_commands:
        return

    if gesture_locked and gesture == last_sent:
        return

    try:
        command = gesture_commands[gesture]
        arduino.write((command + "\n").encode())
        print(f"Sent -> {gesture} [{command}]")

        last_sent = gesture
        gesture_locked = True

    except Exception as e:
        print("Serial Error:", e)


# ==========================================================
# CAMERA INITIALIZATION
# ==========================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Unable to open camera.")
    exit()

print("\n==========================================")
print(" SMART GESTURE CONTROLLED DASHBOARD ")
print("==========================================")
print("Press 'Q' to Quit\n")


# ==========================================================
# MAIN LOOP
# ==========================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        hand_detected = False

        candidate_gesture = None
        candidate_count = 0
        confirmed_gesture = "NO HAND"

        gesture_locked = False
        last_sent = ""
        last_prediction = "NO HAND"
        last_confidence = 0.0

        gesture = "NO HAND"
        confidence = 0.0
    else:
        if not hand_detected:
            hand_detected = True
            hand_start_time = time.time()

        # Ignore predictions during warm-up
        if time.time() - hand_start_time < HAND_WARMUP_TIME:
            cv2.putText(
                frame,
                "Stabilizing Hand...",
                (15, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            cv2.imshow("Smart Gesture Dashboard", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        hand = results.multi_hand_landmarks[0]

        drawer.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        frame_count += 1
        if frame_count % PREDICTION_INTERVAL == 0:
            features = extract_features(hand)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            last_prediction = prediction
            last_confidence = float(np.max(probability))

            if last_confidence >= CONFIDENCE_THRESHOLD:
                stable_gesture = confirm_gesture(prediction)

                if stable_gesture != "NO HAND":
                    if stable_gesture != last_sent:
                        gesture_locked = False

                    send_to_arduino(stable_gesture)

        gesture = confirmed_gesture
        confidence = last_confidence

    # ------------------------------------------
    # Display Information
    # ------------------------------------------

    cv2.rectangle(frame, (0, 0), (640, 90), (40, 40, 40), -1)

    cv2.putText(
        frame,
        "SMART GESTURE DASHBOARD",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Gesture : {gesture}",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence*100:.2f} %",
        (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    cv2.imshow("Smart Gesture Dashboard", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


# ==========================================================
# CLEANUP
# ==========================================================

print("\nClosing application...")

cap.release()
hands.close()

if arduino is not None:
    arduino.close()

cv2.destroyAllWindows()

print("Done.")
