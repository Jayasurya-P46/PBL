import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "No Hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand_landmarks.landmark

            fingers = 0

            # Index
            if lm[8].y < lm[6].y:
                fingers += 1

            # Middle
            if lm[12].y < lm[10].y:
                fingers += 1

            # Ring
            if lm[16].y < lm[14].y:
                fingers += 1

            # Pinky
            if lm[20].y < lm[18].y:
                fingers += 1

            # Thumb (works for both hands reasonably well)
            if abs(lm[4].x - lm[2].x) > 0.08:
                fingers += 1

            if fingers >= 4:
                gesture = "OPEN PALM"

            elif fingers <= 1:
                gesture = "FIST"

            else:
                gesture = "UNKNOWN"

    cv2.putText(frame,
                gesture,
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
