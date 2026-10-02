import csv
import os
import time
import cv2
import mediapipe as mp

from feature_engineering import extract_features_from_mediapipe

DATASET_PATH = os.path.join("dataset", "gesture_dataset_v2.csv")

GESTURES = {
    ord("1"): "OPEN_PALM",
    ord("2"): "FIST",
    ord("3"): "THUMBS_UP",
    ord("4"): "THUMBS_DOWN",
    ord("5"): "VICTORY",
}

FEATURE_COUNT = 105


def create_dataset_file():
    """Create the dataset folder and CSV file with 105 feature columns."""
    os.makedirs("dataset", exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        header = [f"feature_{index}" for index in range(FEATURE_COUNT)]
        header.append("label")

        with open(DATASET_PATH, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)

        print(f"Created new dataset: {DATASET_PATH}")


def validate_dataset_structure():
    """Prevent writing 105-feature data into an incompatible CSV file."""
    if not os.path.exists(DATASET_PATH):
        return

    with open(DATASET_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)

    expected_columns = FEATURE_COUNT + 1

    if header is None or len(header) != expected_columns:
        raise ValueError(
            f"Dataset structure is incorrect.\n"
            f"Expected {expected_columns} columns "
            f"({FEATURE_COUNT} features + label), "
            f"but found {0 if header is None else len(header)}.\n"
            f"Rename or delete this file before continuing:\n"
            f"{DATASET_PATH}"
        )


def save_sample(features, label):
    """Save one engineered feature vector and its gesture label."""
    feature_list = features.tolist()

    if len(feature_list) != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, "
            f"but received {len(feature_list)}."
        )

    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(feature_list + [label])


def draw_interface(frame, selected_label, sample_count):
    """Draw collector instructions and status on the camera frame."""
    selected_text = selected_label if selected_label else "NONE"

    cv2.putText(
        frame,
        f"Selected: {selected_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Saved this session: {sample_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "1: Open  2: Fist  3: Up  4: Down  5: Victory",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SPACE: Save sample | A: Auto-Capture | Q: Quit",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )


def main():
    create_dataset_file()
    validate_dataset_structure()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        hands.close()
        print("Unable to open camera.")
        return

    selected_label = None
    sample_count = 0
    auto_capture = False
    capture_interval = 0.2  # seconds
    last_capture_time = 0

    print("\nControls:")
    print("1 = OPEN_PALM")
    print("2 = FIST")
    print("3 = THUMBS_UP")
    print("4 = THUMBS_DOWN")
    print("5 = VICTORY")
    print("A = Toggle Auto-Capture")
    print("SPACE = Save one sample")
    print("Q = Quit")
    print(f"\nSaving data to: {DATASET_PATH}")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)
            detected_landmarks = None

            if results.multi_hand_landmarks:
                detected_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame,
                    detected_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            draw_interface(frame, selected_label, sample_count)
            cv2.imshow("Gesture Dataset Collector V2", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("a"):
                auto_capture = not auto_capture
                if auto_capture:
                    print("\nAuto Capture: ON")
                else:
                    print("\nAuto Capture: OFF")

            elif key in GESTURES:
                selected_label = GESTURES[key]
                sample_count = 0
                print(f"\nSelected gesture: {selected_label}")

            elif key == 32:  # SPACE Key
                if selected_label is None:
                    print("Select a gesture first using keys 1 to 5.")
                elif detected_landmarks is None:
                    print("No hand detected. Sample not saved.")
                else:
                    try:
                        features = extract_features_from_mediapipe(detected_landmarks)
                        save_sample(features, selected_label)
                        sample_count += 1
                        print(f"Saved {selected_label} sample {sample_count}")
                    except ValueError as error:
                        print(f"Sample not saved: {error}")

            # --- Auto Capture Block (Placed right after manual SPACE block) ---
            current_time = time.time()

            if (
                auto_capture
                and selected_label is not None
                and detected_landmarks is not None
                and current_time - last_capture_time >= capture_interval
            ):
                try:
                    features = extract_features_from_mediapipe(detected_landmarks)
                    save_sample(features, selected_label)
                    sample_count += 1
                    last_capture_time = current_time
                    print(f"Auto-saved {selected_label} sample {sample_count}")
                except ValueError as error:
                    print(f"Auto sample not saved: {error}")

    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
