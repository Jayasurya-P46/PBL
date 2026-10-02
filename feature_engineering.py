import numpy as np


EPSILON = 1e-8


def calculate_distance(point_a, point_b):
    """Calculate 3D Euclidean distance between two landmarks."""
    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)

    return float(np.linalg.norm(point_a - point_b))


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle ABC in degrees.
    Point B is the joint where the angle is measured.
    """
    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)
    point_c = np.asarray(point_c, dtype=np.float64)

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    denominator = (
        np.linalg.norm(vector_ba) *
        np.linalg.norm(vector_bc)
    )

    if denominator < EPSILON:
        return 0.0

    cosine_value = np.dot(vector_ba, vector_bc) / denominator
    cosine_value = np.clip(cosine_value, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine_value)))


def normalize_mediapipe_landmarks(hand_landmarks):
    """
    Convert MediaPipe landmarks into wrist-relative,
    scale-normalized 21x3 coordinates.
    """
    raw_points = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks.landmark
        ],
        dtype=np.float64
    )

    wrist = raw_points[0]

    # Make wrist the origin.
    normalized_points = raw_points - wrist

    # Scale using wrist-to-middle-finger MCP distance.
    scale = np.linalg.norm(normalized_points[9])

    if scale < EPSILON:
        scale = 1.0

    normalized_points = normalized_points / scale

    return normalized_points


def extract_features_from_points(points):
    """
    Generate the final engineered feature vector from
    normalized 21x3 hand landmarks.

    Input shape: (21, 3)
    """

    points = np.asarray(points, dtype=np.float64)

    if points.shape != (21, 3):
        raise ValueError(
            f"Expected landmark shape (21, 3), received {points.shape}"
        )

    features = []

    # 1. Normalized coordinates: 21 × 3 = 63
    features.extend(points.flatten().tolist())

    # 2. Fifteen joint angles
    angle_triplets = [
        # Thumb
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),

        # Index
        (0, 5, 6),
        (5, 6, 7),
        (6, 7, 8),

        # Middle
        (0, 9, 10),
        (9, 10, 11),
        (10, 11, 12),

        # Ring
        (0, 13, 14),
        (13, 14, 15),
        (14, 15, 16),

        # Pinky
        (0, 17, 18),
        (17, 18, 19),
        (18, 19, 20)
    ]

    for point_a, point_b, point_c in angle_triplets:
        angle_value = calculate_angle(
            points[point_a],
            points[point_b],
            points[point_c]
        )

        # Convert degrees to 0–1 range.
        features.append(angle_value / 180.0)

    # 3. Distances between every pair of fingertips: 10
    fingertip_ids = [4, 8, 12, 16, 20]

    for first_index in range(len(fingertip_ids)):
        for second_index in range(first_index + 1, len(fingertip_ids)):
            features.append(
                calculate_distance(
                    points[fingertip_ids[first_index]],
                    points[fingertip_ids[second_index]]
                )
            )

    # 4. Finger curl values: 5
    # Distance from each fingertip to its MCP joint.
    finger_mcp_tip_pairs = [
        (1, 4),
        (5, 8),
        (9, 12),
        (13, 16),
        (17, 20)
    ]

    for mcp_id, tip_id in finger_mcp_tip_pairs:
        features.append(
            calculate_distance(points[mcp_id], points[tip_id])
        )

    # 5. Adjacent fingertip spread: 4
    adjacent_tip_pairs = [
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20)
    ]

    for first_tip, second_tip in adjacent_tip_pairs:
        features.append(
            calculate_distance(points[first_tip], points[second_tip])
        )

    # 6. Palm orientation normal vector: 3
    wrist_to_index = points[5] - points[0]
    wrist_to_pinky = points[17] - points[0]

    palm_normal = np.cross(wrist_to_index, wrist_to_pinky)
    palm_normal_length = np.linalg.norm(palm_normal)

    if palm_normal_length >= EPSILON:
        palm_normal = palm_normal / palm_normal_length
    else:
        palm_normal = np.zeros(3)

    features.extend(palm_normal.tolist())

    # 7. Wrist-to-middle-MCP direction vector: 3
    hand_direction = points[9] - points[0]
    hand_direction_length = np.linalg.norm(hand_direction)

    if hand_direction_length >= EPSILON:
        hand_direction = hand_direction / hand_direction_length
    else:
        hand_direction = np.zeros(3)

    features.extend(hand_direction.tolist())

    # 8. Palm dimensions: 2
    palm_width = calculate_distance(points[5], points[17])
    palm_height = calculate_distance(points[0], points[9])

    features.extend([palm_width, palm_height])

    # Final feature count:
    # 63 + 15 + 10 + 5 + 4 + 3 + 3 + 2 = 105
    return np.asarray(features, dtype=np.float64)


def extract_features_from_mediapipe(hand_landmarks):
    """Complete real-time feature extraction function."""
    normalized_points = normalize_mediapipe_landmarks(hand_landmarks)
    return extract_features_from_points(normalized_points)
