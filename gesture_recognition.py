import cv2
import mediapipe as mp
import math

# ============================================================
# FACE + HAND FINGER GESTURE RECOGNITION
# MediaPipe Hands + OpenCV Face Detection
# Recommended: Python 3.12
# ============================================================

CAMERA_INDEX = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

# OpenCV face detector (built-in Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Finger landmark IDs:
# Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
# For non-thumb fingers, tip is above PIP when the finger is extended.
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def angle(a, b, c):
    """Angle ABC in degrees."""
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0

    value = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(value))


def count_fingers(landmarks, handedness):
    """Count extended fingers from MediaPipe hand landmarks."""

    count = 0
    wrist = landmarks[0]

    # Index, Middle, Ring, Pinky
    finger_data = [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ]

    for mcp_id, pip_id, dip_id, tip_id in finger_data:
        mcp = landmarks[mcp_id]
        pip = landmarks[pip_id]
        dip = landmarks[dip_id]
        tip = landmarks[tip_id]

        pip_angle = angle(mcp, pip, dip)
        dip_angle = angle(pip, dip, tip)

        if (
            pip_angle > 150
            and dip_angle > 145
            and distance(tip, wrist) > distance(pip, wrist) * 1.10
        ):
            count += 1

    # Thumb
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    thumb_mcp_angle = angle(wrist, thumb_mcp, thumb_ip)
    thumb_ip_angle = angle(thumb_mcp, thumb_ip, thumb_tip)

    thumb_extended = (
        thumb_mcp_angle > 115
        and thumb_ip_angle > 145
        and distance(thumb_tip, wrist) >
        distance(thumb_mcp, wrist) * 1.35
    )

    if thumb_extended:
        count += 1

    return max(0, min(5, count))


def gesture_name(count):
    names = {
        0: "FIST",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE"
    }
    return names.get(count, "UNKNOWN")


print("==============================================")
print(" FACE + HAND GESTURE RECOGNITION")
print("==============================================")
print("Starting camera...")
print("Press Q to quit.")
print()

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

# Optional camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Camera frame failed.")
        break

    frame = cv2.flip(frame, 1)

    # ========================================================
    # FACE DETECTION
    # ========================================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=6,
        minSize=(70, 70)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 120, 0),
            2
        )

        cv2.putText(
            frame,
            "FACE",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 120, 0),
            2
        )

    # ========================================================
    # HAND DETECTION
    # ========================================================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_count = 0
    detected_gestures = []

    if results.multi_hand_landmarks:

        hand_count = len(results.multi_hand_landmarks)

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Handedness
            handedness = "Right"

            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = (
                    results.multi_handedness[i]
                    .classification[0]
                    .label
                )

            landmarks = hand_landmarks.landmark

            fingers = count_fingers(
                landmarks,
                handedness
            )

            gesture = gesture_name(fingers)

            detected_gestures.append(
                f"{handedness}: {gesture} ({fingers})"
            )

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Find hand text position
            wrist = landmarks[0]
            px = int(wrist.x * frame.shape[1])
            py = int(wrist.y * frame.shape[0])

            cv2.putText(
                frame,
                f"{gesture} | Fingers: {fingers}",
                (max(10, px - 80), max(30, py - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

    # ========================================================
    # INFORMATION PANEL
    # ========================================================
    cv2.rectangle(
        frame,
        (10, 10),
        (500, 145),
        (0, 0, 0),
        -1
    )

    face_text = f"Faces: {len(faces)}"
    hand_text = f"Hands: {hand_count}"

    cv2.putText(
        frame,
        face_text,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 180, 0),
        2
    )

    cv2.putText(
        frame,
        hand_text,
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    if detected_gestures:
        cv2.putText(
            frame,
            " | ".join(detected_gestures),
            (25, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1
        )
    else:
        cv2.putText(
            frame,
            "Show your hand",
            (25, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

    cv2.putText(
        frame,
        "Q = Quit",
        (frame.shape[1] - 110, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Face + Hand Gesture Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Program stopped.")