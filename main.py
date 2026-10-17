
import cv2
import numpy as np

# ==========================
# 1. VIDEO SETUP
# ==========================

video = cv2.VideoCapture("C:\\Users\\User\\OneDrive - Kaunas University of Technology\\Desktop\\ImageProcessing\\rocket-tracking\\data\\launch.mp4")

if not video.isOpened():
    raise FileNotFoundError("Cannot open video")

THRESHOLD = 150
MIN_AREA = 50

raw_trajectory = []
kalman_trajectory = []

frame_number = 0

# Noise removal kernel
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

# ==========================
# 2. KALMAN FILTER SETUP
# ==========================

# 4 state variables: x, y, vx, vy
# 2 measurements: x, y
kalman = cv2.KalmanFilter(4, 2)

# State transition matrix
# Assume constant velocity between frames
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

# Measurement matrix
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

# Process noise
kalman.processNoiseCov = (
    np.eye(4, dtype=np.float32) * 0.03
)

# Measurement noise
kalman.measurementNoiseCov = (
    np.eye(2, dtype=np.float32) * 10
)

kalman.errorCovPost = (
    np.eye(4, dtype=np.float32)
)

kalman_initialized = False

# ==========================
# 3. PROCESS VIDEO
# ==========================

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    frame = cv2.resize(frame, (800, 450))
    output = frame.copy()

    # Grayscale conversion
    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    # Gaussian smoothing
    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    # Brightness thresholding
    _, flame_mask = cv2.threshold(
        blurred,
        THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Morphological opening
    flame_mask = cv2.morphologyEx(
        flame_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        flame_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    measurement = None

    # ==========================
    # 4. ORIGINAL DETECTION
    # ==========================

    if contours:
        largest = max(
            contours,
            key=cv2.contourArea
        )

        if cv2.contourArea(largest) > MIN_AREA:
            x, y, w, h = cv2.boundingRect(largest)

            center_x = x + w // 2
            top_y = y

            measurement = np.array([
                [np.float32(center_x)],
                [np.float32(top_y)]
            ])

            raw_trajectory.append(
                (center_x, top_y)
            )

            # Bounding box
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2
            )

            # Original detected position: RED
            cv2.circle(
                output,
                (center_x, top_y),
                5,
                (0, 0, 255),
                -1
            )

    # ==========================
    # 5. KALMAN TRACKING
    # ==========================

    if measurement is not None:
        if not kalman_initialized:
            # Initialize at first detection
            kalman.statePost = np.array([
                [measurement[0, 0]],
                [measurement[1, 0]],
                [0],
                [0]
            ], dtype=np.float32)

            kalman.statePre = kalman.statePost.copy()
            kalman_initialized = True

    if kalman_initialized:

        # Predict next position
        prediction = kalman.predict()

        if measurement is not None:
            # Correct prediction with detection
            kalman.correct(measurement)

            # Use corrected position
            state = kalman.statePost
        else:
            # No detection: use prediction
            state = prediction

        kalman_x = int(state[0, 0])
        kalman_y = int(state[1, 0])

        kalman_trajectory.append(
            (kalman_x, kalman_y)
        )

        # Kalman position: GREEN
        cv2.circle(
            output,
            (kalman_x, kalman_y),
            5,
            (0, 255, 0),
            -1
        )

    # ==========================
    # 6. DRAW TRAJECTORIES
    # ==========================

    # Original trajectory: RED
    for i in range(1, len(raw_trajectory)):
        cv2.line(
            output,
            raw_trajectory[i - 1],
            raw_trajectory[i],
            (0, 0, 255),
            2
        )

    # Kalman trajectory: GREEN
    for i in range(1, len(kalman_trajectory)):
        cv2.line(
            output,
            kalman_trajectory[i - 1],
            kalman_trajectory[i],
            (0, 255, 0),
            2
        )

    # ==========================
    # 7. DISPLAY INFORMATION
    # ==========================

    cv2.putText(
        output,
        f"Frame: {frame_number}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "Red: Original",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        "Green: Kalman",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imshow("Rocket Tracking Comparison", output)
    cv2.imshow("Flame Mask", flame_mask)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("Tracking completed!")
print("Original detections:", len(raw_trajectory))
print("Kalman positions:", len(kalman_trajectory))
