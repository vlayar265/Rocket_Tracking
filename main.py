
import cv2
import numpy as np
import time

# ==========================================
# SETTINGS
# ==========================================

VIDEO_PATH = "data/launch_day.mp4"

WIDTH = 1280
HEIGHT = 720

START_FRAME = 1200
MAX_FRAMES = 150

GRABCUT_ITERATIONS = 3
MARGIN = 20

SEARCH_MARGIN = 55
MIN_MATCH_SCORE = 0.45

SAVE_FRAME = 1250
OUTPUT_IMAGE = "rocket_comparison.png"


# ==========================================
# ROCKET TRACKING
# ==========================================

def track_rocket(frame, template, previous_box):
    x, y, w, h = previous_box

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x1 = max(0, x - SEARCH_MARGIN)
    y1 = max(0, y - SEARCH_MARGIN)
    x2 = min(WIDTH, x + w + SEARCH_MARGIN)
    y2 = min(HEIGHT, y + h + SEARCH_MARGIN)

    search = gray[y1:y2, x1:x2]

    if search.shape[0] < h or search.shape[1] < w:
        return None, -1.0

    response = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    _, score, _, location = cv2.minMaxLoc(response)

    return (
        x1 + location[0],
        y1 + location[1],
        w,
        h
    ), float(score)


# ==========================================
# GRABCUT SEGMENTATION
# ==========================================

def segment_rocket(frame, box, modified=False):
    x, y, w, h = box

    x1 = max(0, x - MARGIN)
    y1 = max(0, y - MARGIN)
    x2 = min(WIDTH, x + w + MARGIN)
    y2 = min(HEIGHT, y + h + MARGIN)

    crop = frame[y1:y2, x1:x2].copy()

    rx = x - x1
    ry = y - y1

    mask = np.zeros(crop.shape[:2], np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    if not modified:
        # Original: rectangle initialization
        cv2.grabCut(
            crop,
            mask,
            (rx, ry, w, h),
            bg_model,
            fg_model,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_RECT
        )

    else:
        # Modified: shape-guided initialization
        # Background outside the selected rectangle
        mask[:] = cv2.GC_BGD

        # Everything inside the rectangle starts
        # as probable foreground, not definite foreground
        mask[ry:ry+h, rx:rx+w] = cv2.GC_PR_FGD

        center_x = rx + w // 2

        # A narrow tapered seed INSIDE the rocket
        # We deliberately leave the pointed nose unseeded
        # so GrabCut can decide its actual boundary.

        top = ry + int(0.22 * h)
        bottom = ry + int(0.78 * h)

        if bottom > top:
            seed_points = np.array([
                [center_x - max(1, int(0.025 * w)), top],
                [center_x + max(1, int(0.025 * w)), top],
                [center_x + max(2, int(0.10 * w)), bottom],
                [center_x - max(2, int(0.10 * w)), bottom]
            ], dtype=np.int32)

            cv2.fillPoly(
                mask,
                [seed_points],
                cv2.GC_FGD
            )

        cv2.grabCut(
            crop,
            mask,
            None,
            bg_model,
            fg_model,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_MASK
        )

    binary = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return binary[ry:ry+h, rx:rx+w].copy()


# ==========================================
# VISUALIZATION
# ==========================================

def overlay_mask(frame, box, mask, color):
    result = frame.copy()
    x, y, w, h = box

    roi = result[y:y+h, x:x+w]

    colored = np.zeros_like(roi)
    colored[:] = color

    blended = cv2.addWeighted(
        roi, 0.55,
        colored, 0.45,
        0
    )

    roi[mask > 0] = blended[mask > 0]

    return result


def isolated_rocket(frame, box, mask):
    x, y, w, h = box

    crop = frame[y:y+h, x:x+w]
    result = np.zeros_like(crop)

    result[mask > 0] = crop[mask > 0]

    return result


def make_body_comparison(frame, box, original_mask, modified_mask):
    original = isolated_rocket(frame, box, original_mask)
    modified = isolated_rocket(frame, box, modified_mask)

    scale = 4

    original = cv2.resize(
        original, None,
        fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST
    )

    modified = cv2.resize(
        modified, None,
        fx=scale, fy=scale,
        interpolation=cv2.INTER_NEAREST
    )

    header = 45
    h, w = original.shape[:2]

    comparison = np.zeros(
        (h + header, 2 * w, 3),
        dtype=np.uint8
    )

    comparison[header:, :w] = original
    comparison[header:, w:] = modified

    cv2.putText(
        comparison,
        "Original GrabCut",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        comparison,
        "Tapered GrabCut",
        (w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return comparison


# ==========================================
# VIDEO SETUP
# ==========================================

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise FileNotFoundError(VIDEO_PATH)

video.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

success, first_frame = video.read()

if not success:
    video.release()
    raise RuntimeError("Cannot read starting frame")

first_frame = cv2.resize(first_frame, (WIDTH, HEIGHT))

print("Select ONLY the rocket body.")
print("Include the nose but exclude flame and smoke.")
print("Press ENTER to confirm.")

selection = cv2.selectROI(
    "Select Rocket",
    first_frame,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyWindow("Select Rocket")

x, y, w, h = map(int, selection)

if w <= 2 or h <= 2:
    video.release()
    raise RuntimeError("Invalid ROI")

template = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)[y:y+h, x:x+w].copy()

previous_box = (x, y, w, h)

video.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

original_times = []
modified_times = []

frame_count = 0


# ==========================================
# PROCESS VIDEO
# ==========================================

while frame_count < MAX_FRAMES:
    success, frame = video.read()

    if not success:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame_number = START_FRAME + frame_count

    if frame_count == 0:
        box = previous_box
        score = 1.0
    else:
        box, score = track_rocket(
            frame,
            template,
            previous_box
        )

    if box is None or score < MIN_MATCH_SCORE:
        print(f"Frame {frame_number}: tracking lost")
        frame_count += 1
        continue

    # Original GrabCut
    start = time.perf_counter()

    original_mask = segment_rocket(
        frame,
        box,
        modified=False
    )

    original_time = (
        time.perf_counter() - start
    ) * 1000

    # Modified GrabCut
    start = time.perf_counter()

    modified_mask = segment_rocket(
        frame,
        box,
        modified=True
    )

    modified_time = (
        time.perf_counter() - start
    ) * 1000

    original_times.append(original_time)
    modified_times.append(modified_time)

    previous_box = box

    # Colored segmentation comparison
    original_overlay = overlay_mask(
        frame, box, original_mask, (0, 0, 255)
    )

    modified_overlay = overlay_mask(
        frame, box, modified_mask, (0, 255, 0)
    )

    comparison = np.hstack(
        (original_overlay, modified_overlay)
    )

    comparison = cv2.resize(
        comparison,
        (1280, 360)
    )

    cv2.putText(
        comparison,
        "Original GrabCut",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        comparison,
        "Tapered GrabCut",
        (655, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    body_comparison = make_body_comparison(
        frame,
        box,
        original_mask,
        modified_mask
    )

    cv2.imshow(
        "Rocket Segmentation Comparison",
        comparison
    )

    cv2.imshow(
        "Isolated Rocket Bodies",
        body_comparison
    )

    if frame_number == SAVE_FRAME:
        cv2.imwrite(OUTPUT_IMAGE, body_comparison)
        print(f"Saved {OUTPUT_IMAGE}")

    print(
        f"Frame {frame_number} "
        f"| NCC: {score:.3f} "
        f"| Original: {original_time:.1f} ms "
        f"| Modified: {modified_time:.1f} ms"
    )

    frame_count += 1

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


# ==========================================
# RESULTS
# ==========================================

video.release()
cv2.destroyAllWindows()

print("\n===== EXPERIMENT RESULTS =====")
print("Processed frames:", len(original_times))

if original_times:
    print(
        "Average original time:",
        round(np.mean(original_times), 2),
        "ms"
    )

    print(
        "Average modified time:",
        round(np.mean(modified_times), 2),
        "ms"
    )
