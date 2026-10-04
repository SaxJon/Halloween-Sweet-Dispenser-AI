import os
import sys
import time
import threading
import queue
import logging
import traceback
import requests
from collections import deque

import cv2
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

WEB_HOST = "0.0.0.0"
WEB_PORT = 8989

# Web portal preview settings. These only affect the MJPEG browser feeds.
# Camera capture and AI inference remain unchanged.
WEB_STREAM_WIDTH = 640
WEB_STREAM_HEIGHT = 360
WEB_STREAM_FPS = 8
WEB_JPEG_QUALITY = 10

# Known cameras:
# 0 = PC-LM1E
# 1 = ID Camera (Intel F455)
# 2 = Microsoft LifeCam
TRACKING_CAMERA_INDEX = 0
IDENTIFICATION_CAMERA_INDEX = 1

TRACKING_CAMERA_WIDTH = 1280
TRACKING_CAMERA_HEIGHT = 720
TRACKING_CAMERA_FPS = 30
TRACKING_CAMERA_FLIP_HORIZONTAL = False
TRACKING_CAMERA_FLIP_VERTICAL = False
TRACKING_CAMERA_AUTOFOCUS = False
TRACKING_CAMERA_FOCUS = None

IDENTIFICATION_CAMERA_WIDTH = 1280
IDENTIFICATION_CAMERA_HEIGHT = 720
IDENTIFICATION_CAMERA_FPS = 30
IDENTIFICATION_CAMERA_FLIP_HORIZONTAL = False
IDENTIFICATION_CAMERA_FLIP_VERTICAL = False
IDENTIFICATION_CAMERA_AUTOFOCUS = False
IDENTIFICATION_CAMERA_FOCUS = None

YOLO_IMAGE_SIZE = 320
TRACKING_INFERENCE_INTERVAL = 3
IDENTIFICATION_INFERENCE_IMAGE_SIZE = 320
DISPLAY_LOCAL_WINDOWS = False

SHOW_ORIENTATION_GUIDES = True
SHOW_QUARTER_GUIDES = True
CENTRE_TARGET_SIZE = 25

MODEL = "yolo26n.pt"
CONFIDENCE = 0.45
PERSON_CLASS = 0

CROSSING_DIRECTION = "LEFT_TO_RIGHT"
APPROACH_DISTANCE = 100
CROSSING_TOLERANCE_PIXELS = 40
AUTO_SERVE_NEW_BEYOND_LINE = True

SAMPLES_PER_PERSON = 5
SAMPLE_DELAY_SECONDS = 0.03
APPEARANCE_MATCH_THRESHOLD = 0.82
IDENTIFICATION_MIN_AREA = 30000
IDENTIFICATION_RETRY_SECONDS = 0.30

# Keep FALSE until ready for the real dispenser.
ENABLE_REAL_DISPENSER = os.getenv("HALLOWEEN_AI_LIVE", "false").lower() == "true"
DISPENSER_URL = os.getenv("DISPENSER_URL", "http://127.0.0.1:5000/dispense")
DISPENSER_STATUS_URL = os.getenv("DISPENSER_STATUS_URL", "http://127.0.0.1:5000/status")

TRACK_TIMEOUT_SECONDS = 10
MAX_EVENTS = 100

# ============================================================
# SCARE MIRROR TRIGGER INTEGRATION
# ============================================================

# The mirror remains a separate application. Halloween AI only sends
# lightweight HTTP triggers to its existing local API.
MIRROR_ENABLED = os.getenv("MIRROR_ENABLED", "true").lower() == "true"
MIRROR_BASE_URL = os.getenv("MIRROR_BASE_URL", "http://127.0.0.1:5050")
MIRROR_ROUTE_TRICK_OR_TREAT = "/api/mirror/trick-or-treat"
MIRROR_ROUTE_DISPENSING = "/api/mirror/dispensing"
MIRROR_ROUTE_COLLECT = "/api/mirror/collect"
MIRROR_ROUTE_RANDOM_SCARE = "/api/mirror/scare/random"
MIRROR_ROUTE_RESET = "/api/mirror/reset"
MIRROR_ROUTE_STATUS = "/api/mirror/status"
MIRROR_REQUEST_TIMEOUT = 0.8

MIRROR_SCARE_AFTER_DISPENSE = True
MIRROR_SCARE_DELAY_SECONDS = 3.0
MIRROR_RESET_DELAY_SECONDS = 5.0

# ============================================================
# V2 RELIABILITY / HALLOWEEN COSTUME FALLBACK
# ============================================================

# ID is allowed a bounded time to identify a visitor. If it cannot,
# V2 fails open and serves rather than leaving a visitor without sweets.
ID_FAIL_OPEN_SECONDS = 1.5

# Keep a visitor eligible briefly if ByteTrack loses/reassigns the track.
PENDING_VISITOR_SECONDS = 4.0

# Fixed-camera motion fallback. This is deliberately a fallback, not a
# replacement for YOLO. It helps with bulky / unusual Halloween costumes.
ENABLE_COSTUME_MOTION_FALLBACK = True
MOTION_MIN_AREA = 22000
MOTION_THRESHOLD = 32
MOTION_REQUIRED_FRAMES = 4
MOTION_COOLDOWN_SECONDS = 20.0
MOTION_DIRECTION_MIN_PIXELS = 35
MOTION_BACKGROUND_ALPHA = 0.025

# V2.3 fallback filtering
FALLBACK_MIN_HEIGHT_PIXELS = 180
FALLBACK_MIN_HEIGHT_TO_WIDTH = 0.85
FALLBACK_USE_ID_APPEARANCE = True


# ============================================================
# LOGGING
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "person_detector.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("HalloweenAI")

# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.RLock()
frame_lock = threading.RLock()
identification_camera_lock = threading.RLock()

served_track_ids = set()
previous_x = {}
last_seen = {}
approach_confirmed = {}
crossing_pending = {}
appearance_decisions = {}
identification_last_attempt = {}
served_people = []

next_person_number = 1
total_served = 0
last_served_id = None
last_match_score = None
last_match_person = None
last_dispense_reason = "NONE"
last_id_status = "IDLE"

frame_counter = 0
tracking_inference_count = 0
identification_inference_count = 0
fps_counter = 0
fps_value = 0.0
fps_timer = time.time()

last_tracking_boxes = []
last_tracking_ids = []
last_id_box = None

latest_tracking_jpeg = None
latest_identification_jpeg = None

app_started_at = time.time()
detector_running = False
shutdown_requested = False

events = deque(maxlen=MAX_EVENTS)


# V2 reliability state
approach_started_at = {}
pending_until = {}
fallback_background = None
fallback_centres = deque(maxlen=12)
fallback_active_frames = 0
fallback_last_dispense = 0.0
fallback_status = "IDLE"


mirror_queue = queue.Queue()
mirror_last_action = "IDLE"
mirror_last_ok = None
mirror_last_error = None
mirror_worker_started = False
mirror_approach_tracks = set()



# ============================================================
# EVENT LOG
# ============================================================

def add_event(message, level="INFO"):
    item = {
        "time": time.strftime("%H:%M:%S"),
        "message": str(message),
        "level": level,
    }
    with state_lock:
        events.appendleft(item)

    if level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)

# ============================================================
# FRAME HELPERS
# ============================================================

def set_web_frame(which, frame):
    global latest_tracking_jpeg, latest_identification_jpeg
    if frame is None:
        return
    # Downscale only the portal copy; the detector still uses the original frame.
    web_frame = cv2.resize(
        frame,
        (WEB_STREAM_WIDTH, WEB_STREAM_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    ok, encoded = cv2.imencode(
        ".jpg",
        web_frame,
        [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY],
    )
    if not ok:
        return
    data = encoded.tobytes()
    with frame_lock:
        if which == "tracking":
            latest_tracking_jpeg = data
        else:
            latest_identification_jpeg = data

def apply_camera_flip(frame, flip_horizontal, flip_vertical):
    if frame is None:
        return frame
    if flip_horizontal and flip_vertical:
        return cv2.flip(frame, -1)
    if flip_horizontal:
        return cv2.flip(frame, 1)
    if flip_vertical:
        return cv2.flip(frame, 0)
    return frame

def process_tracking_frame(frame):
    return apply_camera_flip(
        frame,
        TRACKING_CAMERA_FLIP_HORIZONTAL,
        TRACKING_CAMERA_FLIP_VERTICAL,
    )

def process_identification_frame(frame):
    return apply_camera_flip(
        frame,
        IDENTIFICATION_CAMERA_FLIP_HORIZONTAL,
        IDENTIFICATION_CAMERA_FLIP_VERTICAL,
    )

def draw_orientation_guides(frame, camera_label):
    if not SHOW_ORIENTATION_GUIDES:
        return frame

    height, width = frame.shape[:2]
    cx = width // 2
    cy = height // 2

    if SHOW_QUARTER_GUIDES:
        for x in (width // 4, (width * 3) // 4):
            cv2.line(frame, (x, 0), (x, height), (160, 160, 160), 1)
        for y in (height // 4, (height * 3) // 4):
            cv2.line(frame, (0, y), (width, y), (160, 160, 160), 1)

    cv2.line(frame, (cx, 0), (cx, height), (255, 255, 255), 1)
    cv2.line(frame, (0, cy), (width, cy), (255, 255, 255), 1)
    cv2.circle(frame, (cx, cy), CENTRE_TARGET_SIZE, (255, 255, 255), 1)
    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

    cv2.putText(frame, "TOP", (cx + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    cv2.putText(frame, "BOTTOM", (cx + 10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    cv2.putText(frame, "LEFT", (10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    cv2.putText(frame, "RIGHT", (width - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    cv2.putText(frame, camera_label, (cx + 10, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    return frame

# ============================================================
# CAMERA
# ============================================================

def open_camera(index, width, height, fps, camera_name, autofocus=False, manual_focus=None):
    add_event(f"Opening {camera_name} camera index {index}")

    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open {camera_name} camera index {index}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)

    if not autofocus and manual_focus is not None:
        camera.set(cv2.CAP_PROP_FOCUS, manual_focus)

    success, frame = camera.read()
    if not success or frame is None:
        camera.release()
        raise RuntimeError(f"{camera_name} camera opened but returned no image")

    h, w = frame.shape[:2]
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    autofocus_reported = camera.get(cv2.CAP_PROP_AUTOFOCUS)
    focus_reported = camera.get(cv2.CAP_PROP_FOCUS)

    add_event(f"{camera_name} camera OK: {w}x{h} @ {actual_fps:.1f} FPS")
    logger.info("%s autofocus: %s", camera_name, autofocus_reported)
    logger.info("%s focus: %s", camera_name, focus_reported)

    return camera, frame

# ============================================================
# APPEARANCE MATCHING
# ============================================================

def normalise_person_crop(crop):
    if crop is None or crop.size == 0:
        return None
    try:
        return cv2.resize(crop, (160, 320))
    except Exception:
        return None

def create_histogram(person_crop):
    person_crop = normalise_person_crop(person_crop)
    if person_crop is None:
        return None
    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [40, 40], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
    return histogram

def compare_histograms(a, b):
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))

def crop_box(frame, box):
    if frame is None or box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None

def detect_identification_person(model, frame):
    global identification_inference_count
    identification_inference_count += 1

    results = model.predict(
        frame,
        classes=[PERSON_CLASS],
        conf=CONFIDENCE,
        imgsz=IDENTIFICATION_INFERENCE_IMAGE_SIZE,
        verbose=False,
    )

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    best_box = None
    best_area = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None or best_area < IDENTIFICATION_MIN_AREA:
        return None

    return best_box

def capture_appearance_samples(identification_camera, model):
    global last_id_box, last_id_status

    samples = []

    with identification_camera_lock:
        success, frame = identification_camera.read()
        if not success or frame is None:
            last_id_status = "CAMERA READ FAILED"
            return []

        frame = process_identification_frame(frame)
        last_id_status = "IDENTIFYING"

        person_box = detect_identification_person(model, frame)
        if person_box is None:
            last_id_box = None
            last_id_status = "NO PERSON"
            return []

        last_id_box = person_box

        crop = crop_box(frame, person_box)
        histogram = create_histogram(crop)
        if histogram is not None:
            samples.append(histogram)

        attempts = 0
        max_attempts = SAMPLES_PER_PERSON * 3

        while len(samples) < SAMPLES_PER_PERSON and attempts < max_attempts:
            attempts += 1
            time.sleep(SAMPLE_DELAY_SECONDS)

            success, sample_frame = identification_camera.read()
            if not success or sample_frame is None:
                continue

            sample_frame = process_identification_frame(sample_frame)
            sample_crop = crop_box(sample_frame, person_box)
            histogram = create_histogram(sample_crop)

            if histogram is not None:
                samples.append(histogram)

    if samples:
        last_id_status = "CAPTURED"
    else:
        last_id_status = "CAPTURE FAILED"

    return samples

def compare_against_served(current_histograms):
    best_score = -1.0
    best_person = None

    if not current_histograms:
        return None, 0.0

    with state_lock:
        people_snapshot = list(served_people)

    for person in people_snapshot:
        scores = []
        for current in current_histograms:
            for saved in person["histograms"]:
                scores.append(compare_histograms(current, saved))

        if not scores:
            continue

        scores.sort(reverse=True)
        strongest = scores[:3]
        average = sum(strongest) / len(strongest)

        if average > best_score:
            best_score = average
            best_person = person

    return best_person, max(0.0, best_score)

# ============================================================
# VISITOR / DISPENSER
# ============================================================

def register_served_person(track_id, histograms):
    global next_person_number

    with state_lock:
        person = {
            "person_number": next_person_number,
            "track_id": track_id,
            "histograms": histograms,
        }
        served_people.append(person)
        number = next_person_number
        next_person_number += 1

    add_event(f"Registered visitor #{number} track={track_id} samples={len(histograms)}")
    return person

def reset_served_visitors():
    global next_person_number, total_served, last_served_id
    global last_match_score, last_match_person, last_id_status
    global last_id_box, last_dispense_reason

    with state_lock:
        served_people.clear()
        served_track_ids.clear()
        previous_x.clear()
        last_seen.clear()
        approach_confirmed.clear()
        crossing_pending.clear()
        appearance_decisions.clear()
        identification_last_attempt.clear()
        approach_started_at.clear()
        pending_until.clear()
        fallback_centres.clear()
        mirror_approach_tracks.clear()

        next_person_number = 1
        total_served = 0
        last_served_id = None
        last_match_score = None
        last_match_person = None
        last_id_status = "IDLE"
        last_id_box = None
        last_dispense_reason = "NONE"

    add_event("Served visitors reset", "WARNING")

def dispenser_request(reason="MANUAL"):
    global total_served, last_dispense_reason

    if not ENABLE_REAL_DISPENSER:
        add_event(f"SIMULATION dispense: {reason}")
        last_dispense_reason = reason
        return True, "Simulation dispense successful"

    try:
        import requests
        response = requests.post(DISPENSER_URL, timeout=10)
        response.raise_for_status()
        last_dispense_reason = reason
        add_event(f"Physical dispenser activated: {reason}")
        return True, "Dispenser activated"
    except Exception as exc:
        add_event(f"Dispenser request failed: {exc}", "ERROR")
        return False, str(exc)

def dispense(track_id, reason="UNKNOWN"):
    global total_served, last_served_id, last_dispense_reason

    with state_lock:
        if track_id in served_track_ids:
            return False

    queue_mirror(MIRROR_ROUTE_DISPENSING, "DISPENSING")

    if ENABLE_REAL_DISPENSER:
        ok, _ = dispenser_request(reason)
        if not ok:
            return False
    else:
        add_event(f"SIMULATION: would dispense for person {track_id} - {reason}")

    with state_lock:
        served_track_ids.add(track_id)
        total_served += 1
        last_served_id = track_id
        last_dispense_reason = reason

    add_event(f"DISPENSE track={track_id} reason={reason}")
    return True

def complete_dispense(track_id, decision, reason):
    with state_lock:
        if track_id in served_track_ids:
            crossing_pending[track_id] = False
            return False

    if decision is None:
        return False
    if not decision.get("checked", False):
        return False
    if decision.get("blocked", True):
        return False
    if decision.get("served", False):
        return False

    if dispense(track_id, reason):
        register_served_person(track_id, decision["histograms"])

        with state_lock:
            decision["served"] = True
            appearance_decisions[track_id] = decision
            crossing_pending[track_id] = False

        add_event(f"Track {track_id} successfully served and marked complete")
        mirror_after_successful_dispense()
        return True

    return False

def person_at_dispense_area(centre_x, dispense_line_x):
    if CROSSING_DIRECTION == "LEFT_TO_RIGHT":
        return centre_x >= dispense_line_x - CROSSING_TOLERANCE_PIXELS
    return centre_x <= dispense_line_x + CROSSING_TOLERANCE_PIXELS

def cleanup_tracks():
    now = time.time()
    with state_lock:
        expired = [
            track_id for track_id, seen_time in last_seen.items()
            if now - seen_time > TRACK_TIMEOUT_SECONDS
        ]

        for track_id in expired:
            if now <= pending_until.get(track_id, 0):
                continue
            previous_x.pop(track_id, None)
            last_seen.pop(track_id, None)
            approach_confirmed.pop(track_id, None)
            crossing_pending.pop(track_id, None)
            appearance_decisions.pop(track_id, None)
            identification_last_attempt.pop(track_id, None)
            approach_started_at.pop(track_id, None)
            pending_until.pop(track_id, None)

def update_fps():
    global fps_counter, fps_value, fps_timer
    fps_counter += 1
    now = time.time()
    elapsed = now - fps_timer
    if elapsed >= 1.0:
        fps_value = fps_counter / elapsed
        fps_counter = 0
        fps_timer = now

# ============================================================
# SCARE MIRROR HELPERS
# ============================================================

def mirror_request(route, action):
    """Send one mirror API request. Mirror failure must never block sweets."""
    global mirror_last_action, mirror_last_ok, mirror_last_error

    if not MIRROR_ENABLED:
        return False

    url = MIRROR_BASE_URL.rstrip("/") + route

    try:
        response = requests.post(url, timeout=MIRROR_REQUEST_TIMEOUT)
        response.raise_for_status()
        with state_lock:
            mirror_last_action = action
            mirror_last_ok = True
            mirror_last_error = None
        add_event(f"Mirror: {action}")
        return True
    except Exception as exc:
        with state_lock:
            mirror_last_action = action
            mirror_last_ok = False
            mirror_last_error = str(exc)
        add_event(f"Mirror unavailable during {action}: {exc}", "WARNING")
        return False


def mirror_worker():
    while True:
        item = mirror_queue.get()
        try:
            if item is None:
                return

            delay, route, action = item
            if delay:
                time.sleep(delay)
            mirror_request(route, action)
        finally:
            mirror_queue.task_done()


def start_mirror_worker():
    global mirror_worker_started
    if mirror_worker_started:
        return
    threading.Thread(
        target=mirror_worker,
        name="MirrorTriggerWorker",
        daemon=True,
    ).start()
    mirror_worker_started = True


def queue_mirror(route, action, delay=0.0):
    if not MIRROR_ENABLED:
        return
    start_mirror_worker()
    mirror_queue.put((delay, route, action))


def mirror_after_successful_dispense():
    # DISPENSING is queued immediately before the physical/simulated dispense.
    # COLLECT follows success, then optional scare, then reset.
    queue_mirror(MIRROR_ROUTE_COLLECT, "COLLECT")
    if MIRROR_SCARE_AFTER_DISPENSE:
        queue_mirror(
            MIRROR_ROUTE_RANDOM_SCARE,
            "RANDOM SCARE",
            MIRROR_SCARE_DELAY_SECONDS,
        )
        queue_mirror(
            MIRROR_ROUTE_RESET,
            "RESET",
            MIRROR_RESET_DELAY_SECONDS,
        )


def get_mirror_status():
    if not MIRROR_ENABLED:
        return {"enabled": False}

    try:
        response = requests.get(
            MIRROR_BASE_URL.rstrip("/") + MIRROR_ROUTE_STATUS,
            timeout=MIRROR_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return {"enabled": True, "online": True, "data": data}
    except Exception as exc:
        return {
            "enabled": True,
            "online": False,
            "error": str(exc),
        }


# ============================================================
# V2 RELIABILITY HELPERS
# ============================================================

def build_fail_open_decision(track_id):
    """Allow a tracked visitor through when ID could not decide in time."""
    with state_lock:
        decision = appearance_decisions.get(track_id)
        if decision is not None:
            return decision

        decision = {
            "checked": True,
            "blocked": False,
            "served": False,
            "score": 0.0,
            "histograms": [],
            "fail_open": True,
        }
        appearance_decisions[track_id] = decision

    add_event(
        f"Track {track_id} ID timeout - FAIL OPEN / SERVE",
        "WARNING",
    )
    return decision


def motion_costume_fallback(frame, dispense_line_x):
    """
    Fixed-camera presence fallback for costumes YOLO does not recognise.
    Returns True only after a large moving object persists and moves in
    the configured walking direction into the dispense area.
    """
    global fallback_background, fallback_active_frames
    global fallback_last_dispense, fallback_status

    if not ENABLE_COSTUME_MOTION_FALLBACK:
        fallback_status = "DISABLED"
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    gray_f = gray.astype("float32")

    if fallback_background is None:
        fallback_background = gray_f.copy()
        fallback_status = "LEARNING"
        return False

    background_u8 = cv2.convertScaleAbs(fallback_background)
    delta = cv2.absdiff(gray, background_u8)
    _, threshold = cv2.threshold(
        delta,
        MOTION_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )
    threshold = cv2.dilate(threshold, None, iterations=2)

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MOTION_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    # Learn the background slowly. Do not aggressively absorb a large
    # foreground visitor while they are in the dispense area.
    alpha = MOTION_BACKGROUND_ALPHA if best is None else MOTION_BACKGROUND_ALPHA * 0.15
    cv2.accumulateWeighted(gray_f, fallback_background, alpha)

    if best is None:
        fallback_active_frames = 0
        fallback_centres.clear()
        fallback_status = "CLEAR"
        return False

    x, y, w, h = best

    # Reject low/wide motion such as cats, dogs and foxes while remaining
    # permissive enough for bulky Halloween costumes.
    height_to_width = h / max(1.0, float(w))
    if h < FALLBACK_MIN_HEIGHT_PIXELS or height_to_width < FALLBACK_MIN_HEIGHT_TO_WIDTH:
        fallback_active_frames = 0
        fallback_centres.clear()
        fallback_status = f"IGNORED NON-HUMAN SHAPE h={h} ratio={height_to_width:.2f}"
        return False

    cx = x + (w // 2)
    fallback_centres.append(cx)
    fallback_active_frames += 1
    fallback_status = f"PRESENCE area={int(best_area)} h={h} ratio={height_to_width:.2f}"

    if fallback_active_frames < MOTION_REQUIRED_FRAMES:
        return False

    if len(fallback_centres) < MOTION_REQUIRED_FRAMES:
        return False

    movement = fallback_centres[-1] - fallback_centres[0]

    if CROSSING_DIRECTION == "LEFT_TO_RIGHT":
        direction_ok = movement >= MOTION_DIRECTION_MIN_PIXELS
        zone_ok = cx >= dispense_line_x - CROSSING_TOLERANCE_PIXELS
    else:
        direction_ok = movement <= -MOTION_DIRECTION_MIN_PIXELS
        zone_ok = cx <= dispense_line_x + CROSSING_TOLERANCE_PIXELS

    if not direction_ok or not zone_ok:
        return False

    now = time.time()
    if now - fallback_last_dispense < MOTION_COOLDOWN_SECONDS:
        fallback_status = "COOLDOWN"
        return False

    fallback_last_dispense = now
    fallback_status = "COSTUME FALLBACK SERVE"
    return True


def fallback_dispense(identification_camera, model):
    """Fallback serve with repeat-visitor protection from the ID camera."""
    global total_served, last_served_id, last_dispense_reason
    global last_match_score, last_match_person, last_id_status

    reason = "HALLOWEEN COSTUME/PRESENCE FALLBACK"
    histograms = []

    if FALLBACK_USE_ID_APPEARANCE:
        try:
            histograms = capture_appearance_samples(identification_camera, model)

            if histograms:
                matched_person, score = compare_against_served(histograms)
                with state_lock:
                    last_match_score = score
                    last_match_person = matched_person

                if matched_person is not None and score >= APPEARANCE_MATCH_THRESHOLD:
                    with state_lock:
                        last_id_status = f"MATCH {score:.2f}"
                    add_event(
                        f"Fallback blocked - matched visitor "
                        f"#{matched_person['person_number']} score={score:.3f}",
                        "WARNING",
                    )
                    return False

                with state_lock:
                    last_id_status = f"NEW {score:.2f}"
                add_event(f"Fallback visitor identified NEW score={score:.3f}")
            else:
                add_event(
                    "Fallback ID could not identify visitor - FAIL OPEN / SERVE",
                    "WARNING",
                )
        except Exception as exc:
            add_event(
                f"Fallback ID check failed - FAIL OPEN / SERVE: {exc}",
                "WARNING",
            )

    queue_mirror(MIRROR_ROUTE_TRICK_OR_TREAT, "TRICK OR TREAT")
    queue_mirror(MIRROR_ROUTE_DISPENSING, "DISPENSING")

    if ENABLE_REAL_DISPENSER:
        ok, _ = dispenser_request(reason)
        if not ok:
            return False
    else:
        add_event(f"SIMULATION: {reason}")

    with state_lock:
        total_served += 1
        last_served_id = "FALLBACK"
        last_dispense_reason = reason

    if histograms:
        register_served_person("FALLBACK", histograms)

    add_event(f"DISPENSE reason={reason}", "WARNING")
    mirror_after_successful_dispense()
    return True


# ============================================================
# WEB PORTAL
# ============================================================

app = Flask(__name__)

def get_pi_status():
    try:
        import requests
        response = requests.get(DISPENSER_STATUS_URL, timeout=0.7)
        response.raise_for_status()
        try:
            data = response.json()
            return {"reachable": True, "data": data}
        except Exception:
            return {"reachable": True, "data": {"response": response.text[:200]}}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/status")
def api_status():
    with state_lock:
        matched_number = None
        if isinstance(last_match_person, dict):
            matched_number = last_match_person.get("person_number")

        payload = {
            "running": detector_running,
            "uptime_seconds": int(time.time() - app_started_at),
            "mode": "LIVE" if ENABLE_REAL_DISPENSER else "SIMULATION",
            "people": len(last_tracking_ids),
            "served": total_served,
            "remembered": len(served_people),
            "fps": round(fps_value, 1),
            "id_status": last_id_status,
            "last_similarity": None if last_match_score is None else round(last_match_score, 3),
            "last_match_person": matched_number,
            "last_served_id": last_served_id,
            "last_dispense_reason": last_dispense_reason,
            "tracking_camera": TRACKING_CAMERA_INDEX,
            "identification_camera": IDENTIFICATION_CAMERA_INDEX,
            "fallback_status": fallback_status,
            "mirror_enabled": MIRROR_ENABLED,
            "mirror_last_action": mirror_last_action,
            "mirror_last_ok": mirror_last_ok,
            "mirror_last_error": mirror_last_error,
            "events": list(events)[:30],
        }

    # Only check the Pi when explicitly requested to keep dashboard light.
    return jsonify(payload)

@app.route("/api/dispenser-status")
def api_dispenser_status():
    return jsonify(get_pi_status())

@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_served_visitors()
    return jsonify({"ok": True})

@app.route("/api/test-dispense", methods=["POST"])
def api_test_dispense():
    ok, message = dispenser_request("WEB PORTAL TEST")
    return jsonify({"ok": ok, "message": message}), (200 if ok else 500)

@app.route("/api/clear-events", methods=["POST"])
def api_clear_events():
    with state_lock:
        events.clear()
    add_event("Event log cleared")
    return jsonify({"ok": True})

def mjpeg_stream(which):
    while True:
        with frame_lock:
            frame = latest_tracking_jpeg if which == "tracking" else latest_identification_jpeg

        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )
        time.sleep(1.0 / max(1, WEB_STREAM_FPS))

@app.route("/api/mirror-status")
def api_mirror_status():
    return jsonify(get_mirror_status())


@app.route("/api/mirror/trick-or-treat", methods=["POST"])
def api_mirror_trick_or_treat():
    queue_mirror(MIRROR_ROUTE_TRICK_OR_TREAT, "TRICK OR TREAT")
    return jsonify({"ok": True})


@app.route("/api/mirror/dispensing", methods=["POST"])
def api_mirror_dispensing():
    queue_mirror(MIRROR_ROUTE_DISPENSING, "DISPENSING")
    return jsonify({"ok": True})


@app.route("/api/mirror/collect", methods=["POST"])
def api_mirror_collect():
    queue_mirror(MIRROR_ROUTE_COLLECT, "COLLECT")
    return jsonify({"ok": True})


@app.route("/api/mirror/scare", methods=["POST"])
def api_mirror_scare():
    queue_mirror(MIRROR_ROUTE_RANDOM_SCARE, "RANDOM SCARE")
    return jsonify({"ok": True})


@app.route("/api/mirror/reset", methods=["POST"])
def api_mirror_reset():
    queue_mirror(MIRROR_ROUTE_RESET, "RESET")
    return jsonify({"ok": True})


@app.route("/video/tracking")
def video_tracking():
    return Response(
        mjpeg_stream("tracking"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/video/identification")
def video_identification():
    return Response(
        mjpeg_stream("identification"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

def run_web_server():
    add_event(f"Web portal starting on port {WEB_PORT}")
    app.run(
        host=WEB_HOST,
        port=WEB_PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )

# ============================================================
# DETECTOR
# ============================================================

def run_detector():
    global frame_counter, tracking_inference_count
    global last_tracking_boxes, last_tracking_ids
    global last_match_score, last_match_person
    global last_id_status, last_id_box
    global detector_running

    if TRACKING_CAMERA_INDEX == IDENTIFICATION_CAMERA_INDEX:
        raise RuntimeError("Tracking and identification camera indexes must be different.")

    model = YOLO(MODEL)

    tracking_camera, tracking_frame = open_camera(
        TRACKING_CAMERA_INDEX,
        TRACKING_CAMERA_WIDTH,
        TRACKING_CAMERA_HEIGHT,
        TRACKING_CAMERA_FPS,
        "Tracking",
        TRACKING_CAMERA_AUTOFOCUS,
        TRACKING_CAMERA_FOCUS,
    )

    identification_camera, _ = open_camera(
        IDENTIFICATION_CAMERA_INDEX,
        IDENTIFICATION_CAMERA_WIDTH,
        IDENTIFICATION_CAMERA_HEIGHT,
        IDENTIFICATION_CAMERA_FPS,
        "Identification",
        IDENTIFICATION_CAMERA_AUTOFOCUS,
        IDENTIFICATION_CAMERA_FOCUS,
    )

    tracking_frame = process_tracking_frame(tracking_frame)
    tracking_height, tracking_width = tracking_frame.shape[:2]

    dispense_line_x = tracking_width // 2

    if CROSSING_DIRECTION == "LEFT_TO_RIGHT":
        approach_line_x = dispense_line_x - APPROACH_DISTANCE
        tolerance_x = dispense_line_x - CROSSING_TOLERANCE_PIXELS
    else:
        approach_line_x = dispense_line_x + APPROACH_DISTANCE
        tolerance_x = dispense_line_x + CROSSING_TOLERANCE_PIXELS

    add_event(f"Approach line: {approach_line_x}")
    add_event(f"Dispense line: {dispense_line_x}")

    detector_running = True
    add_event("Detector running")

    try:
        while not shutdown_requested:
            now = time.time()
            frame_counter += 1
            update_fps()

            success, frame = tracking_camera.read()
            if not success or frame is None:
                continue

            frame = process_tracking_frame(frame)

            run_inference = frame_counter % TRACKING_INFERENCE_INTERVAL == 0

            if run_inference:
                tracking_inference_count += 1

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[PERSON_CLASS],
                    conf=CONFIDENCE,
                    imgsz=YOLO_IMAGE_SIZE,
                    verbose=False,
                )

                result = results[0]

                if result.boxes is not None and result.boxes.id is not None:
                    last_tracking_boxes = result.boxes.xyxy.cpu().numpy()
                    last_tracking_ids = result.boxes.id.int().cpu().tolist()
                else:
                    last_tracking_boxes = []
                    last_tracking_ids = []

            annotated = frame.copy()

            if run_inference:
                for box, track_id in zip(last_tracking_boxes, last_tracking_ids):
                    x1, y1, x2, y2 = map(int, box)
                    centre_x = int((x1 + x2) / 2)

                    with state_lock:
                        last_seen[track_id] = now

                    if CROSSING_DIRECTION == "LEFT_TO_RIGHT":
                        approached = centre_x <= approach_line_x
                    else:
                        approached = centre_x >= approach_line_x

                    if approached:
                        with state_lock:
                            first_approach = not approach_confirmed.get(track_id, False)
                            approach_confirmed[track_id] = True
                        if first_approach:
                            with state_lock:
                                approach_started_at[track_id] = now
                                pending_until[track_id] = now + PENDING_VISITOR_SECONDS
                            add_event(f"Track {track_id} entered approach area")
                            if track_id not in mirror_approach_tracks:
                                mirror_approach_tracks.add(track_id)
                                queue_mirror(
                                    MIRROR_ROUTE_TRICK_OR_TREAT,
                                    "TRICK OR TREAT",
                                )

                    with state_lock:
                        old_x = previous_x.get(track_id)

                    crossed = False
                    if old_x is not None:
                        if CROSSING_DIRECTION == "LEFT_TO_RIGHT":
                            crossed = old_x < dispense_line_x and centre_x >= dispense_line_x
                        else:
                            crossed = old_x > dispense_line_x and centre_x <= dispense_line_x

                    with state_lock:
                        approached_ok = approach_confirmed.get(track_id, False)
                        already_served = track_id in served_track_ids

                    if crossed and approached_ok and not already_served:
                        with state_lock:
                            crossing_pending[track_id] = True
                        add_event(f"Track {track_id} crossed dispense line")

                    with state_lock:
                        needs_identification = (
                            approach_confirmed.get(track_id, False)
                            and track_id not in appearance_decisions
                            and track_id not in served_track_ids
                        )

                    if needs_identification:
                        with state_lock:
                            last_attempt = identification_last_attempt.get(track_id, 0)

                        if now - last_attempt >= IDENTIFICATION_RETRY_SECONDS:
                            with state_lock:
                                identification_last_attempt[track_id] = now

                            current_histograms = capture_appearance_samples(
                                identification_camera,
                                model,
                            )

                            if current_histograms:
                                matched_person, score = compare_against_served(current_histograms)

                                with state_lock:
                                    last_match_score = score
                                    last_match_person = matched_person

                                if (
                                    matched_person is not None
                                    and score >= APPEARANCE_MATCH_THRESHOLD
                                ):
                                    decision = {
                                        "checked": True,
                                        "blocked": True,
                                        "served": False,
                                        "score": score,
                                        "histograms": current_histograms,
                                        "matched_person": matched_person["person_number"],
                                    }

                                    with state_lock:
                                        appearance_decisions[track_id] = decision
                                        last_id_status = f"MATCH {score:.2f}"

                                    add_event(
                                        f"Track {track_id} matched visitor "
                                        f"#{matched_person['person_number']} score={score:.3f}",
                                        "WARNING",
                                    )
                                else:
                                    decision = {
                                        "checked": True,
                                        "blocked": False,
                                        "served": False,
                                        "score": score,
                                        "histograms": current_histograms,
                                    }

                                    with state_lock:
                                        appearance_decisions[track_id] = decision
                                        last_id_status = f"NEW {score:.2f}"

                                    add_event(f"Track {track_id} identified NEW score={score:.3f}")

                    # V2: if ID cannot see/identify the visitor within the
                    # bounded window, serve rather than fail closed.
                    with state_lock:
                        started = approach_started_at.get(track_id)
                        has_decision = track_id in appearance_decisions
                        approached_now = approach_confirmed.get(track_id, False)

                    if (
                        approached_now
                        and not has_decision
                        and started is not None
                        and now - started >= ID_FAIL_OPEN_SECONDS
                    ):
                        build_fail_open_decision(track_id)

                    with state_lock:
                        decision = appearance_decisions.get(track_id)
                        already_served = track_id in served_track_ids

                    if already_served:
                        with state_lock:
                            crossing_pending[track_id] = False

                    elif decision is not None and decision.get("checked", False):
                        if decision.get("blocked", False):
                            with state_lock:
                                if crossing_pending.get(track_id, False):
                                    crossing_pending[track_id] = False
                                    add_event(f"Track {track_id} blocked - previously served", "WARNING")

                        elif not decision.get("served", False):
                            with state_lock:
                                pending = crossing_pending.get(track_id, False)
                                approached_ok = approach_confirmed.get(track_id, False)

                            if pending:
                                complete_dispense(track_id, decision, "CROSSING DETECTED")

                            elif (
                                AUTO_SERVE_NEW_BEYOND_LINE
                                and approached_ok
                                and person_at_dispense_area(centre_x, dispense_line_x)
                            ):
                                complete_dispense(
                                    track_id,
                                    decision,
                                    "NEW PERSON IN DISPENSE AREA",
                                )

                    with state_lock:
                        previous_x[track_id] = centre_x

            # V2 costume fallback only takes control when YOLO currently
            # has no person track. This avoids double-serving a normal track.
            if not last_tracking_ids:
                if motion_costume_fallback(frame, dispense_line_x):
                    fallback_dispense(identification_camera, model)

            # Draw tracking boxes.
            for box, track_id in zip(last_tracking_boxes, last_tracking_ids):
                x1, y1, x2, y2 = map(int, box)
                centre_x = int((x1 + x2) / 2)
                centre_y = int((y1 + y2) / 2)

                with state_lock:
                    decision = appearance_decisions.get(track_id)
                    served = track_id in served_track_ids
                    approached = approach_confirmed.get(track_id, False)

                if served:
                    status = "SERVED"
                elif decision and decision.get("blocked", False):
                    status = f"BLOCKED {decision['score']:.2f}"
                elif decision and decision.get("checked", False):
                    status = f"NEW {decision['score']:.2f}"
                elif approached:
                    status = "WAITING FOR ID"
                else:
                    status = "TRACKING"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(annotated, (centre_x, centre_y), 6, (0,255,255), -1)
                cv2.putText(
                    annotated,
                    f"Person {track_id} [{status}]",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .6,
                    (255,255,255),
                    2,
                )

            cleanup_tracks()

            draw_orientation_guides(annotated, "TRACKING CENTRE")
            cv2.line(annotated, (approach_line_x, 0), (approach_line_x, tracking_height), (255,255,0), 2)
            cv2.line(annotated, (tolerance_x, 0), (tolerance_x, tracking_height), (0,165,255), 1)
            cv2.line(annotated, (dispense_line_x, 0), (dispense_line_x, tracking_height), (0,0,255), 3)

            with state_lock:
                overlay_served = total_served
                overlay_remembered = len(served_people)
                overlay_id = last_id_status

            mode = "LIVE" if ENABLE_REAL_DISPENSER else "SIMULATION"

            cv2.putText(annotated, f"People: {len(last_tracking_ids)}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.putText(annotated, f"Served: {overlay_served}", (20,65), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.putText(annotated, f"Remembered: {overlay_remembered}", (20,95), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.putText(annotated, f"Mode: {mode}", (20,125), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.putText(annotated, f"FPS: {fps_value:.1f}", (20,155), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            cv2.putText(annotated, f"ID: {overlay_id}", (20,185), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
            cv2.putText(annotated, f"Fallback: {fallback_status}", (20,215), cv2.FONT_HERSHEY_SIMPLEX, .50, (255,255,255), 1)

            set_web_frame("tracking", annotated)

            # ID preview - no continuous YOLO.
            with identification_camera_lock:
                success_id, id_frame = identification_camera.read()

            if success_id and id_frame is not None:
                id_frame = process_identification_frame(id_frame)
                id_display = id_frame.copy()
                draw_orientation_guides(id_display, "ID CENTRE")

                with state_lock:
                    box = last_id_box
                    id_text = last_id_status
                    remembered = len(served_people)
                    similarity = last_match_score

                if box is not None:
                    ix1, iy1, ix2, iy2 = box
                    cv2.rectangle(id_display, (ix1,iy1), (ix2,iy2), (0,255,0), 2)

                cv2.putText(id_display, "ID CAMERA", (20,35), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
                cv2.putText(id_display, f"Status: {id_text}", (20,65), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
                cv2.putText(id_display, f"Remembered: {remembered}", (20,95), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)

                if similarity is not None:
                    cv2.putText(id_display, f"Similarity: {similarity:.3f}", (20,125), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)

                cv2.putText(id_display, "YOLO: ON-DEMAND ONLY", (20,155), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
                set_web_frame("identification", id_display)

                if DISPLAY_LOCAL_WINDOWS:
                    cv2.imshow("Halloween AI - ID", id_display)

            if DISPLAY_LOCAL_WINDOWS:
                cv2.imshow("Halloween AI - Tracking", annotated)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("r"):
                    reset_served_visitors()
            else:
                time.sleep(0.001)

    finally:
        detector_running = False
        tracking_camera.release()
        identification_camera.release()
        cv2.destroyAllWindows()
        add_event("Detector stopped", "WARNING")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    try:
        web_thread = threading.Thread(
            target=run_web_server,
            name="WebPortal",
            daemon=True,
        )
        web_thread.start()

        run_detector()

    except KeyboardInterrupt:
        logger.info("Stopped with Ctrl+C")

    except Exception as exc:
        logger.critical("APPLICATION CRASHED: %s", exc)
        logger.critical(traceback.format_exc())
        print("")
        print("========================================")
        print(" APPLICATION CRASHED")
        print("========================================")
        print(exc)
        print(f"Log: {LOG_FILE}")
        print("")

