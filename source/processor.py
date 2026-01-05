
import os
import time
import threading
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
import mediapipe as mp 

from deepface import DeepFace


DEEPFACE_EVERY_N_FRAMES = 5
EMOTION_SMOOTHING_WINDOW = 7
DEEPFACE_ENFORCE_DETECTION = False
DETECTOR_BACKEND = "opencv"

STICKER_FILES = {
    "angry": "angry_monkey",
    "happy": "happy_cat",
    "sad": "sad_cat",
    "fear": "scared_monkey",
    "disgust": "crazy_cat",
    "ok": "okay_monkey",
    "thinking": "thinking_monkey",
}

EMOTION_MAP = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "fear",
}

@dataclass
class FaceState:
    emotion: str | None = None
    face_box: tuple[int, int, int, int] | None = None
    updated_at: float = 0.0

#YARDIMCI FONKLAR
def try_read_image_any_ext(base_path_no_ext: str):
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = base_path_no_ext + ext
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img, p
    return None, None

def overlay_sticker(frame, sticker, x, y, size):
    if sticker is None:
        return frame

    H, W = frame.shape[:2]
    size = int(max(1, min(size, W, H)))
    x = int(max(0, min(x, W - size)))
    y = int(max(0, min(y, H - size)))

    st = cv2.resize(sticker, (size, size), interpolation=cv2.INTER_AREA)

    if st.ndim == 3 and st.shape[2] == 4:
        bgr = st[:, :, :3].astype(np.float32)
        alpha = (st[:, :, 3:4].astype(np.float32) / 255.0)
        roi = frame[y:y + size, x:x + size].astype(np.float32)
        out = roi * (1.0 - alpha) + bgr * alpha
        frame[y:y + size, x:x + size] = out.astype(np.uint8)
    else:
        frame[y:y + size, x:x + size] = st[:, :, :3]

    return frame

def crazy_tongue_like(frame_bgr, face_box):
    if face_box is None:
        return False

    H, W = frame_bgr.shape[:2]
    x, y, w, h = face_box
    if w < 60 or h < 60:
        return False

    rx1 = int(x + w * 0.22)
    rx2 = int(x + w * 0.78)
    ry1 = int(y + h * 0.52)
    ry2 = int(y + h * 0.95)

    rx1 = max(0, min(rx1, W - 1))
    rx2 = max(0, min(rx2, W))
    ry1 = max(0, min(ry1, H - 1))
    ry2 = max(0, min(ry2, H))

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return False

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)

    red = (((hch <= 15) | (hch >= 165)) & (sch >= 80) & (vch >= 40))
    red_ratio = float(np.mean(red))

    gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.mean(gray < 60))

    # Ağız açıkken içerisi koyu olur; dil çıkınca da kırmızı artar.
    # Eşikleri artırarak diğer duyguları engellemesini önlenir
    if red_ratio > 0.15:  # Was 0.10
        return True
    if red_ratio > 0.10 and dark_ratio > 0.12: # Was 0.06 / 0.08
        return True
    if dark_ratio > 0.25 and red_ratio > 0.06: # Was 0.18 / 0.04
        return True

    return False

##################################################
def _dist(a, b):
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

def _finger_extended(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

def detect_ok_gesture(lm):
    wrist = lm[0]
    middle_mcp = lm[9]
    scale = _dist(wrist, middle_mcp)
    if scale < 1e-6: return False

    d_ti = _dist(lm[4], lm[8])
    if d_ti > scale * 0.45:
        return False

    middle = _finger_extended(lm, 12, 10)
    ring = _finger_extended(lm, 16, 14)
    pinky = _finger_extended(lm, 20, 18)
    return (int(middle) + int(ring) + int(pinky)) >= 2

def detect_thumbs_up(lm):
    thumb_up = lm[4][1] < lm[3][1] < lm[2][1]
    index_closed  = lm[8][1] > lm[6][1]
    middle_closed = lm[12][1] > lm[10][1]
    ring_closed   = lm[16][1] > lm[14][1]
    pinky_closed  = lm[20][1] > lm[18][1]
    closed_count = int(index_closed) + int(middle_closed) + int(ring_closed) + int(pinky_closed)
    return thumb_up and closed_count >= 3

def detect_thinking(lm, frame_w, frame_h, face_box):
    if face_box is None: return False
    fx, fy, fw, fh = face_box
    target = np.array([fx + fw * 0.50, fy + fh * 0.78], dtype=np.float32)
    idx_tip = np.array([lm[8][0] * frame_w, lm[8][1] * frame_h], dtype=np.float32)
    return np.linalg.norm(idx_tip - target) < float(fh * 0.20)

def detect_gesture(lm, frame_w, frame_h, face_box):
    if detect_ok_gesture(lm): return "ok"
    if detect_thumbs_up(lm): return "ok"
    if detect_thinking(lm, frame_w, frame_h, face_box): return "thinking"
    return None

# DDEPFACE WORKERY
def deepface_worker(shared: FaceState, lock: threading.Lock, frame_queue: deque):
    history = deque(maxlen=EMOTION_SMOOTHING_WINDOW)
    while True:
        if not frame_queue:
            time.sleep(0.01)
            continue
        try:
            frame_bgr = frame_queue.pop()
        except IndexError:
            continue
            
        frame_queue.clear()

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(
                rgb,
                actions=["emotion"],
                enforce_detection=DEEPFACE_ENFORCE_DETECTION,
                detector_backend=DETECTOR_BACKEND
            )
            if isinstance(result, list):
                result = result[0]

            dominant = result.get("dominant_emotion", None)
            region = result.get("region", None) or result.get("facial_area", None)

            face_box = None
            if region:
                x = int(region.get("x", 0))
                y = int(region.get("y", 0))
                w = int(region.get("w", region.get("width", 0)))
                h = int(region.get("h", region.get("height", 0)))
                if w > 0 and h > 0:
                    face_box = (x, y, w, h)

            if dominant:
                history.append(str(dominant).lower())
                majority = max(set(history), key=history.count)
                with lock:
                    shared.emotion = majority
                    shared.face_box = face_box
                    shared.updated_at = time.time()
        except Exception:
            continue

#main class
class EmotionProcessor:
    def __init__(self, asset_dir, model_path):
        self.asset_dir = asset_dir
        self.model_path = model_path
        
        # Load stickers
        self.sticker_cache = {}
        for key, base in STICKER_FILES.items():
            img, _ = try_read_image_any_ext(os.path.join(self.asset_dir, base))
            if img is not None:
                self.sticker_cache[key] = img
            else:
                print(f"[UYARI] Sticker okunamadı: {os.path.join(self.asset_dir, base)}")

        # Init DeepFace Loop
        self.shared = FaceState()
        self.lock = threading.Lock()
        self.frame_queue = deque(maxlen=1)
        
        # Start Thread
        t = threading.Thread(target=deepface_worker, args=(self.shared, self.lock, self.frame_queue), daemon=True)
        t.start()
        
        # MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        self.frame_count = 0

    def get_sticker_image(self, key):
        return self.sticker_cache.get(key)

    def process_frame(self, frame, mirror=False):
        """
        Returns: (processed_frame, emotion, gesture, sticker_key)
        """
        if mirror:
            frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]
        self.frame_count += 1

        # DeepFace Queue
        if self.frame_count % DEEPFACE_EVERY_N_FRAMES == 0:
            if not self.frame_queue:
                self.frame_queue.append(frame.copy())
            else:
                self.frame_queue[0] = frame.copy()

        # Read shared state
        with self.lock:
            emotion = self.shared.emotion
            face_box = self.shared.face_box

        # MediaPipe Gesture
        gesture = None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            res = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            if res and res.hand_landmarks and len(res.hand_landmarks) > 0:
                hand0 = res.hand_landmarks[0]
                lm = [(pt.x, pt.y) for pt in hand0]
                gesture = detect_gesture(lm, W, H, face_box)
        except Exception:
            gesture = None

        # Logic
        sticker_key = None
        if gesture in ("ok", "thinking"):
            sticker_key = gesture
        else:
            if crazy_tongue_like(frame, face_box):
                sticker_key = "disgust"
            else:
                if emotion:
                    sticker_key = EMOTION_MAP.get(emotion.lower(), None)

        return frame, emotion, gesture, sticker_key

    def overlay_sticker_on_frame(self, frame, sticker_key):
        sticker = self.sticker_cache.get(sticker_key)
        if sticker is not None:
            H, W = frame.shape[:2]
            margin = 18
            size = int(min(W, H) * 0.28)
            x = W - size - margin
            y = margin
            frame = overlay_sticker(frame, sticker, x, y, size)
        return frame
