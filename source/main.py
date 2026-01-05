import os
import cv2
import time
from processor import EmotionProcessor

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "models", "hand_landmarker.task")

SHOW_DEBUG_TEXT = True

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[HATA] Hand model bulunamadı: {MODEL_PATH}")
        print("assets/models/hand_landmarker.task dosyasını indirip buraya koymalısın.")
        return

    # Initialize Processor
    processor = EmotionProcessor(ASSETS_DIR, MODEL_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı.")
        return

    print("Kamera başlatıldı. Çıkmak için 'q' veya 'ESC' tuşuna basın.")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Sadece CLI/Window modunda mirror isteyebiliriz
        # Eski kodda MIRROR_VIEW = False idi ama aynalama genelde istenir.
        # process_frame içinde mirror argümanı var.
        
        frame, emotion, gesture, sticker_key = processor.process_frame(frame, mirror=False)

        # Overlay sticker for the single-window view
        if sticker_key:
            frame = processor.overlay_sticker_on_frame(frame, sticker_key)

        if SHOW_DEBUG_TEXT:
            H, W = frame.shape[:2]
            label = f"gesture={gesture or 'None'} | emotion={emotion or 'None'} | sticker={sticker_key or 'None'}"
            cv2.putText(frame, label, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("EmotionCamera", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
