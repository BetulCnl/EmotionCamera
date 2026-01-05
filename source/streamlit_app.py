import streamlit as st
import cv2
import numpy as np
import os
import sys

# Add source to path so we can import processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor import EmotionProcessor

# =========================
# CONFIG & STYLES
# =========================
st.set_page_config(
    page_title="Emotion Camera",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a beautiful look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    /* Stylish headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }
    /* Frame container styling */
    .frame-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    /* Sticker container */
    .sticker-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin-top: 20px;
    }
    .emotion-tag {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #4facfe;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# INITIALIZATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "models", "hand_landmarker.task")

@st.cache_resource
def get_processor():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return EmotionProcessor(ASSETS_DIR, MODEL_PATH)

processor = get_processor()

# =========================
# UI LAYOUT
# =========================
st.title("✨ Emotion & Hand Gesture Camera")
st.markdown("Making interactions fun with AI.")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### Live Feed")
    camera_placeholder = st.empty()

with col2:
    st.markdown("### Detected Mood")
    sticker_placeholder = st.empty()
    info_placeholder = st.empty()

# =========================
# MAIN LOOP
# =========================
run = st.checkbox('Start Camera', value=True)

if run and processor:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        st.error("Could not open camera.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Frame read error.")
                break
            
            # Process
            # We DON'T want to mirror logic-wise if we want to read text, but user typically wants mirror like a mirror
            # Let's mirror it for typical webcam feel
            frame, emotion, gesture, sticker_key = processor.process_frame(frame, mirror=False) 
            # Note: processor.process_frame has mirror arg but I set False there and do it manually if needed, 
            # let's trust the processor's mirror arg if we used it.
            # Actually I set mirror=False above. Let's flip for display if desired.
            
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update Left Column (Camera)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update Right Column (Sticker)
            if sticker_key:
                sticker_bgr = processor.get_sticker_image(sticker_key)
                if sticker_bgr is not None:
                    sticker_rgb = cv2.cvtColor(sticker_bgr, cv2.COLOR_BGR2RGB)
                    sticker_placeholder.image(sticker_rgb, caption=sticker_key.upper(), width=300)
                else:
                    sticker_placeholder.empty()
            else:
                sticker_placeholder.empty()
                
            # Update Info Text (Removed as requested)
            info_placeholder.empty()
            
            # Small delay to reduce CPU usage? Streamlit loop is already heavy.
            # cv2.waitKey(1) is for OpenCV windows. Here simple loop is fine.
            # Rerun logic is implicit in Streamlit via st.experimental_rerun() if button clicked, 
            # but for loops we just loop.
            
            # Important: Streamlit re-runs script on interaction. 
            # Inside a while loop prevents interaction unless we use check-runs.
            # But straightforward 'while run' loop blocks other UI elements.
            # For this simple app, it's fine.
            
        cap.release()
else:
    st.info("Click 'Start Camera' to begin.")
