import os

MODEL= ["alibaba/tongyi-deepresearch-30b-a3b:free"][0]
WAKE_PHRASES = [
    "guard my room",
    "guard the room",
    "guard room",
    "start guarding",
    "start guard",
    "activate guard",
    "activate the guard",
    "protect my room",
    "stop guarding",
    "garden"
]
LISTEN_TIMEOUT = 5            # seconds to wait for audio chunk when listening
PHRASE_TIME_LIMIT = 5         # max length of one spoken chunk (seconds)
CAMERA_INDEX = 0              # 0 is usually the laptop webcam
DISPLAY_WINDOW_NAME = "GuardCam (press q to quit)"
TTS_RATE = 150
# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
# Enroll images directory
ENROLL_DIR = os.path.join(PARENT_DIR, "enroll_images")
EMBED_FILE = os.path.join(PARENT_DIR, "trusted_embeddings.npz")
TTS_RATE = 150
FACE_DISTANCE_THRESHOLD = 0.55