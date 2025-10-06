import cv2
import time
import numpy as np
import face_recognition
import threading
import pyttsx3
import importlib
import os

# Import modules from milestones 1 and 2
import actviation_input as activation
import recognition_user_enrollement as recog

# ---------------- CONFIG ----------------
EMBED_FILE = recog.EMBED_FILE
FACE_DISTANCE_THRESHOLD = recog.FACE_DISTANCE_THRESHOLD
ESCALATION_INTERVAL = 10      # seconds between escalation steps
MAX_ESCALATION_LEVEL = 3
TTS_RATE = 150
CAMERA_INDEX = 0

# ----------------------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)

def tts_say(text):
    try:
        print("[Guard says]:", text)
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

# ---------------- ESCALATION LOGIC ----------------
def escalate(level):
    """Speak escalating messages based on threat level."""
    if level == 1:
        tts_say("Hello, I do not recognize you. Who are you?")
    elif level == 2:
        tts_say("Please leave this room immediately.")
    elif level == 3:
        tts_say("This is your final warning. Security will be notified.")
    elif level > 3:
        tts_say("Alarm triggered! Unauthorized person detected!")

def start_guard_monitoring():
    """Continuously monitor using webcam and escalate with unknown presence."""
    trusted_names, trusted_embeddings = recog.load_embeddings(EMBED_FILE)
    if len(trusted_names) == 0:
        print("[ERROR] No trusted users enrolled. Run enrollment first.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[GUARD] Escalation system running...")
    escalation_level = 0
    last_unknown_time = 0
    unknown_detected = False

    while True:
        # Check if guard mode is ON
        if not activation.guard_on:
            cv2.imshow("GuardCam", np.zeros((240, 320, 3), dtype=np.uint8))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame capture failed.")
            break

        # Detect faces
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        unknown_detected = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            match_idx, dist = recog.compare_embedding(face_encoding, trusted_embeddings)
            top *= 2; right *= 2; bottom *= 2; left *= 2
            if match_idx is not None:
                name = trusted_names[match_idx]
                label = f"{name} ({dist:.2f})"
                color = (0, 200, 0)
                escalation_level = 0
            else:
                unknown_detected = True
                label = f"Unknown ({dist:.2f})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left + 4, bottom - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Escalation management
        now = time.time()
        if unknown_detected:
            if escalation_level == 0:
                escalation_level = 1
                last_unknown_time = now
                escalate(escalation_level)
            elif now - last_unknown_time > ESCALATION_INTERVAL and escalation_level < MAX_ESCALATION_LEVEL:
                escalation_level += 1
                last_unknown_time = now
                escalate(escalation_level)
        else:
            escalation_level = 0  # reset when trusted user seen

        # Display
        cv2.putText(frame, f"Guard: {'ON' if activation.guard_on else 'OFF'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if activation.guard_on else (0,0,255), 2)
        cv2.imshow("GuardCam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[GUARD] Quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
def main():
    # Run the activation system in background thread
    activation_thread = threading.Thread(target=activation.main_loop, daemon=True)
    activation_thread.start()

    # Run the monitoring + escalation loop
    start_guard_monitoring()

if __name__ == "__main__":
    main()
