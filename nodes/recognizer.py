from ..state import GuardState
import cv2
import time
import pyttsx3
import numpy as np
import face_recognition
from ..config import *  # for EMBED_FILE, FACE_DISTANCE_THRESHOLD, RATES, etc.
from ..utilities import tts_say
import os

# # Initialize TTS
# tts_engine = pyttsx3.init()
# tts_engine.setProperty("rate", TTS_RATE)

# def tts_say(text):
#     try:
#         tts_engine.say(text)
#         tts_engine.runAndWait()
#     except Exception as e:
#         print("[TTS] Error:", e)

def load_embeddings(npz_path):
    if not os.path.exists(npz_path):
        return [], []
    data = np.load(npz_path, allow_pickle=True)
    return data["names"].tolist(), data["embeddings"]

def compare_embedding(query_emb, known_embeddings, threshold=FACE_DISTANCE_THRESHOLD):
    if len(known_embeddings) == 0:
        return None, None
    dists = face_recognition.face_distance(known_embeddings, query_emb)
    best_idx = np.argmin(dists)
    best_dist = float(dists[best_idx])
    if best_dist <= threshold:
        return int(best_idx), best_dist
    return None, best_dist

def recognizer_node(state: GuardState, camera_index=0) -> GuardState:
    """
    LangGraph node for face recognition.
    Updates GuardState.trusted_user and triggers escalation if intruder.
    """
    if not state.guard_status:
        print("[Recognizer] Guard not active. Returning to activation.")
        return state

    # Load embeddings
    trusted_names, trusted_embeddings = load_embeddings(EMBED_FILE)
    print(f"[Recognizer] Loaded {len(trusted_names)} trusted embeddings.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[Recognizer] Cannot open webcam.")
        return state

    last_spoken = {}
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize frame for speed
            small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            intruder_detected = False

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= 2; right *= 2; bottom *= 2; left *= 2
                match_idx, dist = compare_embedding(face_encoding, trusted_embeddings)

                if match_idx is not None:
                    name = trusted_names[match_idx]
                    label = f"{name} ({dist:.2f})"
                    color = (0, 200, 0)
                    state.trusted_user = True
                    now = time.time()
                    if (name not in last_spoken) or (now - last_spoken[name] > SPOKEN_COOLDOWN):
                        tts_say(f"Welcome {name}")
                        last_spoken[name] = now
                else:
                    label = f"Unknown ({dist:.2f})" if dist is not None else "Unknown"
                    color = (0, 0, 255)
                    state.trusted_user = False
                    intruder_detected = True
                    now = time.time()
                    unk_key = "Unknown"
                    if (unk_key not in last_spoken) or (now - last_spoken[unk_key] > SPOKEN_COOLDOWN):
                        last_spoken[unk_key] = now

                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom-20), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left+4, bottom-4), font, 0.6, (255,255,255), 1)

            cv2.putText(frame, f"Guard: {'ON' if state.guard_status else 'OFF'}", (10,30), font, 1, (0,255,0), 2)
            cv2.imshow("Recognizer Node", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Recognizer] Quit requested")
                break

            if intruder_detected:
                state.escalation_level += 1
                print(f"[Recognizer] Intruder detected! Escalation level: {state.escalation_level}")
                break  # exit loop to trigger escalation node

            # Optional: if guard turned off externally
            if not state.guard_status:
                print("[Recognizer] Guard turned off. Returning to activation.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return state

if __name__ == "__main__":
    state = GuardState()
    state.guard_status = True
    print(state)
    recognizer_node(state)
    print(state)