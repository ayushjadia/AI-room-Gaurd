import cv2
import time
import numpy as np
import face_recognition
import threading
import pyttsx3
import importlib
import os
from .config import *
import speech_recognition as sr
from openai import OpenAI

# Import your previous milestone modules
import actviation_input as activation
import recognition_user_enrollement as recog

# --------------- CONFIG ---------------
EMBED_FILE = recog.EMBED_FILE
FACE_DISTANCE_THRESHOLD = recog.FACE_DISTANCE_THRESHOLD
ESCALATION_INTERVAL = 10
MAX_ESCALATION_LEVEL = 3
TTS_RATE = 150
CAMERA_INDEX = 0
LLM_MODEL = "gpt-4o-mini"  # or "gpt-3.5-turbo" etc.
OPENAI_API_KEY = "YOUR_API_KEY_HERE"

# Initialize TTS and ASR
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)
recognizer = sr.Recognizer()
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- HELPERS -------------------

def tts_say(text):
    """Speak text aloud."""
    print("[Guard says]:", text)
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)

def listen_once(timeout=6):
    """Listen for a single spoken response (returns text)."""
    try:
        with sr.Microphone() as source:
            print("[Guard listening]...")
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)
        print("[Guard transcribing]...")
        text = recognizer.recognize_google(audio)
        print("[Intruder said]:", text)
        return text
    except sr.WaitTimeoutError:
        print("[ASR] Timeout waiting for response.")
        return ""
    except sr.UnknownValueError:
        print("[ASR] Could not understand.")
        return ""
    except Exception as e:
        print("[ASR] Error:", e)
        return ""

def llm_response(prompt):
    """Query the LLM for a dialogue response."""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI guard monitoring a room. Respond politely but firmly to unknown visitors."},
                {"role": "user", "content": prompt},
            ]
        )
        reply = response.choices[0].message.content.strip()
        print("[LLM reply]:", reply)
        return reply
    except Exception as e:
        print("[LLM error]:", e)
        return "Please identify yourself."

# ------------------- ESCALATION LOGIC -------------------

def escalate(level):
    """Speak escalating messages."""
    if level == 1:
        tts_say("Hello, I do not recognize you. Who are you?")
    elif level == 2:
        tts_say("Please leave this room immediately.")
    elif level == 3:
        tts_say("This is your final warning. Security will be notified.")
    elif level > 3:
        tts_say("Alarm triggered! Unauthorized person detected!")

def handle_conversation():
    """Conduct a short dialogue with the unknown person via LLM."""
    tts_say("Hello, I do not recognize you. Who are you?")
    dialogue_history = []

    for turn in range(3):
        human_text = listen_once(timeout=6)
        if not human_text:
            tts_say("I did not hear a response.")
            continue

        dialogue_history.append(f"Visitor: {human_text}")
        guard_prompt = "\n".join(dialogue_history) + "\nGuard:"
        guard_reply = llm_response(guard_prompt)

        tts_say(guard_reply)
        dialogue_history.append(f"Guard: {guard_reply}")

        # If guard_reply sounds like a final warning, break early
        if "leave" in guard_reply.lower() or "warning" in guard_reply.lower():
            break

    tts_say("This is your final warning. I will notify my owner now.")

# ------------------- MONITOR LOOP -------------------

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
    dialogue_done = False

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
                dialogue_done = False
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
                handle_conversation()
                dialogue_done = True
            elif now - last_unknown_time > ESCALATION_INTERVAL and escalation_level < MAX_ESCALATION_LEVEL:
                escalation_level += 1
                last_unknown_time = now
                escalate(escalation_level)
        else:
            escalation_level = 0

        cv2.putText(frame, f"Guard: {'ON' if activation.guard_on else 'OFF'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if activation.guard_on else (0,0,255), 2)
        cv2.imshow("GuardCam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[GUARD] Quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------- MAIN -------------------

def main():
    activation_thread = threading.Thread(target=activation.main_loop, daemon=True)
    activation_thread.start()
    start_guard_monitoring()

if __name__ == "__main__":
    main()
