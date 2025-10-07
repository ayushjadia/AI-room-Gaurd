import threading
import queue
import time
import traceback
import sys
from typing import TypedDict
from ..config import *
from IPython.display import display, clear_output
import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
import ipywidgets as widgets
from ..state import GuardState
from ..utilities import tts_say
 
text_q = queue.Queue()


# WAKE_PHRASES = [p.lower() for p in WAKE_PHRASES]

# Global state
shutdown_flag = False

# Thread-safe queue to pass recognised text to main thread
# text_q = queue.Queue()

# # Initialize TTS
# tts_engine = pyttsx3.init()
# tts_engine.setProperty("rate", TTS_RATE)

# def tts_say(text):
#     """Speak text (non-blocking via startLoop / runAndWait)."""
#     # We will call runAndWait but catch exceptions to avoid blocking forever.
#     try:
#         tts_engine.say(text)
#         tts_engine.runAndWait()
#         tts_engine.stop() 
#     except Exception as e:
#         print("TTS error:", e)

def is_wake_phrase(transcript):
    """Return True if transcript contains any wake phrase."""
    if not transcript:
        return False
    txt = transcript.lower()
    # crude but effective check
    for p in WAKE_PHRASES:
        if p in txt:
            return True
    return False

def asr_listen_loop(recognizer, mic):
    """
    Continuously listen in chunks. When speech is recognized,
    put recognized text into text_q.
    Runs in its own thread.
    """
    global shutdown_flag
    while not shutdown_flag:
        try:
            print("[ASR] Listening for phrase...")
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT)
            # Recognize (Google Web Speech API)
            try:
                print("[ASR] Recognizing...")
                transcript = recognizer.recognize_google(audio)
                print(f"[ASR] Transcript: {transcript}")
                text_q.put(transcript)
            except sr.UnknownValueError:
                # speech unintelligible
                # keep listening
                # print("[ASR] Couldn't understand audio")
                pass
            except sr.RequestError as e:
                # API was unreachable or unresponsive
                print("[ASR] RequestError (maybe no internet):", e)
                # Sleep a bit before retrying
                time.sleep(1.0)
            except Exception as e:
                print("[ASR] Unexpected error during recognition:", e)
                traceback.print_exc()
                time.sleep(0.5)
        except sr.WaitTimeoutError:
            # nothing heard in timeout window: loop again
            continue
        except Exception as e:
            print("[ASR] Fatal error in listen loop:", e)
            traceback.print_exc()
            time.sleep(1.0)
    print("[ASR] Listener thread exiting.")

def set_guard_status(state,status: bool):
    state.guard_status= status

def main_loop(state: GuardState) -> GuardState:
    """Main loop: shows webcam and processes transcripts for activation."""
    global shutdown_flag

    # Setup speech recognizer and microphone
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
    except Exception as e:
        print("Could not open microphone. Make sure it's available.")
        print("Error:", e)
        raise(e)

    # Start ASR thread
    asr_thread = threading.Thread(target=asr_listen_loop, args=(recognizer, mic), daemon=True)
    asr_thread.start()

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemError("Unable to access webcam (index {}). Exiting.".format(CAMERA_INDEX))

    print("Webcam opened. Press 'q' in the window to quit.")

    # Optional: font settings for overlay
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Main display loop
    last_toggle_time = 0
    TOGGLE_COOLDOWN = 1.0  # seconds; avoid accidental repeated toggles

    # For optional demo recording: set to True to save a short clip automatically on activation
    record_on_activate = False
    video_writer = None
    record_start_time = None
    RECORD_SECONDS = 8  # how long to record after activation (if record_on_activate True)

    try:
        while True:
            # Get webcam frame
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed; breaking.")
                break

            # Resize for display (optional)
            display_frame = cv2.resize(frame, (960, 540))

            # Check queue for transcripts
            try:
                transcript = text_q.get_nowait()
            except queue.Empty:
                transcript = None

            if transcript:
                # Look for wake phrase
                if is_wake_phrase(transcript):
                    now = time.time()
                    if now - last_toggle_time > TOGGLE_COOLDOWN:
                        state.guard_status = not state.guard_status  # Toggle through state
                        last_toggle_time = now
                        state_text = "GUARD ACTIVE" if state.guard_status else "GUARD OFF"
                        print(f"[SYSTEM] Toggled guard: {state_text}")
                        tts_say(f"{'Guard mode activated' if state.guard_status else 'Guard mode deactivated'}")
                        # optional demo recording when activated
                        if state.guard_status and record_on_activate:
                            # start recording to file
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            fname = f"guard_activation_{int(time.time())}.mp4"
                            fps = 20.0
                            h, w = display_frame.shape[:2]
                            video_writer = cv2.VideoWriter(fname, fourcc, fps, (w, h))
                            record_start_time = time.time()
                            print(f"[DEMO] Recording activation to {fname}")
                else:
                    # Non-wake phrase recognized. For debugging show transcript briefly
                    print(f"[ASR] Heard (not wake): {transcript}")

            # Update display based on state
            text_color = (0, 255, 0) if state.guard_status else (0, 150, 255)
            cv2.putText(display_frame, f"Guard: {'ON' if state.guard_status else 'OFF'}", (10, 30), font, 1, text_color, 2)
            # show last transcript (if any) for debug:
            if transcript:
                cv2.putText(display_frame, f"Heard: {transcript}", (10, 60), font, 0.6, (255,255,255), 1)

            # frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)  # convert to RGB
            # img = Image.fromarray(frame_rgb)

            # clear_output(wait=True)  # clears old frame
            # display(img)             # shows new frame

            cv2.imshow(DISPLAY_WINDOW_NAME, display_frame)
            # display(Image.fromarray(frame_rgb))

            # If recording, write frames
            if video_writer is not None:
                video_writer.write(display_frame)
                if time.time() - record_start_time > RECORD_SECONDS:
                    video_writer.release()
                    video_writer = None
                    print("[DEMO] Recording saved.")

            # Keyboard handling: q to quit, g to toggle guard manually
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested by user.")
                break
            if key == ord('g'):
                state.guard_status = not state.guard_status
                tts_say(f"{'Guard mode activated' if state.guard_status else 'Guard mode deactivated'}")

            # small sleep to be gentle on CPU
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
    finally:
        shutdown_flag = True
        print("Shutting down...")
        try:
            cap.release()
        except:
            pass
        cv2.destroyAllWindows()
        # stop ASR thread gracefully
        asr_thread.join(timeout=2.0)
        try:
            tts_engine.stop()
        except:
            pass
        print("Exited cleanly.")
    return state  # Return updated state

def activation_node(state: GuardState) -> GuardState:
    """LangGraph node wrapper for activation system"""
    try:
        return main_loop(state)
    except Exception as e:
        print(f"[ActivationNode] Error: {e}")
        return state

# if __name__ == "_
    