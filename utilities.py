import pyttsx3
import queue
from .config import *
# Thread-safe queue to pass recognised text to main thread
# text_q = queue.Queue()

# Initialize TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)


def tts_say(text):
    """Speak text (non-blocking via startLoop / runAndWait)."""
    # We will call runAndWait but catch exceptions to avoid blocking forever.
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_engine.stop() 
    except Exception as e:
        print("TTS error:", e)