from langgraph.graph import StateGraph
import speech_recognition as sr
import pyttsx3
from .nodes.activation import activation_node
from .nodes.recognizer import recognizer_node
from .nodes.escalation import escalation_agent_node
from .state import GuardState
from .config import *

# ---------TTS --------------
import queue
import threading

# ---------------- TTS QUEUE ----------------
tts_queue = queue.Queue()
# Initialize TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)

def tts_say_main_thread(text: str):
    """Queue text to be spoken in main thread."""
    tts_queue.put(text)

def tts_loop():
    """Main-thread loop to speak queued text."""
    while True:
        text = tts_queue.get()
        if text is None:  # sentinel to stop the loop
            break
        print(f"[Guard says]: {text}")
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print("[TTS error]:", e)

# ---- INITIAL STATE ----
state = GuardState(
    guard_status=False,
    trusted_user=False,
    escalation_level=0,
    messages=[]
)

# ---- CREATE GRAPH ----
graph = StateGraph(GuardState)

# Add nodes
graph.add_node("activation", activation_node)
graph.add_node("recognizer", recognizer_node)
graph.add_node("dialogue", escalation_agent_node)

# Connect nodes
graph.set_entry_point("activation")
graph.add_edge("activation", "recognizer")  # start after activation
graph.add_edge("recognizer", "dialogue")    # if intruder detected
graph.add_edge("dialogue", "recognizer")    # loop back for next recognition


# Run graph continuously until escalation_level reaches 4 or trusted_user becomes True
current_state = state


threading.Thread(target=tts_loop, daemon=True).start()
pipline = graph.compile()
tts_queue.put(None)  # sends sentinel to stop tts_loop
# pipline.invoke(current_state)
    # Activation first
    # current_state = graph.run_node("activation", current_state)

    # # Recognizer → Dialogue loop
    # current_state = graph.run_node("recognizer", current_state)

    # if current_state.trusted_user:
    #     print("[Workflow] Trusted user detected. Ending loop.")
    #     break

    # current_state = graph.run_node("dialogue", current_state)

    # # Stop condition: escalation reached alarm level
    # if current_state.escalation_level >= 4:
    #     print("[Workflow] Escalation reached max level! Alarm triggered.")
    #     break

# print("[Final Guard State]:", current_state)
