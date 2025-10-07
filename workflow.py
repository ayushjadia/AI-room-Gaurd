from langgraph.graph import StateGraph
import speech_recognition as sr
import pyttsx3
from .nodes.activation import activation_node
from .nodes.recognizer import recognizer_node
from .nodes.escalation import escalation_agent_node
from .state import GuardState
from .config import *

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


# threading.Thread(target=tts_loop, daemon=True).start()
# pipline = graph.compile()
# pipline.invoke(current_state)
# tts_queue.put(None)  # sends sentinel to stop tts_loop
# pipline.invoke(current_state)
while True:
    # Activation first
    current_state = activation_node(current_state)

    # # Recognizer → Dialogue loop
    current_state = recognizer_node(current_state)

    current_state = escalation_agent_node(current_state)
    
    if current_state.trusted_user:
        print("[Workflow] Trusted user detected. Ending loop.")
        break


    # # Stop condition: escalation reached alarm level
    if current_state.escalation_level >= 4:
        print("[Workflow] Escalation reached max level! Alarm triggered.")
        break

# print("[Final Guard State]:", current_state)
