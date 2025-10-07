from langgraph.graph import StateGraph
import speech_recognition as sr
import pyttsx3
from .nodes.activation import activation_node
from .nodes.recognizer import recognizer_node
from .nodes.escalation import escalation_agent_node
from .state import GuardState
from .config import *

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

pipline = graph.compile()
pipline.invoke(GuardState())