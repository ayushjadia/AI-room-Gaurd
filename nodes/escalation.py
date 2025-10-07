"""
LangGraph node for Escalation Agent (LLM-integrated)
Handles conversation, reasoning, and escalation based on intruder dialogue.
"""

# import time
import threading
import pyttsx3
import speech_recognition as sr
from typing import List
from pydantic import BaseModel
from langchain_cerebras import ChatCerebras
from ..state import GuardState
from dotenv import load_dotenv
load_dotenv()

import time

# import actviation_input as activation
# import recognition_user_enrollement as recognize

# ------------------ INITIALIZE ------------------

TTS_RATE = 150

llm = ChatCerebras(
    model="llama3.1-8b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)



# Shared objects
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)
recognizer = sr.Recognizer()

# Locks to coordinate speech and listening
tts_lock = threading.Lock()
sr_lock = threading.Lock()

# Global flag for TTS status
is_tts_active = False

# ------------------ INIT ------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

recognizer = sr.Recognizer()

# ------------------ TTS ------------------
def tts_say(text: str):
    """Speak text synchronously, blocking until done."""
    print(f"[Guard says]: {text}")
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()  # BLOCKS until speech is finished
    except Exception as e:
        print("[TTS error]:", e)

def listen_once(timeout=6) -> str:
    """
    Record one spoken response.
    Will wait until TTS is finished before starting to listen.
    """
    global is_tts_active
    while is_tts_active:
        time.sleep(0.1)  # wait for guard to finish talking

    with sr_lock:
        try:
            with sr.Microphone() as source:
                print("[Guard listening]...")
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)
            print("[Guard transcribing]...")
            text = recognizer.recognize_google(audio)
            print(f"[Intruder said]: {text}")
            return text
        except Exception as e:
            print("[ASR error]:", e)
            return ""


# ------------------ MAIN NODE ------------------

def escalation_agent_node(state: GuardState) -> GuardState:
    """
    Handles conversation and escalation using LLM reasoning.
    """
    print("\n[EscalationAgent Node] Running LLM dialogue escalation...")

    # Skip if guard off
    if not state.guard_status:
        print("[EscalationAgent] Guard OFF. Exiting node.")
        return state

    # Skip if trusted user
    if state.trusted_user:
        print("[EscalationAgent] Trusted user detected. No escalation.")
        state.escalation_level = 0
        return state

    # # Start conversation
    # if state.escalation_level==0:
    #     tts_say("Hello. I do not recognize you. Who are you and why are you here?")
    #     state.escalation_level+=1
    #     state.messages.append("Guard: Hello. I do not recognize you. Who are you and why are you here?")

    # Dialogue loop
    while state.escalation_level <= 3:
        # print("hi\n")
        intruder_text = listen_once(timeout=7)
        if not intruder_text:
            tts_say("I didn't hear anything.")
            continue

        # Append intruder message
        state.messages.append(f"Intruder: {intruder_text}")

        # Query LLM for reasoning
        prompt_messages = [
            ("system",
             """You are an AI room guard. Respond politely but firmly.
             If the intruder sounds genuine (apology, mistake, friend, etc.), let them go peacefully.
             If they avoid leaving, escalate your tone until they comply."""),
        ]
        for msg in state.messages[-4:]:  # last few exchanges
            role = "assistant" if msg.startswith("Guard") else "user"
            prompt_messages.append((role, msg.split(": ", 1)[-1]))

        # LLM response
        response = llm.invoke(prompt_messages)
        reply = response.content if hasattr(response, "content") else str(response)
        print(f"[LLM reply]: {reply}")

        tts_say(reply)
        state.messages.append(f"Guard: {reply}")

        # Decide escalation logic from LLM + intruder text
        lower_text = intruder_text.lower()
        if any(x in lower_text for x in ["sorry", "friend", "mistake", "wrong room"]):
            print("[EscalationAgent] Genuine response — de-escalating.")
            # state.trusted_user = False
            # state.escalation_level = 2
        return state

    if state.escalation_level > 3:
        tts_say("This is your final warning. Alarm will be triggered.")

    # Final state → let main system handle alarm trigger
    return state

if __name__ == "__main__":
    state = GuardState(guard_status=True,trusted_user=False,escalation_level=0,messages=[])
    escalation_agent_node(state)
    print(state)

