"""
LangGraph node for Escalation Agent (LLM-integrated)
Handles conversation, reasoning, and escalation based on intruder dialogue.
"""
import speech_recognition as sr
from langchain_cerebras import ChatCerebras
from ..state import GuardState
from ..utilities import tts_say
from dotenv import load_dotenv
load_dotenv()


# ------------------ INITIALIZE ------------------

TTS_RATE = 150

llm = ChatCerebras(
    model="llama3.1-8b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

recognizer = sr.Recognizer()

def listen_once(timeout=6) -> str:
    """
    Record one spoken response.
    Will wait until TTS is finished before starting to listen.
    """
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
    tts_say("escalation_agent_node started")
    # Skip if guard off
    if not state.guard_status:
        print("[EscalationAgent] Guard OFF. Exiting node.")
        return state

    # Skip if trusted user
    if state.trusted_user:
        print("[EscalationAgent] Trusted user detected. No escalation.")
        state.escalation_level = 0
        return state

    # Dialogue loop
    while state.escalation_level <= 3:
        tts_say("PLease leave the room")
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
        return state

    if state.escalation_level > 3:
        tts_say("This is your final warning. Alarm will be triggered.")

    # Final state → let main system handle alarm trigger
    return state

if __name__ == "__main__":
    state = GuardState(guard_status=True,trusted_user=False,escalation_level=0,messages=[])
    escalation_agent_node(state)
    print(state)

