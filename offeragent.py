import os
import numpy as np
import time
from typing import TypedDict, Optional
import sounddevice as sd
from google.cloud import texttospeech, speech
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# ------------------------
# CONFIG
# ------------------------
OFFER_TERMS = (
    "You have a Pay Over Time feature on your account that gives you the option to pay certain charges over time with interest. "
    "You can enjoy a lower interest rate on new eligible charges added to your Pay Over Time feature for a limited time.\n\n"
    "For 12 billing cycles, you can enjoy a 8.99% promotional Pay Over Time Annual Percentage Rate on new eligible charges, "
    "automatically added to a Pay Over Time Balance. If you are enrolled in Pay Over Time Select, this promotional rate will also apply "
    "to eligible purchases added to a plan at your request.\n\n"
    "Please note that any charges in your Pay Over Time Balance prior to the promotional period will continue to be charged interest at your "
    "current Pay Over Time APR of 20.99% even during the promotional period. Your current Pay Over Time APR is a variable rate, based on Prime Rate "
    "(plus 18.99%). At the end of the promotional period, any new charges added to a Pay Over Time balance and any remaining Pay Over Time balance "
    "will be subject to the Pay Over Time APR in your Cardmember Agreement, currently 20.99%.\n\n"
    "Annual Membership Fees are not eligible charges for this promotional APR. This promotional rate begins the day after you enroll. "
    "Please note, you cannot be enrolled in more than 1 promotional APR at a time.\n\n"
    "If you enroll in a promotional APR, you may see limited plan duration options during the promotional period when you use Plan It on your account.\n\n"
    "Do you have any questions about the promotional offer?"
)

# Initialize Groq LLM
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=os.getenv("GROQ_API_KEY"))

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./agent.json"

# ------------------------
# HELPERS
# ------------------------
def speak_text(text: str):
    """Convert text to speech and play via speakers."""
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Wavenet-F")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    audio = np.frombuffer(response.audio_content, dtype=np.int16)
    sd.play(audio, samplerate=24000)
    sd.wait()

def record_and_transcribe(duration=4):
    """Record from mic and transcribe using Google STT."""
    client = speech.SpeechClient()
    print("🎤 Listening...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="int16")
    sd.wait()
    audio_bytes = audio.tobytes()

    audio_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )
    audio_file = speech.RecognitionAudio(content=audio_bytes)
    response = client.recognize(config=audio_config, audio=audio_file)

    if response.results:
        transcript = response.results[0].alternatives[0].transcript
        print(f"📝 Customer said: {transcript}")
        return transcript
    else:
        print("❗ No speech recognized.")
    return ""

def analyze_customer_query(query: str):
    messages = [
        HumanMessage(content=f"""
Here are the offer terms: {OFFER_TERMS}

You are an assistant handling credit card offer disclosures.

Examples:
Customer: No, I'm good.
Agent: Do I have your consent to enroll your card account in this Promotional APR Offer?

Customer: No questions.
Agent: Do I have your consent to enroll your card account in this Promotional APR Offer?

Customer: Yes, please proceed.
Agent: CONSENT

Customer: Yes.
Agent: CONSENT

Customer: Go ahead.
Agent: CONSENT

Customer: What is my balance?
Agent: HANDOFF

Customer: Can you increase my credit limit?
Agent: HANDOFF

Customer: No, I'm not interested.
Agent: DECLINE

Customer: I don’t want this offer.
Agent: DECLINE

Customer: Please stop, I don’t want it.
Agent: DECLINE

Instructions:
1. If the customer asks a clarification that is **directly related to the offer terms**, answer briefly using the offer terms above, then ask: "Do you have any questions about the promotional offer?"
2. If the customer says they have no questions (e.g., 'no', 'I'm good', 'no questions', 'everything is clear') → respond ONLY with: "Do I have your consent to enroll your card account in this Promotional APR Offer?"
3. If the customer gives consent (e.g., 'yes', 'I agree', 'proceed', 'sounds good') → respond ONLY with: CONSENT
4. If the customer asks anything **not related to the offer terms** → respond ONLY with: HANDOFF
5. If the customer **clearly declines the offer** (e.g., 'no I'm not interested', 'I don’t want it', 'please stop', 'not interested') → respond ONLY with: DECLINE
6. If unclear, ask politely to repeat.

Customer input: {query}
""")
    ]
    response = llm(messages)
    print(f"💡 Agent response: {response.content.strip()}")
    return response.content.strip()

# ------------------------
# LANGGRAPH NODES
# ------------------------
class OfferAgentState(TypedDict, total=False):
    customer_input: str
    step: str

def read_offer(state: OfferAgentState) -> OfferAgentState:
    print("DEBUG: Reading offer terms...", OFFER_TERMS)
    speak_text(f"Hello. We have an offer for you. {OFFER_TERMS}")
    return {**state, "step": "offer_read"}

def listen_customer(state: OfferAgentState) -> OfferAgentState:
    transcript = record_and_transcribe()
    state["customer_input"] = transcript
    return state

def analyze_response(state: OfferAgentState) -> OfferAgentState:
    transcript = state.get("customer_input", "")
    result = analyze_customer_query(transcript)
    
    # Check for consent
    if result.upper().endswith("CONSENT"):
        speak_text("Thank you for your consent. I am going to submit the application for processing.")
        state["step"] = "consent"

    elif result.upper().endswith("DECLINE"):
        state["step"] = "decline"
    # Check if agent should ask for consent

        # Handoff to live agent
    elif result.upper().endswith("HANDOFF"):
        speak_text("I understand your question is about your account services. Since this isn’t related to the promotional APR offer, I’ll transfer you to a live agent who can further assist you.")
        state["step"] = "handoff"

    elif "Do I have your consent" in result:
        speak_text(result)
        state["step"] = "ask_consent"
    # Otherwise, it's a clarification or repeat
    else:
        speak_text(result)
        state["step"] = "clarification"

    return state

def process_application(state: OfferAgentState) -> OfferAgentState:
    print("DEBUG: Calling the service to process the application...")
    time.sleep(2)
    speak_text("Your enrollment has been approved. Thank you for choosing our promotional offer.")
    return {**state, "step": "process_application"}

def handle_decline(state: OfferAgentState) -> OfferAgentState:
    speak_text("I understand. We will not proceed with the promotional offer. Thank you for your time.")
    state["step"] = "decline"
    return state

def handoff(state: OfferAgentState) -> OfferAgentState:
    print("DEBUG: Escalating to live agent...")
    return {**state, "step": "handoff"}

# ------------------------
# GRAPH
# ------------------------
graph = StateGraph(OfferAgentState)
graph.add_node("read_offer", read_offer)
graph.add_node("listen_customer", listen_customer)
graph.add_node("analyze_response", analyze_response)
graph.add_node("process_application", process_application)
graph.add_node("decline", handle_decline)
graph.add_node("handoff", handoff)


graph.add_edge("read_offer", "listen_customer")
graph.add_edge("listen_customer", "analyze_response")
graph.add_conditional_edges(
    "analyze_response",
    lambda state: (
        "process_application" if state.get("step") == "consent"
        else "handoff" if state.get("step") == "handoff"
        else "decline" if state.get("step") == "decline"
        else "listen_customer"
    ),
    {
        "process_application": "process_application",
        "handoff": "handoff",
        "decline": "decline",
        "listen_customer": "listen_customer"
    }
)

graph.set_entry_point("read_offer")

workflow = graph.compile()

# ------------------------
# RUN
# ------------------------
if __name__ == "__main__":
    print("🚀 Starting Voice Offer POC")
    result = workflow.invoke({})
    print("✅ Finished with state:", result)

# speak_text(f"Hello. We have an offer for you. {OFFER_TERMS}")
# record_and_transcribe()