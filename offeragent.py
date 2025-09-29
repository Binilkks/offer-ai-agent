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
    print(f"🗣️ Agent says: {text}")
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Studio-O")
    audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1.05,  # Slightly faster
    pitch=2.0            # Slightly higher pitch for warmth
)
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

def analyze_intro_offer_query(query: str):
    messages = [
        HumanMessage(content=f"""
You are an assistant introducing a credit card promotional offer to a customer.

Examples:
Customer: Yes, tell me more.
Agent: DETAILS

Customer: Sure, explain the offer.
Agent: DETAILS

Customer: Yes.
Agent: DETAILS

Customer: Okay, go ahead.
Agent: DETAILS

Customer: No, I'm not interested.
Agent: DECLINE

Customer: No thanks.
Agent: DECLINE

Customer: Not now.
Agent: DECLINE

Instructions:
1. If the customer wants to hear more, says 'yes', 'tell me more', 'explain', 'sure', 'okay', or anything similar, respond ONLY with: DETAILS
2. If the customer declines, says 'no', 'not interested', 'no thanks', or anything similar, respond ONLY with: DECLINE
3. If unclear, ask politely: "Would you like to hear more about the promotional offer?"

Customer input: {query}
""")
    ]
    response = llm(messages)
    return response.content.strip()

def analyze_customer_query(query: str):
    messages = [
        HumanMessage(content=f"""
Here are the offer terms: {OFFER_TERMS}

You are an assistant handling customer responses after the offer terms have been read.

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

Customer: Tell me more.
Agent: DETAILS

Customer: Yes, explain the offer.
Agent: DETAILS

Instructions:
1. If the customer asks a clarification that is **directly related to the offer terms**, answer briefly using the offer terms above, then ask: "Do you have any questions about the promotional offer?"
3. If the customer says they have no questions (e.g., 'no', 'I'm good', 'no questions', 'everything is clear') → respond ONLY with: "Do I have your consent to enroll your card account in this Promotional APR Offer?"
4. If the customer gives consent (e.g., 'yes', 'I agree', 'proceed', 'sounds good') → respond ONLY with: CONSENT
5. If the customer asks anything **not related to the offer terms** → respond ONLY with: HANDOFF
6. If the customer **clearly declines the offer** (e.g., 'no I'm not interested', 'I don’t want it', 'please stop', 'not interested') → respond ONLY with: DECLINE
7. If the input is unclear or incomplete, respond ONLY with: "I'm sorry, I didn't catch that. Could you please repeat or clarify your question?"

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

def generate_intro_statement(customer_name, tenure, balance, offer_summary):
    prompt = f"""
You are a friendly customer service agent. Write a brief, natural-sounding introduction for a promotional call.
Personalize it using the following details:
- Customer name: {customer_name}
- Tenure: {tenure}
- Pay Over Time balance: {balance}
- Offer summary: {offer_summary}

Keep the introduction brief and friendly. Do not say "thank you for taking my call." 
Do NOT preface your response with any explanation or meta-text. 
Respond ONLY with the introduction you would say to the customer.
Greet the customer, mention their tenure and balance, introduce the offer, and end by asking if they would like to hear more details.
"""
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def intro_offer(state: OfferAgentState) -> OfferAgentState:
    customer_name = state.get("customer_name", "John")
    tenure = state.get("tenure", "5 years")
    balance = state.get("balance", "$1,500")
    offer_summary = "We have a limited-time offer that could lower your interest rate on new purchases."

    intro = (
        f"Hi {customer_name}, thanks for being with us for {tenure}. "
        f"I see your current Pay Over Time balance is {balance}. "
        "We have a limited-time offer that could lower your interest rate on new purchases. "
        "Would you like me to explain the details?"
    )
    
    intro = generate_intro_statement(customer_name, tenure, balance, offer_summary)
    speak_text(intro)
    return {**state, "step": "intro_offer"}

def generate_offer_summary(balance, new_purchase, current_apr, promo_apr, interest_current, interest_promo, savings):
    prompt = f"""
You are a friendly customer service agent. Write a brief, natural-sounding summary for a credit card promotional APR offer.
Personalize it using the following details:
- Current Pay Over Time balance: ${balance}
- Example new purchase amount: ${new_purchase}
- Current APR: {current_apr}%
- Promotional APR: {promo_apr}%
- Interest at current APR: ${interest_current:.2f}
- Interest at promo APR: ${interest_promo:.2f}
- Savings: ${savings:.2f}

Do NOT greet the customer or say hello. The customer has already been greeted.
Keep it concise, friendly, and easy to understand. End by saying you'll now read the official terms.
"""
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content.strip()

def offer_details(state: OfferAgentState) -> OfferAgentState:
    balance = 1500  # you can pull from customer data API
    new_purchase = 1500
    current_apr = 20.99
    promo_apr = 8.99

    interest_current = new_purchase * (current_apr / 100)
    interest_promo = new_purchase * (promo_apr / 100)
    savings = interest_current - interest_promo

    summary = generate_offer_summary(
        balance, new_purchase, current_apr, promo_apr, interest_current, interest_promo, savings
    )

    speak_text(summary)
    speak_text(OFFER_TERMS)
    return {**state, "step": "listen_customer"}

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
    step = state.get("step", "")
    if step == "intro_offer":
        # Use your intro offer analyzer
        result = analyze_intro_offer_query(transcript)
        if result.upper().endswith("DETAILS"):
            state["step"] = "details"
        elif result.upper().endswith("DECLINE"):
            state["step"] = "decline"
    else:
        result = analyze_customer_query(transcript)
        if result.upper().endswith("CONSENT"):
            speak_text("Thank you for your consent. I am going to submit the application for processing.")
            state["step"] = "consent"
        elif result.upper().endswith("DECLINE"):
            state["step"] = "decline"

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
graph.add_node("intro_offer", intro_offer)
graph.add_node("details", offer_details)
graph.add_node("listen_customer", listen_customer)
graph.add_node("analyze_response", analyze_response)
graph.add_node("process_application", process_application)
graph.add_node("decline", handle_decline)
graph.add_node("handoff", handoff)

graph.add_edge("intro_offer", "listen_customer")
graph.add_edge("listen_customer", "analyze_response")
graph.add_edge("details", "listen_customer")
graph.add_conditional_edges(
    "analyze_response",
    lambda state: (
        "process_application" if state.get("step") == "consent"
        else "handoff" if state.get("step") == "handoff"
        else "decline" if state.get("step") == "decline"
        else "details" if state.get("step") == "details"
        else "listen_customer"
    ),
    {
        "process_application": "process_application",
        "handoff": "handoff",
        "decline": "decline",
        "details": "details",
        "listen_customer": "listen_customer"
    }
)

graph.set_entry_point("intro_offer")

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
