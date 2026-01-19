from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
import os
import json
import httpx
import re
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from datetime import datetime

load_dotenv()

# ------------------ ENV & CONFIG ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RELAY_API_URL = os.getenv("RELAY_API_URL")  # Optional: Relay classification endpoint
RELAY_API_KEY = os.getenv("RELAY_API_KEY")  # Optional: Relay classification key
RELAY_WORKFLOW_URL = os.getenv("RELAY_WORKFLOW_URL")  # Optional: First relay workflow trigger URL
RELAY_WORKFLOW_URL_2 = os.getenv("RELAY_WORKFLOW_URL_2")  # Optional: Second relay workflow trigger URL

# ------------------ INIT SERVICES ------------------
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

llm = None
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )
        print("Gemini LLM initialized successfully")
    except Exception as e:
        print(f"Error initializing LLM: {e}")

# ------------------ FASTAPI SETUP ------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {
        "status": "InclusionBot backend running",
        "health": "/health",
        "chat": "/chat"
    }


# ------------------ MODELS ------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# ------------------ STATE ------------------
sessions: Dict[str, Dict] = {}

# State flags
WAITING_LAWYER_CONSENT = "waiting_lawyer_consent"

REQUIRED_FIELDS = [
    "user_name",
    "user_age",
    "user_gender",
    "user_country",
    "user_query"
]

# Additional state fields
ADDITIONAL_FIELDS = [
    "lawyer_consent"  # Will be set to True/False based on user response
]

QUESTION_PROMPTS = {
    "user_name": "Hi! May I know your name?",
    "user_age": "To help us understand better, could you please share your age?",
    "user_gender": "To help me understand you better, could you please share your gender?",
    "user_country": "Could you please tell me which country you are from?",
    "user_query": "Thank you! What is your query related to gender equality?"
}

# Default values for database safety
DEFAULT_QUERY_TYPE = "general_awareness"
DEFAULT_LAWYER_NEEDED = False

# ------------------ HELPERS ------------------
def save_to_supabase(data: dict):
    if not supabase:
        print("Warning: Supabase not configured")
        return
    try:
        supabase.table("user_queries").insert(data).execute()
        print("Data saved to Supabase successfully")
    except Exception as e:
        print(f"❌ Supabase save error: {e}")

def update_supabase_record(session_id: str, updates: dict):
    if not supabase:
        print("Warning: Supabase not configured")
        return
    try:
        supabase.table("user_queries").update(updates).eq("session_id", session_id).execute()
        print("✅ Supabase record updated")
    except Exception as e:
        print(f"Supabase update error: {e}")

def process_lawyer_consent(user_message: str, state: dict, session_id: str, data: dict) -> str:
    """Process yes/no response for lawyer consent"""
    message_lower = user_message.lower().strip()

    if message_lower in ['yes', 'y', 'true', '1']:
        state["lawyer_consent"] = True
        data["lawyer_consent"] = True
        # Update database with consent
        update_supabase_record(session_id, {"lawyer_consent": True})
        # Trigger second relay workflow
        trigger_relay_workflow_2(data)
        return "✅ Thank you for your consent. A legal expert will contact you soon."

    elif message_lower in ['no', 'n', 'false', '0']:
        state["lawyer_consent"] = False
        data["lawyer_consent"] = False
        # Update database with consent
        update_supabase_record(session_id, {"lawyer_consent": False})
        # Still trigger second relay workflow but with consent=false
        trigger_relay_workflow_2(data)
        return "✅ Understood. Your information has been recorded. If you need legal assistance in the future, feel free to reach out."

    else:
        return "Please answer with 'yes' or 'no'. Would you like to connect with a lawyer?"

def send_to_webhook(data: dict):
    if not WEBHOOK_URL:
        print("Warning: Webhook URL not configured")
        return
    try:
        httpx.post(WEBHOOK_URL, json=data, timeout=10)
        print("✅ Data sent to webhook")
    except Exception as e:
        print(f"❌ Webhook error: {e}")

def trigger_relay_workflow(data: dict) -> dict:
    if not RELAY_WORKFLOW_URL:
        print("Warning: Relay workflow URL not configured")
        return {}
    try:
        response = httpx.post(RELAY_WORKFLOW_URL, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        print("Relay workflow triggered successfully")
        return result
    except Exception as e:
        print(f"Relay workflow error: {e}")
        return {}

def trigger_relay_workflow_2(data: dict):
    if not RELAY_WORKFLOW_URL_2:
        print("Warning: Second relay workflow URL not configured")
        return
    try:
        httpx.post(RELAY_WORKFLOW_URL_2, json=data, timeout=10)
        print("Second relay workflow triggered successfully")
    except Exception as e:
        print(f"Second relay workflow error: {e}")

def classify_with_relay(data: dict) -> dict:
    if not RELAY_API_URL or not RELAY_API_KEY:
        print("Warning: Relay not configured")
        return {}
    safe_data = {k: (v if v is not None else "") for k, v in data.items()}
    try:
        headers = {"Authorization": f"Bearer {RELAY_API_KEY}", "Content-Type": "application/json"}
        response = httpx.post(RELAY_API_URL, json=safe_data, headers=headers, timeout=15)
        response.raise_for_status()
        result = response.json()
        print("Relay response:", result)
        return result
    except Exception as e:
        print(f"Relay error: {e}")
        return {}

def extract_json(text: str) -> dict:
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        else:
            return {}


# ------------------ LLM PROCESSING ------------------
def process_with_llm(state: dict, user_message: str) -> str:
    if not llm:
        return "Warning: LLM is not available. Please check your API key."
    prompt = f"""
You are an empathetic chatbot for a gender equality awareness system.

Your job is ONLY to collect user information.

Required fields:
- user_name
- user_age
- user_gender
- user_country
- user_query

Current collected data:
{json.dumps(state, indent=2)}

User message:
"{user_message}"

Rules:
1. Extract info from user's message.
2. Ask ONLY for the next missing field if any.
3. If all fields are collected, respond with "DONE".
4. Output ONLY valid JSON in this format:

{{
  "fields": {{
    "user_name": "",
    "user_age": "",
    "user_gender": "",
    "user_country": "",
    "user_query": ""
  }},
  "next_question": ""
}}

If all collected, return:
{{
  "fields": {{}} ,
  "next_question": "DONE"
}}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = (response.content or "").strip()
        print("🔍 LLM RAW OUTPUT:", content)
        data = extract_json(content)
        for k, v in data.get("fields", {}).items():
            if k in state and v:
                state[k] = v
        return data.get("next_question", "DONE")
    except Exception as e:
        print(f"LLM processing error: {e}")
        return "Sorry, I couldn’t understand that. Could you rephrase?"

# ------------------ CHAT ENDPOINT ------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in sessions:
            sessions[session_id] = {field: None for field in REQUIRED_FIELDS}
            # Initialize additional state fields
            for field in ADDITIONAL_FIELDS:
                sessions[session_id][field] = None
            sessions[session_id][WAITING_LAWYER_CONSENT] = False

        state = sessions[session_id]

        # Check if we're waiting for lawyer consent
        if state.get(WAITING_LAWYER_CONSENT):
            # Process lawyer consent response
            response_text = process_lawyer_consent(request.message, state, session_id, state.get("collected_data", {}))
            # Clear the waiting flag
            state[WAITING_LAWYER_CONSENT] = False
        else:
            next_question = process_with_llm(state, request.message)

            if next_question == "DONE":
                data = {k: state[k] for k in REQUIRED_FIELDS}
                data["session_id"] = session_id
                data["query_type"] = DEFAULT_QUERY_TYPE
                data["lawyer_needed"] = DEFAULT_LAWYER_NEEDED
                data["timestamp"] = datetime.now().isoformat()

                # Relay classification
                classification_result = classify_with_relay(data)
                data["query_type"] = classification_result.get("query_type", DEFAULT_QUERY_TYPE)
                data["lawyer_needed"] = classification_result.get("lawyer_needed", DEFAULT_LAWYER_NEEDED)

                # Save and send
                save_to_supabase(data)
                send_to_webhook(data)

                # Trigger first relay workflow and get response
                relay_response = trigger_relay_workflow(data)

                # Update data with relay response
                if relay_response:
                    query_type = relay_response.get("query_type")
                    lawyer_needed = relay_response.get("lawyer_needed")

                    if query_type is not None or lawyer_needed is not None:
                        update_data = {}
                        if query_type is not None:
                            update_data["query_type"] = query_type
                            data["query_type"] = query_type
                        if lawyer_needed is not None:
                            update_data["lawyer_needed"] = lawyer_needed
                            data["lawyer_needed"] = lawyer_needed

                        # Update Supabase record
                        update_supabase_record(session_id, update_data)

                        # Store collected data for consent processing
                        state["collected_data"] = data.copy()

                        if data["lawyer_needed"]:
                            # Ask for lawyer consent
                            state[WAITING_LAWYER_CONSENT] = True
                            response_text = "⚖️ This appears to be a legal matter. Would you like to connect with a lawyer? Please answer 'yes' or 'no'."
                        else:
                            # No lawyer needed, trigger second workflow directly
                            trigger_relay_workflow_2(data)
                            response_text = "✅ Thank you! Your details have been submitted successfully."
                else:
                    # No relay response or no updates needed
                    response_text = "✅ Thank you! Your details have been submitted successfully."
            else:
                response_text = next_question

        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ HEALTH CHECK ------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Render provides PORT env var
    uvicorn.run("main:app", host="0.0.0.0", port=port)

