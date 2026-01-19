from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
import os
import json
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from datetime import datetime
import re

load_dotenv()

# ------------------ ENV & CONFIG ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RELAY_WORKFLOW_URL = os.getenv("RELAY_WORKFLOW_URL")
RELAY_WORKFLOW_URL_2 = os.getenv("RELAY_WORKFLOW_URL_2")

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
    return {"status": "InclusionBot backend running", "health": "/health", "chat": "/chat"}

# ------------------ MODELS ------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# ------------------ STATE ------------------
sessions: Dict[str, Dict] = {}

REQUIRED_FIELDS = ["user_name", "user_age", "user_gender", "user_country", "user_query"]
DEFAULT_QUERY_TYPE = "general_awareness"
DEFAULT_LAWYER_NEEDED = False

# ------------------ HELPERS ------------------
# ------------------ HELPERS ------------------
def save_to_supabase(data: dict):
    if not supabase:
        print("Warning: Supabase not configured")
        return
    try:
        # Remove 'timestamp' if present
        data_to_save = data.copy()
        data_to_save.pop("timestamp", None)

        supabase.table("user_queries").insert(data_to_save).execute()
        print("Data saved to Supabase successfully")
    except Exception as e:
        print(f"❌ Supabase save error: {e}")


def update_supabase_record(session_id: str, updates: dict):
    if not supabase:
        print("Warning: Supabase not configured")
        return
    try:
        # Remove 'timestamp' if present in updates
        updates_to_save = updates.copy()
        updates_to_save.pop("timestamp", None)

        supabase.table("user_queries").update(updates_to_save).eq("session_id", session_id).execute()
        print("✅ Supabase record updated")
    except Exception as e:
        print(f"Supabase update error: {e}")


def trigger_relay_workflow(url: str, data: dict) -> dict:
    if not url:
        print("Warning: Relay workflow URL not configured")
        return {}
    try:
        response = httpx.post(url, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Relay workflow error: {e}")
        return {}

def extract_json(text: str) -> dict:
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        return json.loads(match.group()) if match else {}

# ------------------ LLM PROCESSING ------------------
def process_with_llm(state: dict, user_message: str) -> str:
    if not llm:
        return "Warning: LLM is not available."
    prompt = f"""
You are an empathetic chatbot for gender equality awareness.

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
        data = extract_json(content)
        for k, v in data.get("fields", {}).items():
            if k in state and v:
                state[k] = v
        return data.get("next_question", "DONE")
    except Exception as e:
        print(f"LLM error: {e}")
        return "Sorry, I couldn’t understand that. Could you rephrase?"

# ------------------ CHAT ENDPOINT ------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in sessions:
            sessions[session_id] = {field: None for field in REQUIRED_FIELDS}
            sessions[session_id]["lawyer_consult_prompted"] = False

        state = sessions[session_id]

        # ---------------- Lawyer Consultation Response ----------------
        if state.get("lawyer_consult_prompted"):
            msg = request.message.lower().strip()
            if msg in ["yes", "y", "true", "1"]:
                response_text = "📅 Great! We will get you an appointment with a legal expert soon."
            elif msg in ["no", "n", "false", "0"]:
                response_text = "ℹ️ Understood. All your information has been recorded securely."
            else:
                response_text = "⚖️ Please answer 'yes' or 'no' if you want to consult a lawyer."
                return ChatResponse(response=response_text, session_id=session_id)
            state["lawyer_consult_prompted"] = False
            return ChatResponse(response=response_text, session_id=session_id)

        # ---------------- LLM Data Collection ----------------
        next_question = process_with_llm(state, request.message)
        if next_question != "DONE":
            return ChatResponse(response=next_question, session_id=session_id)

        # ---------------- Relay Workflow 1 ----------------
        data = {k: state[k] for k in REQUIRED_FIELDS}
        data.update({
            "session_id": session_id,
            "query_type": DEFAULT_QUERY_TYPE,
            "lawyer_needed": DEFAULT_LAWYER_NEEDED,
        })

        relay_response = trigger_relay_workflow(RELAY_WORKFLOW_URL, data)
        if relay_response:
            data["query_type"] = relay_response.get("query_type", DEFAULT_QUERY_TYPE)
            data["lawyer_needed"] = relay_response.get("lawyer_needed", DEFAULT_LAWYER_NEEDED)

        # ---------------- Save to Supabase ----------------
        save_to_supabase(data)
        update_supabase_record(session_id, {"query_type": data["query_type"], "lawyer_needed": data["lawyer_needed"]})

        # ---------------- Relay Workflow 2 ----------------
        trigger_relay_workflow(RELAY_WORKFLOW_URL_2, data)

        # ---------------- Lawyer Consultation Prompt ----------------
        response_text = "✅ Thank you! Your details have been submitted successfully."
        if data["lawyer_needed"]:
            state["lawyer_consult_prompted"] = True
            response_text += "\n⚖️ This appears to be a legal matter. Would you like to consult a lawyer? Please answer 'yes' or 'no'."

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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


