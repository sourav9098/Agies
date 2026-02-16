from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

from app.security.scanner import SecureScanner
from app.services.gemini import get_gemini_response
from app.database import chat_collection

app = FastAPI(title="Aegis Secure Core")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

scanner = SecureScanner()

class ChatRequest(BaseModel): message: str
class SecurityDetail(BaseModel): scanner_name: str; is_safe: bool; risk_score: float; triggers: List[str]
class ChatResponse(BaseModel): status: str; bot_reply: Optional[str]; security_log: SecurityDetail

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # 1. Security Scan
    is_safe, risk_score, triggers = scanner.scan(request.message)
    
    security_detail = SecurityDetail(
        scanner_name="SecureLLM-Sparse-Ngram", is_safe=is_safe, risk_score=risk_score, triggers=triggers
    )

    # 2. RAG / Generation Decision
    if is_safe:
        status = "success"
        bot_reply = get_gemini_response(request.message)
    else:
        status = "blocked"
        bot_reply = "⚠️ Threat Detected. Prompt intercepted by Aegis Firewall."

    # 3. Database Logging
    log_entry = {
        "user_input": request.message, "is_safe": is_safe, "risk_score": risk_score, 
        "triggers": triggers, "timestamp": datetime.utcnow()
    }
    await chat_collection.insert_one(log_entry)

    return ChatResponse(status=status, bot_reply=bot_reply, security_log=security_detail)

@app.get("/admin/stats")
async def get_stats():
    total = await chat_collection.count_documents({})
    blocked = await chat_collection.count_documents({"is_safe": False})
    recent = await chat_collection.find().sort("timestamp", -1).limit(10).to_list(10)
    for r in recent: r["_id"] = str(r["_id"])
    return {"total_requests": total, "blocked_count": blocked, "recent_logs": recent}