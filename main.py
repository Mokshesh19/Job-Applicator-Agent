# main.py
# Runs a FastAPI server exposing chat endpoints for the Job Applicator Agent.
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from llm_client import OpenAIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Job Applicator Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not config.OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable is not set")
client = OpenAIClient(
    api_key=config.OPENAI_API_KEY,
    model=config.OPENAI_MODEL,
    system_prompt=config.SYSTEM_PROMPT,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    reply: str


class ConversationRequest(BaseModel):
    messages: list[dict] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Single-turn chat: sends a message and returns the assistant's reply.
    """
    try:
        answer = client.generate(req.message)
    except Exception as e:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
    return {"reply": answer}


@app.post("/conversation", response_model=ChatResponse)
def conversation(req: ConversationRequest):
    """
    Multi-turn chat: sends a full conversation history and returns the next reply.
    Useful for maintaining context across multiple exchanges (e.g. iterating on
    a cover letter or resume).
    """
    try:
        answer = client.generate_with_history(req.messages)
    except Exception as e:
        logger.exception("LLM conversation request failed")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
    return {"reply": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
