# main.py
# Runs a FastAPI server exposing a /chat endpoint using our LLMClient.
from fastapi import FastAPI
from pydantic import BaseModel
from llm_client import OpenAIClient

# Initialize FastAPI app
app = FastAPI()

# Instantiate OpenAI client; replace with env var or config
client = OpenAIClient(api_key="YOUR_API_KEY_HERE")

# Request schema for /chat
class ChatRequest(BaseModel):
    message: str  # User's input message

# Response schema
class ChatResponse(BaseModel):
    reply: str    # Bot's reply

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Receives a JSON payload {"message": "..."}, calls LLMClient.generate(),
    and returns {"reply": "..."}.
    """
    answer = client.generate(req.message)
    return {"reply": answer}