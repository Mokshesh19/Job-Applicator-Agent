# config.py
# Centralized configuration loaded from environment variables.
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "You are a job application assistant. You help users craft professional "
    "resumes, write compelling cover letters, prepare for interviews, and "
    "navigate the job application process. Be concise, professional, and "
    "actionable in your responses."
))
