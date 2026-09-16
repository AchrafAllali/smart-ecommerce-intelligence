"""
Configuration for RAG / LLM assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Database
DB_PATH = "database/ecommerce.db"

# LLM provider: "groq", "openai", "ollama", or "auto"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Ollama (local)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Generation
TEMPERATURE = 0.0
MAX_TOKENS = 1000