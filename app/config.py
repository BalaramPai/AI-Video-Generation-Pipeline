import os

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set in the .env file."
    )

# LLM_MODEL = "gpt-5.6-luna"
OLLAMA_MODEL = "llama3.1:8b"