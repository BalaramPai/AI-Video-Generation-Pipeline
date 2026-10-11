import os

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

VIDEO_PROVIDER = os.getenv("VIDEO_PROVIDER", "runway").strip().lower()
RUNWAY_API_SECRET = os.getenv("RUNWAYML_API_SECRET")
RUNWAY_MODEL = os.getenv("RUNWAY_MODEL", "gen4.5").strip()
RUNWAY_POLL_INTERVAL_SECONDS = float(
    os.getenv("RUNWAY_POLL_INTERVAL_SECONDS", "5")
)
RUNWAY_TIMEOUT_SECONDS = float(
    os.getenv("RUNWAY_TIMEOUT_SECONDS", "900")
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VEO_MODEL = os.getenv("VEO_MODEL", "veo-3.1-generate-preview")
VEO_POLL_INTERVAL_SECONDS = float(
    os.getenv("VEO_POLL_INTERVAL_SECONDS", "5")
)
VEO_TIMEOUT_SECONDS = float(
    os.getenv("VEO_TIMEOUT_SECONDS", "900")
)