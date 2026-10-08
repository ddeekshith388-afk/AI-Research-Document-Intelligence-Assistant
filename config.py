import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key exists
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Add your Gemini API key to .env"
    )

print("Gemini API key loaded successfully.")