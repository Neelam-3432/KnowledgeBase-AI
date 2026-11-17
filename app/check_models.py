
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Checking available Gemini models...\n")

try:
    for model in client.models.list():
        actions = getattr(model, "supported_actions", [])

        if "generateContent" in actions:
            print("Model:", model.name)
            print("Supported actions:", actions)
            print("-" * 40)

except Exception as error:
    print("Model listing failed:", error)
