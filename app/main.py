
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.8-flash"


def ask_gemini(question: str) -> str:
    """Send a question to Gemini with basic retry handling."""

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=question
            )

            return response.text or "No text response received."

        except Exception as error:
            error_message = str(error)

            is_temporary = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "disconnected" in error_message.lower()
            )

            if is_temporary and attempt < max_attempts:
                wait_time = 2 ** attempt

                print(
                    f"Temporary error: {error_message}"
                    f"\nRetrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                return (
                    f"Request failed after attempt {attempt}: "
                    f"{error_message}"
                )

    return "Unable to get a response."


if __name__ == "__main__":
    question = input("Ask your AI assistant: ")

    print("\nSending request to Gemini...")

    answer = ask_gemini(question)

    print("\nGemini's response:")
    print(answer)
