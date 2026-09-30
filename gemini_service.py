import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


def get_api_key() -> str:
    """
    Read the Gemini API key from the .env file.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError(
            "Gemini API key is missing. "
            "Open the .env file and set GEMINI_API_KEY."
        )

    return api_key


def get_model_name() -> str:
    """
    Read the Gemini model name from .env.
    """
    return os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    ).strip()


def get_client():
    """
    Create the Google GenAI client.
    """
    return genai.Client(
        api_key=get_api_key()
    )


def generate_text(
    prompt: str,
    max_output_tokens: int = 1200,
    temperature: float = 0.4,
) -> str:
    """
    Send a prompt to Gemini.

    Automatically retries temporary 503/429/5xx errors
    with increasing delays.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = get_client()
    model = get_model_name()

    # Maximum number of attempts.
    max_attempts = 4

    # Delays: 2, 4, 8 seconds.
    retry_delays = [2, 4, 8]

    last_error = None

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

            text = getattr(
                response,
                "text",
                None,
            )

            if not text:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return text.strip()

        except Exception as exc:

            last_error = exc

            error_text = str(exc)

            # Retry only transient errors.
            is_transient = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "500" in error_text
                or "502" in error_text
                or "504" in error_text
            )

            if not is_transient:
                raise

            # No more retries.
            if attempt == max_attempts - 1:
                break

            delay = retry_delays[attempt]

            print(
                f"\nGemini temporarily unavailable."
                f" Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Gemini is temporarily unavailable after "
        f"{max_attempts} attempts.\n\n"
        f"Last error: {last_error}"
    )