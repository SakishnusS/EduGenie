import os

from dotenv import load_dotenv

from gemini_service import generate_text


load_dotenv()


def _use_local_model() -> bool:
    """
    Check whether the optional local LaMini-Flan-T5 model
    should be used for concept explanations.
    """

    return (
        os.getenv(
            "USE_LOCAL_EXPLANATION",
            "false"
        )
        .strip()
        .lower()
        == "true"
    )


def _get_local_model_name() -> str:
    """
    Get the local model name from the .env file.
    """

    return os.getenv(
        "LOCAL_MODEL_NAME",
        "MBZUAI/LaMini-Flan-T5-783M"
    ).strip()


_local_tokenizer = None
_local_model = None


def _load_local_model():
    """
    Load LaMini-Flan-T5 only when it is first required.
    """

    global _local_tokenizer
    global _local_model

    if (
        _local_tokenizer is not None
        and _local_model is not None
    ):
        return _local_tokenizer, _local_model

    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Local explanation dependencies are not installed. "
            "Run: pip install -r requirements-local.txt"
        ) from exc

    model_name = _get_local_model_name()

    _local_tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    _local_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    _local_model.to("cpu")
    _local_model.eval()

    return _local_tokenizer, _local_model


def explain_with_local_model(topic: str) -> str:
    """
    Explain a concept using the local LaMini-Flan-T5 model.
    """

    tokenizer, model = _load_local_model()

    import torch

    prompt = f"""
Explain the following concept to a beginner student.

Topic:
{topic}

Requirements:
- Use simple language.
- Explain what it means.
- Explain how it works.
- Give one simple example.
- Mention the important points.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            num_beams=4,
            early_stopping=True,
        )

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    if not result.strip():
        raise RuntimeError(
            "The local explanation model returned an empty response."
        )

    return result.strip()


def explain_concept(topic: str) -> str:
    """
    Main explanation function used by FastAPI.

    Uses the local LaMini-Flan-T5 model when enabled.
    Otherwise uses Gemini.
    """

    topic = topic.strip()

    if not topic:
        raise ValueError(
            "Please enter a topic to explain."
        )

    # Use local model when explicitly enabled.
    if _use_local_model():

        try:
            return explain_with_local_model(topic)

        except Exception as exc:

            print(
                "Local explanation model failed. "
                f"Falling back to Gemini: {exc}"
            )

    # Gemini explanation
    prompt = f"""
You are EduGenie, an AI educational assistant.

Explain the following topic to a student:

{topic}

Use this structure:

1. Simple Definition
2. How It Works
3. Easy Example
4. Key Points
5. Quick Revision

Requirements:
- Use simple student-friendly language.
- Start from the basics.
- Avoid unnecessary jargon.
- Use an analogy when useful.
- Keep the explanation clear and organized.
- Do not invent facts.
"""

    return generate_text(
        prompt,
        max_output_tokens=1200,
        temperature=0.35,
    )