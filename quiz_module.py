import json
import re

from gemini_service import generate_text


def clean_json_block(text: str) -> str:
    """
    Remove Markdown code fences and extract a JSON array.
    """

    cleaned = text.strip()

    # Remove Markdown fences such as ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Find the JSON array if the model added extra text.
    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    return cleaned.strip()


def _normalize_answer(answer: str, options: list[str]) -> str:
    """
    Convert letter answers such as A/B/C/D to the actual option text.
    """

    answer = answer.strip()

    if answer.upper() in {"A", "B", "C", "D"}:
        index = ord(answer.upper()) - ord("A")

        if index < len(options):
            return options[index]

    # Match the answer to an option ignoring case and whitespace.
    for option in options:
        if answer.casefold() == option.casefold():
            return option

    return answer


def generate_quiz(passage: str) -> list[dict]:
    """
    Generate exactly 3 multiple-choice questions.
    """

    passage = passage.strip()

    if not passage:
        raise ValueError("Please enter a passage or topic for the quiz.")

    prompt = f"""
You are EduGenie, an educational quiz generator.

Generate exactly 3 multiple-choice questions based ONLY on the passage below.

Each question must contain:
- question
- options: exactly 4 options
- answer: the correct option text
- explanation: a short explanation

Rules:
1. Questions must test understanding of the provided content.
2. Each question must have exactly four options.
3. Only one option must be correct.
4. Distractors should be plausible.
5. Return ONLY a valid JSON array.
6. Do not use Markdown.
7. Do not add any text before or after the JSON.

Expected structure:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Correct option text",
    "explanation": "Why this answer is correct."
  }}
]

Passage:
{passage}
"""

    raw_response = generate_text(
        prompt,
        max_output_tokens=1800,
        temperature=0.4,
    )

    cleaned = clean_json_block(raw_response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Quiz generation returned invalid JSON: {exc}"
        ) from exc

    if isinstance(data, dict):
        data = data.get("questions", [])

    if not isinstance(data, list):
        raise RuntimeError("Quiz response is not a JSON list.")

    questions = []

    for item in data:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()

        options = item.get("options", [])
        if not isinstance(options, list):
            continue

        options = [
            str(option).strip()
            for option in options
            if str(option).strip()
        ]

        answer = str(
            item.get("answer", item.get("correct_answer", ""))
        ).strip()

        explanation = str(
            item.get("explanation", "")
        ).strip()

        if not question:
            continue

        if len(options) != 4:
            continue

        answer = _normalize_answer(answer, options)

        if answer not in options:
            continue

        questions.append(
            {
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": explanation,
            }
        )

    if len(questions) < 3:
        raise RuntimeError(
            "Gemini did not return three valid quiz questions. "
            "Please try again."
        )

    return questions[:3]