from gemini_service import generate_text


def summarize_text(text: str) -> str:
    """
    Summarize a long educational passage.
    """

    text = text.strip()

    if not text:
        raise ValueError("Please enter text to summarize.")

    prompt = f"""
You are EduGenie, an AI educational assistant.

Summarize the following educational passage for a student.

Requirements:
1. Keep the important information.
2. Remove repetition and unnecessary details.
3. Use simple and clear language.
4. Preserve important names, concepts, dates, formulas, and facts.
5. Organize the summary with short paragraphs or bullet points where helpful.
6. Do not add information that is not present in the original passage.

Educational passage:
{text}
"""

    return generate_text(
        prompt,
        max_output_tokens=1200,
        temperature=0.3,
    )