from gemini_service import generate_text


def answer_question(question: str) -> str:
    """
    Answer an academic or general educational question.
    """

    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    prompt = f"""
You are EduGenie, an AI educational assistant for students.

Answer the following question accurately and clearly.

Rules:
1. Use simple student-friendly language.
2. Give the direct answer first.
3. Add a short explanation when useful.
4. Use examples when they make the concept easier.
5. Do not invent facts.
6. If the question is ambiguous, explain the ambiguity.
7. Keep the answer concise unless the topic requires more detail.

Student question:
{question}
"""

    return generate_text(
        prompt,
        max_output_tokens=1000,
        temperature=0.3,
    )