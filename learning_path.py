from gemini_service import generate_text


def get_learning_recommendations(topic: str) -> str:
    """
    Generate a structured beginner-to-advanced learning path.
    """

    topic = topic.strip()

    if not topic:
        raise ValueError("Please enter a topic.")

    prompt = f"""
You are EduGenie, an AI educational mentor.

Create a personalized learning path for the topic:

{topic}

Structure the answer as follows:

1. Learning Goal
2. Prerequisites
3. Beginner Level
4. Intermediate Level
5. Advanced Level
6. Practice Projects
7. Revision Strategy
8. Suggested Learning Resources

For each level:
- Explain what should be learned.
- Mention the most important subtopics.
- Give a practical exercise or project.

Make the progression logical from beginner to advanced.

Use simple language suitable for students.

For resources, suggest categories such as:
- official documentation
- books
- tutorials
- videos
- practice platforms

Do not invent fake URLs or pretend that a specific resource was verified.

Learning topic:
{topic}
"""

    return generate_text(
        prompt,
        max_output_tokens=1600,
        temperature=0.5,
    )