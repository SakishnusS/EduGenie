from gemini_service import generate_text


def main():
    print("=" * 60)
    print("EduGenie Gemini API Test")
    print("=" * 60)

    prompt = """
Explain Artificial Intelligence to a first-year college student
in three simple sentences.
"""

    try:
        result = generate_text(
            prompt,
            max_output_tokens=300,
            temperature=0.3,
        )

        print("\nGemini response:\n")
        print(result)

        print("\n" + "=" * 60)
        print("SUCCESS: Gemini API is working.")
        print("=" * 60)

    except Exception as exc:
        print("\nERROR:")
        print(exc)

        print("\nCheck:")
        print("1. Your API key is present in .env")
        print("2. google-genai is installed")
        print("3. Your internet connection is working")


if __name__ == "__main__":
    main()