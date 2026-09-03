from pathlib import Path
from ollama import chat

MODEL = "tinyllama"
DATA_FILE = Path(__file__).parent / "data" / "raw" / "doc1.txt"


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return file.read()


def ask_ai(messages):
    response = chat(
        model=MODEL,
        messages=messages
    )

    return response["message"]["content"]


def main():
    print("================================")
    print("          AI Chatbot")
    print("================================")

    # Load our own data
    document = load_data()

    print("Document loaded successfully.")
    print("Type 'exit' to quit.\n")

    messages = [
    {
        "role": "system",
        "content": f"""
You are a document-based AI assistant.

IMPORTANT RULES:
1. You must answer ONLY using the information in the document below.
2. Do NOT use your own knowledge.
3. Do NOT make assumptions.
4. If the answer cannot be found in the document, reply exactly:
"I don't know based on the provided document."

DOCUMENT:
{document}
"""
    }
]

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("AI: Goodbye!")
            break

        messages.append({
            "role": "user",
            "content": question
        })

        answer = ask_ai(messages)

        messages.append({
            "role": "assistant",
            "content": answer
        })

        print(f" ChatBot: {answer}\n")


if __name__ == "__main__":
    main()