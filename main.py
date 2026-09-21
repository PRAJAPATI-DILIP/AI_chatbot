from pathlib import Path
import json
from ollama import chat

MODEL = "tinyllama"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MEMORY_FILE = DATA_DIR / "chat_history.json"

def load_data():
    documents = []

    for file_path in DATA_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        documents.append(
            f"""
--- {file_path.name} ---
{content}
"""
        )

    return "\n".join(documents)


def load_memory():
    if not MEMORY_FILE.exists():
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=4, ensure_ascii=False)


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

    document = load_data()

    print("Document loaded successfully.")

    # Load previous conversation
    previous_messages = load_memory()

    if previous_messages:
        print("Previous conversation loaded.")
    else:
        print("No previous conversation found.")

    print("Type 'exit' to quit.\n")

    system_message = {
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

    # Start with system instructions
    messages = [system_message]

    # Add previous conversation
    messages.extend(previous_messages)

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            save_memory(messages[1:])
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

        print(f"ChatBot: {answer}\n")

        # Save conversation after every answer
        save_memory(messages[1:])


if __name__ == "__main__":
    main()