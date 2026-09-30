
from pathlib import Path
import json

from ollama import chat
from app.retriever import load_documents, search_documents


MODEL = "tinyllama"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MEMORY_FILE = DATA_DIR / "chat_history.json"


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


def ask_ai(question, relevant_content, previous_messages):
    system_message = {
        "role": "system",
        "content": f"""
You are a strict document-based AI assistant.

RULES:
1. Answer ONLY from the provided information.
2. Do NOT use your own knowledge.
3. Do NOT guess or complete missing information.
4. If the answer is not clearly present in the provided information, reply exactly:
"I don't know based on the provided document."
5. Ignore information from previous conversations if it is not supported by the provided document.
6. Do not introduce unrelated sections from the document.

RELEVANT DOCUMENT INFORMATION:
{relevant_content}
"""
    }

    messages = [system_message]

    # Keep recent conversation for context
    messages.extend(previous_messages[-6:])

    messages.append({
        "role": "user",
        "content": question
    })

    response = chat(
        model=MODEL,
        messages=messages
    )

    return response["message"]["content"]


def main():
    print("================================")
    print("          AI Chatbot")
    print("================================")

    # Load documents
    documents = load_documents(DATA_DIR)

    if not documents:
        print("No documents found.")
        return

    print("Documents loaded successfully.")

    # Load previous conversation
    previous_messages = load_memory()

    if previous_messages:
        print("Previous conversation loaded.")
    else:
        print("No previous conversation found.")

    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("AI: Goodbye!")
            break

        # Search documents
        results = search_documents(documents, question)

        if not results:
            answer = "I don't know based on the provided document."

        else:
            # Take only the relevant sections
            relevant_content = "\n\n".join(
                result["content"]
                for result in results
            )

            answer = ask_ai(
                question,
                relevant_content,
                previous_messages
            )

        print(f"ChatBot: {answer}\n")

        # Save conversation
        previous_messages.append({
            "role": "user",
            "content": question
        })

        previous_messages.append({
            "role": "assistant",
            "content": answer
        })

        save_memory(previous_messages)


if __name__ == "__main__":
    main()
