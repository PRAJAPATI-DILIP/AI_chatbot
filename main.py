
from pathlib import Path
import json

from ollama import chat

from app.retriever import load_documents
from app.vector_store import build_vector_store, search_vectors


MODEL = "qwen3:1.7b"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MEMORY_FILE = DATA_DIR / "chat_history.json"

FALLBACK = "I don't know based on the provided document."

SIMILARITY_THRESHOLD = 0.7


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
4. If the answer is not clearly present, reply exactly:
"{FALLBACK}"
5. Ignore previous conversation information not supported
   by the provided document.
6. Do not introduce unrelated sections.

RELEVANT DOCUMENT INFORMATION:
{relevant_content}
"""
    }

    messages = [system_message]

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

    # Step 1: Load documents
    documents = load_documents(DATA_DIR)

    if not documents:
        print("No documents found.")
        return

    print("Documents loaded successfully.")

    # Step 2: Build vector database
    collection = build_vector_store(documents)

    # Step 3: Load previous conversation
    previous_messages = load_memory()

    if previous_messages:
        print("Previous conversation loaded.")
    else:
        print("No previous conversation found.")

    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("AI: Goodbye!")
            break

        if not question:
            continue

        # Step 4: Search using ChromaDB
        results = search_vectors(collection, question)

        if results is None:
            answer = FALLBACK

        else:
            documents_found = results["documents"][0]
            distances = results["distances"][0]

            # Filter by cosine distance
            relevant_documents = [
                document
                for document, distance in zip(
                    documents_found,
                    distances
                )
                if distance <= SIMILARITY_THRESHOLD
            ]

            if not relevant_documents:
                answer = FALLBACK

            else:
                relevant_content = "\n\n".join(
                    relevant_documents
                )

                # Step 5: Generate answer using Qwen
                answer = ask_ai(
                    question,
                    relevant_content,
                    previous_messages
                )

        print(f"ChatBot: {answer}\n")

        # Step 6: Save conversation
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
