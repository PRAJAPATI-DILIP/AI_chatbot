
from pathlib import Path
import json

from ollama import chat

from app.retriever import load_documents
from app.vector_store import build_vector_store, search_vectors


# Model configuration
MODEL = "qwen3:1.7b"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"

# Separate conversation memory files
GENERAL_MEMORY_FILE = DATA_DIR / "general_history.json"
DOCUMENT_MEMORY_FILE = DATA_DIR / "document_history.json"

FALLBACK = "I don't know based on the provided document."

SIMILARITY_THRESHOLD = 0.4


# Step 1: Load conversation memory
def load_memory(memory_file):

    if not memory_file.exists():
        return []

    try:
        with open(memory_file, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return []


# Step 2: Save conversation memory
def save_memory(messages, memory_file):

    with open(memory_file, "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=4, ensure_ascii=False)


def build_contextual_query(question, previous_messages):

    # Get the most recent user question
    previous_questions = [
        message["content"]
        for message in previous_messages
        if message["role"] == "user"
    ]

    if not previous_questions:
        return question

    last_question = previous_questions[-1]

    # Combine previous question with current question
    contextual_query = last_question + " " + question

    return contextual_query



# Step 3: Generate AI response
def ask_ai(
    question,
    relevant_content=None,
    previous_messages=None,
    mode="document"
):

    if mode == "general":

        system_message = {
            "role": "system",
            "content": """
You are a helpful AI assistant.

RULES:
1. Answer questions clearly and naturally.
2. Use your general knowledge.
3. Explain concepts with examples when useful.
4. If you are uncertain, say so.
5. Do not invent facts.
"""
        }

    else:

        system_message = {
            "role": "system",
            "content": f"""
You are a strict document-based AI assistant.

RULES:
1. Answer ONLY from the provided information.
2. Do NOT use your own knowledge.
3. Do NOT guess or complete missing information.
4. If the answer is not clearly present, reply exactly:
"I don't know based on the provided document."
5. Do not introduce unrelated information.

RELEVANT DOCUMENT INFORMATION:
{relevant_content}
"""
        }

    messages = [system_message]

    # Include only the last 6 messages from the selected mode
    messages.extend((previous_messages or [])[-6:])

    messages.append({
        "role": "user",
        "content": question
    })

    response = chat(
        model=MODEL,
        messages=messages
    )

    return response["message"]["content"]


# Step 4: Main chatbot
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

    # Build vector database
    collection = build_vector_store(documents)

    # Load separate conversation histories
    general_messages = load_memory(GENERAL_MEMORY_FILE)
    document_messages = load_memory(DOCUMENT_MEMORY_FILE)

    if general_messages:
        print("General conversation loaded.")
    else:
        print("No general conversation found.")

    if document_messages:
        print("Document conversation loaded.")
    else:
        print("No document conversation found.")

    print("\nType 'exit' to quit.\n")

    # Main chat loop
    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("AI: Goodbye!")
            break

        if not question:
            continue

        # Select mode
        mode = input(
            "Select mode (general/document): "
        ).strip().lower()

        # GENERAL MODE
        if mode == "general":

            answer = ask_ai(
                question,
                previous_messages=general_messages,
                mode="general"
            )

        # DOCUMENT MODE
        elif mode == "document":
            
            contextual_query = build_contextual_query(question,document_messages)

            results = search_vectors(collection, contextual_query)

            if results is None:

                answer = FALLBACK

            else:

                documents_found = results["documents"][0]
                distances = results["distances"][0]
                metadatas = results["metadatas"][0]

                print("\n--- Retrieval Details ---")

                for metadata, distance in zip(metadatas, distances):

                    print(
                        f"File: {metadata['filename']} | "
                        f"Distance: {distance:.4f}"
                    )

                print("------------------------")

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

                    answer = ask_ai(
                        question,
                        relevant_content,
                        document_messages,
                        mode="document"
                    )

        # INVALID MODE
        else:

            print(
                "Invalid mode. Please enter general or document."
            )
            continue

        # Display answer
        print(f"ChatBot: {answer}\n")

        # Save conversation according to selected mode
        if mode == "general":

            general_messages.append({
                "role": "user",
                "content": question
            })

            general_messages.append({
                "role": "assistant",
                "content": answer
            })

            save_memory(
                general_messages,
                GENERAL_MEMORY_FILE
            )

        elif mode == "document":

            document_messages.append({
                "role": "user",
                "content": question
            })

            document_messages.append({
                "role": "assistant",
                "content": answer
            })

            save_memory(
                document_messages,
                DOCUMENT_MEMORY_FILE
            )


if __name__ == "__main__":
    main()
