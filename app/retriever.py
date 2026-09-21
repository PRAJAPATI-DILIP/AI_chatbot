from pathlib import Path


def load_documents(data_dir):
    documents = {}

    for file_path in data_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            documents[file_path.name] = file.read()

    return documents


def search_documents(documents, question):
    question_words = set(question.lower().split())

    results = []

    for filename, content in documents.items():
        content_words = set(content.lower().split())

        matching_words = question_words.intersection(content_words)

        if matching_words:
            results.append({
                "filename": filename,
                "content": content,
                "matches": len(matching_words)
            })

    results.sort(
        key=lambda item: item["matches"],
        reverse=True
    )

    return results