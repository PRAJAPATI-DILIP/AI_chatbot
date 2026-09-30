
from pathlib import Path


def load_documents(data_dir):
    documents = {}

    for file_path in data_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            documents[file_path.name] = file.read()

    return documents


def split_into_sections(content):
    sections = []
    current_section = []

    for line in content.splitlines():
        line = line.strip()

        # A numbered heading starts a new section
        if line and line[0].isdigit() and "." in line:
            if current_section:
                sections.append("\n".join(current_section))

            current_section = [line]

        elif line:
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section))

    return sections


def search_documents(documents, question):
    question_words = set(question.lower().split())

    # Ignore common words
    stop_words = {
        "who", "what", "is", "the", "a", "an",
        "of", "in", "on", "for", "to", "and"
    }

    question_words = question_words - stop_words

    results = []

    for filename, content in documents.items():

        sections = split_into_sections(content)

        for section in sections:
            section_words = set(section.lower().split())

            matching_words = question_words.intersection(section_words)

            if len(matching_words)>=2:
                results.append({
                    "filename": filename,
                    "content": section,
                    "matches": len(matching_words)
                })

    results.sort(
        key=lambda item: item["matches"],
        reverse=True
    )

    return results[:3]


if __name__ == "__main__":
    data_dir = Path("../data/raw")

    documents = load_documents(data_dir)

    print("Documents found:", list(documents.keys()))

    question = input("Enter your question: ")

    results = search_documents(documents, question)

    if results:
        for result in results:
            print("\nFile:", result["filename"])
            print("Matches:", result["matches"])
            print("Content:")
            print(result["content"])
    else:
        print("\nNo relevant information found.")