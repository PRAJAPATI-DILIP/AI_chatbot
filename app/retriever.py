
import re
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

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks




def search_documents(documents, question):
    question_words = set(re.findall(r'\b\w+\b', question.lower()))

    stop_words = {
        "who", "what", "is", "the", "a", "an",
        "of", "in", "on", "for", "to", "and"
    }

    question_words -= stop_words

    results = []

    for filename, content in documents.items():

        sections = split_into_sections(content)

        for section in sections:

            chunks = chunk_text(section)

            for chunk in chunks:
                chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))

                matching_words = question_words.intersection(chunk_words)

                if len(matching_words) >= 1:
                    results.append({
                        "filename": filename,
                        "content": chunk,
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


