
import chromadb

from app.embeddings import generate_embedding
from app.retriever import chunk_text, split_into_sections

from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "data" / "chroma_db")
)


def build_vector_store(documents):

    # Rebuild the collection using the current documents
    try:
        client.delete_collection("documents")
    except Exception:
        pass

    collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

    ids = []
    texts = []
    embeddings = []
    metadatas = []

    index = 0

    for filename, content in documents.items():

        sections = split_into_sections(content)

        for section in sections:

            chunks = chunk_text(section)

            for chunk in chunks:

                vector = generate_embedding(chunk)

                ids.append(f"chunk_{index}")
                texts.append(chunk)
                embeddings.append(vector)

                metadatas.append({
                    "filename": filename
                })

                index += 1

    if texts:
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(f"Vector database created: {len(texts)} chunks")

    return collection


def search_vectors(collection, question):

    if collection.count() == 0:
        return None

    question_embedding = generate_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results



if __name__ == "__main__":

    from app.retriever import load_documents

    data_dir = BASE_DIR / "data" / "raw"

    documents = load_documents(data_dir)

    print("Documents:", list(documents.keys()))

    collection = build_vector_store(documents)

    test_questions = [
       "hello"
    ]
    print("Current test:", test_questions)

    for question in test_questions:

        print("\n================================")
        print(" TEST QUESTION:", question)

        results = search_vectors(collection, question)

        if results is None:
            print("No documents available.")
            continue

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            print("\nFile:", metadata["filename"])
            print("Distance:", round(distance, 4))
            print("Content:", document)
