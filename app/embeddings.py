
from ollama import embed


MODEL = "nomic-embed-text"


def generate_embedding(text):
    response = embed(
        model=MODEL,
        input=text
    )

    return response["embeddings"][0]
