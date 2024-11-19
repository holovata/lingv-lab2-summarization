import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings


def fetch_embeddings_from_db(db_name="data/user_history.db"):
    """
    Fetches all embeddings, topics, and file paths from the database.

    Parameters:
    - db_name (str): Path to the SQLite database file.

    Returns:
    - List of tuples containing ID, EmphasisTopic, FilePath, and Embedding.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT ID, EmphasisTopic, FilePath, Embedding FROM UserHistory WHERE Embedding IS NOT NULL")
        records = cursor.fetchall()
        # Decode embeddings from binary blobs
        records = [(record[0], record[1], record[2], np.frombuffer(record[3], dtype=np.float32)) for record in records]
        return records
    except sqlite3.Error as e:
        print(f"Error fetching embeddings: {e}")
        return []
    finally:
        conn.close()


def find_similar_topic(input_text, db_name="data/user_history.db"):
    """
    Finds the most similar topic in the database to the input text using cosine similarity.

    Parameters:
    - input_text (str): The text to compare against stored topics.
    - db_name (str): Path to the SQLite database file.

    Returns:
    - Dictionary containing the most similar topic, file path, and similarity score.
    """
    # Initialize the embedding model
    model = OllamaEmbeddings(model="nomic-embed-text:latest")

    # Compute the embedding for the input text
    input_embedding = np.array(model.embed_query(input_text), dtype=np.float32).reshape(1, -1)

    # Fetch embeddings from the database
    data = fetch_embeddings_from_db(db_name)

    if not data:
        print("No embeddings found in the database.")
        return None

    # Prepare data for similarity computation
    embeddings = np.array([record[3] for record in data])
    topics = [record[1] for record in data]
    file_paths = [record[2] for record in data]

    # Compute cosine similarity
    similarities = cosine_similarity(input_embedding, embeddings)[0]

    # Find the most similar topic
    most_similar_idx = np.argmax(similarities)
    most_similar_topic = topics[most_similar_idx]
    most_similar_file = file_paths[most_similar_idx]
    similarity_score = similarities[most_similar_idx]

    return {
        "Most Similar Topic": most_similar_topic,
        "File Path": most_similar_file,
        "Similarity Score": similarity_score
    }


if __name__ == "__main__":
    # Example usage
    input_text = "The architecture for the model."
    result = find_similar_topic(input_text)
    if result:
        print("Most Similar Topic:", result["Most Similar Topic"])
        print("File Path:", result["File Path"])
        print("Similarity Score:", result["Similarity Score"])
