import sqlite3
import numpy as np
# from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings

def create_database(db_name="data/user_history.db"):
    """
    Creates a SQLite database with a table to store file paths, focus topics, and embeddings.

    Parameters:
    - db_name (str): The name of the database file.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create a table for storing file paths, focus topics, and embeddings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS UserHistory (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        FilePath TEXT NOT NULL,
        EmphasisTopic TEXT NOT NULL,
        Embedding BLOB,
        Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
    print(f"Database '{db_name}' and table 'UserHistory' created successfully.")


def add_to_user_history(db_name="data/user_history.db", file_path="", topic="", embedding=None):
    """
    Adds a record to the UserHistory table in the database.

    Parameters:
    - db_name (str): Path to the SQLite database file.
    - file_path (str): Path to the related text file.
    - topic (str): The topic of emphasis.
    - embedding (numpy array): The embedding vector to store.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Convert embedding to bytes
        embedding_blob = embedding.tobytes() if embedding is not None else None

        cursor.execute("""
            INSERT INTO UserHistory (FilePath, EmphasisTopic, Embedding)
            VALUES (?, ?, ?)
        """, (file_path, topic, embedding_blob))

        conn.commit()
        conn.close()
        print(f"Record added successfully: FilePath='{file_path}', EmphasisTopic='{topic}'")
    except sqlite3.Error as e:
        print(f"Error while inserting into the database: {e}")


def compute_embeddings_and_store(db_name="data/user_history.db"):
    """
    Computes embeddings for each topic in the database and stores them in the Embedding column.

    Parameters:
    - db_name (str): The name of the SQLite database file.
    """
    # Initialize the embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        # Fetch all records without embeddings
        cursor.execute("SELECT ID, EmphasisTopic FROM UserHistory WHERE Embedding IS NULL")
        records = cursor.fetchall()

        if not records:
            print("No records without embeddings found. All topics are already embedded.")
            return

        for record_id, topic in records:
            try:
                # Compute embedding for the topic
                single_vector = embeddings.embed_query(topic)
                single_vector = np.array(single_vector, dtype=np.float32)

                # Store the embedding as a blob
                cursor.execute("UPDATE UserHistory SET Embedding = ? WHERE ID = ?",
                               (single_vector.tobytes(), record_id))
                print(f"Embedding computed and stored for ID {record_id}, Topic: {topic}")

            except Exception as e:
                print(f"Error computing embedding for ID {record_id}, Topic: {topic}. Error: {e}")

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()
        print("Embeddings computation process completed.")

def fetch_user_history(db_name="data/user_history.db"):
    """
    Fetches and displays all records from the UserHistory table, excluding embeddings.

    Parameters:
    - db_name (str): Path to the SQLite database file.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Query to fetch all records excluding the Embedding column
        cursor.execute("""
            SELECT ID, FilePath, EmphasisTopic, Timestamp
            FROM UserHistory
        """)
        records = cursor.fetchall()

        # Check if there are records to display
        if not records:
            print("The UserHistory table is empty.")
            return

        print("User History Records:")
        print("-" * 80)
        print(f"{'ID':<5} {'FilePath':<60} {'EmphasisTopic':<40} {'Timestamp':<25}")
        print("-" * 80)

        for record in records:
            record_id, file_path, topic, timestamp = record
            print(f"{record_id:<5} {file_path:<60} {topic:<40} {timestamp:<25}")

        print("-" * 80)

    except sqlite3.Error as e:
        print(f"Error fetching data from the database: {e}")
    finally:
        conn.close()

# Example usage




if __name__ == "__main__":
    fetch_user_history("data/user_history.db")
    # Step 1: Create the database
    # create_database()

    # Step 2: Add records to the database
    db_name = "data/user_history.db"

    file_paths = [
        r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\summaries\summaries_bank\roberta_ARCH_summary.txt",
        r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\summaries\summaries_bank\roberta_FTaMPE_summary.txt",
        r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\summaries\summaries_bank\roberta_NLPAPP_summary.txt",
        r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\summaries\summaries_bank\roberta_PT_summary.txt",
        r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\summaries\summaries_bank\roberta_PTaFT_summary.txt"
    ]

    topics = [
        "Architecture and Pre-training",
        "Fine-Tuning and Model Performance Evaluation",
        "NLP Applications and Evaluation",
        "Model Pre-training",
        "Pre-training and Fine-tuning Procedures"
    ]

    # for file_path, topic in zip(file_paths, topics):
        # add_to_user_history(db_name, file_path, topic)

    # Step 3: Compute and store embeddings
    # compute_embeddings_and_store(db_name)
