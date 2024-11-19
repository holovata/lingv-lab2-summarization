from topics.topic_extractor import (
    create_topic_extraction_prompt,
    extract_topics_from_chunks,
    save_to_csv
)
from topics.topic_generalizer import (
    get_topic_list,
    generate_summarized_topics,
    get_generalized_topic_list,
    reduce_topics_to_core,
    save_final_topics_to_csv
)
import csv


def extract_topics(llm, chunks, output_csv_path):
    """
    Extracts topics from given text chunks and saves to a CSV file.

    Parameters:
    - chunks (list): The list of text chunks to analyze.
    - output_csv_path (str): The path to save the extracted topics CSV file.
    """
    try:
        print("Creating prompt for topic extraction...")
        prompt = create_topic_extraction_prompt()

        print("Extracting topics from chunks...")
        results = extract_topics_from_chunks(llm, chunks, prompt)

        print(f"Saving results to CSV at {output_csv_path}...")
        save_to_csv(results, output_csv_path)
        print(f"Topics successfully saved to {output_csv_path}.")

    except Exception as e:
        print(f"Error during topic extraction: {e}")


def generalize_topics(llm, extracted_topics_path, generalized_topics_path, final_topics_path):
    """
    Generalizes topics from extracted topics and reduces them to core topics.

    Parameters:
    - extracted_topics_path (str): Path to the CSV file with extracted topics.
    - generalized_topics_path (str): Path to save generalized topics CSV file.
    - final_topics_path (str): Path to save final reduced topics CSV file.
    """
    try:
        print(f"Reading extracted topics from {extracted_topics_path}...")
        topic_list = get_topic_list(extracted_topics_path)

        if not topic_list:
            print("No topics found in the extracted topics file.")
            return

        print("Generating summarized topics...")
        summarized_topics = generate_summarized_topics(llm, topic_list)
        print(f"Saving summarized topics to {generalized_topics_path}...")
        save_to_csv(summarized_topics, generalized_topics_path)

        print(f"Reading generalized topics from {generalized_topics_path}...")
        generalized_topic_list = get_generalized_topic_list(generalized_topics_path)

        if not generalized_topic_list:
            print("No generalized topics found. Aborting reduction.")
            return

        topics_for_reduction = "\n".join(generalized_topic_list)

        print("Reducing topics to core topics...")
        final_topics = reduce_topics_to_core(llm, topics_for_reduction)

        print(f"Saving final reduced topics to {final_topics_path}...")
        save_final_topics_to_csv(final_topics, final_topics_path)
        print(f"Final topics successfully saved to {final_topics_path}.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error during topic generalization: {e}")


def display_topics_from_csv(csv_path):
    """
    Reads topics from a CSV file and displays them in a human-readable format.

    Parameters:
    - csv_path (str): Path to the CSV file containing topics and descriptions.
    """
    try:
        # Open the CSV file for reading
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            topics = list(reader)

        # Check if the file contains any data
        if not topics:
            print("The CSV file is empty or contains no valid data.")
            return

        # Display the topics with a clear format
        print("Topics from the file:")
        for i, topic in enumerate(topics, start=1):
            print(f"{i}. {topic['Topic']}\n   Description: {topic['Description']}\n")
    except FileNotFoundError:
        # Handle the case when the file is not found
        print(f"The file at path '{csv_path}' was not found.")
    except KeyError as e:
        # Handle the case when the CSV structure is incorrect
        print(f"Invalid CSV structure: Missing column {e}.")
    except Exception as e:
        # Handle any other exceptions
        print(f"An error occurred while reading the file: {e}")
