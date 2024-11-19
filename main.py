import os
import warnings
import pandas as pd
import sqlite3
from langchain_core.prompts import PromptTemplate
from topics.topic_extractor import initialize_llm
from saved_history.topic_search import find_similar_topic
from saved_history.database_utils import add_to_user_history
from utils.text_preprocessor import process_pdf, process_text
from utils.topic_list_generator import extract_topics, generalize_topics, display_topics_from_csv
from summaries.summary_generalizer import create_final_summary,create_final_summary_with_example, get_gen_summary_list, get_summary_list,generate_generalized_summaries, save_generalized_summaries
from summaries.chunks_summary import create_summary_prompt,summarize_chunks,save_to_csv,create_relevance_prompt

warnings.filterwarnings("ignore")


def main():
    # Step 1: Console welcomes the user and asks for input text or PDF path
    print("Welcome to the NLP Summarization Assistant!")
    print("Please provide a path to a PDF file.")
    user_input = input("Enter the path to the PDF: ")

    if os.path.isfile(user_input) and user_input.endswith(".pdf"):
        # If a valid PDF path is provided
        chunks = process_pdf(user_input, 1500, 100)
    else:
        print("Error: No valid input provided. Exiting...")
        return

    if not chunks:
        print("Error: No valid input provided. Exiting...")
        return

    # Step 2: Extract topics from the input text
    llm = initialize_llm()

    extracted_topics_path = r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\last_interaction\LAST_extracted_topics.csv"
    generalized_topics_path = r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\last_interaction\LAST_generalized_topics.csv"
    final_topics_path = r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\last_interaction\LAST_final_topics.csv"

    extract_topics(llm, chunks, extracted_topics_path)
    generalize_topics(llm, extracted_topics_path, generalized_topics_path, final_topics_path)
    display_topics_from_csv(final_topics_path)

    # Step 3: Ask user to select or input the emphasis topic
    user_topic = input("\nEnter the topic you want to emphasize: ")
    print(f"Selected topic for emphasis: {user_topic}")

    # Step 4: Search for a similar topic in the database
    db_path = "saved_history/data/user_history.db"
    result = find_similar_topic(user_topic, db_path)

    chunks_summaries_path = r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\last_interaction\LAST_chunks_summaries.csv"
    # Define the prompt for summarizing each chunk with an emphasis on the topic
    sum_prompt = create_summary_prompt(user_topic)
    rel_prompt = create_relevance_prompt(user_topic)

    # Summarize each chunk and store the results
    results = summarize_chunks(llm, chunks, sum_prompt, rel_prompt)

    # Save results to a CSV file
    save_to_csv(results, chunks_summaries_path)

    # Load relevance scores from CSV
    df = pd.read_csv(chunks_summaries_path)
    chunk_summaries = df['Summary'].tolist()
    relevance_scores = df['Relevance Score'].tolist()

    # Generate generalized summaries
    generalized_summaries = generate_generalized_summaries(llm, chunk_summaries, user_topic)

    generalized_summaries_path = r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\last_interaction\LAST_generalized_summaries.csv"
    # Save generalized summaries to CSV
    save_generalized_summaries(generalized_summaries,generalized_summaries_path)

    generalized_summaries_list = get_gen_summary_list(generalized_summaries_path)
    summaries_for_reduction = "\n".join(generalized_summaries_list)

    if result and result["Similarity Score"] > 0.35:
        similar_topic = result["Most Similar Topic"]
        print("SIMILAR TOPIC:" + similar_topic)
        file_path = result["File Path"]
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                similar_topic_summary = file.read()

            # Step 5: Generate summary using the example
            final_summary = create_final_summary_with_example(
                llm,
                user_topic,
                summaries_for_reduction,
                similar_topic,
                similar_topic_summary,
            )
        else:
            print(f"Warning: File not found for similar topic. Generating new summary.")
            final_summary = create_final_summary(llm, user_topic, summaries_for_reduction)
    else:
        print("No closely related topic found in the database. Generating new summary.")
        final_summary = create_final_summary(llm, user_topic, summaries_for_reduction)

    # Output the summary
    print("\nGenerated Summary:")
    print(final_summary)

    # Step 6: Save the summary to a text file
    save_dir = "saved_history/data/user_texts"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{user_topic.replace(' ', '_')}_summary.txt")

    # Write the final_summary directly to the file
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(final_summary)

    print(f"Summary saved to: {save_path}")

    # Step 7: Add the summary and topic to the database
    add_to_user_history(db_path, save_path, user_topic)
    print("\nThe summary and topic have been saved to the database. Thank you for using the NLP Summarization Assistant!")


if __name__ == "__main__":
    main()
