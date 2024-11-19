import warnings
import pandas as pd
from langchain_core.prompts import PromptTemplate
from topics.topic_extractor import initialize_llm
from topics.topic_generalizer import chunk_list
from saved_history.topic_search import find_similar_topic
import os

warnings.filterwarnings("ignore")


def get_summary_list(file_path="data/chunks_summaries.csv"):
    """
    Reads the CSV file and returns a list of summaries from the 'Summary' column.
    """
    df = pd.read_csv(file_path)
    return df['Summary'].tolist()


def get_gen_summary_list(file_path="data/generalized_summaries.csv"):
    """
    Reads the CSV file and returns a list of gensummaries from the 'GenSummary' column.
    """
    df = pd.read_csv(file_path)
    return df['GenSummary'].tolist()


def generate_generalized_summaries(llm, summary_list, emphasis_topic, chunk_size=20, overlap=5):
    """
    Generates a generalized summary from the list of summaries.
    """


    # Define prompt template with the emphasis topic directly embedded
    template = f"""
    Summarize the following set of summaries with an emphasis on the topic: "{emphasis_topic}". 

    Requirements:
    - Focus on details relevant to "{emphasis_topic}", but include other significant points as well.
    - Format the summary in **concise bullet points**.
    - Omit introductory or closing remarks, and avoid general statements not related to the main content.
    - **Avoid adding interpretations or excessive explanations**; each bullet point should be clear and factual.
    - **Do not generalize excessively**: Keep points specific, maintaining a balanced view that captures all essential details.

    Content:
    {{summary_list_part}}
    """

    prompt = PromptTemplate(input_variables=["summary_list_part"], template=template)
    summary_chunks = chunk_list(summary_list, chunk_size, overlap)
    generalized_summaries = []

    for index, chunk in enumerate(summary_chunks):
        summary_list_part = "\n".join(chunk)
        summary_prompt = prompt.format(summary_list_part=summary_list_part)
        generalized_summary = llm(summary_prompt).strip()

        generalized_summaries.append({
            "Chunk Index": index + 1,
            "GenSummary": generalized_summary
        })

    return generalized_summaries


def save_generalized_summaries(generalized_summaries, file_path="data/generalized_summaries.csv"):
    """
    Saves the generalized summaries to a CSV file.
    """
    df = pd.DataFrame(generalized_summaries)
    df.to_csv(file_path, index=False)
    print(f"Generalized summaries saved to {file_path}")


def create_final_summary(llm, emphasis_topic, summaries_for_reduction):
    """
    Creates a final summary based on the generalized summaries, with an emphasis on the specified topic.
    """
    reduction_template = f"""
    Write a concise summary of the following set of summaries, focusing on the topic: "{emphasis_topic}". 

    Requirements:
    - Summarize in **concise bullet points**.
    - Focus on "{emphasis_topic}", but include any other critical details.
    - Avoid introductory or closing remarks, and maintain specificity in each bullet point.

    Content:
    {{summaries_for_reduction}}

    BULLET POINT SUMMARY:
    """

    reduction_prompt = PromptTemplate(input_variables=["summaries_for_reduction"], template=reduction_template)
    final_summary_prompt = reduction_prompt.format(summaries_for_reduction=summaries_for_reduction)
    final_summary = llm(final_summary_prompt).strip()

    return final_summary


def create_final_summary_with_example(llm, emphasis_topic, summaries_for_reduction, similar_topic, similar_topic_summary):
    reduction_template = f"""
                The user has requested a summary focusing on the topic: "{emphasis_topic}". 
                
                Below is a summary of another text with a focus on a closely related topic: "{similar_topic}".

                Example Summary with a focus on a related topic:
                {similar_topic_summary}

                TASK:
                Based on the user's input and the Example Summary provided:
                - Combine the following summaries into a single, well-structured summary.
                - Follow the **style and structure** of the Example Summary.
                - Ensure the summary focuses on "{emphasis_topic}" while incorporating all major key points from the provided summaries.
                - Format the output in **concise bullet points**.
                - Avoid adding unrelated details or generalizations.
                - Use clear, professional language that directly addresses the key points.

                Content to summarize:
                {{summaries_for_reduction}}

                OUTPUT SUMMARY:
                """
    reduction_prompt = PromptTemplate(input_variables=["summaries_for_reduction"], template=reduction_template)
    final_summary_prompt = reduction_prompt.format(summaries_for_reduction=summaries_for_reduction)
    final_summary = llm(final_summary_prompt).strip()

    return final_summary


def save_final_summary_to_txt(final_summary, file_path):
    """
    Saves the final summary to a text file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(final_summary)
    print(f"Final summary saved to {file_path}")


def main():
    # Initialize the LLM
    llm = initialize_llm()

    # Define the topic to emphasize in the summary
    emphasis_topic = "Pre-training and Fine-tuning Procedures for NLP Models"

    # Load summaries from CSV
    # chunk_summaries = get_summary_list()

    # Generate generalized summaries
    # generalized_summaries = generate_generalized_summaries(llm, chunk_summaries, emphasis_topic)

    # Save generalized summaries to CSV
    # save_generalized_summaries(generalized_summaries)

    # Load the generalized summaries from CSV for final reduction
    generalized_summaries_list = get_gen_summary_list()
    summaries_for_reduction = "\n".join(generalized_summaries_list)

    similarity_threshold = 0.3  # Define a threshold for similarity

    result = find_similar_topic(emphasis_topic,
                                r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\saved_history\data\user_history.db")
    if result:
        # If a similar topic is found
        similar_topic = result["Most Similar Topic"]
        file_path = result["File Path"]
        similarity_score = result["Similarity Score"]

        print("Most Similar Topic:", similar_topic)
        print("File Path:", file_path)
        print("Similarity Score:", similarity_score)

        if similarity_score >= similarity_threshold:
            # If similarity score meets or exceeds the threshold
            if os.path.exists(file_path):
                # Read the content of the file with the similar topic summary
                with open(file_path, "r", encoding="utf-8") as file:
                    similar_topic_summary = file.read()

                # Generate the final summary using the example
                final_summary = create_final_summary_with_example(
                    llm,
                    emphasis_topic,
                    summaries_for_reduction,
                    similar_topic,
                    similar_topic_summary
                )
            else:
                print(f"Error: File {file_path} not found.")
                return
        else:
            # If similarity score is below the threshold
            print(f"No sufficiently similar topic found (similarity score: {similarity_score:.2f}).")
            final_summary = create_final_summary(
                llm,
                emphasis_topic,
                summaries_for_reduction
            )
    else:
        # If no similar topic is found
        print("No similar topic found in the database.")
        final_summary = create_final_summary(
            llm,
            emphasis_topic,
            summaries_for_reduction
        )

    # Print and save the final summary
    print("Final Summary:\n", final_summary)
    save_final_summary_to_txt(final_summary, r"C:\Work\mi41\linguistics\lab2_summarization\lingv_lab2_smmarization\saved_history\data\user_texts\bert_PTaFT_summary_example.txt")


# Run the main function
if __name__ == "__main__":
    main()
