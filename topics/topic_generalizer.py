import re
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from topics.topic_extractor import initialize_llm


def get_topic_list(file_path="data/extracted_topics.csv"):
    """
    Reads the CSV file and returns a list of topics from the 'Topic' column.
    """
    df = pd.read_csv(file_path)
    return df['Topic'].tolist()


def chunk_list(list, chunk_size=25, overlap=5):
    """
    Divides the list into chunks with overlap.
    """
    chunks = []
    for i in range(0, len(list), chunk_size - overlap):
        chunk = list[i:i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(list):
            break
    return chunks


def extract_main_topics(summary_text):
    """
    Extracts main topics from the text in the specified format.
    """
    pattern = r"\d+\.\s\*\*(.+?)\*\*:\s*(.+)"
    return re.findall(pattern, summary_text)


def generate_summarized_topics(llm, topic_list, chunk_size=15, overlap=5):
    """
    Generates summarized main topics from the list of topics.
    """
    template = """
    Based on the following list of topics, identify 2 main overarching themes that best represent the essential content.
    Please respond in the following format exactly, without adding extra explanations or information:

    Format:

    1. **Main Theme 1**: Description of main theme 1
    2. **Main Theme 2**: Description of main theme 2

    Example:

    1. **Data Privacy Concerns**: This theme covers the importance of protecting personal information in digital platforms, addressing challenges and solutions for maintaining data security.
    2. **Advances in AI Applications**: This theme discusses recent advancements in artificial intelligence, including improvements in language models and ethical considerations.

    Now, using this exact format, identify 2 main overarching themes from the provided list:

    List of topics:
    {topic_list_part}
    """

    prompt = PromptTemplate(input_variables=["topic_list_part"], template=template)
    topic_chunks = chunk_list(topic_list, chunk_size, overlap)
    summarized_topics = []

    for index, chunk in enumerate(topic_chunks):
        topic_list_part = "\n".join(chunk)
        summary_prompt = prompt.format(topic_list_part=topic_list_part)
        summary = llm(summary_prompt).strip()
        main_topics = extract_main_topics(summary)

        for topic, description in main_topics:
            summarized_topics.append({
                "Chunk Index": index + 1,
                "Topic": topic,
                "Description": description
            })

    return summarized_topics


def get_generalized_topic_list(file_path="data/generalized_topics.csv"):
    """
    Reads the generalized topics from a CSV file and returns a formatted list of topics and descriptions.
    """
    df = pd.read_csv(file_path)
    return [f"**{row['Topic']}**: {row['Description']}" for _, row in df.iterrows()]


def reduce_topics_to_core(llm, topics_for_reduction):
    """
    Reduces the list of topics to the 6-7 most essential ones and returns the final topics and descriptions.
    """
    reduction_template = """
    The following is a list of the top 10 main themes identified in the document. Please reduce this list to the 6-7 most essential themes that capture the primary content and significance of the document.
    
        Please respond in the following format exactly, without adding extra explanations or information:

    Format:

    1. **Main Theme 1**: Description of main theme 1
    2. **Main Theme 2**: Description of main theme 2

    Example:

    1. **Data Privacy Concerns**: This theme covers the importance of protecting personal information in digital platforms, addressing challenges and solutions for maintaining data security.
    2. **Advances in AI Applications**: This theme discusses recent advancements in artificial intelligence, including improvements in language models and ethical considerations.

    Now, using this exact format, identify 6-7 most essential themes from the provided list:

    List of topics:
    {topics_for_reduction}

    List of essential topics:
    """

    reduction_prompt = PromptTemplate(input_variables=["topics_for_reduction"], template=reduction_template)
    reduction_summary_prompt = reduction_prompt.format(topics_for_reduction=topics_for_reduction)
    reduced_summary = llm(reduction_summary_prompt).strip()

    # Extract the final core topics using regular expressions
    final_topics = extract_main_topics(reduced_summary)
    return final_topics


def save_final_topics_to_csv(final_topics, file_path="data/final_topics.csv"):
    """
    Saves the final reduced topics to a CSV file.
    """
    data = [{"Topic": topic, "Description": description} for topic, description in final_topics]
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Final reduced topics saved to {file_path}")


def main():
    # Step 1: Initialize the LLM
    llm = initialize_llm()

    # Step 2: Get initial topic list and generate summarized topics
    # topic_list = get_topic_list()
    # summarized_topics = generate_summarized_topics(llm, topic_list)
    # save_to_csv(summarized_topics)

    # Step 3: Read generalized topics and prepare for reduction
    generalized_topic_list = get_generalized_topic_list()
    topics_for_reduction = "\n".join(generalized_topic_list)

    # Step 4: Reduce to 6-7 core topics
    final_topics = reduce_topics_to_core(llm, topics_for_reduction)

    # Step 5: Save final reduced topics to CSV
    save_final_topics_to_csv(final_topics)


# Run the main function
if __name__ == "__main__":
    main()
