import warnings
from langchain_core.prompts import PromptTemplate
from topics.topic_extractor import split_text_into_chunks, initialize_llm, load_pdf, save_to_csv

warnings.filterwarnings("ignore")


def create_summary_prompt(emphasis_topic):
    """
    Defines the prompt template for summarizing each text chunk with an emphasis on a specific topic.
    """
    return PromptTemplate(
        input_variables=["chunk"],
        template=f"""
        Summarize the following section of text with a focus on the topic: "{emphasis_topic}". 
        If the text contains sections that resemble a bibliography, reference list, or citations, please ignore them 
        and focus only on the main content.

        Provide only the summary in bullet points, without additional commentary or overly general statements.
        Keep the summary concise and specific to the content provided.

        Here is the text:
        {{chunk}}
        """
    )


def create_relevance_prompt(emphasis_topic):
    """
    Defines the prompt template for evaluating the relevance of a text chunk to a specific topic.
    """
    return PromptTemplate(
        input_variables=["chunk"],
        template=f"""
        Evaluate how explicitly the following text relates to the topic: "{emphasis_topic}".
        Rate the relevance on a scale from 1 to 5, where:
        1 - Not relevant, 2 - Slightly relevant, 3 - Somewhat relevant, 4 - Very relevant, 5 - Highly relevant.

        Provide only the relevance score as a number.

        Here is the text:
        {{chunk}}
        """
    )


def evaluate_relevance(llm, chunk, prompt):
    """
    Evaluates the relevance of a text chunk to the specified topic using the LLM.

    Parameters:
    - llm: The initialized language model.
    - chunk: The text chunk to evaluate.
    - prompt: The prompt template for evaluating relevance.

    Returns:
    - Relevance score as a float (1-5).
    """
    relevance_prompt = prompt.format(chunk=chunk.page_content)
    relevance_score = llm(relevance_prompt).strip()
    try:
        return float(relevance_score)
    except ValueError:
        return 0.0  # Default to 0 if the model returns invalid data


def summarize_chunks(llm, chunks, summary_prompt, relevance_prompt, relevance_threshold=4):
    """
    Summarizes each text chunk and evaluates its relevance to the specified topic.

    Parameters:
    - llm: The initialized language model.
    - chunks: List of text chunks to process.
    - summary_prompt: The prompt template for summarizing.
    - relevance_prompt: The prompt template for evaluating relevance.

    Returns:
    - List of dictionaries containing 'Chunk ID', 'Text Chunk', 'Summary', and 'Relevance Score' for each chunk.
    """
    results = []
    for index, chunk in enumerate(chunks, start=1):
        # Evaluate relevance
        relevance_score = evaluate_relevance(llm, chunk, relevance_prompt)

        # Skip if relevance_score < relevance_threshold
        if relevance_score < relevance_threshold:
            print(f"// Chunk {index}: rel.score = {relevance_score} - skipped (below threshold)")
            continue

        # Generate summary
        summary_prompt_text = summary_prompt.format(chunk=chunk.page_content)
        summary = llm(summary_prompt_text).strip()

        results.append({
            "Chunk ID": index,
            "Text Chunk": chunk.page_content,
            "Summary": summary,
            "Relevance Score": relevance_score
        })
        print(f"// Chunk {index}: rel.score = {relevance_score} - summary generated")
    return results


def main():
    # Initialize the LLM
    llm = initialize_llm()

    # Load and process the PDF
    pdf_file = "../document.pdf"
    pages = load_pdf(pdf_file)

    # Split text into chunks
    chunks = split_text_into_chunks(pages, 1000, 100)

    # Define the topic to emphasize in the summary
    emphasis_topic = "Pre-training and Fine-tuning Procedures for NLP Models"

    # Define the prompt for summarizing each chunk with an emphasis on the topic
    summary_prompt = create_summary_prompt(emphasis_topic)

    relevance_prompt = create_relevance_prompt(emphasis_topic)

    # Summarize each chunk and store the results
    results = summarize_chunks(llm, chunks, summary_prompt, relevance_prompt)

    # Save results to a CSV file
    save_to_csv(results, file_path="data/chunks_summaries.csv")


# Run the main function
if __name__ == "__main__":
    main()
