import warnings
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")


def initialize_llm():
    """
    Initializes the LLM model with a callback manager.
    """
    return Ollama(model="llama3.2:latest", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


def load_pdf(file_path):
    """
    Loads a PDF file and returns the pages.
    """
    print(f"Loading PDF file: {file_path}")
    pdf_loader = PyPDFLoader(file_path)
    return pdf_loader.load()


def split_text_into_chunks(pages, chunk_size=700, overlap=50):
    """
    Splits text from PDF pages into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_documents(pages)
    print(f"Number of pages: {len(pages)}, Number of chunks: {len(chunks)}")
    return chunks


def create_topic_extraction_prompt():
    """
    Defines the prompt template for topic extraction.
    """
    return PromptTemplate(
        input_variables=["chunk"],
        template="""
        You are a qualitative researcher analyzing a section of text to identify the main topic. Focus on identifying the 1 central topic, with specific details unique to this chunk.

        The topic should:
        1. Reflect the main idea specific to this section of the text.
        2. Be detailed and clearly relevant to the unique content of this text.

        Provide only the title of the topic without any additional text and quotes. Ensure that it is specific and not overly generalized.

        Here is the text:
        {chunk}
        """
    )


def extract_topics_from_chunks(llm, chunks, prompt):
    """
    Extracts topics from each chunk of text using the LLM.
    """
    results = []
    for index, chunk in enumerate(chunks):
        summary_prompt = prompt.format(chunk=chunk.page_content)
        topic = llm(summary_prompt).strip()
        results.append({
            "Chunk Index": index + 1,
            "Text Chunk": chunk.page_content,
            "Topic": topic
        })
        print(f"// Chunk {index + 1} - topic detected")
    return results


def save_to_csv(data, file_path="data/extracted_topics.csv"):
    """
    Saves extracted topics data to a CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")


def main():
    # Initialize the LLM
    llm = initialize_llm()

    # Load and process the PDF
    pdf_file = "../document.pdf"
    pages = load_pdf(pdf_file)

    # Split text into chunks
    chunks = split_text_into_chunks(pages)

    # Define the prompt for extracting topics
    prompt = create_topic_extraction_prompt()

    # Extract topics from each chunk
    results = extract_topics_from_chunks(llm, chunks, prompt)

    # Save results to a CSV file
    save_to_csv(results)


# Run the main function
if __name__ == "__main__":
    main()
