from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from topics.topic_extractor import load_pdf, split_text_into_chunks

def process_pdf(file_path, chunk_size=700, overlap=50):
    print(f"Processing PDF file: {file_path}")
    pages = load_pdf(file_path)

    chunks = split_text_into_chunks(pages, chunk_size=chunk_size, overlap=overlap)

    print(f"PDF processed: {len(pages)} pages, {len(chunks)} chunks created.")
    return chunks


def process_text(input_text, chunk_size=700, overlap=50):
    print(f"Processing text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.create_documents(input_text)

    print("Number of chunks: {len(chunks)}")
    return chunks


