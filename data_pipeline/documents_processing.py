
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import embeddings


def extract_text_from_pdf(file_name):
    print(file_name)
    loader = PyPDFLoader(file_name)
    documents = loader.load()
    return documents


def chunk_text(documents, chunk_size=512):
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            separators=["\n", "\n\n"]
        )
    chunks = splitter.split_documents(documents)
    return chunks


def documents_processing():
    # Below code can be extended if we have multiple files and with different file format. The additional logic can be added to read different file types.
    documents_directory = "data/"
    documents = []
    for file_name in os.listdir(documents_directory):
        if file_name.endswith(".pdf"):  # Change based on your file types
            documents.extend(extract_text_from_pdf(os.path.join(documents_directory,file_name)))
    
    chunks = chunk_text(documents, chunk_size=512)
    embeddings.Embeddings().embed_documents(chunks)
