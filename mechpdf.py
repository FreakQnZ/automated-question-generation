import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from utils.pdf_utils import pdf_to_text, text_to_chunk

if __name__ == "__main__":
    print("Hello Langchain")
    load_dotenv()

    # Getting Text from PDF
    pdf_path = "mech.pdf"
    document = pdf_to_text(pdf_path)

    # Converting Text to Chunks / chunk size = 420
    docs = text_to_chunk(document)

    # Converting to embeddings to store in Pinecone
    embeddings = HuggingFaceEmbeddings()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    PineconeVectorStore.from_documents(docs, embeddings, index_name="mech")
    print("Added to Pinecone")