from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def pdf_to_text(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    print(f"...")
    return documents


def text_to_chunk(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=420, chunk_overlap=30, separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(document)
    print(f"Spliited into {len(docs)} chunks")
    print("...")
    return docs
