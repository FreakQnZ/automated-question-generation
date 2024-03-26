import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains import RetrievalQA

if __name__ == "__main__":
    load_dotenv()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    embeddings = HuggingFaceEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(index_name="mech", embedding=embeddings)

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    mixtral_llm = HuggingFaceEndpoint(
        endpoint_url=repo_id,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        top_k=30,
        repetition_penalty=1.02,
    )

    qa = RetrievalQA.from_chain_type(llm=mixtral_llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

    res = qa.invoke("What are the different types of heat engine")
    print(res["result"])