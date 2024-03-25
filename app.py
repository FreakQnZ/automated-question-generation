from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain

if __name__ == "__main__":
    print("Hello Langchain")
    load_dotenv()

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # repo_id = "openchat/openchat-3.5-0106"

    mixtral_llm = HuggingFaceEndpoint(
        endpoint_url=repo_id,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        top_k=30,
        repetition_penalty=1.02,
    )

    pdf_path = "evs.pdf"

    loader = PyPDFLoader(file_path=pdf_path)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("pilot_run")

    saved_vectorstore = FAISS.load_local(
        "pilot_run", embeddings, allow_dangerous_deserialization=True
    )

    qa = RetrievalQA.from_chain_type(
        llm=mixtral_llm,
        chain_type="stuff",
        retriever=saved_vectorstore.as_retriever(),
        return_source_documents=True,
    )

    template = input("Question: ")
    # template = "What happened on 22nd july 1947"

    res = qa.invoke(template)
    print(res["result"])

    while True:
        template = input("Question: ")

        if template == "exit":
            print("Exiting...")
            break

        res = qa.invoke(template)
        print(res["result"])

    # prompt_template = PromptTemplate(template=template, input_variables=["person"])

    # chain = LLMChain(llm=mixtral_llm, prompt=prompt_template)

    # res = chain.invoke(input={"person": "United States of America"})
    # print(res)
