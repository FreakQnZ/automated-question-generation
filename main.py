import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from openai import embeddings
from pinecone import Pinecone

if __name__ == "__main__":
    print("Hello Langchain")
    load_dotenv()

    # pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # loader = TextLoader("blog.txt")
    # document = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # texts = text_splitter.split_documents(document)
    # print(len(texts))

    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    # embeddings = HuggingFaceEmbeddings()

    # docsearch = PineconeVectorStore.from_documents(
    #     texts, embeddings, index_name="medum-blog"
    # )

    # template = """
    # Who is the current Prime minister of India. summarise his life in 50 words
    # """

    # prompt_template = PromptTemplate(template=template, input_variables=[])

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    mixtral_llm = HuggingFaceEndpoint(
        endpoint_url=repo_id,
        task="text-generation",
        # model_kwargs={
        max_new_tokens=512,
        temperature=0.1,
        top_k=30,
        repetition_penalty=1.02,
        # },
    )

    # chain = LLMChain(llm=mixtral_llm, prompt=prompt_template)

    # res = chain.invoke(input={})
    # print(res)

    # qa = RetrievalQA.from_chain_type(
    #     llm=mixtral_llm,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True,
    # )

    # query = "What is a Vector DB? Give me a 15 word answer for a beginner"

    # results = qa({"query": query})

    # print(results)
