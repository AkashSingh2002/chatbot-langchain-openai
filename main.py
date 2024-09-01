import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
import pickle
import faiss

load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

def load_documents(urls):
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    return data

def split_documents(data):
    text_splitter = CharacterTextSplitter(separator='\n', 
                                          chunk_size=1000, 
                                          chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    return docs

def create_faiss_store(docs, embeddings):
    vector_store = FAISS.from_documents(docs, embeddings)
    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store

def load_faiss_store():
    with open("faiss_store_openai.pkl", "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

def init_qa_chain(llm, vector_store):
    retriever = vector_store.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    return chain

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("question")
    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    llm = OpenAI(temperature=0, model_name='')

    vector_store = load_faiss_store()

    chain = init_qa_chain(llm, vector_store)

    result = chain({"question": user_input}, return_only_outputs=True)
    
    return jsonify(result)

if __name__ == "__main__":
    urls = [
        "Enter your link here"
    ]

    data = load_documents(urls)

    docs = split_documents(data)

    embeddings = OpenAIEmbeddings()

    vector_store = create_faiss_store(docs, embeddings)

    app.run(host="0.0.0.0", port=5000)
