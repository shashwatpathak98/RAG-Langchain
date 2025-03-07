import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Function to load documents
def load_documents(directory="./data"):
    from langchain_community.document_loaders import TextLoader
    import glob
    
    # Get all txt files in the directory
    txt_files = glob.glob(f"{directory}/**/*.txt", recursive=True)
    documents = []
    
    # Load each file
    for file_path in txt_files:
        try:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    
    print(f"Loaded {len(documents)} documents")
    return documents

# Function to split documents
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks")
    return splits

# Function to create vector store
def create_vector_store(splits):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

# Function to create RAG chain
def create_rag_chain(vector_store):
    # Create Gemini model instance
    llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
    
    # Create prompt template
    template = """
    You are a helpful assistant specialized in explaining RAG (Retrieval-Augmented Generation) systems. 
    Use the following context to provide a detailed answer to the question.
    If the information is not explicitly in the context, do your best to synthesize what's available.
    If you truly don't know, just say you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# Function to debug retrieval
def debug_retrieval(query, vector_store):
    print("\n===== DEBUG RETRIEVAL =====")
    docs = vector_store.similarity_search(query, k=5)
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content)
    print("\n===== END DEBUG =====")
    return docs


# Main execution
if __name__ == "__main__":
    # Process documents and create vector store
    documents = load_documents()
    splits = split_documents(documents)
    vector_store = create_vector_store(splits)
    
    # Create RAG chain
    qa_chain = create_rag_chain(vector_store)
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'debug':
            debug_query = input("Enter query to debug: ")
            relevant_docs = vector_store.similarity_search(debug_query, k=5)
            print("\nRetrieved documents:")
            for i, doc in enumerate(relevant_docs):
                print(f"\nDoc {i+1}:\n{doc.page_content[:300]}...")
            continue
        result = qa_chain.invoke({"query": query})
        print("\nAnswer:", result["result"])