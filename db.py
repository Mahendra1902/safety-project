from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import glob

def build_vectordb():
    docs = []

    # --- Load .txt Files ---
    try:
        txt_loader = DirectoryLoader(
            "incident_docs", 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        txt_docs = txt_loader.load()
        docs.extend(txt_docs)
    except Exception as e:
        print(f"⚠️ Failed to load .txt files: {e}")

    # --- Load .csv Files ---
    try:
        csv_files = glob.glob(os.path.join("incident_docs", "**/*.csv"), recursive=True)
        for csv_file in csv_files:
            loader = CSVLoader(file_path=csv_file)
            csv_docs = loader.load()
            docs.extend(csv_docs)
    except Exception as e:
        print(f"⚠️ Failed to load .csv files: {e}")

    print(f"📄 Total raw documents loaded: {len(docs)}")

    if not docs:
        print("❌ No documents found in 'incident_docs'. Exiting.")
        return

    # --- Split into chunks ---
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    print(f"✂️ Split into {len(split_docs)} chunks.")

    if not split_docs:
        print("❌ No content to embed after splitting. Exiting.")
        return

    # --- Embedding + FAISS Index ---
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # Change to "cuda" if using GPU
        )
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local("incident_faiss_index")
        print("✅ Vector DB created and saved as 'incident_faiss_index'")
    except Exception as e:
        print(f"❌ Error while building vector DB: {e}")

# Run the builder
if __name__ == "__main__":
    build_vectordb()
