import os
import glob
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

# Document Loaders and Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Setup Logging & Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("indexer")

def index_docs():
    """
    Reads PDFs from backend/data, chunks them, and uploads vectors to Azure AI Search.
    """
    # 2. Define Paths
    # We look for the 'data' folder relative to this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "../../backend/data")
    
    # 3. Debug: Check Environment Variables
    logger.info("=" * 60)
    logger.info("Environment Configuration Check:")
    logger.info(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set'}")
    logger.info(f"Embedding Model: text-embedding-3-small")
    logger.info(f"Vector Store Path: {os.getenv('VECTOR_STORE_PATH', './chroma_db')}")
    logger.info("=" * 60)

    # 4. Validate Required Environment Variables
    required_vars = [
        "OPENAI_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file and ensure all variables are set.")
        return
    
    # 5. Initialize Embedding Model (The "Translator")
    # This turns text into numbers (vectors).
    try:
        logger.info("Initializing OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("✓ Embeddings model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        logger.error("Please verify your OpenAI API key.")
        return

    # 6. Initialize Chroma Vector Store (Local Database)
    try:
        logger.info("Initializing Chroma vector store...")
        persist_directory = os.getenv("VECTOR_STORE_PATH", "./chroma_db")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        logger.info(f"✓ Vector store initialized at: {persist_directory}")
    except Exception as e:
        logger.error(f"Failed to initialize Chroma: {e}")
        logger.error("Please verify your configuration.")
        return
    
    # 7. Find PDF Files
    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {data_folder}. Please add files.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDFs to process: {[os.path.basename(f) for f in pdf_files]}")
    
    all_splits = []
    
    # 8. Process Each PDF
    for pdf_path in pdf_files:
        try:
            logger.info(f"Loading: {os.path.basename(pdf_path)}...")
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()
            
            # 9. Chunking Strategy
            # We split text into 1000-character chunks with 200-character overlap
            # to ensure context isn't lost between cuts.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(raw_docs)
            
            # Tag the source for citation later
            for split in splits:
                split.metadata["source"] = os.path.basename(pdf_path)
            
            all_splits.extend(splits)
            logger.info(f" -> Split into {len(splits)} chunks.")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
    
    # 10. Upload to Chroma
    if all_splits:
        logger.info(f"Uploading {len(all_splits)} chunks to Chroma vector store...")
        try:
            # Chroma accepts batches automatically via this method
            vector_store.add_documents(documents=all_splits)
            logger.info("=" * 60)
            logger.info("✅ Indexing Complete! The Knowledge Base is ready.")
            logger.info(f"Total chunks indexed: {len(all_splits)}")
            logger.info(f"Stored in: {persist_directory}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to upload documents to Chroma: {e}")
            logger.error("Please check your configuration and try again.")
    else:
        logger.warning("No documents were processed.")

if __name__ == "__main__":
    index_docs()