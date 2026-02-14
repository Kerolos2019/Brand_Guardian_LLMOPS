import json
import os
import logging
import re  # <--- Added Regex for cleaning
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Import the State schema .
from backend.src.graph.state import VideoAuditState, ComplianceIssue

# Import the Service
from backend.src.services.video_indexer import VideoIndexerService

# Configure Logger
logger = logging.getLogger("brand-guardian")
logging.basicConfig(level=logging.INFO)

# --- NODE 1: THE INDEXER ---
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Downloads YouTube video and extracts insights using OpenAI Whisper and Vision API.
    """
    video_url = state.get("video_url")
    video_id_input = state.get("video_id", "vid_demo")

    logger.info(f"--- [Node: Indexer] Processing: {video_url} ---")

    local_filename = "temp_audit_video.mp4"

    try:
        vi_service = VideoIndexerService()

        # 1. DOWNLOAD
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Please provide a valid YouTube URL for this test.")

        # 2. PROCESS VIDEO (Transcription + OCR)
        logger.info(f"Processing video with OpenAI Whisper and Vision API...")
        video_data = vi_service.process_video(local_path, video_name=video_id_input)

        # 3. CLEANUP
        if os.path.exists(local_path):
            os.remove(local_path)

        logger.info("--- [Node: Indexer] Extraction Complete ---")
        return video_data

    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "transcript": "",
            "ocr_text": []
        }

# --- NODE 2: THE COMPLIANCE AUDITOR ---
def audit_content_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    Performs Retrieval-Augmented Generation (RAG) to audit the content.
    """
    logger.info("--- [Node: Auditor] querying Knowledge Base & LLM ---")
    
    transcript = state.get("transcript", "")
    
    if not transcript:
        logger.warning("No transcript available. Skipping Audit.")
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped because video processing failed (No Transcript)."
        }

    # Initialize Clients
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Use Chroma for local vector storage
    persist_directory = os.getenv("VECTOR_STORE_PATH", "./chroma_db")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # RAG Retrieval
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {' '.join(ocr_text)}"
    docs = vector_store.similarity_search(query_text, k=3)
    
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])
    
    # --- UPDATED PROMPT WITH STRICT SCHEMA ---
    system_prompt = f"""
    You are a Senior Brand Compliance Auditor.
    
    OFFICIAL REGULATORY RULES:
    {retrieved_rules}
    
    INSTRUCTIONS:
    1. Analyze the Transcript and OCR text below.
    2. Identify ANY violations of the rules.
    3. Return strictly JSON in the following format:
    
    {{
        "compliance_results": [
            {{
                "category": "Claim Validation",
                "severity": "CRITICAL",
                "description": "Explanation of the violation..."
            }}
        ],
        "status": "FAIL", 
        "final_report": "Summary of findings..."
    }}

    If no violations are found, set "status" to "PASS" and "compliance_results" to [].
    """

    user_message = f"""
    VIDEO METADATA: {state.get('video_metadata', {})}
    TRANSCRIPT: {transcript}
    ON-SCREEN TEXT (OCR): {ocr_text}
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        # --- FIX: Clean Markdown if present (```json ... ```) ---
        content = response.content
        if "```" in content:
            # Regex to find JSON inside code blocks
            content = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL).group(1)
            
        audit_data = json.loads(content.strip())
        
        return {
            "compliance_results": audit_data.get("compliance_results", []),
            "final_status": audit_data.get("status", "FAIL"),
            "final_report": audit_data.get("final_report", "No report generated.")
        }

    except Exception as e:
        logger.error(f"System Error in Auditor Node: {str(e)}")
        # Log the raw response to see what went wrong
        logger.error(f"Raw LLM Response: {response.content if 'response' in locals() else 'None'}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL"
        }