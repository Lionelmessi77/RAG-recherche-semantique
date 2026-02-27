"""
RAG Ingestion Pipeline with Qdrant Cloud
Extracts text from PDFs, chunks it, creates embeddings with OpenAI, stores in Qdrant Cloud
"""

import os
import warnings
import glob
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

import pypdf
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
DATA_FOLDER = "Data"
COLLECTION_NAME = "agro_documents"

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def get_all_pdfs(data_folder: str) -> List[str]:
    """Get all PDF files from the data folder."""
    pdf_files = glob.glob(os.path.join(data_folder, "**/*.pdf"), recursive=True)
    return pdf_files


def create_embedding(text: str) -> Optional[List[float]]:
    """Create embedding using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"  OpenAI API error: {e}")
        return None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Chunk text into smaller pieces."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def init_qdrant():
    """Initialize Qdrant Cloud client and create collection."""
    try:
        # Clean URL - remove port if present
        url = QDRANT_URL.replace(":6333", "") if QDRANT_URL and QDRANT_URL.endswith(":6333") else QDRANT_URL

        qdrant = QdrantClient(
            url=url,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False
        )

        # Test connection
        collections = qdrant.get_collections().collections
        collection_names = [col.name for col in collections]

        if COLLECTION_NAME not in collection_names:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            print(f"✅ Created Qdrant Cloud collection: {COLLECTION_NAME}")
        else:
            print(f"✅ Using existing Qdrant Cloud collection: {COLLECTION_NAME}")

        # Get existing points count
        info = qdrant.get_collection(COLLECTION_NAME)
        print(f"   Existing vectors: {info.points_count}")

        return qdrant, info.points_count

    except Exception as e:
        print(f"❌ Qdrant Cloud connection failed: {e}")
        print("   Falling back to local storage...")
        return None, 0


def ingest_documents():
    """Main ingestion function."""
    print("🚀 Starting RAG ingestion with OpenAI + Qdrant Cloud\n")

    # Get all PDF files
    pdf_files = get_all_pdfs(DATA_FOLDER)
    print(f"📄 Found {len(pdf_files)} PDF files\n")

    # Initialize Qdrant
    qdrant, start_id = init_qdrant()

    use_local = qdrant is None
    local_docs = []

    # Process each PDF
    processed = 0
    for pdf_path in pdf_files:
        print(f"📖 Processing: {Path(pdf_path).name}")

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text.strip()) < 10:
            print("   ⏭️  Skipped (no text)\n")
            continue

        # Get metadata
        filename = Path(pdf_path).name
        folder = Path(pdf_path).parent.name

        # Chunk the text
        chunks = chunk_text(text)
        print(f"   📦 Extracted {len(chunks)} chunks")

        # Create embeddings and upload
        points = []
        for chunk_idx, chunk in enumerate(chunks):
            embedding = create_embedding(chunk)
            if embedding:
                point = PointStruct(
                    id=start_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": filename,
                        "folder": folder,
                        "chunk_index": chunk_idx
                    }
                )
                points.append(point)
                start_id += 1

        # Upload to Qdrant or save locally
        if points:
            if not use_local and qdrant:
                try:
                    qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    print(f"   ✅ Uploaded {len(points)} vectors to Qdrant Cloud")
                except Exception as e:
                    print(f"   ⚠️  Upload failed: {e}, saving locally...")
                    local_docs.extend(points)
            else:
                local_docs.extend(points)
                print(f"   💾 Saved {len(points)} vectors locally")

            processed += 1
        print()

    # Final summary
    print("="*50)
    print("✅ Ingestion complete!")
    print(f"   Documents processed: {processed}")
    print(f"   Total vectors created: {start_id}")

    if not use_local and qdrant:
        print(f"   📍 Storage: Qdrant Cloud")
    else:
        print(f"   📍 Storage: Local (vectors.pkl)")
        # Save to pickle as backup
        with open("vectors.pkl", "wb") as f:
            pickle.dump(local_docs, f)
        print(f"   💾 Saved to vectors.pkl")

    print("="*50)


if __name__ == "__main__":
    ingest_documents()
