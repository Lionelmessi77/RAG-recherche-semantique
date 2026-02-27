
import os
import pickle
from typing import List, Optional, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "agro_documents"

openai_client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class SearchResult:
    """Search result with text and score."""
    text: str
    score: float
    source: str
    folder: str


class RAGEngine:
    """RAG engine with Qdrant Cloud support."""

    def __init__(self):
        """Initialize the engine - tries Qdrant Cloud first, then local."""
        self.qdrant = self._init_qdrant()
        self.use_local = self.qdrant is None
        self.client = openai_client

        if not self.use_local:
            print("✅ Connected to Qdrant Cloud")
        else:
            print("💾 Using local storage (vectors.pkl)")

    def _init_qdrant(self):
        """Initialize Qdrant Cloud client."""
        try:
            url = QDRANT_URL.replace(":6333", "") if QDRANT_URL and QDRANT_URL.endswith(":6333") else QDRANT_URL

            qdrant = QdrantClient(
                url=url,
                api_key=QDRANT_API_KEY,
                prefer_grpc=False
            )

            # Test connection
            qdrant.get_collection(COLLECTION_NAME)
            return qdrant

        except Exception as e:
            print(f"Qdrant Cloud unavailable: {e}")
            return None

    def _load_local_docs(self):
        """Load documents from local pickle file."""
        try:
            with open("vectors.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("❌ No local data found. Run qdrant_ingest.py first!")
            return []

    def create_query_embedding(self, query: str) -> Optional[List[float]]:
        """Create embedding using OpenAI."""
        try:
            response = self.client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Search for relevant document fragments."""
        query_vector = self.create_query_embedding(query)
        if not query_vector:
            return []

        if not self.use_local and self.qdrant:
            # Search Qdrant Cloud - try different API methods
            try:
                # Try newer API first
                search_result = self.qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True
                )
                points = search_result.points
            except Exception:
                # Try older API
                search_result = self.qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True
                )
                points = search_result

            return [
                SearchResult(
                    text=p.payload.get("text", ""),
                    score=p.score,
                    source=p.payload.get("source", "Unknown"),
                    folder=p.payload.get("folder", "")
                )
                for p in points
            ]
        else:
            # Search local
            docs = self._load_local_docs()
            if not docs:
                return []

            # Calculate cosine similarity
            import numpy as np
            results = []
            for doc in docs:
                a = np.array(query_vector)
                b = np.array(doc["embedding"])
                score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                results.append(SearchResult(
                    text=doc["text"],
                    score=score,
                    source=doc["source"],
                    folder=doc["folder"]
                ))

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

    def generate_answer(self, query: str, results: List[SearchResult], conversation_history: List[Dict] = None) -> str:
        """Generate answer using OpenAI LLM."""
        if not results:
            return "Désolé, je n'ai pas trouvé d'informations pertinentes."

        context = "\n\n".join([
            f"Fragment {i+1} (source: {r.source}, score: {r.score:.3f}):\n{r.text}"
            for i, r in enumerate(results)
        ])

        messages = [
            {
                "role": "system",
                "content": """Tu es AgroAI, assistant expert pour STE AGRO MELANGE TECHNOLOGIE.

Réponds en français clair et professionnel.
Utilise des émojis pour organiser (📌 points clés, ✅ recommandations, 🔬 technique).
PAS de markdown (**bold**).
Structure en paragraphes clairs.
Base tes réponses sur les fragments fournis."""
            }
        ]

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-4:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({
            "role": "user",
            "content": f"Question: {query}\n\nFragments:\n{context}\n\nRéponds de manière structurée."
        })

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=800,
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur: {e}"

    def get_collection_info(self) -> Dict:
        """Get collection info."""
        if not self.use_local and self.qdrant:
            info = self.qdrant.get_collection(COLLECTION_NAME)
            return {
                "vectors_count": info.points_count,
                "storage": "Qdrant Cloud ☁️"
            }
        else:
            docs = self._load_local_docs()
            return {
                "vectors_count": len(docs),
                "storage": "Local 💾"
            }


if __name__ == "__main__":
    engine = RAGEngine()

    queries = [
        "Quels enzymes pour le pain?",
        "Comment doser l'amylase?"
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)

        results = engine.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Score: {r.score:.3f} | {r.source}")
            print(f"   {r.text[:100]}...")
