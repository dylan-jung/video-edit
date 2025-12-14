import json
import os
from typing import Dict, List

from langchain_core.tools import Tool

# from src.modules.chat.config import PROJECT_ID
from src.shared.infrastructure.ai.vector_db import VectorDB
from src.modules.indexing.infrastructure.embedding.google_embeddings import \
from src.modules.indexing.infrastructure.embedding.google_embeddings import \
    GoogleEmbeddingGenerator
from shared.infrastructure.ai.openai_embedding_service import \
    OpenAIEmbeddingService


class SementicVisionSearchTool:
    name = "sementic_vision_search"
    description = (
        "Perform semantic search across all video scenes in the project to find content matching a query. "
        "This tool searches through visual content, OCR text, objects, actions, emotions, and context "
        "across all indexed video scenes using AI-powered semantic understanding. "
        "Returns the most relevant scenes with similarity scores and detailed information. "
        "The information includes video_id, scene_id, start_time, end_time, similarity_score, background, objects, ocr_text, actions, emotions, context, highlights. "
        "This is useful for finding specific visual content, objects, activities, or themes "
        "without knowing exact video IDs or timestamps. "
        "This indexed data is not accurate, so you should not rely on the results too much."
        "Input: query (str) - the search query describing what you're looking for "
        "Output: search_results (JSON) - list of matching scenes with similarity scores and metadata"
    )

    def __init__(self, project_id: str):
        """Initialize the semantic vision search tool with embedding generator."""
        self.project_id = project_id
        # Try to use OpenAI embeddings first, fallback to Google embeddings
        try:
            if "OPENAI_API_KEY" in os.environ:
                self.embedding_generator = OpenAIEmbeddingService()
                self.dimension = 3072  # OpenAI text-embedding-3-large dimension
                print("✅ Using OpenAI embeddings for semantic search")
            else:
                self.embedding_generator = GoogleEmbeddingGenerator()
                self.dimension = 768  # Google embedding dimension
                print("✅ Using Google embeddings for semantic search")
        except Exception as e:
            print(f"⚠️ Failed to initialize embedding generator: {e}")
            # Use Google as fallback
            self.embedding_generator = GoogleEmbeddingGenerator()
            self.dimension = 768

    def _get_vector_db_path(self, project_id: str) -> str:
        """Get the vector database file path for the project."""
        return f"projects/{project_id}/vision_vector_db.faiss"

    def _load_vector_db(self, project_id: str) -> VectorDB:
        """Load the vector database for the project."""
        vector_db_path = self._get_vector_db_path(project_id)
        
        try:
            vector_db = VectorDB.load(vector_db_path, dimension=self.dimension)
            print(f"✅ Vector database loaded: {vector_db.get_stats()}")
            return vector_db
        except Exception as e:
            print(f"❌ Failed to load vector database from {vector_db_path}: {e}")
            # Return empty database if file doesn't exist
            return VectorDB(dimension=self.dimension)

    def _generate_query_embedding(self, query: str):
        """Generate embedding for the search query."""
        try:
            embedding = self.embedding_generator.generate_text_embedding(query)
            return embedding
        except Exception as e:
            print(f"❌ Failed to generate query embedding: {e}")
            raise e

    def _format_search_results(self, results: List[Dict]) -> List[Dict]:
        """Format search results for better readability."""
        formatted_results = []
        
        for result in results:
            formatted_result = {
                'video_id': result.get('video_id'),
                'scene_id': result.get('scene_id'),
                'start_time': result.get('start_time'),
                'end_time': result.get('end_time'),
                'similarity_score': round(result.get('similarity_score', 0), 4),
                'background': result.get('background'),
                'objects': result.get('objects', []),
                'ocr_text': result.get('ocr_text', []),
                'actions': result.get('actions', []),
                'emotions': result.get('emotions', []),
                'context': result.get('context'),
                'highlights': result.get('highlight', [])
            }
            
            # Remove empty fields for cleaner output
            formatted_result = {k: v for k, v in formatted_result.items() 
                              if v is not None and v != [] and v != ''}
            
            formatted_results.append(formatted_result)
        
        return formatted_results

    def call(self, query: str) -> str:
        """
        Perform semantic vision search across video scenes.
        
        Args:
            query: The search query string
            
        Returns:
            JSON string containing search results
        """
        if not query or not query.strip():
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)
        
        query = query.strip()
        print(f"🔍 Performing semantic search for: '{query}'")
        
        try:
            # Load vector database
            vector_db = self._load_vector_db(self.project_id)
            
            # Check if database has any data
            stats = vector_db.get_stats()
            if stats['total_vectors'] == 0:
                return json.dumps({
                    "query": query,
                    "results": [],
                    "message": "No video scenes are indexed yet. Please analyze some videos first.",
                    "stats": stats
                }, ensure_ascii=False)
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            
            # Perform semantic search
            raw_results = vector_db.search_similar_scenes(query_embedding, k=10)
            
            # Format results
            formatted_results = self._format_search_results(raw_results)
            
            # Prepare response
            response = {
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "database_stats": stats
            }
            
            print(f"✅ Found {len(formatted_results)} matching scenes")
            
            return json.dumps(response, ensure_ascii=False)
            
        except Exception as e:
            error_msg = f"Semantic search failed: {str(e)}"
            print(f"❌ {error_msg}")
            return json.dumps({
                "query": query,
                "error": error_msg,
                "results": []
            }, ensure_ascii=False)
        
    def as_tool(self) -> Tool:
        """Convert the tool to a LangChain-compatible tool format."""
        def tool_func(*args, **kwargs) -> str:
            print(f"🔍 Tool called with args: {args}, kwargs: {kwargs}")
            return self.call(*args, **kwargs)
        
        return Tool(
            name=self.name,
            description=self.description,
            func=tool_func
        )