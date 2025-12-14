import json
import os
from typing import Annotated, Dict, List
from langgraph.prebuilt import InjectedState
from langchain_core.tools import StructuredTool, Tool

# from src.modules.chat.config import PROJECT_ID
from src.shared.infrastructure.ai.vector_db_registry import VectorDBRegistry
from src.shared.infrastructure.ai.openai_embedding_service import \
    OpenAIEmbeddingService


class SementicSpeechSearchTool:
    name = "sementic_speech_search"
    description = (
        "Perform semantic search across all speech/audio content in the project to find content matching a query. "
        "This tool searches through speech analysis data including text, summaries, keywords, topics, emotions, sentiment, and context "
        "across all indexed audio segments using AI-powered semantic understanding. "
        "Returns the most relevant speech chunks with similarity scores and detailed information. "
        "The information includes video_id, start_time, end_time, similarity_score, summary, keywords, topics, sentiment, importance, context, text. "
        "This is useful for finding specific spoken content, topics, emotions, or themes "
        "without knowing exact video IDs or timestamps. "
        "This indexed data is analyzed from speech, so accuracy may vary depending on speech quality."
        "Input: query (str) - the search query describing what you're looking for "
        "Output: search_results (JSON) - list of matching speech chunks with similarity scores and metadata"
    )

    def __init__(self):
        """Initialize the semantic speech search tool with embedding generator."""
        # Use OpenAI embeddings directly
        try:
            self.embedding_generator = OpenAIEmbeddingService()
            self.dimension = 3072  # OpenAI text-embedding-3-large dimension
            print("✅ Using OpenAI embeddings for speech search")
        except Exception as e:
            print(f"⚠️ Failed to initialize embedding generator: {e}")
            raise e


    def _load_vector_db(self, project_id: str):
        """Load the speech vector database from registry."""
        registry = VectorDBRegistry.get_instance()
        return registry.get_speech_db(project_id)

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
                'start_time': result.get('start_time'),
                'end_time': result.get('end_time'),
                'similarity_score': round(result.get('similarity_score', 0), 4),
                'summary': result.get('summary'),
                'keywords': result.get('keywords', []),
                'topics': result.get('topics', []),
                'sentiment': result.get('sentiment'),
                'importance': result.get('importance'),
                'context': result.get('context'),
                'text': result.get('text', [])
            }
            
            # Remove empty fields for cleaner output
            formatted_result = {k: v for k, v in formatted_result.items() 
                              if v is not None and v != [] and v != ''}
            
            formatted_results.append(formatted_result)
        
        return formatted_results

    def call(self, query: str, project_id: str) -> str:
        """
        Perform semantic speech search across audio content.
        
        Args:
            query: The search query string
            project_id: The project ID injected from state
            
        Returns:
            JSON string containing search results
        """
        if not query or not query.strip():
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)
        
        query = query.strip()
        print(f"🔍 Performing semantic speech search for: '{query}'")
        
        try:
            # Load speech vector database
            vector_db = self._load_vector_db(project_id)
            
            # Check if database has any data
            stats = vector_db.get_stats()
            if stats['total_vectors'] == 0:
                return json.dumps({
                    "query": query,
                    "results": [],
                    "message": "No speech content is indexed yet. Please analyze some audio/videos first.",
                    "stats": stats
                }, ensure_ascii=False)
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            
            # Perform semantic search
            raw_results = vector_db.search_similar_speech(query_embedding, k=10)
            
            # Format results
            formatted_results = self._format_search_results(raw_results)
            
            # Prepare response
            response = {
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "database_stats": stats
            }
            
            print(f"✅ Found {len(formatted_results)} matching speech chunks")
            
            return json.dumps(response, ensure_ascii=False)
            
        except Exception as e:
            error_msg = f"Speech semantic search failed: {str(e)}"
            print(f"❌ {error_msg}")
            return json.dumps({
                "query": query,
                "error": error_msg,
                "results": []
            }, ensure_ascii=False)
        
    def as_tool(self) -> StructuredTool:
        """Convert the tool to a LangChain-compatible tool format."""
        def tool_func(query: str, project_id: Annotated[str, InjectedState("project_id")]) -> str:
            print(f"🔍 Speech Search Tool called with query: {query}")
            return self.call(query, project_id)
        
        return StructuredTool.from_function(
            func=tool_func,
            name=self.name,
            description=self.description,
        )