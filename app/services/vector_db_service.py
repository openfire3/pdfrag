import os
import uuid
from typing import List, Tuple, Dict, Any, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import Config
from app.logger_config import logger

class QuadrantService():
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT
            #url=Config.QDRANT_PATH,
            #api_key=Config.QDRANT_API_KEY
        )
    
    def get_collections(self) -> List[Dict[str, Any]]:
        try:
            collections = self.qdrant_client.get_collections()
            result = []
            for collection in collections.collections:
                metadata = self._get_collection_metadata(collection.name)
                if metadata:
                    result.append({
                        'collection_name': collection.name,
                        'filename': metadata.get('filename'),
                        'created_at': metadata.get('created_at'),
                        'pages_count': metadata.get('pages_count'),
                        'size_bytes': metadata.get('size_bytes')
                    })
            return result
        except Exception as e:
            logger.error(f"Error receiving collections: {str(e)}")
            raise
    
    def _get_collection_metadata(self, collection_name: str) -> Optional[Dict[str, Any]]:
        try:
            points = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=1
            )[0]
            if points:
                return points[0].payload.get('metadata', {})
            return None
        except Exception:
            return None

    def create_collection(self, collection_name):
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Created collection {collection_name}")
        
    def save_point(self,collection_name,embedding, page_num, chunk_text, chunk_part, total_chunks, metadata):
        try:
            point_id = uuid.uuid4().int & ((1 << 64) - 1)
            logger.info(f"Adding to vector DB - Collection: {collection_name}, Page: {page_num}, Chunk: {chunk_part}/{total_chunks}")
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'page_num': page_num,
                        'text': chunk_text,
                        'chunk_part': chunk_part,
                        'total_chunks': total_chunks,
                        'metadata': metadata
                    }
                )]
            )
            logger.info(f"Successfully added to Qdrant - Page: {page_num}, Chunk: {chunk_part}/{total_chunks}")
        except Exception as e:
            logger.error(f"Error saving to vector DB - Page {page_num}: {str(e)}")
            raise
        
    def search(self, collection_name, query_embedding, page_range_start, page_range_end, top_k):
        range_params = {}
        query_filter = None  # Initialize with None by default
        
        if page_range_start is not None:
            range_params["gte"] = page_range_start
        if page_range_end is not None:
            range_params["lte"] = page_range_end

        # Create filter only if we have range parameters
        if range_params:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="page_num",
                        range=models.Range(**range_params)
                    )
                ]
            )
        
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,  # Now it's always defined
            limit=top_k
        )
        logger.info(f"Found {len(search_results)} results")
        
        return search_results