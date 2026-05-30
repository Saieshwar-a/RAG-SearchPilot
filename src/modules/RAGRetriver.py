import os
from typing import List, Dict, Any
from .VectorStore import VectorStore
from .EmbeddingManager import EmbeddingManager


class RAGRetriver:

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:

        print(f"Retrieving documents for query: {query}")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")

        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        # search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            #Process results

            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (document, meta, distance, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                    # Convert distance to similarity score cosine similarity
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': meta,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                print(f"Retrived {len(retrieved_docs)} documents after filtering.")

            else:
                print("No documents retrieved from vector store.")

            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise