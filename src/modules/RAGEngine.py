from ollama import Client
from datetime import datetime
import re
import requests
from sympy import Dict
from traitlets import Any

from src import RAGRetriver

class RAGEngine:
    def __init__(self, retriever: RAGRetriver, model_name: str = "mistral:latest", ollama_host: str = "http://localhost:11434"):
        self.retriever = retriever
        self.model_name = model_name
        self.client = Client(host=ollama_host)
    
    def generate_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate a response using Ollama LLM with retrieved context from RAG
        
        Args:
            query: The user's question
            top_k: Number of top documents to retrieve
            
        Returns:
            Dictionary containing the query, retrieved context, and generated response
        """
        print(f"Generating response for query: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        # Prepare context from retrieved documents
        context = "\n".join([doc['content'] for doc in retrieved_docs])
        
        # Construct the prompt
        prompt = f"""Based on the following context, answer the user's question.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        try:
            # Call Ollama Client
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            generated_text = response.get('response', '')
            
            return {
                'query': query,
                'context_docs': retrieved_docs,
                'context': context,
                'response': generated_text,
                'num_docs_retrieved': len(retrieved_docs),
                'timestamp': datetime.now().isoformat()
            }
                
        except Exception as e:
            print(f"Error generating response: {e}")
            raise