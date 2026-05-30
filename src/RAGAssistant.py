from src import DataProcessor, VectorStore, RAGRetriver, RAGEngine,EmbeddingManager
import numpy as np


class RAGAssistant:
    def __init__(self, directory: str, rebuild: bool = False):
        """
        Initialize the RAG Assistant.
        
        Args:
            directory: Path to the directory containing documents
            rebuild: If True, rebuild the knowledge base even if embeddings exist.
                     If False (default), reuse existing embeddings from vector store.
        """
        self.directory = directory

        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()

        # Only build knowledge base if vector store is empty or rebuild is requested
        if rebuild or self.vector_store.is_empty():
            print("Building knowledge base from documents...")
            self._build_knowledge_base()
        else:
            print(f"Using existing embeddings from vector store ({self.vector_store.collection.count()} documents)")

        self.retriever = RAGRetriver(
            self.vector_store,
            self.embedding_manager
        )

        self.engine = RAGEngine(self.retriever)

    def _build_knowledge_base(self):
        processor = DataProcessor(self.directory)
        documents = processor.split_documents()

        embeddings = [
            self.embedding_manager.generate_embedding(doc.page_content)
            for doc in documents
        ]

        self.vector_store.add_documents(
            documents,
            np.array(embeddings)
        )

    def ask(self, question: str):
        return self.engine.generate_response(question)

    def retrieve(self, query: str, top_k: int = 5):
        return self.retriever.retrieve(query, top_k=top_k)