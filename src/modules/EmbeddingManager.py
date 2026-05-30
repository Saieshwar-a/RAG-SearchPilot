import numpy as np
import sentence_transformers as SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Lazy load the embedding model on first use."""
        if self.model is None:
            try:
                self.model = SentenceTransformer.SentenceTransformer(self.model_name)
                print(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                raise

    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            # Load model only when we need to generate embeddings
            self._load_model()
            
            print(f"Generating embedding for text: {text[:50]}...")
            embedding = self.model.encode(text, show_progress_bar=True)
            print(f"Generated embedding with shape {embedding.shape}")
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise