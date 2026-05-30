from src import RAGAssistant


rag = RAGAssistant("./Data/PdfData")

response = rag.ask(
    "What is the Attention Mechanism in Neural Networks?"
)

print(response["response"])