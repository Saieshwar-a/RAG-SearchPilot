import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.documents = []
        self._process_all_pdf()
     
    def _process_all_pdf(self):

            for filename in os.listdir(self.directory):
                if filename.endswith(".pdf"):
                    try:
                        filepath = os.path.join(self.directory, filename)

                        loader = PyPDFLoader(filepath)
                        doc = loader.load()

                        print(f"Loaded {len(doc)} pages from {filename}")

                        for d in doc:
                            d.metadata["source_file"] = filename
                            d.metadata["file_type"] = "pdf"

                        self.documents.extend(doc)

                    except Exception as e:
                        print(f"Error loading {filename}: {e}")

    def split_documents(self, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            separators=["\n\n", "\n", " ", ""], 
            length_function=len)
        split_docs = text_splitter.split_documents(self.documents)

        print(f"Split {len(self.documents)} documents into {len(split_docs)} chunks.")
        if split_docs:
            print(f"Example chunk: {split_docs[0].page_content[:200]}...")
            print(f"Metadata of example chunk: {split_docs[0].metadata}")
        return split_docs
    


