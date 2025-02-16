from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class Embeddings():
    """Embedding functionality."""
    def __init__(self):
        self.modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        self.model_kwargs = {'device':'cpu'}
        self.encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(
                            model_name=self.modelPath, 
                            model_kwargs=self.model_kwargs, 
                            encode_kwargs=self.encode_kwargs 
                        )
        
        
    def embed_documents(self, chunks):
        """embed and store embeddings."""
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(folder_path = "../model", index_name= "faiss_vectorstore")

    
    def embed_query(self, query: str):
        """embed the incoming query."""
        embedding_vector = self.embeddings.embed_query(query)
        return embedding_vector
    