from FlagEmbedding import BGEM3FlagModel
import numpy as np

class CustomBGEM3FlagModel:
    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.model_name = model_name
        self.use_fp16 = use_fp16

    def embed(self, texts):
        # Assuming 'encode' is a method of BGEM3FlagModel that returns a dictionary with 'dense_vecs'
        return self.model.encode(texts)['dense_vecs']

    def embed_documents(self, documents):
        texts = [doc['text'] if isinstance(doc, dict) else doc for doc in documents]
        embeddings = self.embed(texts)
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    
    def embed_query(self, query):
        return self.embed([query])[0].tolist()