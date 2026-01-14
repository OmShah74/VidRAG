import os
from nano_vectordb import NanoVectorDB
from config import Config

class VectorEngine:
    def __init__(self, collection_name="videorag_vectors"):
        self.db_path = os.path.join(Config.STORAGE_FOLDER, f"{collection_name}.json")
        # Dimension 1536 for OpenAI, 1024 for ImageBind (adjust based on usage)
        # For this prototype, we store text embeddings (1536) in one DB
        self.db = NanoVectorDB(embed_dim=1536, storage_file=self.db_path)

    def upsert(self, data_items):
        """
        data_items: list of dicts {'id': str, 'vector': list, 'metadata': dict}
        """
        self.db.upsert(data_items)
        self.db.save()

    def search(self, query_vector, top_k=5):
        return self.db.query(query_vector, top_k=top_k)