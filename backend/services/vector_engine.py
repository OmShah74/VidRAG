import os
from nano_vectordb import NanoVectorDB
from config import Config

class VectorEngine:
    def __init__(self):
        self.text_db_path = os.path.join(Config.STORAGE_FOLDER, "text_index.json")
        self.visual_db_path = os.path.join(Config.STORAGE_FOLDER, "visual_index.json")
        
        # FIX: Pass the dimension as the first positional argument
        # 1536 for OpenAI Text Embeddings
        self.text_db = NanoVectorDB(1536, storage_file=self.text_db_path)
        
        # 512 for OpenCLIP Visual Embeddings
        self.visual_db = NanoVectorDB(512, storage_file=self.visual_db_path)

    def upsert_text(self, data):
        """Upsert into Textual Index"""
        self.text_db.upsert(data)
        self.text_db.save()

    def upsert_visual(self, data):
        """Upsert into Visual Context Index"""
        self.visual_db.upsert(data)
        self.visual_db.save()

    def search_text(self, vector, top_k=5):
        return self.text_db.query(vector, top_k=top_k)

    def search_visual(self, vector, top_k=5):
        return self.visual_db.query(vector, top_k=top_k)