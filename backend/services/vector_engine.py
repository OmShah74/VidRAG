import os
from nano_vectordb import NanoVectorDB
from config import Config

class VectorEngine:
    def __init__(self):
        # 1. Textual Index (Dimensions: 1536 for OpenAI)
        self.text_db = NanoVectorDB(embed_dim=1536, storage_file=os.path.join(Config.STORAGE_FOLDER, "text_index.json"))
        
        # 2. Visual Index (Dimensions: 512 for ViT-B-32)
        self.visual_db = NanoVectorDB(embed_dim=512, storage_file=os.path.join(Config.STORAGE_FOLDER, "visual_index.json"))

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