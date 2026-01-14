import torch
import open_clip
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
from config import Config
import os

# Initialize OpenAI Client
client = OpenAI(api_key=Config.OPENAI_API_KEY)

class AIEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Loading AI Engine on {self.device}...")

        # 1. VISUAL EMBEDDING MODEL (Multi-Modal Context Encoding)
        # We use OpenCLIP (ViT-B-32) to map images to vector space, similar to ImageBind
        print("Loading Visual Encoder (OpenCLIP)...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # 2. VISION-LANGUAGE MODEL (Vision-Text Grounding)
        # Using MiniCPM-V 2.6 for detailed captioning
        self.vlm_model = None
        self.vlm_tokenizer = None
        try:
            print("Loading VLM (MiniCPM-V)...")
            self.vlm_model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)
            self.vlm_tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
            self.vlm_model.eval()
        except Exception as e:
            print(f"⚠️ VLM Load Failed (Check GPU Memory): {e}")

    def get_visual_embedding(self, image_path):
        """Generates a dense vector for a video frame (Visual Channel)."""
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        return image_features.cpu().numpy()[0].tolist()

    def get_text_embedding_clip(self, text):
        """Generates a vector for a text query in the Visual Space."""
        text_tokens = self.clip_tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        return text_features.cpu().numpy()[0].tolist()

    def get_text_embedding_openai(self, text):
        """Standard Text Embedding for the Textual Channel."""
        response = client.embeddings.create(input=text, model=Config.EMBEDDING_MODEL)
        return response.data[0].embedding

    def generate_detailed_caption(self, image_path):
        """Generates detailed scene description (Vision-Text Grounding)."""
        if not self.vlm_model: return "Visual description unavailable."
        
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': 'Describe this video scene in extreme detail, focusing on objects, actions, and text visible.'}]
        
        res = self.vlm_model.chat(image=image, msgs=msgs, tokenizer=self.vlm_tokenizer)
        return res

    def transcribe_audio(self, audio_path):
        """Extracts spoken content."""
        audio_file = open(audio_path, "rb")
        transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcription.text

    def extract_graph_entities(self, text):
        """LLM-based Entity-Relation Extraction for Graph Construction."""
        prompt = f"""
        Analyze the following video transcript/caption:
        "{text}"
        
        Extract a Knowledge Graph. Identify:
        1. Entities (Key concepts, people, objects, locations)
        2. Relationships (How they are connected)
        
        Output strictly valid JSON:
        {{
            "entities": [ {{"name": "EntityName", "type": "Person/Object/Concept"}} ],
            "relations": [ {{"source": "EntityName", "target": "EntityName", "relation": "verb_phrase"}} ]
        }}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)

    def decompose_query(self, user_query):
        """Decomposes user query into Visual, Textual, and Graph sub-queries."""
        prompt = f"""
        You are a query planner for a Video RAG system.
        User Query: "{user_query}"
        
        Generate 3 distinct search queries:
        1. visual_query: A short description of a visual scene to search for (e.g., "A red car driving in rain").
        2. keyword_query: Keywords for text search (e.g., "climate change carbon emissions").
        3. entities: A list of key entities to search in the Knowledge Graph.
        
        Output JSON: {{ "visual_query": "...", "keyword_query": "...", "entities": ["..."] }}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)