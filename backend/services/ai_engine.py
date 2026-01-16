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

        # 1. VISUAL EMBEDDING MODEL (OpenCLIP)
        print("Loading Visual Encoder (OpenCLIP)...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # 2. VISION-LANGUAGE MODEL (Safe Load)
        self.vlm_model = None
        self.vlm_tokenizer = None
        
        # Only try loading VLM if on GPU, otherwise skip to save RAM/Time
        if self.device == "cuda":
            try:
                print("Loading VLM (MiniCPM-V)...")
                # Note: This still requires HF Login, but the try/except blocks the crash
                self.vlm_model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)
                self.vlm_tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
                self.vlm_model.eval()
            except Exception as e:
                print(f"⚠️ VLM Load Skipped: {e}")
                print("   -> System will run in 'Audio + Vector' mode (faster/lighter).")
        else:
            print("⚠️ CPU detected: Skipping heavy VLM to prevent crash. Using basic indexing.")

    def get_visual_embedding(self, image_path):
        """Generates a dense vector for a video frame."""
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
        response = client.embeddings.create(input=text, model=Config.EMBEDDING_MODEL)
        return response.data[0].embedding

    def generate_detailed_caption(self, image_path):
        """Safe captioning: returns placeholder if VLM is off."""
        if not self.vlm_model: 
            return "Visual description unavailable (Running on CPU/No VLM)."
        
        try:
            image = Image.open(image_path).convert('RGB')
            msgs = [{'role': 'user', 'content': 'Describe this video scene in detail.'}]
            res = self.vlm_model.chat(image=image, msgs=msgs, tokenizer=self.vlm_tokenizer)
            return res
        except:
            return "Error analyzing frame."

    def transcribe_audio(self, audio_path):
        audio_file = open(audio_path, "rb")
        transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcription.text

    def extract_graph_entities(self, text):
        # ... (Keep existing code) ...
        prompt = f"""
        Extract Entities and Relations from: "{text}"
        Output strictly valid JSON:
        {{ "entities": [ {{"name": "X", "type": "Y"}} ], "relations": [ {{"source": "X", "target": "Y", "relation": "Z"}} ] }}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)

    def decompose_query(self, user_query):
        # ... (Keep existing code) ...
        prompt = f"""
        Decompose query: "{user_query}"
        Output JSON: {{ "visual_query": "...", "keyword_query": "...", "entities": ["..."] }}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)