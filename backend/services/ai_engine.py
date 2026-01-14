import torch
from transformers import AutoModel, AutoTokenizer
# import imagebind # Uncomment if installed
# from imagebind import data
from openai import OpenAI
from config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

class AIEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading AI Engine on {self.device}...")
        
        # 1. Initialize Whisper (Using OpenAI API for simplicity in prototype, or local)
        # Local: self.whisper = ... 
        
        # 2. Initialize MiniCPM-V (Visual Captioning)
        # Note: Requires ~8GB VRAM. If crashing, skip and use simple frame descriptions.
        try:
            self.vlm_model = AutoModel.from_pretrained(
                "openbmb/MiniCPM-V-2_6", 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            self.vlm_tokenizer = AutoTokenizer.from_pretrained(
                "openbmb/MiniCPM-V-2_6", 
                trust_remote_code=True
            )
            self.vlm_model.eval()
            print("MiniCPM-V Loaded.")
        except Exception as e:
            print(f"Skipping VLM Load (GPU issue?): {e}")
            self.vlm_model = None

    def transcribe_audio(self, audio_path):
        """Uses OpenAI Whisper API"""
        audio_file = open(audio_path, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        return transcription.text

    def get_text_embedding(self, text):
        response = client.embeddings.create(
            input=text,
            model=Config.EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def generate_caption(self, image_path):
        """Generate description for a frame"""
        if not self.vlm_model:
            return "Visual analysis unavailable."
        
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': 'Describe this video frame in detail.'}]
        
        res = self.vlm_model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.vlm_tokenizer
        )
        return res

    def extract_entities_gpt(self, text):
        """Extract entities for Knowledge Graph"""
        prompt = f"""
        Extract key entities (Concepts, People, Objects) and their relationships from this text.
        Return JSON format: {{ "entities": [{"name": "X", "type": "Y"}], "relations": [{"source": "X", "target": "Z", "relation": "rel"}] }}
        Text: {text}
        """
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "system", "content": "You are a Knowledge Graph extractor."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)