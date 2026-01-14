import os
import uuid
from moviepy.editor import VideoFileClip
from services.ai_engine import AIEngine
from services.vector_engine import VectorEngine
from services.graph_engine import GraphEngine
from config import Config

ai = AIEngine()
vec_db = VectorEngine()
graph_db = GraphEngine()

def process_video(video_path):
    print(f"Processing {video_path}...")
    
    # 1. Split Video into 30s chunks
    clip = VideoFileClip(video_path)
    duration = clip.duration
    chunk_len = 30
    
    chunks_metadata = []
    
    for start in range(0, int(duration), chunk_len):
        end = min(start + chunk_len, duration)
        subclip = clip.subclip(start, end)
        
        chunk_id = str(uuid.uuid4())
        
        # Save Temp Artifacts
        temp_audio = os.path.join(Config.STORAGE_FOLDER, f"{chunk_id}.mp3")
        temp_frame = os.path.join(Config.STORAGE_FOLDER, f"{chunk_id}.jpg")
        
        subclip.audio.write_audiofile(temp_audio, logger=None)
        subclip.save_frame(temp_frame, t=(end-start)/2) # Middle frame
        
        # 2. Transcribe & Caption
        transcript = ai.transcribe_audio(temp_audio)
        caption = ai.generate_caption(temp_frame)
        
        combined_text = f"Transcript: {transcript}\nVisual Context: {caption}"
        
        # 3. Vector Indexing
        embedding = ai.get_text_embedding(combined_text)
        vec_db.upsert([{
            "id": chunk_id,
            "vector": embedding,
            "metadata": {
                "video_path": video_path,
                "start": start,
                "end": end,
                "text": combined_text
            }
        }])
        
        # 4. Graph Construction
        kg_data = ai.extract_entities_gpt(combined_text)
        
        for ent in kg_data.get('entities', []):
            graph_db.add_entity(ent['name'], ent.get('type', 'concept'), chunk_id)
            
        for rel in kg_data.get('relations', []):
            graph_db.add_relation(rel['source'], rel['target'], rel['relation'])
        
        # Cleanup temp
        os.remove(temp_audio)
        os.remove(temp_frame)
        
    graph_db.save()
    print("Indexing Complete.")