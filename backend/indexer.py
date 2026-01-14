import os
import uuid
from moviepy.editor import VideoFileClip
from services.ai_engine import AIEngine
from services.vector_engine import VectorEngine
from services.graph_engine import GraphEngine
from config import Config

# Initialize Engines
ai = AIEngine()
vec_db = VectorEngine()
graph_db = GraphEngine()

def process_video(video_path):
    print(f"🎬 Starting Advanced Indexing for: {video_path}")
    
    clip = VideoFileClip(video_path)
    duration = clip.duration
    chunk_len = 30  # Analyze every 30 seconds
    
    for start in range(0, int(duration), chunk_len):
        end = min(start + chunk_len, duration)
        mid_point = start + (end - start) / 2
        chunk_id = str(uuid.uuid4())
        
        print(f"   Processing Chunk {start}-{end}s...")

        # 1. Extract Modalities
        temp_audio = os.path.join(Config.STORAGE_FOLDER, f"{chunk_id}.mp3")
        temp_frame = os.path.join(Config.STORAGE_FOLDER, f"{chunk_id}.jpg")
        
        # Save Audio & Frame
        subclip = clip.subclip(start, end)
        subclip.audio.write_audiofile(temp_audio, logger=None)
        clip.save_frame(temp_frame, t=mid_point)

        # 2. Information Extraction
        transcript = ai.transcribe_audio(temp_audio)
        visual_caption = ai.generate_detailed_caption(temp_frame) # Vision-Text Grounding
        
        # Combined semantic context
        full_text_context = f"Time: {start}-{end}s. Transcript: {transcript}. Visual Scene: {visual_caption}"

        # 3. CHANNEL 1: Multi-Modal Context Encoding (Visual Vector)
        # We embed the raw image directly into vector space
        visual_embedding = ai.get_visual_embedding(temp_frame)
        vec_db.upsert_visual([{
            "id": chunk_id,
            "vector": visual_embedding,
            "metadata": {"text": full_text_context, "start": start, "end": end}
        }])

        # 4. CHANNEL 2: Textual Embedding (Semantic Vector)
        text_embedding = ai.get_text_embedding_openai(full_text_context)
        vec_db.upsert_text([{
            "id": chunk_id,
            "vector": text_embedding,
            "metadata": {"text": full_text_context, "start": start, "end": end}
        }])

        # 5. CHANNEL 3: Graph Construction (Structured Knowledge)
        # Extract entities from the combined text context
        kg_data = ai.extract_graph_entities(full_text_context)
        graph_db.add_knowledge(
            kg_data.get('entities', []), 
            kg_data.get('relations', []), 
            chunk_id
        )

        # Cleanup
        if os.path.exists(temp_audio): os.remove(temp_audio)
        if os.path.exists(temp_frame): os.remove(temp_frame)

    graph_db.save()
    print("✅ Indexing Complete. Knowledge Graph & Vectors Updated.")