import os
import uuid
from moviepy.video.io.VideoFileClip import VideoFileClip
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
    
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            chunk_len = 30
            
            for start in range(0, int(duration), chunk_len):
                end = min(start + chunk_len, duration)
                mid_point = start + (end - start) / 2
                chunk_id = str(uuid.uuid4())
                
                print(f"   Processing Chunk {start}-{end}s...")

                # Define Temp Paths
                temp_audio = os.path.join(Config.STORAGE_FOLDER, f"{chunk_id}.mp3")
                temp_frame = os.path.join(Config.STORAGE_FOLDER, f"{chunk_id}.jpg")
                
                subclip = clip.subclip(start, end)
                
                # Audio Handling
                transcript = "[No Speech Detected]"
                if subclip.audio is not None:
                    try:
                        subclip.audio.write_audiofile(temp_audio, logger=None)
                        transcript = ai.transcribe_audio(temp_audio)
                    except Exception as e:
                        print(f"      ⚠️ Audio extraction failed: {e}")
                    finally:
                        if os.path.exists(temp_audio): os.remove(temp_audio)
                else:
                    print("      ℹ️ No audio track found.")

                # Frame Handling
                clip.save_frame(temp_frame, t=mid_point)
                visual_caption = ai.generate_detailed_caption(temp_frame) 
                
                full_text_context = f"Time: {start}-{end}s. Transcript: {transcript}. Visual Scene: {visual_caption}"

                # --- FIX: Use Internal Keys (__vector__, __id__) ---

                # 3. CHANNEL 1: Visual Vector
                try:
                    visual_embedding = ai.get_visual_embedding(temp_frame)
                    vec_db.upsert_visual([{
                        "__id__": chunk_id,
                        "__vector__": visual_embedding,
                        "metadata": {"text": full_text_context, "start": start, "end": end}
                    }])
                except Exception as e:
                    print(f"      ⚠️ Visual embedding failed: {e}")

                # 4. CHANNEL 2: Textual Vector
                try:
                    text_embedding = ai.get_text_embedding_openai(full_text_context)
                    vec_db.upsert_text([{
                        "__id__": chunk_id,
                        "__vector__": text_embedding,
                        "metadata": {"text": full_text_context, "start": start, "end": end}
                    }])
                except Exception as e:
                    print(f"      ⚠️ Text embedding failed: {e}")

                # 5. CHANNEL 3: Graph Construction
                try:
                    kg_data = ai.extract_graph_entities(full_text_context)
                    graph_db.add_knowledge(
                        kg_data.get('entities', []), 
                        kg_data.get('relations', []), 
                        chunk_id
                    )
                except Exception as e:
                    print(f"      ⚠️ Graph extraction failed: {e}")

                if os.path.exists(temp_frame): os.remove(temp_frame)

            graph_db.save()
            print("✅ Indexing Complete.")

    except Exception as e:
        print(f"❌ Critical Error processing video: {e}")