from services.ai_engine import AIEngine, client
from services.vector_engine import VectorEngine
from services.graph_engine import GraphEngine
from config import Config

ai = AIEngine()
vec_db = VectorEngine()
graph_db = GraphEngine()

def query_videorag(user_query):
    print(f"🔍 Processing Query: {user_query}")

    # 1. Query Reformulation
    plan = ai.decompose_query(user_query)
    print(f"   Plan: {plan}")

    candidate_chunks = {} 

    # 2. Path A: Visual Retrieval
    if plan.get('visual_query'):
        vis_query_emb = ai.get_text_embedding_clip(plan['visual_query'])
        vis_results = vec_db.search_visual(vis_query_emb, top_k=3)
        for res in vis_results:
            # FIX: Access using __id__ and metadata key
            chunk_id = res.get('__id__')
            if chunk_id:
                candidate_chunks[chunk_id] = res.get('metadata')

    # 3. Path B: Textual Retrieval
    text_query_emb = ai.get_text_embedding_openai(plan['keyword_query'])
    text_results = vec_db.search_text(text_query_emb, top_k=3)
    for res in text_results:
        # FIX: Access using __id__
        chunk_id = res.get('__id__')
        if chunk_id:
            candidate_chunks[chunk_id] = res.get('metadata')

    # 4. Path C: Graph Retrieval (Simplified for Prototype)
    # In a full production system, we would query metadata for these graph IDs.
    # Here we skip explicitly fetching if not found in vector search to avoid complexity.
    
    # 5. Context Integration
    unique_sources = [v for v in candidate_chunks.values() if v is not None]
    
    # Sort by timestamp
    try:
        unique_sources.sort(key=lambda x: x['start'])
    except:
        pass # Handle cases where start time might be missing safely
    
    context_str = ""
    for src in unique_sources:
        context_str += f"[Timestamp {src['start']}-{src['end']}s]: {src['text']}\n\n"

    if not context_str:
        context_str = "No relevant video segments found."

    # 6. LLM Response Generation
    system_prompt = """
    You are VideoRAG, an advanced video assistant. 
    Answer the user's question based strictly on the provided Video Context.
    Cite the timestamps (e.g., [10s-40s]) for every claim you make.
    If the context is empty, say you don't know.
    """

    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nUser Question: {user_query}"}
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": unique_sources
    }