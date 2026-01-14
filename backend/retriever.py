from services.ai_engine import AIEngine, client
from services.vector_engine import VectorEngine
from services.graph_engine import GraphEngine
from config import Config

ai = AIEngine()
vec_db = VectorEngine()
graph_db = GraphEngine()

def query_videorag(user_query):
    print(f"🔍 Processing Query: {user_query}")

    # 1. Query Reformulation & Decomposition
    # Break the query into Visual intent, Textual intent, and Graph entities
    plan = ai.decompose_query(user_query)
    print(f"   Plan: {plan}")

    candidate_chunks = {} # Map chunk_id -> metadata

    # 2. Path A: Visual Retrieval (Cross-Modal)
    # Search for visual scenes described in the query
    if plan.get('visual_query'):
        vis_query_emb = ai.get_text_embedding_clip(plan['visual_query'])
        vis_results = vec_db.search_visual(vis_query_emb, top_k=3)
        for res in vis_results:
            candidate_chunks[res['id']] = res['metadata']

    # 3. Path B: Textual Retrieval (Semantic)
    # Standard RAG search
    text_query_emb = ai.get_text_embedding_openai(plan['keyword_query'])
    text_results = vec_db.search_text(text_query_emb, top_k=3)
    for res in text_results:
        candidate_chunks[res['id']] = res['metadata']

    # 4. Path C: Graph Retrieval (Structured)
    # Find chunks related to the entities in the query
    if plan.get('entities'):
        graph_chunk_ids = graph_db.retrieve_context(plan['entities'], hops=1)
        # Note: In a real DB, we'd fetch metadata for these IDs. 
        # For prototype, we verify if they exist in our loaded text results or skip specific metadata fetch 
        # (Assuming overlap with Vector Search, or we would need a Key-Value store lookup here).
        # For MVP, we skip strictly fetching metadata for ONLY graph hits if not in vector hits to save complexity.

    # 5. Context Integration
    unique_sources = list(candidate_chunks.values())
    
    # Sort by timestamp to give linear context
    unique_sources.sort(key=lambda x: x['start'])
    
    context_str = ""
    for src in unique_sources:
        context_str += f"[Timestamp {src['start']}-{src['end']}s]: {src['text']}\n\n"

    # 6. LLM Response Generation
    system_prompt = """
    You are VideoRAG, an advanced video assistant. 
    Answer the user's question based strictly on the provided Video Context.
    Cite the timestamps (e.g., [10s-40s]) for every claim you make.
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