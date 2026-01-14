from services.ai_engine import AIEngine, client
from services.vector_engine import VectorEngine
from services.graph_engine import GraphEngine
from config import Config

ai = AIEngine()
vec_db = VectorEngine()
graph_db = GraphEngine()

def query_videorag(user_query):
    # 1. Query Reformulation (Simplified)
    # Get keywords for Graph Search
    keywords = ai.extract_entities_gpt(user_query).get('entities', [])
    keyword_names = [k['name'] for k in keywords]
    
    # 2. Retrieval
    
    # A. Vector Search (Semantic)
    query_emb = ai.get_text_embedding(user_query)
    vector_results = vec_db.search(query_emb, top_k=5)
    
    # B. Graph Search (Contextual)
    graph_chunk_ids = graph_db.get_related_chunks(keyword_names)
    
    # 3. Fusion & Context Building
    # In a real app, fetch chunk metadata for graph_chunk_ids from a KV store. 
    # Here we rely mostly on Vector Results for the MVP text context.
    
    context_text = ""
    sources = []
    
    for res in vector_results:
        meta = res['metadata']
        context_text += f"[Time {meta['start']}-{meta['end']}]: {meta['text']}\n\n"
        sources.append(meta)
        
    # 4. Generation
    system_prompt = "You are a video assistant. Use the provided video context to answer the user query. Cite timestamps."
    
    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
        ]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }