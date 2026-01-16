import json
from openai import OpenAI
from config import Config

class LLMEngine:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.LLM_MODEL  # e.g., "gpt-4o-mini"

    def extract_knowledge_graph(self, text_context):
        """
        Implements 'Graph-based Textual Knowledge Grounding' from the paper.
        Extracts structured Entities and Relationships from a video chunk's text.
        """
        system_prompt = """
        You are a Knowledge Graph Extractor for a video understanding system.
        Analyze the provided video transcript and visual description.
        
        Tasks:
        1. Identify key ENTITIES (People, Objects, Locations, Concepts, Events).
        2. Identify RELATIONSHIPS between these entities (e.g., "is wearing", "is located in", "discusses").
        
        Output strictly valid JSON with this schema:
        {
            "entities": [{"name": "Entity Name", "type": "Person/Object/Location/Concept"}],
            "relations": [{"source": "Entity A", "target": "Entity B", "relation": "verb_phrase"}]
        }
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {text_context}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1 # Low temp for consistent JSON
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM Graph Extraction Error: {e}")
            return {"entities": [], "relations": []}

    def decompose_query(self, user_query):
        """
        Implements 'Query Reformulation' for the Dual-Channel Retrieval.
        Breaks a user question into Visual, Textual, and Graph search intents.
        """
        system_prompt = """
        You are a Query Planner for a Multi-Modal RAG system.
        The user will ask a question about a video. You must generate 3 distinct search strategies:

        1. visual_query: A short, descriptive sentence describing a SCENE to look for visually (e.g., "A red car driving in the rain").
        2. keyword_query: Semantic keywords for text/transcript search (e.g., "climate change carbon emissions").
        3. entities: A list of specific Proper Nouns or key concepts to search in the Knowledge Graph.

        Output strictly valid JSON:
        {
            "visual_query": "string",
            "keyword_query": "string",
            "entities": ["string", "string"]
        }
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Query: {user_query}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM Query Decomposition Error: {e}")
            # Fallback
            return {
                "visual_query": user_query,
                "keyword_query": user_query,
                "entities": []
            }

    def check_relevance(self, user_query, chunk_text):
        """
        Implements 'LLM-based Video Clip Filtering' (Section 3.2 of the paper).
        Verifies if a retrieved video chunk actually contains the answer.
        Returns: Boolean
        """
        prompt = f"""
        Query: "{user_query}"
        Retrieved Video Context: "{chunk_text}"

        Does this context contain information relevant to answering the query? 
        Answer strictly with JSON: {{"is_relevant": true}} or {{"is_relevant": false}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("is_relevant", False)
        except:
            return True # Fallback to keeping it if check fails

    def generate_final_response(self, user_query, context_str):
        """
        Generates the final answer with timestamp citations.
        """
        system_prompt = """
        You are VideoRAG, an intelligent video assistant.
        1. Answer the User Query based STRICTLY on the provided Context.
        2. The Context is a list of video segments with timestamps.
        3. You must CITE the timestamps for every fact you state. 
           Format: "The speaker argues that AI is evolving [10s-15s]."
        4. If the answer is not in the context, state "I cannot find the answer in the video."
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nUser Query: {user_query}"}
                ],
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"