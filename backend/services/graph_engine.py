import networkx as nx
import json
import os
from config import Config

class GraphEngine:
    def __init__(self):
        self.graph_path = os.path.join(Config.STORAGE_FOLDER, "knowledge_graph.json")
        self.G = nx.Graph()
        self.load()

    def add_knowledge(self, entities, relations, source_chunk_id):
        """
        Constructs the graph iteratively.
        Nodes = Entities.
        Edges = Relations.
        Metadata = Link back to the video chunk (source_chunk_id).
        """
        for ent in entities:
            name = ent['name'].lower()
            if not self.G.has_node(name):
                self.G.add_node(name, type=ent['type'], chunks=[])
            
            # Link entity to video chunk
            if source_chunk_id not in self.G.nodes[name]['chunks']:
                self.G.nodes[name]['chunks'].append(source_chunk_id)

        for rel in relations:
            src = rel['source'].lower()
            tgt = rel['target'].lower()
            if self.G.has_node(src) and self.G.has_node(tgt):
                self.G.add_edge(src, tgt, relation=rel['relation'])

    def retrieve_context(self, entities, hops=1):
        """
        Traverses the graph to find related video chunks.
        Input: List of entities from the user query.
        Output: Set of chunk IDs that contain these entities or their neighbors.
        """
        relevant_chunks = set()
        for entity in entities:
            entity = entity.lower()
            if self.G.has_node(entity):
                # 0-hop: The entity itself
                relevant_chunks.update(self.G.nodes[entity]['chunks'])
                
                # 1-hop: Neighbors
                if hops > 0:
                    for neighbor in self.G.neighbors(entity):
                        relevant_chunks.update(self.G.nodes[neighbor]['chunks'])
        
        return list(relevant_chunks)

    def save(self):
        data = nx.node_link_data(self.G)
        with open(self.graph_path, 'w') as f:
            json.dump(data, f)

    def load(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
            self.G = nx.node_link_graph(data)