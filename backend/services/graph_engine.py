import networkx as nx
import json
import os
from config import Config

class GraphEngine:
    def __init__(self):
        self.graph_path = os.path.join(Config.STORAGE_FOLDER, "knowledge_graph.json")
        self.G = nx.Graph()
        self.load()

    def add_entity(self, entity, entity_type, source_chunk_id):
        if not self.G.has_node(entity):
            self.G.add_node(entity, type=entity_type, chunks=[])
        
        # Link entity to the text chunk where it was found
        if source_chunk_id not in self.G.nodes[entity]['chunks']:
            self.G.nodes[entity]['chunks'].append(source_chunk_id)

    def add_relation(self, source, target, relation):
        self.G.add_edge(source, target, relation=relation)

    def get_related_chunks(self, entities, hops=1):
        """Retrieve chunk IDs related to specific entities."""
        relevant_chunks = set()
        for entity in entities:
            if self.G.has_node(entity):
                # Add direct chunks
                relevant_chunks.update(self.G.nodes[entity]['chunks'])
                # Traverse neighbors
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