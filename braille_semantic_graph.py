"""
Braille Semantic Graph: Core Implementation

A living, evolving graph that stores and reasons about braille patterns.
The graph's topology IS the system's knowledge.
"""

import time
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import math


@dataclass
class BrailleNode:
    """A node in the semantic graph representing a braille pattern."""
    pattern: str  # Braille sequence (e.g., "⠓⠑⠇⠇⠕")
    frequency: int = 1  # How many times observed
    timestamp: float = field(default_factory=time.time)  # Last access time
    semantic_density: float = 0.0  # Local connectivity measure
    
    def __hash__(self):
        return hash(self.pattern)
    
    def __eq__(self, other):
        if isinstance(other, BrailleNode):
            return self.pattern == other.pattern
        return False
    
    def to_dict(self):
        return asdict(self)


@dataclass
class BrailleEdge:
    """An edge in the semantic graph representing a relationship between patterns."""
    source_pattern: str
    target_pattern: str
    weight: float = 0.5  # Strength of relationship (0.0 - 1.0)
    relationship_type: str = "similarity"  # sequential, similarity, transformational
    reinforcement_count: int = 0  # How many times reinforced
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self):
        return asdict(self)


class BrailleSemanticGraph:
    """
    A dynamic graph that learns and evolves through interaction.
    
    Core principle: All operations are braille-native.
    No decoding to float-space during reasoning.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, decay_factor: float = 0.95):
        self.nodes: Dict[str, BrailleNode] = {}
        self.edges: Dict[Tuple[str, str], BrailleEdge] = {}
        self.similarity_threshold = similarity_threshold
        self.decay_factor = decay_factor
        self.learning_rate = 0.1
        self.min_edge_weight = 0.05
        self.interaction_count = 0
    
    # ========================================================================
    # Braille-Native Metrics
    # ========================================================================
    
    @staticmethod
    def hamming_similarity(pattern1: str, pattern2: str) -> float:
        """
        Compute Hamming similarity between two braille patterns.
        
        Returns a score from 0.0 (completely different) to 1.0 (identical).
        """
        if pattern1 == pattern2:
            return 1.0
        
        # Pad to same length
        max_len = max(len(pattern1), len(pattern2))
        p1 = pattern1.ljust(max_len, chr(0x2800))
        p2 = pattern2.ljust(max_len, chr(0x2800))
        
        # Count matching bits
        matches = 0
        for c1, c2 in zip(p1, p2):
            code1 = ord(c1) - 0x2800 if 0x2800 <= ord(c1) <= 0x28FF else 0
            code2 = ord(c2) - 0x2800 if 0x2800 <= ord(c2) <= 0x28FF else 0
            
            # XOR to find differing bits
            xor = code1 ^ code2
            # Count matching bits (8 - popcount of XOR)
            matches += 8 - bin(xor).count('1')
        
        total_bits = max_len * 8
        return matches / total_bits if total_bits > 0 else 0.0
    
    @staticmethod
    def edit_distance(pattern1: str, pattern2: str) -> int:
        """Compute Levenshtein distance between two patterns."""
        m, n = len(pattern1), len(pattern2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pattern1[i-1] == pattern2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def composite_similarity(pattern1: str, pattern2: str) -> float:
        """
        Compute composite similarity using Hamming and edit distance.
        
        Combines:
        - Hamming similarity (0.6 weight): bit-level similarity
        - Normalized edit distance (0.4 weight): sequence-level similarity
        """
        hamming_sim = BrailleSemanticGraph.hamming_similarity(pattern1, pattern2)
        
        # Edit distance similarity
        max_len = max(len(pattern1), len(pattern2))
        edit_dist = BrailleSemanticGraph.edit_distance(pattern1, pattern2)
        edit_sim = 1.0 - (edit_dist / max_len) if max_len > 0 else 1.0
        
        # Composite
        return 0.6 * hamming_sim + 0.4 * edit_sim
    
    def compute_semantic_density(self, pattern: str) -> float:
        """
        Compute semantic density for a pattern.
        
        Measures how well-connected a pattern is in the graph.
        High density = central, meaningful concept.
        Low density = peripheral, isolated pattern.
        """
        if pattern not in self.nodes:
            return 0.0
        
        # Count outgoing and incoming edges
        outgoing = sum(1 for (src, tgt) in self.edges.keys() if src == pattern)
        incoming = sum(1 for (src, tgt) in self.edges.keys() if tgt == pattern)
        
        # Sum of edge weights
        outgoing_weight = sum(self.edges[(src, tgt)].weight 
                             for (src, tgt) in self.edges.keys() if src == pattern)
        incoming_weight = sum(self.edges[(src, tgt)].weight 
                             for (src, tgt) in self.edges.keys() if tgt == pattern)
        
        # Degree centrality
        total_nodes = len(self.nodes)
        if total_nodes <= 1:
            return 0.0
        
        degree_centrality = (outgoing + incoming) / (2 * (total_nodes - 1))
        
        # Weighted centrality
        max_weight = 2.0  # Max possible weight sum
        weighted_centrality = (outgoing_weight + incoming_weight) / max_weight if max_weight > 0 else 0.0
        
        # Composite
        density = 0.5 * degree_centrality + 0.5 * weighted_centrality
        return min(density, 1.0)
    
    # ========================================================================
    # Graph Operations
    # ========================================================================
    
    def add_node(self, pattern: str) -> BrailleNode:
        """Add or retrieve a node for a braille pattern."""
        if pattern not in self.nodes:
            self.nodes[pattern] = BrailleNode(pattern=pattern)
        else:
            # Update timestamp and frequency
            self.nodes[pattern].frequency += 1
            self.nodes[pattern].timestamp = time.time()
        
        return self.nodes[pattern]
    
    def add_edge(self, source: str, target: str, relationship_type: str = "similarity", 
                 weight: Optional[float] = None) -> BrailleEdge:
        """Add or update an edge between two patterns."""
        # Ensure nodes exist
        self.add_node(source)
        self.add_node(target)
        
        edge_key = (source, target)
        
        if edge_key not in self.edges:
            # New edge
            if weight is None:
                weight = self.composite_similarity(source, target)
            self.edges[edge_key] = BrailleEdge(
                source_pattern=source,
                target_pattern=target,
                weight=weight,
                relationship_type=relationship_type
            )
        else:
            # Reinforce existing edge
            edge = self.edges[edge_key]
            edge.reinforcement_count += 1
            edge.timestamp = time.time()
            
            # Update weight using learning rule
            if weight is not None:
                edge.weight = edge.weight + (self.learning_rate * (weight - edge.weight))
            else:
                # Reinforce based on similarity
                sim = self.composite_similarity(source, target)
                edge.weight = edge.weight + (self.learning_rate * (sim - edge.weight))
        
        return self.edges[edge_key]
    
    def ingest_sequence(self, sequence: str, n_gram_size: int = 2):
        """
        Ingest a braille sequence into the graph.
        
        Breaks the sequence into n-grams and updates the graph.
        """
        self.interaction_count += 1
        
        # Create n-grams
        n_grams = []
        for i in range(len(sequence) - n_gram_size + 1):
            n_gram = sequence[i:i+n_gram_size]
            n_grams.append(n_gram)
        
        if not n_grams:
            return
        
        # Add nodes for each n-gram
        for n_gram in n_grams:
            self.add_node(n_gram)
        
        # Add sequential edges
        for i in range(len(n_grams) - 1):
            self.add_edge(n_grams[i], n_grams[i+1], relationship_type="sequential")
        
        # Add similarity edges to existing patterns
        for n_gram in n_grams:
            # Find similar patterns in the graph
            for existing_pattern in list(self.nodes.keys()):
                if existing_pattern != n_gram:
                    sim = self.composite_similarity(n_gram, existing_pattern)
                    if sim > self.similarity_threshold:
                        self.add_edge(n_gram, existing_pattern, relationship_type="similarity", weight=sim)
        
        # Update semantic density for all nodes
        for pattern in self.nodes.keys():
            self.nodes[pattern].semantic_density = self.compute_semantic_density(pattern)
    
    def prune_edges(self, age_threshold: float = 3600.0):
        """
        Prune old, weak edges to prevent graph from becoming too dense.
        
        age_threshold: Edges older than this (in seconds) have their weight decayed.
        """
        current_time = time.time()
        edges_to_remove = []
        
        for edge_key, edge in self.edges.items():
            age = current_time - edge.timestamp
            
            # Decay weight based on age
            decay = self.decay_factor ** (age / age_threshold)
            edge.weight *= decay
            
            # Mark for removal if weight is too low
            if edge.weight < self.min_edge_weight:
                edges_to_remove.append(edge_key)
        
        # Remove marked edges
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
    
    # ========================================================================
    # Reasoning Interface
    # ========================================================================
    
    def find_similar_patterns(self, pattern: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find the k most similar patterns in the graph."""
        similarities = []
        
        for existing_pattern in self.nodes.keys():
            if existing_pattern != pattern:
                sim = self.composite_similarity(pattern, existing_pattern)
                similarities.append((existing_pattern, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def predict_next_patterns(self, pattern: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the most likely next patterns given an input.
        
        Uses sequential edges from the graph.
        """
        candidates = []
        
        # Find outgoing sequential edges
        for (src, tgt) in self.edges.keys():
            if src == pattern and self.edges[(src, tgt)].relationship_type == "sequential":
                edge = self.edges[(src, tgt)]
                candidates.append((tgt, edge.weight))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
    
    def get_neighborhood(self, pattern: str, depth: int = 1) -> Dict[str, float]:
        """
        Get all patterns within a certain graph distance.
        
        Returns a dict mapping pattern -> distance score.
        """
        neighborhood = {pattern: 1.0}
        frontier = {pattern}
        
        for _ in range(depth):
            next_frontier = set()
            for current in frontier:
                # Find neighbors
                for (src, tgt) in self.edges.keys():
                    if src == current:
                        if tgt not in neighborhood:
                            edge_weight = self.edges[(src, tgt)].weight
                            neighborhood[tgt] = edge_weight
                            next_frontier.add(tgt)
                    elif tgt == current:
                        if src not in neighborhood:
                            edge_weight = self.edges[(src, tgt)].weight
                            neighborhood[src] = edge_weight
                            next_frontier.add(src)
            frontier = next_frontier
        
        return neighborhood
    
    # ========================================================================
    # Statistics and Introspection
    # ========================================================================
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the graph."""
        if not self.nodes:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'avg_degree': 0.0,
                'avg_semantic_density': 0.0,
                'avg_edge_weight': 0.0,
                'interaction_count': self.interaction_count
            }
        
        # Compute degree statistics
        degrees = defaultdict(int)
        for (src, tgt) in self.edges.keys():
            degrees[src] += 1
            degrees[tgt] += 1
        
        avg_degree = sum(degrees.values()) / len(self.nodes) if self.nodes else 0.0
        
        # Compute semantic density statistics
        densities = [node.semantic_density for node in self.nodes.values()]
        avg_semantic_density = sum(densities) / len(densities) if densities else 0.0
        
        # Compute edge weight statistics
        edge_weights = [edge.weight for edge in self.edges.values()]
        avg_edge_weight = sum(edge_weights) / len(edge_weights) if edge_weights else 0.0
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'avg_degree': avg_degree,
            'avg_semantic_density': avg_semantic_density,
            'avg_edge_weight': avg_edge_weight,
            'interaction_count': self.interaction_count,
            'graph_density': len(self.edges) / (len(self.nodes) * (len(self.nodes) - 1)) if len(self.nodes) > 1 else 0.0
        }
    
    def get_top_nodes_by_density(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get the k most semantically dense nodes."""
        nodes_by_density = [(node.pattern, node.semantic_density) 
                           for node in self.nodes.values()]
        nodes_by_density.sort(key=lambda x: x[1], reverse=True)
        return nodes_by_density[:k]
    
    def to_dict(self) -> Dict:
        """Serialize the graph to a dictionary."""
        return {
            'nodes': {pattern: node.to_dict() for pattern, node in self.nodes.items()},
            'edges': {f"{src}->{tgt}": edge.to_dict() for (src, tgt), edge in self.edges.items()},
            'interaction_count': self.interaction_count
        }


# ============================================================================
# Test and Demonstration
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("BRAILLE SEMANTIC GRAPH: CORE IMPLEMENTATION TEST")
    print("=" * 80)
    
    # Initialize graph
    graph = BrailleSemanticGraph()
    
    # Test 1: Basic node and edge operations
    print("\n[TEST 1] Basic Node and Edge Operations")
    print("-" * 80)
    
    node1 = graph.add_node("⠓⠑⠇⠇⠕")
    node2 = graph.add_node("⠓⠑⠇⠏⠕")
    print(f"Added node 1: {node1.pattern}")
    print(f"Added node 2: {node2.pattern}")
    
    edge = graph.add_edge("⠓⠑⠇⠇⠕", "⠓⠑⠇⠏⠕", relationship_type="similarity")
    print(f"Added edge: {edge.source_pattern} -> {edge.target_pattern} (weight: {edge.weight:.3f})")
    
    # Test 2: Braille-native similarity metrics
    print("\n[TEST 2] Braille-Native Similarity Metrics")
    print("-" * 80)
    
    pattern1 = "⠓⠑⠇⠇⠕"
    pattern2 = "⠓⠑⠇⠏⠕"
    
    hamming_sim = graph.hamming_similarity(pattern1, pattern2)
    edit_dist = graph.edit_distance(pattern1, pattern2)
    composite_sim = graph.composite_similarity(pattern1, pattern2)
    
    print(f"Pattern 1: {pattern1}")
    print(f"Pattern 2: {pattern2}")
    print(f"Hamming similarity: {hamming_sim:.3f}")
    print(f"Edit distance: {edit_dist}")
    print(f"Composite similarity: {composite_sim:.3f}")
    
    # Test 3: Sequence ingestion
    print("\n[TEST 3] Sequence Ingestion and Graph Evolution")
    print("-" * 80)
    
    test_sequence = "⠓⠑⠇⠇⠕⠠⠃⠗⠁⠊⠇⠇⠑"
    print(f"Ingesting sequence: {test_sequence}")
    graph.ingest_sequence(test_sequence, n_gram_size=2)
    
    stats = graph.get_statistics()
    print(f"Graph now has {stats['num_nodes']} nodes and {stats['num_edges']} edges")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Average semantic density: {stats['avg_semantic_density']:.3f}")
    
    # Test 4: Semantic density
    print("\n[TEST 4] Semantic Density Computation")
    print("-" * 80)
    
    top_nodes = graph.get_top_nodes_by_density(k=5)
    print("Top 5 nodes by semantic density:")
    for pattern, density in top_nodes:
        print(f"  {pattern}: {density:.3f}")
    
    # Test 5: Similarity search
    print("\n[TEST 5] Similarity-Based Pattern Retrieval")
    print("-" * 80)
    
    query = "⠓⠑⠇⠇⠕"
    similar = graph.find_similar_patterns(query, k=3)
    print(f"Patterns similar to {query}:")
    for pattern, sim in similar:
        print(f"  {pattern}: {sim:.3f}")
    
    # Test 6: Next pattern prediction
    print("\n[TEST 6] Sequential Pattern Prediction")
    print("-" * 80)
    
    query = "⠓⠑"
    next_patterns = graph.predict_next_patterns(query, k=3)
    print(f"Likely patterns after {query}:")
    if next_patterns:
        for pattern, weight in next_patterns:
            print(f"  {pattern}: {weight:.3f}")
    else:
        print("  (No sequential edges found)")
    
    # Test 7: Graph statistics
    print("\n[TEST 7] Graph Statistics")
    print("-" * 80)
    
    stats = graph.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
    print("\nKey Achievement:")
    print("The braille semantic graph is now operational.")
    print("It learns from sequences, computes braille-native similarities,")
    print("and evolves its internal structure through interaction.")
    print("=" * 80)
