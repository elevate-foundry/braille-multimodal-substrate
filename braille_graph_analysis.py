"""
Phase 3: Semantic Density Analysis and Pattern Clustering

This module analyzes the braille semantic graph to identify emergent concepts
and visualize the structure of the evolving symbolic ecosystem.
"""

from braille_semantic_graph import BrailleSemanticGraph
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import math


class BrailleGraphAnalyzer:
    """
    Analyzes the structure and dynamics of a braille semantic graph.
    """
    
    def __init__(self, graph: BrailleSemanticGraph):
        self.graph = graph
    
    # ========================================================================
    # Community Detection (Clustering)
    # ========================================================================
    
    def label_propagation_clustering(self, max_iterations: int = 100) -> Dict[str, int]:
        """
        Simple label propagation algorithm to detect communities/clusters.
        
        Returns a dict mapping pattern -> cluster_id.
        """
        if not self.graph.nodes:
            return {}
        
        # Initialize each node with its own label
        labels = {pattern: i for i, pattern in enumerate(self.graph.nodes.keys())}
        
        for iteration in range(max_iterations):
            new_labels = labels.copy()
            changed = False
            
            for pattern in self.graph.nodes.keys():
                # Get neighbors
                neighbors = []
                for (src, tgt) in self.graph.edges.keys():
                    if src == pattern:
                        neighbors.append((tgt, self.graph.edges[(src, tgt)].weight))
                    elif tgt == pattern:
                        neighbors.append((src, self.graph.edges[(src, tgt)].weight))
                
                if neighbors:
                    # Count labels of neighbors, weighted by edge weight
                    label_counts = defaultdict(float)
                    for neighbor, weight in neighbors:
                        label_counts[labels[neighbor]] += weight
                    
                    # Assign most common label
                    if label_counts:
                        most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
                        if most_common_label != new_labels[pattern]:
                            new_labels[pattern] = most_common_label
                            changed = True
            
            labels = new_labels
            if not changed:
                break
        
        # Relabel clusters to be contiguous (0, 1, 2, ...)
        unique_labels = set(labels.values())
        label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
        return {pattern: label_mapping[label] for pattern, label in labels.items()}
    
    def get_clusters(self) -> Dict[int, List[str]]:
        """
        Get clusters of patterns.
        
        Returns a dict mapping cluster_id -> list of patterns.
        """
        clusters_dict = self.label_propagation_clustering()
        
        clusters = defaultdict(list)
        for pattern, cluster_id in clusters_dict.items():
            clusters[cluster_id].append(pattern)
        
        return dict(clusters)
    
    # ========================================================================
    # Semantic Density Analysis
    # ========================================================================
    
    def compute_global_semantic_density(self) -> float:
        """
        Compute the overall semantic density of the graph.
        
        High density = graph is tightly clustered (meaningful structure).
        Low density = graph is sparse (little structure).
        """
        if not self.graph.nodes:
            return 0.0
        
        densities = [node.semantic_density for node in self.graph.nodes.values()]
        return sum(densities) / len(densities) if densities else 0.0
    
    def get_density_distribution(self) -> Dict[str, float]:
        """Get semantic density for each pattern."""
        return {pattern: node.semantic_density for pattern, node in self.graph.nodes.items()}
    
    def get_density_percentiles(self) -> Dict[str, float]:
        """Get percentile ranks of patterns by semantic density."""
        densities = sorted([node.semantic_density for node in self.graph.nodes.values()])
        
        percentiles = {}
        for pattern, node in self.graph.nodes.items():
            # Find percentile
            rank = sum(1 for d in densities if d <= node.semantic_density)
            percentile = (rank / len(densities)) * 100 if densities else 0.0
            percentiles[pattern] = percentile
        
        return percentiles
    
    # ========================================================================
    # Graph Dynamics
    # ========================================================================
    
    def get_hub_nodes(self, k: int = 5) -> List[Tuple[str, int]]:
        """
        Get the k most connected nodes (hubs).
        
        These are the central concepts in the graph.
        """
        degree_count = defaultdict(int)
        
        for (src, tgt) in self.graph.edges.keys():
            degree_count[src] += 1
            degree_count[tgt] += 1
        
        sorted_hubs = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_hubs[:k]
    
    def get_peripheral_nodes(self, k: int = 5) -> List[Tuple[str, int]]:
        """
        Get the k most isolated nodes (periphery).
        
        These are patterns that haven't integrated into the main structure.
        """
        degree_count = defaultdict(int)
        
        for (src, tgt) in self.graph.edges.keys():
            degree_count[src] += 1
            degree_count[tgt] += 1
        
        # Nodes with low degree
        all_nodes = set(self.graph.nodes.keys())
        for node in all_nodes:
            if node not in degree_count:
                degree_count[node] = 0
        
        sorted_peripheral = sorted(degree_count.items(), key=lambda x: x[1])
        return sorted_peripheral[:k]
    
    def get_bridge_nodes(self, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get nodes that connect different clusters (bridges).
        
        These are patterns that link disparate concepts.
        """
        clusters = self.get_clusters()
        
        bridge_scores = {}
        for pattern in self.graph.nodes.keys():
            # Find which clusters this pattern connects to
            connected_clusters = set()
            
            for (src, tgt) in self.graph.edges.keys():
                if src == pattern:
                    for cluster_id, members in clusters.items():
                        if tgt in members:
                            connected_clusters.add(cluster_id)
                elif tgt == pattern:
                    for cluster_id, members in clusters.items():
                        if src in members:
                            connected_clusters.add(cluster_id)
            
            # Bridge score = number of clusters connected
            bridge_scores[pattern] = len(connected_clusters)
        
        sorted_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_bridges[:k]
    
    # ========================================================================
    # Pattern Relationships
    # ========================================================================
    
    def get_pattern_context(self, pattern: str, depth: int = 2) -> Dict[str, float]:
        """
        Get the semantic context of a pattern.
        
        Returns all patterns within a certain graph distance and their weights.
        """
        return self.graph.get_neighborhood(pattern, depth=depth)
    
    def find_pattern_analogs(self, pattern1: str, pattern2: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find patterns that are analogous to pattern1 in the same way pattern2 relates to pattern1.
        
        This is a simple analogical reasoning operation.
        """
        # Get neighbors of pattern1
        neighbors1 = self.graph.get_neighborhood(pattern1, depth=1)
        
        # Get neighbors of pattern2
        neighbors2 = self.graph.get_neighborhood(pattern2, depth=1)
        
        # Find patterns in neighbors2 that are not in neighbors1
        candidates = {}
        for pattern, weight in neighbors2.items():
            if pattern not in neighbors1 and pattern != pattern1 and pattern != pattern2:
                candidates[pattern] = weight
        
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:k]
    
    # ========================================================================
    # Visualization and Reporting
    # ========================================================================
    
    def generate_text_report(self) -> str:
        """Generate a text report of the graph's structure."""
        report = []
        report.append("=" * 80)
        report.append("BRAILLE SEMANTIC GRAPH ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Overall statistics
        stats = self.graph.get_statistics()
        report.append("\n[GRAPH STATISTICS]")
        report.append(f"Nodes: {stats['num_nodes']}")
        report.append(f"Edges: {stats['num_edges']}")
        report.append(f"Average degree: {stats['avg_degree']:.2f}")
        report.append(f"Average semantic density: {stats['avg_semantic_density']:.3f}")
        report.append(f"Graph density: {stats['graph_density']:.3f}")
        
        # Clusters
        clusters = self.get_clusters()
        report.append(f"\n[CLUSTERING]")
        report.append(f"Number of clusters: {len(clusters)}")
        for cluster_id, members in sorted(clusters.items()):
            report.append(f"  Cluster {cluster_id}: {len(members)} patterns")
            # Show top 3 members by density
            top_members = sorted(members, key=lambda p: self.graph.nodes[p].semantic_density, reverse=True)[:3]
            for member in top_members:
                density = self.graph.nodes[member].semantic_density
                report.append(f"    - {member} (density: {density:.3f})")
        
        # Hub nodes
        hubs = self.get_hub_nodes(k=5)
        report.append(f"\n[HUB NODES (Central Concepts)]")
        for pattern, degree in hubs:
            density = self.graph.nodes[pattern].semantic_density
            report.append(f"  {pattern}: degree={degree}, density={density:.3f}")
        
        # Peripheral nodes
        peripheral = self.get_peripheral_nodes(k=5)
        report.append(f"\n[PERIPHERAL NODES (Isolated Patterns)]")
        for pattern, degree in peripheral:
            if degree == 0:
                report.append(f"  {pattern}: isolated")
            else:
                density = self.graph.nodes[pattern].semantic_density
                report.append(f"  {pattern}: degree={degree}, density={density:.3f}")
        
        # Bridge nodes
        bridges = self.get_bridge_nodes(k=5)
        report.append(f"\n[BRIDGE NODES (Connecting Concepts)]")
        for pattern, num_clusters in bridges:
            if num_clusters > 1:
                density = self.graph.nodes[pattern].semantic_density
                report.append(f"  {pattern}: connects {num_clusters} clusters, density={density:.3f}")
        
        # Top nodes by density
        top_nodes = self.graph.get_top_nodes_by_density(k=10)
        report.append(f"\n[TOP NODES BY SEMANTIC DENSITY]")
        for pattern, density in top_nodes:
            report.append(f"  {pattern}: {density:.3f}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def visualize_graph_ascii(self, max_nodes: int = 20) -> str:
        """
        Generate an ASCII visualization of the graph.
        
        Shows nodes and their connections.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("BRAILLE SEMANTIC GRAPH VISUALIZATION (ASCII)")
        lines.append("=" * 80)
        
        # Get top nodes by degree
        hubs = self.get_hub_nodes(k=min(max_nodes, len(self.graph.nodes)))
        hub_patterns = [p for p, _ in hubs]
        
        lines.append(f"\nShowing top {len(hub_patterns)} nodes by connectivity:\n")
        
        for i, pattern in enumerate(hub_patterns):
            density = self.graph.nodes[pattern].semantic_density
            
            # Show node
            lines.append(f"{i+1:2}. {pattern} [density: {density:.2f}]")
            
            # Show outgoing edges
            outgoing = []
            for (src, tgt) in self.graph.edges.keys():
                if src == pattern and tgt in hub_patterns:
                    weight = self.graph.edges[(src, tgt)].weight
                    outgoing.append((tgt, weight))
            
            if outgoing:
                outgoing.sort(key=lambda x: x[1], reverse=True)
                for tgt, weight in outgoing[:3]:
                    bar_length = int(weight * 10)
                    bar = "█" * bar_length + "░" * (10 - bar_length)
                    lines.append(f"    └─> {tgt} [{bar}] {weight:.2f}")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


# ============================================================================
# Test and Demonstration
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("BRAILLE SEMANTIC GRAPH ANALYSIS")
    print("=" * 80)
    
    # Create and populate a graph
    graph = BrailleSemanticGraph()
    
    # Ingest multiple sequences to build structure
    sequences = [
        "⠓⠑⠇⠇⠕⠠⠃⠗⠁⠊⠇⠇⠑",
        "⠓⠑⠇⠇⠕⠠⠺⠕⠗⠇⠙",
        "⠠⠃⠗⠁⠊⠇⠇⠑⠠⠍⠁⠞⠞⠑⠗",
        "⠠⠍⠁⠞⠞⠑⠗⠠⠑⠝⠑⠗⠛⠽",
    ]
    
    for seq in sequences:
        graph.ingest_sequence(seq, n_gram_size=2)
    
    # Prune old edges
    graph.prune_edges()
    
    # Create analyzer
    analyzer = BrailleGraphAnalyzer(graph)
    
    # Test 1: Generate report
    print("\n[TEST 1] Comprehensive Analysis Report")
    print("-" * 80)
    report = analyzer.generate_text_report()
    print(report)
    
    # Test 2: Clustering
    print("\n[TEST 2] Pattern Clustering")
    print("-" * 80)
    clusters = analyzer.get_clusters()
    print(f"Identified {len(clusters)} clusters:")
    for cluster_id, members in sorted(clusters.items()):
        print(f"  Cluster {cluster_id}: {members}")
    
    # Test 3: ASCII Visualization
    print("\n[TEST 3] Graph Visualization")
    print("-" * 80)
    viz = analyzer.visualize_graph_ascii()
    print(viz)
    
    # Test 4: Semantic density analysis
    print("\n[TEST 4] Semantic Density Analysis")
    print("-" * 80)
    global_density = analyzer.compute_global_semantic_density()
    print(f"Global semantic density: {global_density:.3f}")
    
    percentiles = analyzer.get_density_percentiles()
    print(f"\nDensity percentiles (sample):")
    for pattern, percentile in list(percentiles.items())[:5]:
        print(f"  {pattern}: {percentile:.1f}th percentile")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
