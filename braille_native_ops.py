"""
Stage 2: Braille-Native Operations

This module implements operations that work directly in braille space,
without requiring a decode-to-float step. These are the primitives that
enable the transition from braille-as-carrier to braille-as-substrate.

Key operations:
- Topological metrics (Hamming distance, neighborhood)
- Morphological operators (dilation, erosion, convolution)
- Sequence-level metrics (edit distance, structural loss)
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import json


class BrailleTopology:
    """
    Defines the topology of braille space.
    
    The 256 braille characters (U+2800 to U+28FF) form a discrete space.
    We define distance metrics and neighborhood relationships directly on this space.
    """
    
    BRAILLE_START = 0x2800
    BRAILLE_END = 0x28FF
    BRAILLE_RANGE = 256
    
    @staticmethod
    def hamming_distance(char1: int, char2: int) -> int:
        """
        Compute Hamming distance between two braille characters.
        
        Hamming distance = number of bit positions where the characters differ.
        This is a native metric for braille space.
        
        Args:
            char1: First braille character (0-255)
            char2: Second braille character (0-255)
            
        Returns:
            Hamming distance (0-8, since braille is 8-bit)
        """
        xor = char1 ^ char2
        return bin(xor).count('1')
    
    @staticmethod
    def hamming_similarity(char1: int, char2: int) -> float:
        """
        Compute normalized Hamming similarity (0-1).
        
        Similarity = 1 - (hamming_distance / 8)
        """
        distance = BrailleTopology.hamming_distance(char1, char2)
        return 1.0 - (distance / 8.0)
    
    @staticmethod
    def nearest_neighbors(char: int, k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the k nearest braille characters to a given character.
        
        Uses Hamming distance as the metric.
        
        Args:
            char: Reference braille character (0-255)
            k: Number of neighbors to return
            
        Returns:
            List of (neighbor_char, similarity) tuples, sorted by similarity (descending)
        """
        neighbors = []
        for other in range(BrailleTopology.BRAILLE_RANGE):
            if other != char:
                sim = BrailleTopology.hamming_similarity(char, other)
                neighbors.append((other, sim))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:k]
    
    @staticmethod
    def neighborhood_graph(k: int = 3) -> Dict[int, Set[int]]:
        """
        Compute the k-nearest-neighbor graph for all braille characters.
        
        Returns a dictionary where each braille character maps to its k nearest neighbors.
        This defines the topology of braille space.
        
        Args:
            k: Number of nearest neighbors for each character
            
        Returns:
            Dictionary mapping each character to its k nearest neighbors
        """
        graph = {}
        for char in range(BrailleTopology.BRAILLE_RANGE):
            neighbors = BrailleTopology.nearest_neighbors(char, k)
            graph[char] = set(n[0] for n in neighbors)
        return graph


class BrailleMorphology:
    """
    Morphological operators that work directly on braille patterns.
    
    These operations are defined on 2D grids of braille characters,
    using bitwise operations on the 8-bit patterns.
    """
    
    @staticmethod
    def braille_dilate(grid: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Morphological dilation in braille space.
        
        For each braille character, dilate by taking the bitwise OR
        with all neighbors in the grid.
        
        Args:
            grid: 2D array of braille characters (0-255)
            iterations: Number of dilation iterations
            
        Returns:
            Dilated grid
        """
        result = grid.copy()
        h, w = grid.shape
        
        for _ in range(iterations):
            new_result = result.copy()
            for i in range(h):
                for j in range(w):
                    # Collect all neighbors (including diagonals)
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                neighbors.append(result[ni, nj])
                    
                    # Dilate: bitwise OR of all neighbors
                    dilated = 0
                    for neighbor in neighbors:
                        dilated |= neighbor
                    new_result[i, j] = dilated
            result = new_result
        
        return result
    
    @staticmethod
    def braille_erode(grid: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Morphological erosion in braille space.
        
        For each braille character, erode by taking the bitwise AND
        with all neighbors in the grid.
        
        Args:
            grid: 2D array of braille characters (0-255)
            iterations: Number of erosion iterations
            
        Returns:
            Eroded grid
        """
        result = grid.copy()
        h, w = grid.shape
        
        for _ in range(iterations):
            new_result = result.copy()
            for i in range(h):
                for j in range(w):
                    # Collect all neighbors
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                neighbors.append(result[ni, nj])
                    
                    # Erode: bitwise AND of all neighbors
                    eroded = 255  # Start with all bits set
                    for neighbor in neighbors:
                        eroded &= neighbor
                    new_result[i, j] = eroded
            result = new_result
        
        return result
    
    @staticmethod
    def braille_convolve(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Convolution in braille space.
        
        Apply a kernel to the integer representation of braille patterns.
        The result is clipped to [0, 255] and converted back to braille.
        
        Args:
            grid: 2D array of braille characters (0-255)
            kernel: 2D convolution kernel
            
        Returns:
            Convolved grid (values clipped to 0-255)
        """
        h, w = grid.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        result = np.zeros_like(grid, dtype=float)
        
        for i in range(h):
            for j in range(w):
                value = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        ii = i + ki - pad_h
                        jj = j + kj - pad_w
                        if 0 <= ii < h and 0 <= jj < w:
                            value += grid[ii, jj] * kernel[ki, kj]
                result[i, j] = np.clip(value, 0, 255)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def braille_edge_detect(grid: np.ndarray) -> np.ndarray:
        """
        Edge detection in braille space using Sobel-like operator.
        
        Uses a simple 3x3 kernel to detect edges.
        
        Args:
            grid: 2D array of braille characters
            
        Returns:
            Edge-detected grid
        """
        # Sobel kernel (simplified)
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=float)
        
        return BrailleMorphology.braille_convolve(grid, kernel)


class BrailleSequenceMetrics:
    """
    Metrics and operations on braille sequences (1D).
    
    These operations work on sequences of braille tokens,
    computing distances and structural properties without decoding.
    """
    
    @staticmethod
    def braille_edit_distance(seq1: str, seq2: str) -> int:
        """
        Levenshtein distance between two braille sequences.
        
        This is the edit distance computed directly on braille tokens,
        without decoding to float space.
        
        Args:
            seq1: First braille sequence (string of braille characters)
            seq2: Second braille sequence
            
        Returns:
            Edit distance
        """
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        
        return dp[m, n]
    
    @staticmethod
    def braille_popcount(seq: str) -> int:
        """
        Total "energy" of a braille sequence, measured by total bit count.
        
        Popcount = sum of the number of dots in each braille character.
        
        Args:
            seq: Braille sequence
            
        Returns:
            Total popcount
        """
        total = 0
        for char in seq:
            code = ord(char)
            if 0x2800 <= code <= 0x28FF:
                value = code - 0x2800
                total += bin(value).count('1')
        return total
    
    @staticmethod
    def braille_structural_loss(seq1: str, seq2: str) -> float:
        """
        Structural loss between two braille sequences.
        
        Compares:
        1. Edit distance (token-level)
        2. Popcount difference (energy-level)
        3. Length difference
        
        Returns a combined loss that captures structural similarity.
        
        Args:
            seq1: First braille sequence
            seq2: Second braille sequence
            
        Returns:
            Structural loss (0-1, where 0 is identical)
        """
        edit_dist = BrailleSequenceMetrics.braille_edit_distance(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        
        popcount1 = BrailleSequenceMetrics.braille_popcount(seq1)
        popcount2 = BrailleSequenceMetrics.braille_popcount(seq2)
        max_popcount = max(popcount1, popcount2, 1)  # Avoid division by zero
        
        edit_loss = edit_dist / max(max_len, 1)
        popcount_loss = abs(popcount1 - popcount2) / max_popcount
        length_loss = abs(len(seq1) - len(seq2)) / max(max_len, 1)
        
        # Weighted combination
        total_loss = 0.4 * edit_loss + 0.3 * popcount_loss + 0.3 * length_loss
        return min(total_loss, 1.0)
    
    @staticmethod
    def braille_similarity(seq1: str, seq2: str) -> float:
        """
        Normalized similarity between two braille sequences (0-1).
        
        Similarity = 1 - structural_loss
        """
        loss = BrailleSequenceMetrics.braille_structural_loss(seq1, seq2)
        return 1.0 - loss


class BrailleNativeEngine:
    """
    Braille-native reasoning engine.
    
    Combines topology, morphology, and sequence metrics to enable
    reasoning that stays entirely within braille space.
    """
    
    def __init__(self):
        self.topology = BrailleTopology()
        self.morphology = BrailleMorphology()
        self.metrics = BrailleSequenceMetrics()
        self.neighborhood_graph = self.topology.neighborhood_graph(k=3)
    
    def find_similar_patterns(self, pattern: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar braille patterns in a corpus.
        
        Uses braille-native similarity metric.
        """
        # This is a placeholder; in a real system, this would search a corpus
        # For now, we'll generate synthetic similar patterns
        similar = []
        for _ in range(k):
            # Generate a pattern by mutating the input
            mutated = list(pattern)
            for i in range(len(mutated)):
                if np.random.rand() < 0.2:  # 20% chance to mutate
                    char_code = ord(mutated[i])
                    if 0x2800 <= char_code <= 0x28FF:
                        value = char_code - 0x2800
                        # Flip a random bit
                        bit = np.random.randint(0, 8)
                        value ^= (1 << bit)
                        mutated[i] = chr(0x2800 + value)
            
            mutated_str = ''.join(mutated)
            sim = self.metrics.braille_similarity(pattern, mutated_str)
            similar.append((mutated_str, sim))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def fuse_patterns(self, patterns: List[str]) -> str:
        """
        Fuse multiple braille patterns into a single pattern.
        
        Uses bitwise OR to combine all patterns.
        """
        if not patterns:
            return ''
        
        # Pad all patterns to the same length
        max_len = max(len(p) for p in patterns)
        padded = [p.ljust(max_len, chr(0x2800)) for p in patterns]
        
        # Fuse using bitwise OR
        fused = []
        for i in range(max_len):
            combined = 0
            for pattern in padded:
                char_code = ord(pattern[i])
                if 0x2800 <= char_code <= 0x28FF:
                    value = char_code - 0x2800
                    combined |= value
            fused.append(chr(0x2800 + combined))
        
        return ''.join(fused)
    
    def get_statistics(self) -> Dict:
        """
        Return statistics about the braille-native engine.
        """
        return {
            'braille_alphabet_size': self.topology.BRAILLE_RANGE,
            'topology_type': 'Hamming distance (8-bit)',
            'morphological_operators': ['dilate', 'erode', 'convolve', 'edge_detect'],
            'sequence_metrics': ['edit_distance', 'popcount', 'structural_loss', 'similarity'],
            'neighborhood_graph_size': len(self.neighborhood_graph)
        }


# ============================================================================
# Test and Demonstration
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("STAGE 2: BRAILLE-NATIVE OPERATIONS")
    print("=" * 80)
    
    # Initialize the engine
    engine = BrailleNativeEngine()
    
    # Test 1: Hamming Distance and Similarity
    print("\n[TEST 1] Hamming Distance and Similarity")
    print("-" * 80)
    char1, char2 = 0, 255
    distance = BrailleTopology.hamming_distance(char1, char2)
    similarity = BrailleTopology.hamming_similarity(char1, char2)
    print(f"Character 1: {char1:08b} (⠀)")
    print(f"Character 2: {char2:08b} (⠿)")
    print(f"Hamming distance: {distance}")
    print(f"Hamming similarity: {similarity:.3f}")
    
    # Test 2: Nearest Neighbors
    print("\n[TEST 2] Nearest Neighbors")
    print("-" * 80)
    char = 128
    neighbors = BrailleTopology.nearest_neighbors(char, k=5)
    print(f"Reference character: {char:08b}")
    for i, (neighbor, sim) in enumerate(neighbors):
        print(f"  Neighbor {i+1}: {neighbor:08b} (similarity: {sim:.3f})")
    
    # Test 3: Morphological Dilation
    print("\n[TEST 3] Morphological Dilation")
    print("-" * 80)
    test_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 255, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 255, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    
    dilated = BrailleMorphology.braille_dilate(test_grid, iterations=1)
    print("Original grid (center values):")
    print(test_grid)
    print("\nDilated grid (center values):")
    print(dilated)
    
    # Test 4: Sequence Edit Distance
    print("\n[TEST 4] Braille Sequence Edit Distance")
    print("-" * 80)
    seq1 = "⠓⠑⠇⠇⠕"  # "Hello" in braille
    seq2 = "⠓⠑⠇⠏⠕"  # "Helpo" in braille (one character different)
    edit_dist = BrailleSequenceMetrics.braille_edit_distance(seq1, seq2)
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    print(f"Edit distance: {edit_dist}")
    
    # Test 5: Sequence Similarity
    print("\n[TEST 5] Braille Sequence Similarity")
    print("-" * 80)
    similarity = BrailleSequenceMetrics.braille_similarity(seq1, seq2)
    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")
    print(f"Similarity: {similarity:.3f}")
    
    # Test 6: Pattern Fusion
    print("\n[TEST 6] Pattern Fusion")
    print("-" * 80)
    patterns = ["⠓⠑⠇⠇⠕", "⠓⠑⠇⠇⠕", "⠓⠑⠇⠇⠕"]
    fused = engine.fuse_patterns(patterns)
    print(f"Pattern 1: {patterns[0]}")
    print(f"Pattern 2: {patterns[1]}")
    print(f"Pattern 3: {patterns[2]}")
    print(f"Fused: {fused}")
    
    # Test 7: Engine Statistics
    print("\n[TEST 7] Braille-Native Engine Statistics")
    print("-" * 80)
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print("STAGE 2 TESTS COMPLETE")
    print("=" * 80)
