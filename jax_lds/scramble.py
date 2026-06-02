import numpy as np
from typing import List, Dict, Tuple

def get_digits_from_point(x: float, b: int, m: int) -> np.ndarray:
    """Extracts m digits of point x in base b."""
    digits = np.zeros(m, dtype=int)
    for i in range(m):
        x *= b
        digit = int(x)
        digits[i] = digit
        x -= digit
        if x < 1e-15:
            break
    return digits

def get_point_from_digits(digits: np.ndarray, b: int) -> float:
    """Converts array of digits in base b back to float."""
    m = len(digits)
    powers = b ** (-np.arange(1, m + 1, dtype=np.float64))
    return np.dot(digits, powers)

class OwenScrambler:
    """
    Class for Owen scrambling.
    Stores generated permutation trees.
    """
    def __init__(self, s: int, b: int, m: int, seed: int = None):
        if not (isinstance(b, int) and b >= 2):
            raise ValueError("Base 'b' must be an integer >= 2.")
        if not (isinstance(m, int) and m >= 1):
            raise ValueError("Precision 'm' must be an integer >= 1.")
        
        self.s = s
        self.b = b
        self.m = m
        self.rng = np.random.default_rng(seed)
        
        self.permutation_trees: List[Dict[Tuple[int, ...], np.ndarray]] = []
        self._generate_all_trees()

    def _generate_permutation(self) -> np.ndarray:
        p = np.arange(self.b)
        self.rng.shuffle(p)
        return p

    def _generate_tree_for_dimension(self) -> Dict[Tuple[int, ...], np.ndarray]:
        tree = {}
        tree[()] = self._generate_permutation()
        
        for k in range(1, self.m):
            num_prefixes = self.b ** k
            for i in range(num_prefixes):
                prefix_digits = []
                temp_i = i
                for _ in range(k):
                    prefix_digits.insert(0, temp_i % self.b)
                    temp_i //= self.b
                
                prefix = tuple(prefix_digits)
                tree[prefix] = self._generate_permutation()
        return tree

    def _generate_all_trees(self):
        self.permutation_trees = [self._generate_tree_for_dimension() for _ in range(self.s)]

    def scramble_point(self, point: np.ndarray) -> np.ndarray:
        if len(point) != self.s:
            raise ValueError(f"Point must have dimension {self.s}")

        scrambled_point = np.zeros(self.s)
        
        for j in range(self.s):
            original_digits = get_digits_from_point(point[j], self.b, self.m)
            scrambled_digits = np.zeros(self.m, dtype=int)
            
            for k in range(self.m):
                prefix = tuple(original_digits[:k])
                permutation = self.permutation_trees[j][prefix]
                scrambled_digits[k] = permutation[original_digits[k]]
            
            scrambled_point[j] = get_point_from_digits(scrambled_digits, self.b)
            
        return scrambled_point

def scramble_points(points: np.ndarray, m: int, b: int, seed: int = None) -> np.ndarray:
    s = points.shape[1]
    scrambler = OwenScrambler(s=s, b=b, m=m, seed=seed)
    
    scrambled = np.zeros_like(points)
    for i in range(points.shape[0]):
        scrambled[i] = scrambler.scramble_point(points[i])
        
    return scrambled
