import jax
import jax.numpy as jnp
import numpy as np
import math
import galois
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import combinations_with_replacement, permutations, product
from typing import Dict, Tuple, List, Optional
from fractions import Fraction

def generate_D_vectors(t: int, m: int, s: int) -> np.ndarray:
    n = m - t
    D_list = []
    for D in combinations_with_replacement(range(n + 1), s):
        if sum(D) == n:
            D_list.extend(set(permutations(D)))
    D_array = np.array(D_list)
    return np.unique(D_array, axis=0)

def generate_D_A_pairs(t: int, m: int, s: int, b: int) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    D_array = generate_D_vectors(t, m, s)
    D_A_to_index = {}
    for D in D_array:
        A_ranges = [range(b ** d_i) for d_i in D]
        D_A_to_index[tuple(D)] = list(product(*A_ranges))
    return D_A_to_index

def convert_points_to_fractions(points: np.ndarray, b: int) -> np.ndarray:
    points_fractions = []
    for point in points:
        point_fractions_row = []
        for x in point:
            if isinstance(x, Fraction):
                point_fractions_row.append(x)
                continue
            x_float = float(f"{x:.20f}".rstrip('0').rstrip('.'))
            if x_float < 1e-10:
                point_fractions_row.append(Fraction(0, 1))
            elif x_float > 1 - 1e-10:
                point_fractions_row.append(Fraction(1, 1))
            else:
                found = False
                for n in range(1, 15):
                    denominator = b**n
                    numerator = int(round(x_float * denominator))
                    if abs(x_float - numerator / denominator) < 1e-10:
                        point_fractions_row.append(Fraction(numerator, denominator))
                        found = True
                        break
                if not found:
                    point_fractions_row.append(Fraction(str(x_float)))
        points_fractions.append(point_fractions_row)
    return np.array(points_fractions)

def _is_tms_network(points_fractions: np.ndarray, t: int, m: int, s: int, b: int) -> bool:
    D_A_pairs = generate_D_A_pairs(t, m, s, b)
    for D, A_list in D_A_pairs.items():
        for A in A_list:
            count = 0
            lower_bounds = [Fraction(A[j], b**D[j]) for j in range(s)]
            upper_bounds = [Fraction(A[j] + 1, b**D[j]) for j in range(s)]
            for point in points_fractions:
                if all(lower_bounds[j] <= point[j] < upper_bounds[j] for j in range(s)):
                    count += 1
            
            if count != b**t:
                return False 
                
    return True

class TMSNet:
    def __init__(self, t, m, s, b, points):
        self.t = t
        self.m = m
        self.s = s
        self.b = b
        self.points = np.asarray(points) # Convert JAX array to numpy if needed
        print(f"A ({self.t}, {self.m}, {self.s})-network is created on base {self.b}, containing {self.points.shape[0]} points.")

    def visualize(self):
        if self.points.shape[1] < 2:
            print("Visualization is only possible for s >= 2.")
            return
            
        if self.points.shape[1] == 2:
            df = pd.DataFrame(self.points, columns=["x", "y"])
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x="x", y="y", s=20).set_title(f'({self.t}, {self.m}, {self.s})-net')
            plt.grid(True)
            plt.show()
            
        elif self.points.shape[1] == 3:
            df = pd.DataFrame(self.points, columns=["x", "y", "z"])
            fig = px.scatter_3d(df, x='x', y='y', z='z', title=f'({self.t}, {self.m}, {self.s})-net')
            fig.update_traces(marker=dict(size=3))
            fig.show()
        else:
            print(f"Visualization for s={self.s} is not supported. Only the first 2 dimensions will be shown.")
            df = pd.DataFrame(self.points[:, :2], columns=["x", "y"])
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x="x", y="y", s=20).set_title(f'({self.t}, {self.m}, {self.s})-net (first 2 dimensions)')
            plt.grid(True)
            plt.show()

    def verify(self) -> bool:
        print(f"--- Starting verification for t = {self.t}... ---")
        if self.points.shape[0] != self.b**self.m:
            print(f"Critical error: For a (t, m, s)-network of radix {self.b} with m={self.m} there should be {self.b**self.m} points, but {self.points.shape[0]} is provided.")
            return False

        points_fractions = convert_points_to_fractions(self.points, self.b)
        best_t_found = -1
        for new_t in range(self.m + 1):
            if _is_tms_network(points_fractions, new_t, self.m, self.s, self.b):
                best_t_found = new_t
                break

        if best_t_found == -1:
            print("Verification failed: No suitable value for t could be found.")
            return False
        if best_t_found == self.t:
            print(f"Verification is successful: the points indeed form a ({self.t}, {self.m}, {self.s})-network.")
            return True
        else:
            old_t = self.t
            self.t = best_t_found
            print(f"Initial check for t={old_t} failed.")
            print(f"---Updating---")
            print(f"The best value of t for a given set of points is found: t = {self.t}.")
            print(f"The object's t parameter was updated from {old_t} to {self.t}.")
            return True