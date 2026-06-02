import jax
import jax.numpy as jnp
import numpy as np
import galois
from itertools import product

# Utility functions for generating Niederreiter matrices using galois
def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0: return False
    return True

def is_prime_power(n):
    if n < 2: return False
    if is_prime(n): return True
    for p in range(2, int(n ** 0.5) + 1):
        if is_prime(p):
            power = p
            while power <= n:
                if power == n: return True
                if n % power != 0: break
                power *= p
    return False

def _e_param(t, m, s):
    n = t + s
    x = s
    if n < x: return None
    result = [1] * x
    remaining = n - x
    i = 0
    while remaining > 0:
        result[i] += 1
        remaining -= 1
        i = (i + 1) % x
    return result

def _generate_excellent_poly(b, e, s):
    assert is_prime_power(b), "b must be a prime power"
    pi = []
    unique_polys = {degree: set() for degree in set(e)}

    for deg in set(e):
        all_irred = list(galois.irreducible_polys(b, deg))
        all_irred = [p for p in all_irred if p != galois.Poly([1, 0], field=galois.GF(b))]
        if len(all_irred) < e.count(deg):
            raise ValueError(f"There are not enough irreducible polynomials of degree {deg} in GF({b}).")
        unique_polys[deg] = set(all_irred)

    used = {deg: set() for deg in set(e)}

    for deg in e: 
        available = sorted(list(unique_polys[deg] - used[deg]), key=lambda p: tuple(p.coeffs))
        P = available[0] 
        used[deg].add(P)
        pi.append(P)
    return pi

def _generate_recurrent_sequence(poly, u, m):
    e = poly.degree
    degree = e * u
    poly_u = poly ** u
    coeffs = poly_u.coeffs

    GF = poly.field
    alpha = [GF(0)] * (e * (u - 1))
    alpha += [GF(1)] + [GF(0)] * (degree - (e * (u - 1)) - 1)

    while len(alpha) < m + degree:
        acc = GF(0)
        for k in range(1, degree + 1):
            acc -= coeffs[k] * alpha[-k]
        alpha.append(acc)
    return alpha

def _build_generator_matrix(poly, m):
    e = poly.degree
    num_sections = (m + e - 1) // e  
    G = np.zeros((m, m), dtype=int)
    for u in range(1, num_sections + 1):
        alpha = _generate_recurrent_sequence(poly, u, m)
        r_h = e - 1 if u < num_sections else (m - 1) % e
        for r in range(r_h + 1):
            j = e * (u - 1) + r
            if j >= m:
                break
            for k in range(m):
                G[j, k] = int(alpha[r + k])
    return G

def generate_generator_matrices(b, t, m, s, verbose=False):
    e = _e_param(t, m, s)
    assert e is not None, "Wrong params: t, m, s"
    assert t <= m, "t must be less or equal m"
    pi_list = _generate_excellent_poly(b, e, s)
    if verbose:
        print("list of polys:", pi_list)
    matrices = []
    for i in range(s):
        G = _build_generator_matrix(pi_list[i], m)
        matrices.append(G)
    return np.array(matrices)

def random_invertible_matrix(n, q):
    GF = galois.GF(q)
    while True:
        A = GF.Random((n, n))
        if np.linalg.det(A) != 0:
            return A

# JAX accelerated parts
@jax.jit
def _get_points_jax_prime(b: int, m: int, s: int, G: jnp.ndarray):
    """
    Computes (G @ vecs) % b for prime bases using JAX, and converts to fractions.
    G shape: (s, m, m)
    """
    N = b ** m
    n_values = jnp.arange(N)
    
    # Compute vecs in base b
    powers = b ** jnp.arange(m)
    # Shape: (N, m)
    vecs = (n_values[:, None] // powers[None, :]) % b
    
    # Transpose and reverse rows as requested in vecbm_opt
    # original logic: vecs_gf = gf((vecs.T)[::-1]) shape (m, N)
    vecs_t_rev = vecs.T[::-1]
    
    # Multiply matrices: G has shape (s, m, m)
    # result shape: (s, m, N)
    result = jnp.matmul(G, vecs_t_rev) % b
    
    powers_desc = b ** jnp.arange(m-1, -1, -1)
    
    # rnums calculation
    # original logic: rnums = np.tensordot(result, powers, axes=(1, 0)) # shape (s, N)
    rnums = jnp.tensordot(result, powers_desc, axes=(1, 0))
    
    # original logic: points = (rnums.T) * (b**(-m))
    points = rnums.T * (b ** (-m))
    
    return points

def get_points_opt_jax(b, t, m, s, scramble="No", verbose=False):
    """
    Constructing points using base matrices and JAX acceleration for performance.
    """
    G = generate_generator_matrices(b, t, m, s, verbose=verbose)
    
    GF = galois.GF(b)
    G_gf = GF(G)
    
    if scramble == "Linear":
        for i in range(s):
            A = random_invertible_matrix(m, b)
            G_gf[i] = A @ G_gf[i]

    # Check if prime. If prime, we use pure JAX arithmetic.
    # Otherwise (e.g. GF(4), GF(8), we must use galois arrays for correct addition / multiplication).
    if is_prime(b):
        G_jax = jnp.array(G_gf, dtype=jnp.int32)
        points = _get_points_jax_prime(b, m, s, G_jax)
        # JAX returns a unified array, we can return as a python numpy array
        return np.array(points)
    else:
        # Fallback to pure numpy + galois for prime power bases (e.g. GF(4))
        # because standard JAX `jnp.matmul` modulo b is wrong for Galois field arithmetic
        n_values = np.arange(b**m)
        powers = b ** np.arange(m)
        vecs = (n_values[:, None] // powers[None, :]) % b
        vecs_gf = GF((vecs.T)[::-1])
        
        result = np.empty((s, m, b**m), dtype=object)
        for i in range(s):
            result[i] = G_gf[i] @ vecs_gf
            
        powers_desc = b ** np.arange(m-1, -1, -1)
        rnums = np.tensordot(result, powers_desc, axes=(1, 0))
        points = (rnums.T) * (b**(-m))
        return points

class PolynomialNetConstructor:
    @staticmethod
    def construct_niederreiter(b, t, m, s, verbose=False, scramble="No"):
        points = get_points_opt_jax(b, t, m, s, scramble=scramble, verbose=verbose)
        return points
    
    @staticmethod
    def construct_rosenbloom_tsfasman(q, m, s, beta=None):
        GF = galois.GF(q)

        if s > q:
            raise ValueError("s must be least or equal q")
            
        S_gf = GF.elements[:s]
        is_default_beta = beta is None

        if not is_default_beta:
            beta_keys = np.array(sorted(list(beta.keys())))
            if np.array_equal(beta_keys, np.arange(q)):
                beta_lookup_array = np.array([beta[i] for i in range(q)], dtype=np.float64)
                beta_map_func = lambda x_int_array: beta_lookup_array[x_int_array.astype(np.int64)]
            else:
                _beta_vec = np.vectorize(lambda val_int: beta.get(val_int), otypes=[np.float64])
                beta_map_func = lambda x_int_array: _beta_vec(x_int_array)

        coeffs_int_tuples = product(range(q), repeat=m)
        poly_list = [galois.Poly(list(c_tuple)[::-1], field=GF) for c_tuple in coeffs_int_tuples]
        
        num_polynomials = q**m
        points = np.zeros((num_polynomials, s), dtype=np.float64)
        q_powers = q**(-(np.arange(m, dtype=np.float64) + 1.0))
        eval_matrix_int = np.zeros((s, m), dtype=np.int64)

        for idx_f, f_poly in enumerate(poly_list):
            current_deriv_poly = f_poly
            for j_deriv_order in range(m):
                eval_results_gf = current_deriv_poly(S_gf)
                eval_matrix_int[:, j_deriv_order] = eval_results_gf.astype(np.int64)
                if j_deriv_order < m - 1:
                    current_deriv_poly = current_deriv_poly.derivative()
            
            if is_default_beta:
                digits_matrix = eval_matrix_int.astype(np.float64)
            else:
                digits_matrix = beta_map_func(eval_matrix_int)
                
            points[idx_f, :] = np.dot(digits_matrix, q_powers)
            
        return points
