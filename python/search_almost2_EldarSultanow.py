import math
import numpy as np
from multiprocessing import Pool
from numba import njit, prange

@njit
def isqrt(n):
    """Schnelle, Numba-kompatible Quadratwurzel mit NumPy."""
    return int(np.sqrt(n))  

@njit
def is_perfect_square(n):
    """Überprüft effizient, ob n eine perfekte Quadratzahl ist."""
    root = isqrt(n)  
    return root * root == n

@njit
def is_known_family(w, x, y, z):
    """Prüft effizient, ob die Werte zur bekannten (p,q)-Familie gehören."""
    for p in range(1, 100):  
        for q in range(1, p):
            w_known = (p**4 - 9*q**4)**2 - 64*p**4*q**4
            x_known = 4*p*q * (p**2 + 3*q**2) * (p**4 - 2*p**2*q**2 + 9*q**4)
            y_known = (p**4 - 9*q**4)**2 + 64*p**4*q**4
            z_known = (p**2 + q**2) * (p**2 + 9*q**2) * (p**4 - 2*p**2*q**2 + 9*q**4)
            if {w, x, y, z} == {w_known, x_known, y_known, z_known}:
                return True
    return False

@njit(parallel=True)
def find_almost_solutions(range_limit):
    """Findet Near-Solutions, bei denen genau 5 der 6 Bedingungen erfüllt sind."""
    solutions = []  
    
    for w in prange(1, range_limit):
        for x in range(w + 1, range_limit):  
            for y in range(x + 1, range_limit):  
                for z in range(y + 1, range_limit):  
                    conditions = [
                        (-x**2 + y**2),
                        (-x**2 + z**2),
                        (-y**2 + z**2),
                        (-w**2 + x**2),
                        (-w**2 + y**2),
                        (-w**2 + z**2)
                    ]

                    valid = 0  # Explizite Zählvariable
                    for num in conditions:
                        if is_perfect_square(num):
                            valid += 1

                    if valid >= 5:  # Mindestens 5 der 6 Bedingungen müssen Quadrate sein
                        # if not is_known_family(w, x, y, z):  
                        solutions.append((w, x, y, z))  

    return solutions  

def process_range(range_limit):
    """Wrapper-Funktion, um Near-Solutions zu berechnen und auszugeben."""
    solutions = find_almost_solutions(range_limit)
    for sol in solutions:
        print(f"{sol[0]}, {sol[1]}, {sol[2]}, {sol[3]}")

if __name__ == "__main__":
    range_limit = 500  

    print("w, x, y, z")

    with Pool() as pool:
        pool.map(process_range, [range_limit])