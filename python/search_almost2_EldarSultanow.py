import math
from multiprocessing import Pool
from numba import njit, prange

@njit
def isqrt(n):
    """Numba-kompatible Integer-Quadratwurzel."""
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

@njit
def is_perfect_square(n):
    """Überprüft effizient, ob n eine perfekte Quadratzahl ist."""
    if n < 0:
        return False
    root = isqrt(n)  # Ersetze math.isqrt(n) durch numba-kompatible Funktion
    return root * root == n

@njit
def is_known_family(w1, w2, x, y, z):
    """Prüft effizient, ob die Werte zur bekannten (p,q)-Familie gehören."""
    # Ggf. mit dynamischer Obergrenze
    # for p in range(1, int(math.sqrt(range_limit))):
    for p in prange(1, 100):
        for q in range(1, p):
            w1_known = (p**4 - 9*q**4)**2 - 64*p**4*q**4
            w2_known = 16*p**2 * (p**4 - 9*q**4)
            x_known = 4*p*q * (p**2 + 3*q**2) * (p**4 - 2*p**2*q**2 + 9*q**4)
            y_known = (p**4 - 9*q**4)**2 + 64*p**4*q**4
            z_known = (p**2 + q**2) * (p**2 + 9*q**2) * (p**4 - 2*p**2*q**2 + 9*q**4)
            if {w1, w2, x, y, z} == {w1_known, w2_known, x_known, y_known, z_known}:
                return True
    return False

@njit(parallel=True)
def find_almost_solutions(range_limit):
    """Findet Near-Solutions, die nicht zur bekannten Familie gehören."""
    solutions = []
    
    for w1 in prange(1, range_limit):
        for w2 in range(1, range_limit):
            for x in range(1, range_limit):
                for y in range(x + 1, range_limit):  # y > x
                    for z in range(y + 1, range_limit):  # z > y
                        # Prüfe die 9 Bedingungen
                        conditions = [
                            (-x**2 + y**2, True),
                            (-x**2 + z**2, True),
                            (-y**2 + z**2, True),
                            (-w1**2 + x**2, False),
                            (-w1**2 + y**2, True),
                            (-w1**2 + z**2, True),
                            (-w2**2 + x**2, True),
                            (-w2**2 + y**2, True),
                            (-w2**2 + z**2, False)
                        ]

                        valid = 0
                        for num, check in conditions:
                            if is_perfect_square(num) == check:
                                valid += 1

                        # Falls mindestens 8 von 9 Bedingungen stimmen
                        if valid >= 8:
                            if not is_known_family(w1, w2, x, y, z):
                                solutions.append((w1, w2, x, y, z))

    return solutions

def process_range(range_limit):
    """Wrapper-Funktion, um Near-Solutions zu berechnen und auszugeben."""
    solutions = find_almost_solutions(range_limit)
    for sol in solutions:
        print(f"{sol[0]}, {sol[1]}, {sol[2]}, {sol[3]}, {sol[4]}")

if __name__ == "__main__":
    range_limit = 500

    # Kopfzeile ausgeben
    print("w1, w2, x, y, z", flush=True)

    with Pool() as pool:
        pool.map(process_range, [range_limit])