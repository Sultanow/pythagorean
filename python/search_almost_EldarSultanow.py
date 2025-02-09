import math
from multiprocessing import Pool

def is_perfect_square(n):
    """Überprüft, ob n eine perfekte Quadratzahl ist."""
    if n < 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n

def generate_near_solutions(p, q):
    """Berechnet (w1, w2, x, y, z) und überprüft, ob es eine Near-Solution ist."""
    w1 = (p**4 - 9*q**4)**2 - 64*p**4*q**4
    w2 = 16*p**2 * (p**4 - 9*q**4)
    x = 4*p*q * (p**2 + 3*q**2) * (p**4 - 2*p**2*q**2 + 9*q**4)
    y = (p**4 - 9*q**4)**2 + 64*p**4*q**4
    z = (p**2 + q**2) * (p**2 + 9*q**2) * (p**4 - 2*p**2*q**2 + 9*q**4)

    # Prüfen, ob 5 von 6 Gleichungen Quadrate sind
    checks = [
        (-x**2 + y**2, True),
        (-x**2 + z**2, True),
        (-y**2 + z**2, True),
        (-w1**2 + x**2, False),  # Eine dieser beiden soll *kein* Quadrat sein
        (-w1**2 + y**2, True),
        (-w1**2 + z**2, True),
        (-w2**2 + x**2, True),
        (-w2**2 + y**2, True),
        (-w2**2 + z**2, False)  # Diese ebenfalls nicht
    ]

    valid = sum(is_perfect_square(num) == check for num, check in checks)

    if valid >= 8:  # Falls mindestens 8 von 9 Bedingungen stimmen
        return (w1, w2, x, y, z)

    return None

def find_solutions(range_p, range_q):
    """Sucht nach Near-Solutions im Bereich p, q."""
    for p in range(1, range_p):
        for q in range(1, p):  # q < p, um doppelte Lösungen zu vermeiden
            solution = generate_near_solutions(p, q)
            if solution:
                print(f"{solution[0]}, {solution[1]}, {solution[2]}, {solution[3]}, {solution[4]}", flush=True)


if __name__ == "__main__":
    range_p = 100
    range_q = 100

    # Kopfzeile ausgeben
    print("w1, w2, x, y, z", flush=True)

    with Pool() as pool:
        pool.starmap(find_solutions, [(range_p, range_q)])