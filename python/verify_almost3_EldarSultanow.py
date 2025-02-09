import math

def is_perfect_square(n):
    """Überprüft, ob eine Zahl eine perfekte Quadratzahl ist."""
    if n < 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n

def verify_solution(w, x, y, z2):
    """Prüft, ob mindestens 5 von 6 Bedingungen erfüllt sind."""
    conditions = [
        (-x**2 + y**2),
        (-x**2 + z2),
        (-y**2 + z2),
        (-w**2 + x**2),
        (-w**2 + y**2),
        (-w**2 + z2)
    ]

    valid = sum(1 for num in conditions if is_perfect_square(num))

    return valid >= 5  # Mindestens 5 Bedingungen müssen erfüllt sein

def belongs_to_family(w, x, y, z2):
    """Prüft, ob der gefundene Wert zur bekannten Familie gehört."""
    for p in range(1, 500):  # Wir testen p bis 500
        q = 1  # q ist fest auf 1 gesetzt

        w1_known = (p**4 - 9*q**4)**2 - 64*p**4*q**4
        x_known = 16*p**2 * (p**4 - 9*q**4)
        y_known = 4*p*q * (p**2 + 3*q**2) * (p**4 - 2*p**2*q**2 + 9*q**4)
        z2_known = (p**2 + q**2) * (p**2 + 9*q**2) * (p**4 - 2*p**2*q**2 + 9*q**4)

        if {w, x, y, z2} == {w1_known, x_known, y_known, z2_known}:
            return True  # Gehört zur Familie

    return False  # Kein passendes p gefunden

def verify_almost_solutions(input_file, output_file):
    """Liest die Datei ein, überprüft jede Zeile und speichert die validen Lösungen mit Familienzugehörigkeit."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        header = infile.readline().strip()  # Kopfzeile überspringen
        outfile.write(header + ", valid, family_member\n")  # Neue Datei mit zusätzlichen Spalten

        for line in infile:
            try:
                w, x, y, z2 = map(int, line.strip().split(", "))  # Werte einlesen
                is_valid = verify_solution(w, x, y, z2)  # Prüfen ob mindestens 5 erfüllt sind
                is_family = belongs_to_family(w, x, y, z2)  # Prüfen ob zur Familie gehörig

                outfile.write(f"{w}, {x}, {y}, {z2}, {is_valid}, {is_family}\n")  # Ergebnis speichern

            except ValueError:
                print(f"⚠ Fehlerhafte Zeile: {line.strip()}")

if __name__ == "__main__":
    input_file = "near_solutions6.txt"
    output_file = "verified_solutions6.txt"
    verify_almost_solutions(input_file, output_file)
    print(f"Verifizierung abgeschlossen! Die validierten Lösungen wurden in '{output_file}' gespeichert.")
