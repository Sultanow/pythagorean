import json

def extract_almost_solutions(input_file, output_file):
    """Liest die Datei ein und speichert alle Almost Solutions in eine neue Datei."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        outfile.write("w, x, y, z\n")

        for line in infile:
            try:
                data = json.loads(line.strip())  # JSON aus Zeile parsen
                if "res" in data and len(data["res"]) > 0:  # Prüfen, ob es ein Ergebnis gibt
                    for res_entry in data["res"]:
                        if all(key in res_entry for key in ["w", "x", "y", "z2"]):
                            w, x, y, z2 = res_entry["w"], res_entry["x"], res_entry["y"], res_entry["z2"]
                            outfile.write(f"{w}, {x}, {y}, {z2}\n")  # Speichert die Werte in die Datei

            except json.JSONDecodeError:
                print(f"Fehlerhafte Zeile (kein gültiges JSON): {line.strip()}")

if __name__ == "__main__":
    input_file = "C:/Users/esultano/git/pythagorean/data/almost_solutions_Arty_3M.txt"
    output_file = "near_solutions6.txt"
    extract_almost_solutions(input_file, output_file)
    print(f"Extraktion abgeschlossen! Die Almost Solutions wurden in '{output_file}' gespeichert.")
