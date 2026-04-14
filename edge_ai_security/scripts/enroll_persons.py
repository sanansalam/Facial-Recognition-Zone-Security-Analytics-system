"""
enroll_persons.py
Terminal enrollment tool — label each face cluster and save to database.

Run from project root:
  python3 scripts/enroll_persons.py

Controls:
  Type name + Enter → enroll person
  s + Enter         → skip (save as Customer)
  q + Enter         → quit
"""

import os
import sys
import json
import sqlite3
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FACES_DIR    = PROJECT_ROOT / "data" / "extracted_faces"
DB_PATH      = PROJECT_ROOT / "data" / "security.db"

ROLES = [
    "Manager", "Senior Staff", "Sales Staff",
    "Cashier", "Security Guard", "Cleaner",
    "Customer", "Unknown",
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            role          TEXT NOT NULL,
            embedding     TEXT NOT NULL,
            person_folder TEXT,
            enrolled_at   TEXT DEFAULT (datetime('now')),
            active        INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database: {DB_PATH}\n")

def get_enrolled_folders():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT person_folder FROM persons WHERE person_folder IS NOT NULL"
        ).fetchall()
        return {r[0] for r in rows}
    except:
        return set()
    finally:
        conn.close()

def enroll(name, role, folder_name, embedding):
    emb_json = json.dumps(
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    )
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO persons (name, role, embedding, person_folder) VALUES (?,?,?,?)",
        (name, role, emb_json, folder_name)
    )
    conn.commit()
    conn.close()

def count_enrolled():
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    except:
        return 0
    finally:
        conn.close()

def read_info(folder):
    info = {"appearances": "?", "cameras": "?"}
    info_file = folder / "info.txt"
    if info_file.exists():
        for line in info_file.read_text().splitlines():
            if "Appearances:" in line:
                info["appearances"] = line.split(":")[1].strip()
            elif "Cameras:" in line:
                info["cameras"] = line.split(":")[1].strip()
    return info

def pick_role():
    print("\n  Roles:")
    for i, role in enumerate(ROLES, 1):
        print(f"    {i}. {role}")
    while True:
        choice = input("\n  Enter role number: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(ROLES):
                return ROLES[idx]
        print("  Invalid. Try again.")

def print_summary():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT name, role FROM persons ORDER BY role, name"
        ).fetchall()
    except:
        rows = []
    finally:
        conn.close()
    if not rows:
        return
    print("\n" + "="*50)
    print("ENROLLED PERSONS:")
    print(f"  {'Name':<25} {'Role'}")
    print(f"  {'-'*25} {'-'*20}")
    for name, role in rows:
        print(f"  {name:<25} {role}")

def main():
    print("\n" + "="*55)
    print("  FACE ENROLLMENT — Jewellery Shop Security")
    print("="*55 + "\n")

    if not FACES_DIR.exists():
        print(f"ERROR: {FACES_DIR} not found")
        sys.exit(1)

    init_db()

    folders = sorted([
        f for f in FACES_DIR.iterdir()
        if f.is_dir() and f.name.startswith("person_")
           and (f / "embedding.pkl").exists()
    ])

    if not folders:
        print("ERROR: No embeddings found.")
        print("Run first: python3 scripts/generate_embeddings.py")
        sys.exit(1)

    enrolled_folders = get_enrolled_folders()
    pending = [f for f in folders if f.name not in enrolled_folders]

    print(f"Total persons with embeddings: {len(folders)}")
    print(f"Already enrolled:              {len(enrolled_folders)}")
    print(f"Pending:                       {len(pending)}")

    if not pending:
        print("\nAll persons already enrolled!")
        print_summary()
        return

    print("\nControls: type name → enroll | s → skip | q → quit\n")

    enrolled_now = 0

    for i, folder in enumerate(pending, 1):
        print("\n" + "─"*55)
        info = read_info(folder)
        print(f"\n  {folder.name.upper()}  |  "
              f"Seen {info['appearances']} times  |  "
              f"Camera: {info['cameras']}")
        print(f"  Progress: {i}/{len(pending)}  "
              f"({count_enrolled()} total enrolled)\n")
        print(f"  View images:")
        for j in range(1, 6):
            img = folder / f"sample_{j}.jpg"
            if img.exists():
                print(f"    {img}")

        # Try to open image viewer
        img1 = folder / "sample_1.jpg"
        if img1.exists():
            os.system(f'xdg-open "{img1}" 2>/dev/null &')

        print()
        action = input("  Name (or 's' skip / 'q' quit): ").strip()

        if action.lower() == "q":
            print("\nQuitting. Progress saved.")
            break

        if action.lower() == "s" or action == "":
            with open(folder / "embedding.pkl", "rb") as f:
                embedding = pickle.load(f)
            enroll("Unknown", "Customer", folder.name, embedding)
            print(f"  Saved as Customer")
            enrolled_now += 1
            continue

        name = action
        role = pick_role()

        with open(folder / "embedding.pkl", "rb") as f:
            embedding = pickle.load(f)

        enroll(name, role, folder.name, embedding)
        print(f"\n  Enrolled: {name} ({role})")
        enrolled_now += 1

    print("\n" + "="*55)
    print(f"Session done. Enrolled: {enrolled_now}  "
          f"Total: {count_enrolled()}")
    print_summary()

if __name__ == "__main__":
    main()
