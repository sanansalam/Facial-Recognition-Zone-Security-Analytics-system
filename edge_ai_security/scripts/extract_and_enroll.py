"""
extract_and_enroll.py
Extracts unique faces directly from video and enrolls them.

Run from project root:
  python3 scripts/extract_and_enroll.py
"""

import os, sys, json, sqlite3, pickle
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_PATH   = PROJECT_ROOT / "data" / "videos" / \
    "219_8_Jewellery_IPC6_aef7458404ad4b2c89a99ec41882bdca_20260214182622.avi"
FACES_DIR    = PROJECT_ROOT / "data" / "extracted_faces"
MODELS_DIR   = PROJECT_ROOT / "models"
DB_PATH      = PROJECT_ROOT / "data" / "security.db"

# ── Settings ───────────────────────────────────────────────────────────────────
SAMPLE_EVERY_N_SECONDS = 2    # grab one frame every N seconds
MIN_FACE_SIZE          = 80   # minimum face width/height in pixels
DET_THRESHOLD          = 0.35 # face detection confidence threshold
SIMILARITY_THRESHOLD   = 0.5  # below this = different person

ROLES = [
    "Manager", "Senior Staff", "Sales Staff",
    "Cashier", "Security Guard", "Cleaner",
    "Customer", "Unknown"
]

def load_insightface():
    from insightface.app import FaceAnalysis
    print("Loading InsightFace (RetinaFace + ArcFace)...")
    app = FaceAnalysis(
        name="buffalo_sc",
        root=str(MODELS_DIR / "insightface"),
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("Loaded.\n")
    return app

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def is_new_person(embedding, known_embeddings):
    for known_emb in known_embeddings:
        if cosine_similarity(embedding, known_emb) > SIMILARITY_THRESHOLD:
            return False
    return True

def extract_faces_from_video(app):
    print(f"Opening video: {VIDEO_PATH.name}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("ERROR: Cannot open video file")
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    step     = int(fps * SAMPLE_EVERY_N_SECONDS)

    print(f"Video: {duration:.0f}s at {fps:.0f}fps")
    print(f"Sampling every {SAMPLE_EVERY_N_SECONDS}s "
          f"({total_frames // step} frames to check)\n")

    FACES_DIR.mkdir(parents=True, exist_ok=True)

    known_embeddings = []
    persons = []
    frame_idx = 0
    checked = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        checked += 1
        timestamp = frame_idx / fps

        # Resize to 640x640 for inference
        frame_resized = cv2.resize(frame, (640, 640))

        try:
            faces = app.get(frame_resized)
        except Exception as e:
            continue

        for face in faces:
            if float(face.det_score) < DET_THRESHOLD:
                continue

            # Check face size
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            w = x2 - x1
            h = y2 - y1
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            embedding = face.embedding

            if not is_new_person(embedding, known_embeddings):
                continue

            # New unique person found
            person_id = len(persons) + 1
            folder = FACES_DIR / f"person_{person_id:02d}"
            folder.mkdir(exist_ok=True)

            # Save face crop
            pad = 20
            fx1 = max(0, x1 - pad)
            fy1 = max(0, y1 - pad)
            fx2 = min(frame_resized.shape[1], x2 + pad)
            fy2 = min(frame_resized.shape[0], y2 + pad)
            face_crop = frame_resized[fy1:fy2, fx1:fx2]
            cv2.imwrite(str(folder / "face.jpg"), face_crop)

            # Save full frame with face highlighted
            annotated = frame_resized.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"Person {person_id}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            cv2.imwrite(str(folder / "context.jpg"), annotated)

            # Save embedding
            with open(folder / "embedding.pkl", "wb") as f:
                pickle.dump(embedding, f)

            # Save info
            with open(folder / "info.txt", "w") as f:
                f.write(f"Person {person_id}\n")
                f.write(f"First seen: {timestamp:.1f}s\n")
                f.write(f"Face confidence: {face.det_score:.2f}\n")
                f.write(f"Face size: {w}x{h}px\n")

            known_embeddings.append(embedding)
            persons.append({
                "id": person_id,
                "folder": folder,
                "timestamp": timestamp,
                "confidence": float(face.det_score)
            })

            print(f"  → New person {person_id:02d} at {timestamp:.1f}s "
                  f"(conf={face.det_score:.2f} size={w}x{h})")

        if checked % 10 == 0:
            pct = (frame_idx / total_frames) * 100
            print(f"  Progress: {pct:.0f}% | "
                  f"Unique persons found: {len(persons)}")

    cap.release()
    print(f"\nDone. Found {len(persons)} unique persons.")
    return persons

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

def enroll(name, role, folder_name, embedding):
    emb_json = json.dumps(
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    )
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO persons (name, role, embedding, person_folder) "
        "VALUES (?,?,?,?)",
        (name, role, emb_json, folder_name)
    )
    conn.commit()
    conn.close()

def pick_role():
    for i, role in enumerate(ROLES, 1):
        print(f"    {i}. {role}")
    while True:
        choice = input("\n  Role number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(ROLES):
            return ROLES[int(choice)-1]
        print("  Invalid. Try again.")

def enroll_persons(persons):
    print("\n" + "="*55)
    print("  ENROLLMENT — Label each unique person")
    print("="*55)
    print("  Controls: type name → enroll | s → skip | q → quit\n")

    init_db()
    enrolled = 0

    for i, person in enumerate(persons, 1):
        folder = person["folder"]
        print(f"\n{'─'*55}")
        print(f"  Person {person['id']:02d}  |  "
              f"First seen at {person['timestamp']:.1f}s  |  "
              f"Confidence: {person['confidence']:.2f}")
        print(f"  Progress: {i}/{len(persons)}\n")
        print(f"  Face image:    {folder}/face.jpg")
        print(f"  In context:    {folder}/context.jpg")

        # Try to open image
        os.system(f'xdg-open "{folder}/face.jpg" 2>/dev/null &')

        print()
        action = input("  Name (s=skip, q=quit): ").strip()

        if action.lower() == "q":
            break
        if action.lower() == "s" or action == "":
            with open(folder / "embedding.pkl", "rb") as f:
                emb = pickle.load(f)
            enroll("Unknown", "Customer", folder.name, emb)
            print("  Saved as Customer")
            enrolled += 1
            continue

        name = action
        print("\n  Roles:")
        role = pick_role()

        with open(folder / "embedding.pkl", "rb") as f:
            emb = pickle.load(f)
        enroll(name, role, folder.name, emb)
        print(f"\n  ✓ Enrolled: {name} ({role})")
        enrolled += 1

    print(f"\n{'='*55}")
    print(f"Enrollment complete. {enrolled} persons saved to database.")

    # Print summary
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT name, role FROM persons ORDER BY role, name"
    ).fetchall()
    conn.close()
    if rows:
        print(f"\n{'Name':<25} {'Role'}")
        print(f"{'-'*25} {'-'*20}")
        for name, role in rows:
            print(f"{name:<25} {role}")

def main():
    print("\n" + "="*55)
    print("  FACE EXTRACTION + ENROLLMENT")
    print("  Jewellery Shop Security System")
    print("="*55 + "\n")

    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        sys.exit(1)

    # Clean previous extraction
    if FACES_DIR.exists():
        import shutil
        shutil.rmtree(FACES_DIR)
        print("Cleaned previous extraction.\n")

    app = load_insightface()
    persons = extract_faces_from_video(app)

    if not persons:
        print("\nNo faces found. Try lowering DET_THRESHOLD or MIN_FACE_SIZE.")
        sys.exit(1)

    enroll_persons(persons)

if __name__ == "__main__":
    main()
