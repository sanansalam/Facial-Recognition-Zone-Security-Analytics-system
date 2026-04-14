import os, sys, pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FACES_DIR    = PROJECT_ROOT / "data" / "extracted_faces"
MODELS_DIR   = PROJECT_ROOT / "models"

def load_insightface():
    from insightface.app import FaceAnalysis
    print("Loading InsightFace...")
    app = FaceAnalysis(
        name="buffalo_sc",
        root=str(MODELS_DIR / "insightface"),
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("InsightFace loaded.\n")
    return app

def get_best_embedding(app, folder):
    import cv2
    best_embedding = None
    best_score = 0.0
    best_img = None
    for i in range(1, 6):
        img_path = folder / f"sample_{i}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < 80 or w < 80:
            img = cv2.resize(img, (112, 112))
        try:
            faces = app.get(img)
        except Exception as e:
            continue
        if not faces:
            continue
        face = faces[0]
        score = float(face.det_score)
        if score > best_score:
            best_score = score
            best_embedding = face.embedding
            best_img = img_path.name
    return best_embedding, best_score, best_img

def main():
    if not FACES_DIR.exists():
        print(f"ERROR: {FACES_DIR} not found")
        sys.exit(1)

    folders = sorted([
        f for f in FACES_DIR.iterdir()
        if f.is_dir() and f.name.startswith("person_")
    ])

    if not folders:
        print("ERROR: No person folders found")
        sys.exit(1)

    print(f"Found {len(folders)} person folders.\n")

    try:
        app = load_insightface()
    except Exception as e:
        print(f"ERROR loading InsightFace: {e}")
        sys.exit(1)

    success = 0
    failed = 0
    skipped = 0

    for folder in folders:
        emb_path = folder / "embedding.pkl"
        if emb_path.exists():
            print(f"  [{folder.name}] Already exists — skipping")
            skipped += 1
            continue

        print(f"  [{folder.name}] Processing...", end=" ", flush=True)
        embedding, score, best_img = get_best_embedding(app, folder)

        if embedding is None:
            print(f"FAILED — no face detected")
            failed += 1
            continue

        with open(emb_path, "wb") as f:
            pickle.dump(embedding, f)
        print(f"OK — {best_img} confidence={score:.2f}")
        success += 1

    print(f"\n{'='*50}")
    print(f"Success: {success}  Failed: {failed}  Skipped: {skipped}")
    print(f"\nNext: python3 scripts/enroll_persons.py")

if __name__ == "__main__":
    main()
