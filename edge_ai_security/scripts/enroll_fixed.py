import os, sys, json, sqlite3, shutil, logging
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Suppress Qt font warnings and other noise
os.environ["QT_QPA_PLATFORM"] = "offscreen" if "DISPLAY" not in os.environ else ""
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts=false"

BASE_DIR = "/home/sana/Bank project/edge_ai_security"
os.chdir(BASE_DIR)

# --- LOAD SOURCES FROM .env ---
VIDEOS = []
VIDEO_LABELS = []
VIDEO_CAM_IDS = []

env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        env_lines = f.readlines()
    
    env_dict = {}
    for line in env_lines:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env_dict[k.strip()] = v.strip().strip('"').strip("'")
    
    for i in range(15):
        src_key = f"CAM_{i}_SOURCE"
        lbl_key = f"CAM_{i}_LABEL"
        if src_key in env_dict:
            src = env_dict[src_key]
            if src.startswith("/data/"):
                src = src.replace("/data/", "data/", 1)
            
            VIDEOS.append(src)
            VIDEO_LABELS.append(env_dict.get(lbl_key, f"Camera {i}"))
            VIDEO_CAM_IDS.append(f"cam_{i}")

if not VIDEOS:
    print("!!! NO VIDEOS FOUND IN .env. Falling back to default.")
    VIDEOS = ["data/test_videos/cctv_sample.mp4"]
    VIDEO_LABELS = ["Test Cam"]
    VIDEO_CAM_IDS = ["cam_0"]
else:
    print(f"Loaded {len(VIDEOS)} sources from .env.")

MODEL_ROOT     = "models/insightface"
_data_dir = os.environ.get("DATA_DIR", "data")
DB_PATH   = os.path.join(_data_dir, "security.db")
FACES_DIR      = "data/extracted_faces"
SAMPLE_EVERY_N = 30
DET_THRESH     = 0.45
MIN_FACE_PX    = 35
MIN_APPEAR     = 2
CLUSTER_THRESH = 0.50   # 0.55 was a bit too strict, 0.50 merges more duplicates
ROLES = ["Manager","Staff","Cashier","Security","Cleaner","Customer","Unknown"]

print("=" * 55)
print(" FACE EXTRACTION + ENROLLMENT TOOL")
print(" 3 videos | tighter clustering")
print("=" * 55)
print()
print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_sc",
    root=MODEL_ROOT,
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_thresh=DET_THRESH, det_size=(1280, 1280))
print("Model loaded (Optimized for 1280x1280).\n")

print("--- START OPTIONS ---")
print("  0. CLEAR EXISTING DATABASE (Start from scratch)")
print("  1. Full Run (Extraction -> Labeling -> Zones)")
print("  3. Skip to Labeling (Uses existing extracted faces)")
print("  4. Skip to Zone Drawing (Uses existing database)")
print("  5. Skip to Access Assignment (Use existing zones)")
try:
    START_STAGE = int(input("Select stage to start from [1]: ").strip() or "1")
except ValueError:
    START_STAGE = 1

clusters = []

def add_to_cluster(emb, crop, cam_label):
    best_c   = None
    best_sim = -1.0
    for c in clusters:
        sim = float(np.dot(emb, c["centroid"]))
        if sim > best_sim:
            best_sim = sim
            best_c   = c
    if best_c and best_sim >= CLUSTER_THRESH:
        if len(best_c["embeddings"]) < 20:
            best_c["embeddings"].append(emb)
        cent = np.mean(best_c["embeddings"], axis=0)
        n    = np.linalg.norm(cent)
        best_c["centroid"] = cent/n if n > 0 else cent
        best_c["count"] += 1
        best_c["cams"].add(cam_label)
        if len(best_c["samples"]) < 5:
            best_c["samples"].append({"crop": crop, "cam": cam_label})
    else:
        clusters.append({
            "id":         len(clusters),
            "embeddings": [emb],
            "centroid":   emb.copy(),
            "samples":    [{"crop": crop, "cam": cam_label}],
            "count":      1,
            "cams":       set([cam_label]),
        })

# --- STAGE 1: Video Extraction ---
if START_STAGE <= 1:
    print("=" * 55)
    print(" STAGE 1: Extracting faces from videos")
    print(f" CLUSTER_THRESH={CLUSTER_THRESH} MIN_APPEAR={MIN_APPEAR}")
    print("=" * 55)

    for vf, cam_label in zip(VIDEOS, VIDEO_LABELS):
        if not os.path.exists(vf):
            print(f"  [{cam_label}] NOT FOUND: {vf}")
            continue

        cap   = cv2.VideoCapture(vf)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n[{cam_label}] {total/fps:.0f}s | {W}x{H}")
        print(f"  Sampling every {SAMPLE_EVERY_N} frames")

        frame_no, sampled, found = 0, 0, 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_no += 1
            if frame_no % SAMPLE_EVERY_N != 0: continue
            sampled += 1

            faces = app.get(frame)
            for face in faces:
                x1,y1,x2,y2 = face.bbox.astype(int)
                if (x2-x1) < MIN_FACE_PX: continue
                if face.normed_embedding is None: continue
                pad = 15
                fx1, fy1 = max(0, x1-pad), max(0, y1-pad)
                fx2, fy2 = min(W, x2+pad), min(H, y2+pad)
                crop = frame[fy1:fy2, fx1:fx2].copy()
                if crop.size > 0: crop = cv2.resize(crop, (112,112))
                add_to_cluster(face.normed_embedding, crop, cam_label)
                found += 1

            if sampled % 20 == 0:
                print(f"  {int(frame_no/total*100)}% | faces={found} | clusters={len(clusters)}", end="\r", flush=True)

        cap.release()
        print(f"\n  Done: {found} faces | {len(clusters)} clusters total")

    before = len(clusters)
    clusters = [c for c in clusters if c["count"] >= MIN_APPEAR]
    print(f"\n Extraction complete! Before: {before}, After filter: {len(clusters)}")

    if not clusters:
        print("No persons found. Stopping.")
        sys.exit(0)

    if os.path.exists(FACES_DIR): shutil.rmtree(FACES_DIR)
    os.makedirs(FACES_DIR)

    for c in clusters:
        pid = c["id"] + 1
        pdir = os.path.join(FACES_DIR, f"person_{pid:02d}")
        os.makedirs(pdir)
        for i, s in enumerate(c["samples"]):
            cv2.imwrite(os.path.join(pdir, f"s{i+1}.jpg"), s["crop"])
        c["samples"] = [] # Clear RAM
        with open(os.path.join(pdir,"info.txt"),"w") as f:
            f.write(f"Person {pid}\nAppearances: {c['count']}\nCameras: {sorted(c['cams'])}\n")
    print(f"Face photos saved to: {FACES_DIR}/\n")

# Stage 1 Complete. FREE UP MEMORY (InsightFace takes a lot of RAM)
if 'app' in globals():
    del app
import gc
gc.collect()

if START_STAGE <= 3:
    print()
    input("Press ENTER when ready to start labeling...")

print(f"\n{'='*55}")
print(f" STAGE 3: Label {len(clusters)} persons")
print(f"{'='*55}\n")

# Stage 3 uses a database to store results. We allow resuming by not deleting it.
print(f"Using database: {DB_PATH}")
os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
db = sqlite3.connect(DB_PATH, timeout=30)
db.executescript("""
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    embedding TEXT NOT NULL,
    face_image TEXT,
    appearances INTEGER DEFAULT 0,
    cameras_seen TEXT,
    cluster_id INTEGER DEFAULT -1,
    active INTEGER DEFAULT 1,
    enrolled_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    cam_id TEXT NOT NULL,
    polygon_points TEXT DEFAULT '[]',
    restricted_roles TEXT DEFAULT '[]',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS zone_access (
    person_id INTEGER,
    zone_id INTEGER,
    granted_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (person_id, zone_id)
)
""")

# Migration: add any missing columns for backward compatibility
cursor = db.execute("PRAGMA table_info(persons)")
columns = [row[1] for row in cursor.fetchall()]
if "cluster_id" not in columns:
    db.execute("ALTER TABLE persons ADD COLUMN cluster_id INTEGER DEFAULT -1")
if "active" not in columns:
    db.execute("ALTER TABLE persons ADD COLUMN active INTEGER DEFAULT 1")
    db.execute("UPDATE persons SET active=1 WHERE active IS NULL")
# Migrate old cam_label column to cam_id in zones table (if from old schema)
cursor = db.execute("PRAGMA table_info(zones)")
zone_cols = [row[1] for row in cursor.fetchall()]
if "cam_label" in zone_cols and "cam_id" not in zone_cols:
    db.execute("ALTER TABLE zones ADD COLUMN cam_id TEXT")
    db.execute("UPDATE zones SET cam_id=cam_label")
db.commit()

if START_STAGE == 0:
    confirm = input("\n!!! WARNING: This will DELETE all existing names and zones. Are you sure? (y/n): ").strip().lower()
    if confirm == 'y':
        db.execute("DELETE FROM persons")
        db.execute("DELETE FROM zones")
        db.execute("DELETE FROM zone_access")
        db.commit()
        print("Database cleared. Starting Full Run...")
        START_STAGE = 1
    else:
        print("Abort.")
        sys.exit(0)

# Show enrolled list once to help user remember what names were used
already = db.execute("SELECT id, name, role FROM persons WHERE active=1 AND role!='FALSE_POSITIVE'").fetchall()
if already:
    print("\n--- ALREADY ENROLLED ---")
    for p in already:
        print(f"  [{p[0]}] {p[1]} ({p[2]})")
    print("------------------------\n")

enrolled = {}

if START_STAGE <= 3:
    for i, c in enumerate(clusters):
        cid  = c["id"]
        pid  = cid + 1
        seen = c["count"]
        cams = sorted(c["cams"])

        # Resume check
        existing = db.execute("SELECT name, role FROM persons WHERE cluster_id = ?", (cid,)).fetchone()
        if existing:
            print(f"--- Person {i+1} of {len(clusters)} (ID {pid}) ---")
            print(f"  ALREADY ENROLLED: {existing[0]} ({existing[1]})")
            enrolled[cid] = True
            continue

        print(f"--- Person {i+1} of {len(clusters)} (ID {pid}) ---")
        print(f"  Seen    : {seen}x")
        print(f"  Cameras : {cams}")
        print(f"  Photo   : {FACES_DIR}/person_{pid:02d}/s1.jpg")

        # Build Gallery View (Tile up to 5 photos)
        pdir = os.path.join(FACES_DIR, f"person_{pid:02d}")
        photos = []
        if os.path.exists(pdir):
            for fname in sorted(os.listdir(pdir)):
                if fname.endswith(".jpg"):
                    img = cv2.imread(os.path.join(pdir, fname))
                    if img is not None:
                        # Resize to standard size for tiling
                        img = cv2.resize(img, (150, 150))
                        # Add a small border
                        img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(50, 50, 50))
                        photos.append(img)
        
        if photos:
            # Tile images horizontally
            gallery = np.hstack(photos[:5])
            win_title = f"Person {i+1}/{len(clusters)} | {seen} appearances"
            cv2.imshow(win_title, gallery)
            # Move window to top left so it doesn't block terminal
            cv2.moveWindow(win_title, 10, 10)
            cv2.waitKey(500)

        name = input("  Name (or skip/reject): ").strip()
        if name.lower() == "skip" or not name:
            cv2.destroyAllWindows()
            print("  Skipped.\n")
            continue

        if name.lower() == "reject":
            cv2.destroyAllWindows()
            db.execute("INSERT INTO persons (name, role, embedding, cluster_id, active) VALUES (?, ?, ?, ?, 0)", 
                       ("REJECTED", "FALSE_POSITIVE", "[]", cid))
            db.commit()
            print("  Permanently rejected.\n")
            continue

        # -- Smart Merging Logic --
        existing_p = db.execute("SELECT id, role, embedding, appearances FROM persons WHERE name = ? AND role != 'FALSE_POSITIVE' LIMIT 1", (name,)).fetchone()
        
        role = "Unknown"
        if existing_p:
            e_id, e_role, e_emb_json, e_apps = existing_p
            print(f"  !!! Person '{name}' already exists (ID {e_id}, Role: {e_role})")
            merge = input(f"  Merge this cluster into ID {e_id}? (y/n): ").strip().lower()
            if merge == 'y':
                role = e_role
                new_emb = max(c["embeddings"], key=lambda e: float(np.dot(e, c["centroid"])))
                try:
                    old_emb = np.array(json.loads(e_emb_json))
                    # Weighted average for the new embedding
                    merged_emb = (old_emb * e_apps + new_emb) / (e_apps + 1)
                    merged_emb = merged_emb / np.linalg.norm(merged_emb)
                    db.execute("UPDATE persons SET embedding = ?, appearances = appearances + ?, cluster_id = -1 WHERE id = ?",
                               (json.dumps(merged_emb.tolist()), seen, e_id))
                    db.commit()
                    print(f"  Merged into ID {e_id}. Appearances updated.\n")
                    cv2.destroyAllWindows()
                    enrolled[cid] = e_id
                    continue
                except:
                    print("  Merge failed (embedding error). Creating new entry.")
            else:
                print("  Creating separate entry for this person.")
        
        if role == "Unknown":
            print("  Role:")
            for idx, r in enumerate(ROLES):
                print(f"    {idx+1}. {r}")
            while True:
                try:
                    ch = int(input(f"  Select 1-{len(ROLES)}: "))
                    if 1 <= ch <= len(ROLES):
                        role = ROLES[ch-1]
                        break
                except ValueError: pass
                print("  Enter a number.")

        cv2.destroyAllWindows()
        best_emb = max(c["embeddings"], key=lambda e: float(np.dot(e, c["centroid"])))
        img_path = os.path.join(FACES_DIR, f"person_{pid:02d}", "s1.jpg")
        cur = db.execute(
            "INSERT INTO persons (name,role,embedding,face_image,appearances,cameras_seen,cluster_id,active) VALUES(?,?,?,?,?,?,?,1)",
            (name, role, json.dumps(best_emb.tolist()), img_path, seen, json.dumps(cams), cid)
        )
        db.commit()
        enrolled[cid] = cur.lastrowid
        print(f"  Saved: {name} | {role} | ID={cur.lastrowid} | active=1\n")

        # Aggressive cleanup after each person
        import gc
        gc.collect()

    print(f" Enrolled {len(enrolled)} persons.")

if START_STAGE <= 4:
    print(f"\n{'='*55}")
    print(" STAGE 4: Draw zones on camera frames")
    print(f"{'='*55}")
    print(" LEFT CLICK = add point")
    print(" ENTER = save zone")
    print(" R = clear and redraw")
    print(" S = skip this camera")
    input("\nPress ENTER to start drawing zones...")

    zone_pts = []
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            zone_pts.append([x, y])

    zone_ids = []

    for vf, cam_label, cam_id in zip(VIDEOS, VIDEO_LABELS, VIDEO_CAM_IDS):
        if not os.path.exists(vf):
            continue
        cap   = cv2.VideoCapture(vf)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total*0.3))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        orig_h, orig_w = frame.shape[:2]
        disp = cv2.resize(frame, (1280, 720))
        sx = orig_w / 1280
        sy = orig_h / 720
        print(f"\n[{cam_label}] Draw zone boundaries.")

        while True:
            zone_pts.clear()
            win = f"[{cam_label}] Click zone | ENTER=save 'A'=AUTO R=redo S=skip"
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, on_mouse)

            while True:
                d = disp.copy()
                for pt in zone_pts:
                    cv2.circle(d, tuple(pt), 5, (0,255,0), -1)
                if len(zone_pts) >= 2:
                    pts = np.array(zone_pts, np.int32)
                    cv2.polylines(d, [pts.reshape(-1,1,2)], len(zone_pts)>=3, (0,255,255), 2)
                cv2.putText(d, f"Pts:{len(zone_pts)} ENTER:Save 'A':AUTO(Full) R:Redo S:Skip",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow(win, d)
                key = cv2.waitKey(30) & 0xFF
                if key == 13 and len(zone_pts) >= 3:
                    break
                elif key == ord("a"):
                    # Auto-zone: Full frame
                    zone_pts = [[0,0], [1280,0], [1280,720], [0,720]]
                    break
                elif key == ord("r"):
                    zone_pts.clear()
                elif key == ord("s"):
                    zone_pts.clear()
                    break

            cv2.destroyAllWindows()
            if len(zone_pts) < 3:
                print(f"  Skipped {cam_label}.")
                break

            poly  = [[int(p[0]*sx), int(p[1]*sy)] for p in zone_pts]
            zname = input("  Zone name: ").strip() or f"Zone_{cam_id}"
            print("  Restricted roles (numbers, comma-sep, ENTER=none):")
            for i,r in enumerate(ROLES):
                print(f"    {i+1}. {r}")
            restr = []
            rs = input("  Restricted: ").strip()
            if rs:
                for x in rs.split(","):
                    try:
                        idx = int(x.strip())-1
                        if 0 <= idx < len(ROLES):
                            restr.append(ROLES[idx])
                    except ValueError: pass

            cur = db.execute(
                "INSERT INTO zones (name,cam_id,polygon_points,restricted_roles) VALUES(?,?,?,?)",
                (zname, cam_id, json.dumps(poly), json.dumps(restr))
            )
            db.commit()
            zid = cur.lastrowid
            zone_ids.append(zid)
            print(f"  Saved: {zname} | cam_id={cam_id} | ID={zid} | Restricted:{restr}")

            more = input(f"  Add another zone on {cam_label}? (y/n): ").strip().lower()
            if more != "y":
                break

if START_STAGE <= 5:
    persons_db = db.execute("SELECT * FROM persons").fetchall()
    zones_db   = db.execute("SELECT * FROM zones").fetchall()

    print(f"\n{'='*55}")
    print(" STAGE 5: Assign zone access per person")
    print(f"{'='*55}")

    if persons_db and zones_db:
        for p in persons_db:
            print(f"\n  {p[1]} | Role: {p[2]}")
            for i,z in enumerate(zones_db):
                print(f"    {i+1}. {z[1]} (cam:{z[2]})")
            rs = input("  Allowed zones (comma-sep, ENTER=none): ").strip()
            if rs:
                for x in rs.split(","):
                    try:
                        idx = int(x.strip())-1
                        if 0 <= idx < len(zones_db):
                            db.execute("INSERT OR IGNORE INTO zone_access (person_id,zone_id) VALUES(?,?)",
                                       (p[0], zones_db[idx][0]))
                            db.commit()
                            print(f"  Granted: {p[1]} -> {zones_db[idx][1]}")
                    except ValueError:
                        pass
    else:
        print("  No zones or persons — skipping.")

    pf = db.execute("SELECT id,name,role,appearances FROM persons").fetchall()
    zf = db.execute("SELECT id,name,cam_id FROM zones").fetchall()
    af = db.execute("SELECT * FROM zone_access").fetchall()

    print(f"\n{'='*55}")
    print(" ENROLLMENT COMPLETE")
    print(f"{'='*55}")
    print(f" Persons : {len(pf)}")
    for p in pf:
        print(f"   [{p[0]}] {p[1]} | {p[2]} | seen {p[3]}x")
    print(f" Zones   : {len(zf)}")
    for z in zf:
        print(f"   [{z[0]}] {z[1]} | cam={z[2]}")
    print(f" Access  : {len(af)} rules")
    print(f" DB      : {DB_PATH}")
    print(f"{'='*55}")
db.close()
