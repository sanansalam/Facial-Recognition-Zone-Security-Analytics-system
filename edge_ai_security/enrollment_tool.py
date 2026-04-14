#!/usr/bin/env python3
"""
enrollment_tool.py
Edge AI CCTV Security System — Enrollment Tool

Stage 1: Extract faces from all 4 videos
Stage 2: Cluster similar faces (same person = 1 cluster)
Stage 3: You label each cluster (name + role)
Stage 4: Draw zones on camera frame + assign access

Run with: python3 enrollment_tool.py
Dependencies: pip install opencv-python onnxruntime numpy
"""

import os
import sys
import cv2
import json
import time
import shutil
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# InsightFace via ONNX Runtime directly
# (no internet — loads from local model files only)
import onnxruntime as ort

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/home/sana/Bank project/edge_ai_security"

VIDEOS = [
    "data/videos/219_8_Jewellery_IPC6_aef7458404ad4b2c89a99ec41882bdca_20260214182622.avi",
    # "data/videos/219_8_Jewellery_IPC6_ef337a9e9bdf415ab7137c2db0852780_20260330125041.avi",
    # "data/videos/219_8_Jewellery_IPC6_a56efa055afe4b17b9564f37355d6077_20260214120324.avi",
    # "data/videos/219_8_Jewellery_IPC6_8891a760756e4bd580aec41b059e2f28_20260214192449.avi",
]

VIDEO_LABELS = ["cam_0", "cam_1", "cam_2", "cam_3"]

MODEL_DIR = "models/insightface/models/buffalo_sc"
DET_MODEL = "det_500m.onnx"       # RetinaFace
REC_MODEL = "w600k_mbf.onnx"     # ArcFace
DB_PATH   = "data/enrollment.db"
FACES_DIR = "data/extracted_faces"

SAMPLE_EVERY_N_FRAMES = 50       # 1 frame every 2 seconds at 25fps
DET_THRESHOLD         = 0.35     # RetinaFace confidence
CLUSTER_THRESHOLD     = 0.45     # cosine distance — same person
MIN_FACE_SIZE         = 30       # ignore tiny faces (pixels)
INPUT_SIZE            = (640, 640)

AVAILABLE_ROLES = [
    "Manager",
    "Staff",
    "Cashier",
    "Security",
    "Cleaner",
    "Customer",
    "Unknown",
]

# ============================================================
# CLASS: FaceProcessor
# ============================================================

class FaceProcessor:
    """
    Loads RetinaFace + ArcFace from local ONNX files.
    No internet. No insightface library needed.
    Uses onnxruntime directly.
    """

    def __init__(self, model_dir: str):
        det_path = os.path.join(model_dir, DET_MODEL)
        rec_path = os.path.join(model_dir, REC_MODEL)

        # Verify both model files exist
        for p in [det_path, rec_path]:
            if not os.path.exists(p):
                print(f"ERROR: Model not found: {p}")
                sys.exit(1)

        print(f"Loading RetinaFace from {DET_MODEL}...")
        self.det_session = ort.InferenceSession(
            det_path,
            providers=["CPUExecutionProvider"]
        )

        print(f"Loading ArcFace from {REC_MODEL}...")
        self.rec_session = ort.InferenceSession(
            rec_path,
            providers=["CPUExecutionProvider"]
        )

        # Get input names for both models
        self.det_input = self.det_session.get_inputs()[0].name
        self.rec_input = self.rec_session.get_inputs()[0].name

        print("Models loaded. Running on CPU.")

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Run RetinaFace (SCRFD det_500m) on a frame.
        Returns list of dicts:
          { bbox:[x1,y1,x2,y2], score:float, kps:[[x,y]x5] }
        Only returns faces larger than MIN_FACE_SIZE.
        """
        h, w = frame.shape[:2]

        resized = cv2.resize(frame, INPUT_SIZE)
        img = resized.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)[np.newaxis]

        raw_outputs = self.det_session.run(None, {self.det_input: img})

        scores_list = []
        bboxes_list = []
        kps_list = []

        for out in raw_outputs:
            arr = out.squeeze(0) if out.ndim == 3 else out.reshape(-1, out.shape[-1])
            last = arr.shape[-1] if arr.ndim == 2 else 1
            if last == 1:
                scores_list.append(arr.reshape(-1))
            elif last == 4:
                bboxes_list.append(arr.reshape(-1, 4))
            elif last == 10:
                kps_list.append(arr.reshape(-1, 10))

        scores_list.sort(key=lambda x: -x.shape[0])
        bboxes_list.sort(key=lambda x: -x.shape[0])
        kps_list.sort(key=lambda x: -x.shape[0])

        strides = [8, 16, 32]
        faces   = []

        for idx, stride in enumerate(strides):
            if idx >= len(scores_list) or idx >= len(bboxes_list):
                break

            scores_flat = scores_list[idx]
            bboxes_flat = bboxes_list[idx]
            kps_flat    = kps_list[idx] if idx < len(kps_list) else None

            feat_h, feat_w = INPUT_SIZE[1] // stride, INPUT_SIZE[0] // stride
            num_anchors = 2 if len(scores_flat) == (feat_h * feat_w * 2) else 1

            anchor_centers = []
            for row in range(feat_h):
                for col in range(feat_w):
                    for _ in range(num_anchors):
                        anchor_centers.append([(col + 0.5) * stride, (row + 0.5) * stride])
            anchor_centers = np.array(anchor_centers, dtype=np.float32)

            n = min(len(scores_flat), len(bboxes_flat), len(anchor_centers))
            for i in range(n):
                score = float(scores_flat[i])
                if score < DET_THRESHOLD:
                    continue

                cx, cy = anchor_centers[i]
                dx1, dy1, dx2, dy2 = bboxes_flat[i]

                # Distance to L, T, R, B
                x1 = (cx - dx1 * stride) * w / INPUT_SIZE[0]
                y1 = (cy - dy1 * stride) * h / INPUT_SIZE[1]
                x2 = (cx + dx2 * stride) * w / INPUT_SIZE[0]
                y2 = (cy + dy2 * stride) * h / INPUT_SIZE[1]

                if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                    continue

                face: dict = {
                    "bbox":  [int(x1), int(y1), int(x2), int(y2)],
                    "score": score,
                    "kps":   None,
                }

                if kps_flat is not None and i < len(kps_flat):
                    kps = kps_flat[i].reshape(5, 2).copy()
                    # Apply center offset to keypoints as well
                    kps[:, 0] = (kps[:, 0] * stride + cx) * w / INPUT_SIZE[0]
                    kps[:, 1] = (kps[:, 1] * stride + cy) * h / INPUT_SIZE[1]
                    face["kps"] = kps.astype(int).tolist()

                faces.append(face)

        # NMS to remove overlapping boxes
        return self._nms(faces)

    def _nms(self, faces: list, iou_threshold: float = 0.4) -> list:
        """
        Non-maximum suppression.
        Removes duplicate overlapping detections.
        """
        if not faces:
            return []

        faces = sorted(faces, key=lambda f: f["score"], reverse=True)
        keep = []
        for face in faces:
            suppress = False
            for kept in keep:
                if self._iou(face["bbox"], kept["bbox"]) > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                keep.append(face)
        return keep

    def _iou(self, box_a: list, box_b: list) -> float:
        """Intersection over union between two boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter  = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def get_embedding(self,
                      frame: np.ndarray,
                      bbox: list) -> np.ndarray:
        """
        Crop face region and run ArcFace.
        Returns L2-normalized 512-dim embedding.
        Returns None if crop is invalid.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add padding around face
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize to ArcFace input: 112x112
        face_img = cv2.resize(crop, (112, 112))
        # BGR → RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1]
        face_img = (face_img.astype(np.float32) - 127.5) / 128.0
        # HWC → NCHW
        face_img = face_img.transpose(2, 0, 1)[np.newaxis]

        # Run ArcFace
        output    = self.rec_session.run(None, {self.rec_input: face_img})
        embedding = output[0].flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)


# ============================================================
# CLASS: FaceClusterer
# ============================================================

class FaceClusterer:
    """
    Groups extracted face embeddings into clusters.
    Each cluster = one unique person.
    Uses greedy clustering: if cosine distance to existing
    cluster centroid < CLUSTER_THRESHOLD → same person.
    """

    def __init__(self):
        # Each cluster: list of embeddings + metadata
        # cluster = {
        #   "id": int,
        #   "embeddings": [np.array, ...],
        #   "centroid": np.array,
        #   "samples": [
        #     {"crop": np.array (small crop),
        #      "cam_label": str,
        #      "video_idx": int,
        #      "frame_no": int,
        #      "score": float}
        #   ]
        # }
        self.clusters = []

    def add_face(self,
                 embedding:  np.ndarray,
                 frame_crop: np.ndarray,
                 cam_label:  str,
                 video_idx:  int,
                 frame_no:   int,
                 score:      float):
        """
        Add a face embedding to the best matching cluster.
        Creates a new cluster if no match found.
        Keeps max 5 sample images per cluster.
        """
        best_cluster = None
        best_dist    = float("inf")

        for cluster in self.clusters:
            dist = 1.0 - float(np.dot(embedding, cluster["centroid"]))
            if dist < best_dist:
                best_dist    = dist
                best_cluster = cluster

        if best_cluster is not None and best_dist < CLUSTER_THRESHOLD:
            # Add to existing cluster
            best_cluster["embeddings"].append(embedding)
            # Update centroid as mean of all embeddings
            best_cluster["centroid"] = np.mean(
                best_cluster["embeddings"], axis=0)
            # Normalize centroid
            norm = np.linalg.norm(best_cluster["centroid"])
            if norm > 0:
                best_cluster["centroid"] /= norm
            # Save sample (max 5 per cluster)
            if len(best_cluster["samples"]) < 5:
                best_cluster["samples"].append({
                    "crop":      frame_crop,
                    "cam_label": cam_label,
                    "video_idx": video_idx,
                    "frame_no":  frame_no,
                    "score":     score,
                })
        else:
            # Create new cluster
            cluster_id = len(self.clusters)
            self.clusters.append({
                "id":         cluster_id,
                "embeddings": [embedding],
                "centroid":   embedding.copy(),
                "samples":    [{
                    "crop":      frame_crop,
                    "cam_label": cam_label,
                    "video_idx": video_idx,
                    "frame_no":  frame_no,
                    "score":     score,
                }],
            })

    def get_best_embedding(self, cluster: dict) -> np.ndarray:
        """
        Return the embedding closest to the centroid.
        This is the most representative face.
        """
        centroid = cluster["centroid"]
        best_emb = None
        best_sim = -1.0
        for emb in cluster["embeddings"]:
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_emb = emb
        return best_emb

    def summary(self) -> str:
        lines = []
        for c in self.clusters:
            cams = set(s["cam_label"] for s in c["samples"])
            lines.append(
                f"  Person {c['id']+1}: "
                f"{len(c['embeddings'])} appearances, "
                f"seen on {cams}"
            )
        return "\n".join(lines)


# ============================================================
# CLASS: Database
# ============================================================

class Database:
    """
    Manages the SQLite database at data/enrollment.db.
    Creates all tables on first run.
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        print(f"Database ready: {db_path}")

    def _create_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            role         TEXT NOT NULL,
            embedding    TEXT NOT NULL,
            face_image   TEXT,
            appearances  INTEGER DEFAULT 0,
            cameras_seen TEXT,
            enrolled_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS zones (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL,
            cam_label        TEXT NOT NULL,
            polygon_points   TEXT NOT NULL,
            restricted_roles TEXT DEFAULT '[]',
            created_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS zone_access (
            person_id  INTEGER REFERENCES persons(id),
            zone_id    INTEGER REFERENCES zones(id),
            granted_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (person_id, zone_id)
        );
        """)
        self.conn.commit()

    def save_person(self,
                    name:            str,
                    role:            str,
                    embedding:       np.ndarray,
                    face_image_path: str,
                    appearances:     int,
                    cameras_seen:    list) -> int:
        """
        Save an enrolled person to the database.
        Returns the new person_id.
        """
        emb_json  = json.dumps(embedding.tolist())
        cams_json = json.dumps(list(cameras_seen))

        cur = self.conn.execute("""
            INSERT INTO persons
              (name, role, embedding, face_image, appearances, cameras_seen)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, role, emb_json, face_image_path, appearances, cams_json))
        self.conn.commit()
        return cur.lastrowid

    def save_zone(self,
                  name:             str,
                  cam_label:        str,
                  polygon:          list,
                  restricted_roles: list) -> int:
        """Save a zone polygon to the database."""
        cur = self.conn.execute("""
            INSERT INTO zones
              (name, cam_label, polygon_points, restricted_roles)
            VALUES (?, ?, ?, ?)
        """, (name, cam_label, json.dumps(polygon),
              json.dumps(restricted_roles)))
        self.conn.commit()
        return cur.lastrowid

    def grant_zone_access(self, person_id: int, zone_id: int):
        """Grant a person access to a zone."""
        self.conn.execute("""
            INSERT OR IGNORE INTO zone_access (person_id, zone_id)
            VALUES (?, ?)
        """, (person_id, zone_id))
        self.conn.commit()

    def get_all_persons(self) -> list:
        return self.conn.execute(
            "SELECT * FROM persons"
        ).fetchall()

    def get_all_zones(self) -> list:
        return self.conn.execute(
            "SELECT * FROM zones"
        ).fetchall()

    def close(self):
        self.conn.close()


# ============================================================
# STAGE 1: Extract faces from videos
# ============================================================

def extract_faces_from_videos(processor: FaceProcessor,
                               clusterer: FaceClusterer):
    """
    Read all 4 videos, sample every 50 frames,
    detect faces, generate embeddings, cluster them.
    Shows a live progress line in terminal.
    No frames saved to disk during this stage.
    """
    print("\n" + "=" * 55)
    print(" STAGE 1: Extracting faces from videos")
    print("=" * 55)

    total_frames_processed = 0
    total_faces_found      = 0

    for vid_idx, (video_file, cam_label) in \
            enumerate(zip(VIDEOS, VIDEO_LABELS)):

        video_path = os.path.join(BASE_DIR, video_file)

        if not os.path.exists(video_path):
            print(f"  WARNING: Not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ERROR: Cannot open {video_path}")
            continue

        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps              = cap.get(cv2.CAP_PROP_FPS)
        duration_min     = total_vid_frames / max(fps, 1) / 60

        print(f"\n  [{cam_label}] "
              f"{os.path.basename(video_file)[-25:]}")
        print(f"  Duration: {duration_min:.1f}min  "
              f"Sampling every {SAMPLE_EVERY_N_FRAMES} frames")

        frame_no     = 0
        faces_in_vid = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1

            # Only process every Nth frame
            if frame_no % SAMPLE_EVERY_N_FRAMES != 0:
                continue

            # Detect faces
            faces = processor.detect_faces(frame)

            for face in faces:
                emb = processor.get_embedding(frame, face["bbox"])
                if emb is None:
                    continue

                # Crop face for display later
                x1, y1, x2, y2 = face["bbox"]
                pad = 10
                fx1  = max(0, x1 - pad)
                fy1  = max(0, y1 - pad)
                fx2  = min(frame.shape[1], x2 + pad)
                fy2  = min(frame.shape[0], y2 + pad)
                crop = frame[fy1:fy2, fx1:fx2].copy()

                # Resize crop to thumbnail
                if crop.size > 0:
                    crop = cv2.resize(crop, (80, 80))

                clusterer.add_face(
                    embedding  = emb,
                    frame_crop = crop,
                    cam_label  = cam_label,
                    video_idx  = vid_idx,
                    frame_no   = frame_no,
                    score      = face["score"],
                )
                faces_in_vid += 1

            total_frames_processed += 1

            # Progress update every 50 sampled frames
            sampled = frame_no // SAMPLE_EVERY_N_FRAMES
            if sampled % 50 == 0:
                clusters_so_far = len(clusterer.clusters)
                print(
                    f"  Frame {frame_no}/{total_vid_frames} | "
                    f"Faces found: {faces_in_vid} | "
                    f"Unique persons: {clusters_so_far}",
                    end="\r",
                )

        cap.release()
        total_faces_found += faces_in_vid
        print(f"\n  Done: {faces_in_vid} faces found")

    print(f"\n{'=' * 55}")
    print(f" Extraction complete!")
    print(f" Total frames processed: {total_frames_processed}")
    print(f" Total faces found     : {total_faces_found}")
    print(f" Unique persons found  : {len(clusterer.clusters)}")
    print(f"{'=' * 55}")


# ============================================================
# STAGE 2: Save face samples to disk
# ============================================================

def save_face_samples(clusterer: FaceClusterer):
    """
    Save best face crop per cluster to disk so user can
    see who each cluster represents.
    Saved to: data/extracted_faces/person_N/
    """
    faces_dir = os.path.join(BASE_DIR, FACES_DIR)
    if os.path.exists(faces_dir):
        shutil.rmtree(faces_dir)
    os.makedirs(faces_dir, exist_ok=True)

    for cluster in clusterer.clusters:
        person_dir = os.path.join(
            faces_dir, f"person_{cluster['id']+1:02d}")
        os.makedirs(person_dir, exist_ok=True)

        # Save all sample crops
        for i, sample in enumerate(cluster["samples"]):
            img_path = os.path.join(person_dir, f"sample_{i+1}.jpg")
            cv2.imwrite(img_path, sample["crop"])

        # Save cluster info as text file
        cams = set(s["cam_label"] for s in cluster["samples"])
        info = (
            f"Person {cluster['id']+1}\n"
            f"Appearances: {len(cluster['embeddings'])}\n"
            f"Cameras: {', '.join(sorted(cams))}\n"
        )
        with open(os.path.join(person_dir, "info.txt"), "w") as f:
            f.write(info)

    print(f"\nFace samples saved to: {faces_dir}")
    print("Open this folder to see each detected person.")


# ============================================================
# STAGE 3: Label persons interactively
# ============================================================

def label_persons_interactively(clusterer: FaceClusterer,
                                 db: Database) -> dict:
    """
    For each cluster, show the user a window with the best
    face samples and ask for name + role.
    Saves each person to the database.
    Returns dict: cluster_id → person_id
    """
    print("\n" + "=" * 55)
    print(" STAGE 3: Label each detected person")
    print("=" * 55)
    print(f" Found {len(clusterer.clusters)} unique persons.")
    print(" A window will open for each person.")
    print(" Enter name and role in terminal.")
    print(" Press any key to close each window.")
    print("=" * 55)

    cluster_to_person = {}

    for cluster in clusterer.clusters:
        cluster_id  = cluster["id"]
        appearances = len(cluster["embeddings"])
        cams_seen   = set(s["cam_label"] for s in cluster["samples"])

        print(f"\n--- Person {cluster_id+1} of "
              f"{len(clusterer.clusters)} ---")
        print(f"  Appearances : {appearances}")
        print(f"  Cameras     : {', '.join(sorted(cams_seen))}")

        # Build display image with sample crops
        samples = cluster["samples"]
        n       = len(samples)
        grid_w  = min(n, 5) * 82 + 10
        grid_h  = 100
        display = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        display[:] = (40, 40, 40)

        for i, sample in enumerate(samples[:5]):
            crop = sample["crop"]
            if crop.size == 0:
                continue
            x_off = i * 82 + 5
            # Resize crop to 80x80 if needed
            c = cv2.resize(crop, (80, 80))
            display[5:85, x_off:x_off + 80] = c

            # Label with cam name
            cv2.putText(
                display,
                sample["cam_label"],
                (x_off, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (180, 180, 180), 1,
            )

        # Add title
        title = f"Person {cluster_id+1} | {appearances} appearances"
        # Save display image and open with system viewer
        img_path = f"/tmp/person_{cluster_id+1}_preview.jpg"
        cv2.imwrite(img_path, display)
        import subprocess
        subprocess.Popen(["eog", img_path])
        import time
        time.sleep(1)

        # Get name from user
        while True:
            name = input(
                f"  Enter name (or 'skip' to skip): "
            ).strip()
            if name:
                break

        if name.lower() == "skip":
            cv2.destroyAllWindows()
            print(f"  Skipped person {cluster_id+1}")
            continue

        # Show role options
        print("  Select role:")
        for i, role in enumerate(AVAILABLE_ROLES):
            print(f"    {i+1}. {role}")

        while True:
            try:
                choice = int(input(
                    f"  Enter number (1-{len(AVAILABLE_ROLES)}): "))
                if 1 <= choice <= len(AVAILABLE_ROLES):
                    role = AVAILABLE_ROLES[choice - 1]
                    break
            except ValueError:
                pass
            print("  Invalid. Try again.")

        cv2.destroyAllWindows()

        # Get best embedding for this cluster
        best_emb = clusterer.get_best_embedding(cluster)

        # Save best face image
        faces_dir     = os.path.join(BASE_DIR, FACES_DIR)
        face_img_path = os.path.join(
            faces_dir,
            f"person_{cluster_id+1:02d}",
            "enrolled.jpg",
        )
        if cluster["samples"]:
            cv2.imwrite(face_img_path, cluster["samples"][0]["crop"])

        # Save to database
        person_id = db.save_person(
            name            = name,
            role            = role,
            embedding       = best_emb,
            face_image_path = face_img_path,
            appearances     = appearances,
            cameras_seen    = list(cams_seen),
        )

        cluster_to_person[cluster_id] = person_id
        print(f"  Saved: {name} | {role} | ID={person_id}")

    return cluster_to_person


# ============================================================
# STAGE 4: Zone drawing
# ============================================================

zone_points = []   # global for mouse callback


def mouse_callback(event, x, y, flags, param):
    global zone_points
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append([x, y])


def setup_zones_interactively(db: Database) -> list:
    """
    For each video, show one representative frame.
    User clicks to draw a polygon zone boundary.
    User names the zone and sets restricted roles.
    Returns list of zone_ids created.
    """
    global zone_points

    print("\n" + "=" * 55)
    print(" STAGE 4: Set up zones")
    print("=" * 55)
    print(" For each camera, you will:")
    print("  1. See a frame from that camera")
    print("  2. LEFT CLICK to draw zone boundary")
    print("  3. Press ENTER when done drawing")
    print("  4. Press R to reset and redraw")
    print("  5. Press S to skip this camera")
    print("=" * 55)

    zone_ids = []

    for vid_idx, (video_file, cam_label) in \
            enumerate(zip(VIDEOS, VIDEO_LABELS)):

        video_path = os.path.join(BASE_DIR, video_file)
        if not os.path.exists(video_path):
            continue

        # Get a frame from middle of video
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # Resize for display
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))

        print(f"\n  [{cam_label}] Draw zone boundaries.")
        print(f"  Click points → ENTER to save → R to reset → S to skip")

        # Allow multiple zones per camera
        while True:
            zone_points    = []
            drawing_frame  = frame.copy()

            win_name = (f"[{cam_label}] "
                        f"Draw zone — ENTER=save R=reset S=skip")
            cv2.namedWindow(win_name)
            cv2.setMouseCallback(win_name, mouse_callback)

            print(f"\n  Drawing zone on {cam_label}...")
            print("  Click to add points. "
                  "ENTER=done, R=reset, S=skip camera")

            while True:
                display = drawing_frame.copy()

                # Draw clicked points
                for pt in zone_points:
                    cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)

                # Draw polygon lines
                if len(zone_points) >= 2:
                    pts = np.array(zone_points, np.int32)
                    cv2.polylines(
                        display,
                        [pts.reshape(-1, 1, 2)],
                        False,
                        (0, 255, 0), 2,
                    )

                # Draw closing line preview
                if len(zone_points) >= 3:
                    pts = np.array(zone_points, np.int32)
                    cv2.polylines(
                        display,
                        [pts.reshape(-1, 1, 2)],
                        True,
                        (0, 200, 255), 2,
                    )

                # Instructions overlay
                cv2.putText(
                    display,
                    f"Points: {len(zone_points)} "
                    f"| ENTER=save R=reset S=skip",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2,
                )

                cv2.imshow(win_name, display)
                key = cv2.waitKey(30) & 0xFF

                if key == 13:  # ENTER
                    if len(zone_points) >= 3:
                        break
                    else:
                        print("  Need at least 3 points. Keep clicking.")

                elif key == ord('r'):
                    zone_points = []
                    print("  Reset. Click again.")

                elif key == ord('s'):
                    cv2.destroyAllWindows()
                    zone_points = []
                    break

            cv2.destroyAllWindows()

            if not zone_points or len(zone_points) < 3:
                break   # skip this camera

            # Get zone name
            zone_name = input(
                f"  Zone name for {cam_label} "
                f"(e.g. Showroom, Vault): "
            ).strip()
            if not zone_name:
                zone_name = f"Zone_{cam_label}"

            # Get restricted roles
            print("  Which roles are RESTRICTED from this zone?")
            print("  (Enter numbers separated by commas, "
                  "or press ENTER for none)")
            for i, role in enumerate(AVAILABLE_ROLES):
                print(f"    {i+1}. {role}")

            restricted = []
            choice_str = input("  Restricted roles: ").strip()
            if choice_str:
                for c in choice_str.split(","):
                    try:
                        idx = int(c.strip()) - 1
                        if 0 <= idx < len(AVAILABLE_ROLES):
                            restricted.append(AVAILABLE_ROLES[idx])
                    except ValueError:
                        pass

            # Save zone to database
            zone_id = db.save_zone(
                name             = zone_name,
                cam_label        = cam_label,
                polygon          = zone_points,
                restricted_roles = restricted,
            )
            zone_ids.append(zone_id)
            print(
                f"  Saved zone: {zone_name} on {cam_label} | "
                f"ID={zone_id} | Restricted: {restricted}"
            )

            # Ask if user wants to add another zone on this camera
            another = input(
                f"  Add another zone on {cam_label}? (y/n): "
            ).strip().lower()
            if another != 'y':
                break

    return zone_ids


# ============================================================
# STAGE 5: Assign zone access per person
# ============================================================

def assign_zone_access(db: Database,
                       cluster_to_person: dict):
    """
    For each enrolled person, ask which zones they are
    allowed to access. Saves to zone_access table.
    """
    print("\n" + "=" * 55)
    print(" STAGE 5: Assign zone access per person")
    print("=" * 55)

    persons = db.get_all_persons()
    zones   = db.get_all_zones()

    if not zones:
        print("  No zones configured. Run zone setup first.")
        return

    if not persons:
        print("  No persons enrolled. Enroll persons first.")
        return

    print(f"  Persons: {len(persons)}")
    print(f"  Zones  : {len(zones)}")

    for person in persons:
        print(f"\n  Person: {person['name']} | Role: {person['role']}")
        print("  Which zones can this person access?")

        for i, zone in enumerate(zones):
            print(f"    {i+1}. {zone['name']} "
                  f"(cam: {zone['cam_label']})")

        print("  Enter zone numbers separated by commas "
              "(or ENTER for none):")
        choice_str = input("  Allowed zones: ").strip()

        if choice_str:
            for c in choice_str.split(","):
                try:
                    idx = int(c.strip()) - 1
                    if 0 <= idx < len(zones):
                        zone = zones[idx]
                        db.grant_zone_access(
                            person_id = person["id"],
                            zone_id   = zone["id"],
                        )
                        print(f"  Granted: {person['name']} → "
                              f"{zone['name']}")
                except ValueError:
                    pass


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 55)
    print(" ENROLLMENT TOOL")
    print(" Edge AI CCTV Security System")
    print("=" * 55)
    print(f" Project: {BASE_DIR}")
    print(f" Videos : {len(VIDEOS)}")
    print(f" DB     : {DB_PATH}")
    print("=" * 55)

    # Change to project directory
    os.chdir(BASE_DIR)

    # Verify video files exist
    missing = [v for v in VIDEOS if not os.path.exists(v)]
    if missing:
        print("\nERROR — Missing video files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # Initialize components
    processor = FaceProcessor(MODEL_DIR)
    clusterer = FaceClusterer()
    db        = Database(DB_PATH)

    # Stage 1: Extract faces
    extract_faces_from_videos(processor, clusterer)

    if not clusterer.clusters:
        print("\nNo faces detected. "
              "Check video files and model paths.")
        db.close()
        sys.exit(1)

    # Stage 2: Save face samples to disk
    save_face_samples(clusterer)

    # Stage 3: Label each person interactively
    cluster_to_person = label_persons_interactively(clusterer, db)

    if not cluster_to_person:
        print("\nNo persons enrolled. Exiting.")
        db.close()
        sys.exit(0)

    # Stage 4: Set up zones
    zone_ids = setup_zones_interactively(db)

    # Stage 5: Assign zone access
    if zone_ids:
        assign_zone_access(db, cluster_to_person)

    # Final summary
    print("\n" + "=" * 55)
    print(" ENROLLMENT COMPLETE")
    print("=" * 55)
    persons = db.get_all_persons()
    zones   = db.get_all_zones()
    print(f" Persons enrolled : {len(persons)}")
    print(f" Zones configured : {len(zones)}")
    print(f" Database         : "
          f"{os.path.join(BASE_DIR, DB_PATH)}")
    print("=" * 55)
    print("\nNext step:")
    print("  The system will now recognize enrolled")
    print("  persons and make real zone decisions.")
    print("  Run: docker compose up")
    print("=" * 55)

    db.close()


if __name__ == "__main__":
    main()
