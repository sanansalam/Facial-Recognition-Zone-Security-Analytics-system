"""
enrollment_dashboard.py
Edge AI Security System — Web-Based Enrollment UI

A Gradio interface for adding persons to the security database
via webcam capture or photo upload.

Launch: python3 enrollment_dashboard.py
Open:   http://localhost:7860
"""

import os
import sys
import json
import sqlite3
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

# ── Ensure shared module is on the path ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import gradio as gr
from insightface.app import FaceAnalysis

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ROOT = os.path.join(
    os.path.dirname(BASE_DIR), "edge_ai_security", "models", "insightface"
)
DB_PATH = os.path.join(BASE_DIR, "sop_state_machine", "security.db")
FACES_DIR = os.path.join(BASE_DIR, "data", "enrolled_faces")
os.makedirs(FACES_DIR, exist_ok=True)

ROLES = ["Manager", "Staff", "Cashier", "Security", "Cleaner", "Customer", "Visitor", "Unknown"]

# ── Face Model ────────────────────────────────────────────────────────────────
print("Loading InsightFace models…")
_fa = FaceAnalysis(
    name="buffalo_sc",
    root=MODEL_ROOT,
    providers=["CPUExecutionProvider"],
)
_fa.prepare(ctx_id=-1, det_thresh=0.4, det_size=(640, 640))
print("Models ready.\n")


# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            role         TEXT NOT NULL,
            embedding    TEXT NOT NULL,
            face_image   TEXT,
            appearances  INTEGER DEFAULT 1,
            cameras_seen TEXT DEFAULT '[]',
            enrolled_at  TEXT DEFAULT (datetime('now')),
            active       INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS zones (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL,
            cam_id           TEXT NOT NULL,
            polygon_points   TEXT DEFAULT '[]',
            restricted_roles TEXT DEFAULT '[]',
            created_at       TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS zone_access (
            person_id  INTEGER,
            zone_id    INTEGER,
            granted_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (person_id, zone_id)
        );
    """)
    conn.commit()
    conn.close()


_ensure_tables()


def list_persons():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, role, enrolled_at FROM persons WHERE active=1 ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def list_zones():
    conn = get_db()
    rows = conn.execute("SELECT id, name, cam_id FROM zones ORDER BY id").fetchall()
    conn.close()
    return rows


def get_person_zones(person_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT z.id, z.name FROM zone_access za JOIN zones z ON za.zone_id=z.id WHERE za.person_id=?",
        (person_id,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def save_person(name, role, embedding, face_crop_bgr, zone_ids):
    """Insert or update a person in the DB and write their face image."""
    face_path = None
    tag = f"{name.replace(' ','_')}_{int(datetime.now().timestamp())}"

    if face_crop_bgr is not None:
        face_path = os.path.join(FACES_DIR, f"{tag}.jpg")
        cv2.imwrite(face_path, face_crop_bgr)

    emb_json = json.dumps(embedding.tolist())

    conn = get_db()
    cur = conn.execute(
        "INSERT INTO persons (name, role, embedding, face_image, enrolled_at, active) "
        "VALUES (?, ?, ?, ?, datetime('now'), 1)",
        (name, role, emb_json, face_path)
    )
    person_id = cur.lastrowid
    for zid in zone_ids:
        conn.execute(
            "INSERT OR IGNORE INTO zone_access (person_id, zone_id) VALUES (?, ?)",
            (person_id, zid)
        )
    conn.commit()
    conn.close()
    return person_id


def delete_person(person_id):
    conn = get_db()
    conn.execute("UPDATE persons SET active=0 WHERE id=?", (person_id,))
    conn.commit()
    conn.close()


# ── Face extraction ───────────────────────────────────────────────────────────
def extract_embedding(img_bgr):
    """Return (embedding, face_crop_bgr, message)."""
    if img_bgr is None:
        return None, None, "❌ No image provided."
    faces = _fa.get(img_bgr)
    if not faces:
        return None, None, "⚠️ No face detected. Try a clearer / closer photo."

    # Largest face wins
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    face = faces[0]
    emb = face.normed_embedding

    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = img_bgr.shape[:2]
    pad = 20
    crop = img_bgr[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)].copy()
    return emb, crop, f"✅ Face detected (confidence {face.det_score:.2f})"


# ── Gradio state ──────────────────────────────────────────────────────────────
class EnrollState:
    """Holds captured frames across Gradio events."""
    def __init__(self):
        self.embeddings = []
        self.best_crop = None   # BGR

    def add(self, emb, crop):
        self.embeddings.append(emb)
        if self.best_crop is None:
            self.best_crop = crop

    def get_mean_embedding(self):
        if not self.embeddings:
            return None
        m = np.mean(self.embeddings, axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 0 else m

    def reset(self):
        self.embeddings.clear()
        self.best_crop = None
        return self


# ── Helpers for Gradio ────────────────────────────────────────────────────────
def bgr_to_rgb(img_bgr):
    """Convert OpenCV BGR to RGB for Gradio display."""
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr


def zone_choices():
    rows = list_zones()
    return [(f"{r['name']} ({r['cam_id']})", r["id"]) for r in rows]


def persons_table_data():
    rows = list_persons()
    data = [[r["id"], r["name"], r["role"], r["enrolled_at"]] for r in rows]
    return data


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(
    title="Edge AI — Enrollment Dashboard",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    css="""
        .panel { border-radius: 12px; padding: 12px; }
        .status-ok  { color: #22c55e; font-weight: bold; }
        .status-err { color: #ef4444; font-weight: bold; }
        footer { display: none !important; }
    """
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.Markdown("""
    # 🔐 Edge AI Security — Enrollment Dashboard
    Add or remove people from the security system.  
    Each enrolled person gets a **face embedding** stored in the database  
    that the live inference engine uses for real-time recognition.
    """)

    enroll_state = gr.State(EnrollState())

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 — Webcam Capture
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("📷 Webcam Capture"):
            gr.Markdown("### Step 1 — Capture face samples")
            gr.Markdown(
                "Click **Capture Frame** several times from different angles "
                "for a more robust embedding (3–5 shots recommended)."
            )
            with gr.Row():
                webcam_in = gr.Image(
                    sources=["webcam"], streaming=False, type="numpy", label="Live Camera",
                    elem_classes="panel"
                )
                preview_out = gr.Image(label="Last Captured Face", elem_classes="panel")

            cam_status = gr.Textbox(label="Status", interactive=False)
            capture_count = gr.Textbox(label="Captured frames", value="0", interactive=False)

            btn_capture = gr.Button("📸 Capture Frame", variant="primary")
            btn_reset_capture = gr.Button("🔄 Reset Captures", variant="secondary")

            gr.Markdown("### Step 2 — Fill in details and enroll")
            with gr.Row():
                with gr.Column():
                    cam_name_in  = gr.Textbox(label="Full Name *", placeholder="e.g. Aisha Khan")
                    cam_role_in  = gr.Dropdown(ROLES, label="Role *", value="Staff")
                    cam_zones_in = gr.CheckboxGroup(
                        choices=zone_choices(), label="Allowed Zones"
                    )
                with gr.Column():
                    cam_enroll_status = gr.Textbox(label="Enrollment Result", interactive=False, lines=3)

            btn_cam_enroll = gr.Button("✅ Enroll Person", variant="primary", size="lg")

            # Events
            def on_capture(image, state: EnrollState):
                if image is None:
                    return None, "❌ No camera input.", str(len(state.embeddings)), state
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                emb, crop, msg = extract_embedding(img_bgr)
                if emb is None:
                    return None, msg, str(len(state.embeddings)), state
                state.add(emb, crop)
                return bgr_to_rgb(crop), msg, str(len(state.embeddings)), state

            btn_capture.click(
                on_capture,
                inputs=[webcam_in, enroll_state],
                outputs=[preview_out, cam_status, capture_count, enroll_state],
            )

            def on_reset_capture(state: EnrollState):
                state.reset()
                return None, "Captures reset.", "0", state

            btn_reset_capture.click(
                on_reset_capture,
                inputs=[enroll_state],
                outputs=[preview_out, cam_status, capture_count, enroll_state],
            )

            def on_cam_enroll(name, role, zone_ids, state: EnrollState):
                name = name.strip()
                if not name:
                    return "❌ Please enter a name.", state
                emb = state.get_mean_embedding()
                if emb is None:
                    return "❌ No face captures yet. Capture at least 1 frame.", state
                pid = save_person(name, role, emb, state.best_crop, zone_ids)
                state.reset()
                return f"✅ {name} ({role}) enrolled successfully!\n   Person ID: {pid}", state

            btn_cam_enroll.click(
                on_cam_enroll,
                inputs=[cam_name_in, cam_role_in, cam_zones_in, enroll_state],
                outputs=[cam_enroll_status, enroll_state],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 — Photo Upload
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("🖼️ Upload Photo"):
            gr.Markdown("### Upload a clear, front-facing photo")
            with gr.Row():
                upload_in  = gr.Image(type="numpy", label="Upload Photo", elem_classes="panel")
                upload_preview = gr.Image(label="Detected Face", elem_classes="panel")

            upload_status = gr.Textbox(label="Detection Status", interactive=False)

            with gr.Row():
                with gr.Column():
                    up_name_in  = gr.Textbox(label="Full Name *", placeholder="e.g. Rajan Sharma")
                    up_role_in  = gr.Dropdown(ROLES, label="Role *", value="Staff")
                    up_zones_in = gr.CheckboxGroup(
                        choices=zone_choices(), label="Allowed Zones"
                    )
                with gr.Column():
                    up_result = gr.Textbox(label="Enrollment Result", interactive=False, lines=3)

            btn_detect   = gr.Button("🔍 Detect Face", variant="secondary")
            btn_up_enroll = gr.Button("✅ Enroll Person", variant="primary", size="lg")

            _upload_state = gr.State({"emb": None, "crop": None})

            def on_detect(image, _):
                if image is None:
                    return None, "❌ Please upload an image first.", {"emb": None, "crop": None}
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                emb, crop, msg = extract_embedding(img_bgr)
                if emb is None:
                    return None, msg, {"emb": None, "crop": None}
                return bgr_to_rgb(crop), msg, {"emb": emb.tolist(), "crop": (
                    cv2.imencode(".jpg", crop)[1].tobytes().hex()
                )}

            btn_detect.click(
                on_detect,
                inputs=[upload_in, _upload_state],
                outputs=[upload_preview, upload_status, _upload_state],
            )

            def on_up_enroll(name, role, zone_ids, state):
                name = name.strip()
                if not name:
                    return "❌ Please enter a name.", state
                emb_list = state.get("emb")
                if emb_list is None:
                    return "❌ Detect a face first.", state
                emb = np.array(emb_list)
                # Reconstruct crop from hex
                crop_bgr = None
                crop_hex = state.get("crop")
                if crop_hex:
                    buf = bytes.fromhex(crop_hex)
                    arr = np.frombuffer(buf, np.uint8)
                    crop_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                pid = save_person(name, role, emb, crop_bgr, zone_ids)
                return f"✅ {name} ({role}) enrolled!\n   Person ID: {pid}", {"emb": None, "crop": None}

            btn_up_enroll.click(
                on_up_enroll,
                inputs=[up_name_in, up_role_in, up_zones_in, _upload_state],
                outputs=[up_result, _upload_state],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — Manage Persons
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("👥 Manage Persons"):
            gr.Markdown("### Enrolled persons")
            persons_tbl = gr.Dataframe(
                headers=["ID", "Name", "Role", "Enrolled At"],
                datatype=["number", "str", "str", "str"],
                value=persons_table_data(),
                interactive=False,
                wrap=True,
            )
            btn_refresh = gr.Button("🔄 Refresh List", variant="secondary")
            btn_refresh.click(lambda: persons_table_data(), outputs=persons_tbl)

            gr.Markdown("---")
            gr.Markdown("### Remove a person")
            with gr.Row():
                del_id   = gr.Number(label="Person ID to remove", precision=0)
                del_msg  = gr.Textbox(label="Result", interactive=False)
            btn_delete = gr.Button("🗑️ Remove Person", variant="stop")

            def on_delete(pid):
                if not pid:
                    return "❌ Enter a valid ID."
                delete_person(int(pid))
                return f"✅ Person ID {int(pid)} deactivated."

            btn_delete.click(
                on_delete,
                inputs=[del_id],
                outputs=[del_msg],
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 4 — About
        # ══════════════════════════════════════════════════════════════════
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(f"""
            ### System Info
            | Item | Value |
            |------|-------|
            | **Database** | `{DB_PATH}` |
            | **Face Images** | `{FACES_DIR}` |
            | **Model** | InsightFace `buffalo_sc` (ArcFace + SCRFD) |
            | **Zone Source** | Security DB — `zones` table |

            ### How zone access works
            - Zones are drawn during **Zone Setup** and stored in the database.
            - When you check a zone here, the person is **allowed** in that zone.
            - Roles listed in a zone's **restricted_roles** column are blocked
              even if they appear in the zone — the SOP engine enforces this at runtime.

            ### Tips for good enrollment
            - Use 3–5 captures from slightly different angles.
            - Good lighting, no occlusions (mask, hat pulled down).
            - The face should be at least **80–100 px** tall in the frame.
            """)

    gr.Markdown("<small>Edge AI Security Enrollment Dashboard</small>")


if __name__ == "__main__":
    print(f"Starting Enrollment Dashboard…")
    print(f"Database : {DB_PATH}")
    print(f"Faces    : {FACES_DIR}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
