import os
import sys

# EXTREME OPTIMIZATION: Limit all underlying math libraries to 1-2 threads
# This prevents the CPU from locking up and freezing your system!
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import cv2
import numpy as np
import sqlite3
import json
import datetime
import time
import subprocess
import shutil
import gc
from insightface.app import FaceAnalysis

# Constants
VIDEOS = {
    "cam_0": "data/videos/219_8_Jewellery_IPC6_aef7458404ad4b2c89a99ec41882bdca_20260214182622.avi",
    "cam_1": "data/videos/219_8_Jewellery_IPC6_ef337a9e9bdf415ab7137c2db0852780_20260330125041.avi",
    "cam_2": "data/videos/219_8_Jewellery_IPC6_a56efa055afe4b17b9564f37355d6077_20260214120324.avi",
    "cam_3": "data/videos/219_8_Jewellery_IPC6_8891a760756e4bd580aec41b059e2f28_20260214192449.avi"
}

DB_PATH = "data/enrollment.db"
FACES_DIR = "data/extracted_faces"
TMP_CROP_DIR = "data/tmp_crops"

ROLES = {
    1: "Manager",
    2: "Staff",
    3: "Cashier",
    4: "Security",
    5: "Cleaner",
    6: "Customer",
    7: "Unknown"
}

def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or '.', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        role TEXT,
        embedding TEXT,
        appearances INTEGER,
        cameras_seen TEXT,
        enrolled_at TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS zones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        cam_label TEXT,
        polygon_points TEXT,
        restricted_roles TEXT,
        created_at TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS zone_access (
        person_id INTEGER,
        zone_id INTEGER,
        granted_at TEXT,
        PRIMARY KEY (person_id, zone_id)
    )
    """)
    conn.close()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_faces():
    print("STAGE 1: Extracting faces...")
    
    # Clean previous temp directory if interrupted previously
    if os.path.exists(TMP_CROP_DIR):
        shutil.rmtree(TMP_CROP_DIR)
    os.makedirs(TMP_CROP_DIR, exist_ok=True)
    
    app = FaceAnalysis(name='buffalo_sc', root='models/insightface', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    extracted_data = []
    
    for cam_id, video_path in VIDEOS.items():
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found. Skipping {cam_id}.")
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}. Skipping {cam_id}.")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * 2)
        if frame_interval == 0: frame_interval = 50
        
        frame_idx = 0
        faces_found = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            if frame_idx % frame_interval == 0:
                faces = app.get(frame)
                ts_seconds = frame_idx / fps
                
                for f_idx, face in enumerate(faces):
                    box = face.bbox.astype(int)
                    w, h = box[2] - box[0], box[3] - box[1]
                    if w < 30 or h < 30: continue
                        
                    faces_found += 1
                    
                    # MEMORY FIX: The array slice holds a reference to the ENTIRE 6MB 1080p frame!
                    # If we store it directly, it leaks memory incredibly fast.
                    crop = frame[max(0, box[1]):min(frame.shape[0], box[3]),
                                 max(0, box[0]):min(frame.shape[1], box[2])]
                    
                    crop_filename = f"{cam_id}_{frame_idx}_{f_idx}.jpg"
                    crop_path = os.path.join(TMP_CROP_DIR, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    extracted_data.append({
                        "cam_id": cam_id,
                        "timestamp": ts_seconds,
                        "embedding": face.embedding,
                        "crop_path": crop_path, # STORE DISK PATH INSTEAD OF IMAGE ARRAY!
                        "frame_number": frame_idx,
                        "confidence": face.det_score
                    })
                    del crop # Dereference explicitly
                    
                del faces
                # Run lightweight collection periodically to keep RAM ultra low
                if frame_idx % (frame_interval * 20) == 0:
                    gc.collect()
                    
                print(f"\r[{cam_id}] Frame {frame_idx}/{total_frames} | faces={faces_found}", end="")
            frame_idx += 1
        print()
        if faces_found == 0:
            print(f"Warning: No faces found in {cam_id}.")
        cap.release()
    return extracted_data

def cluster_faces(extracted_data):
    print("STAGE 2: Clustering...")
    clusters = []
    
    for face in extracted_data:
        emb = face['embedding']
        best_sim = -1
        best_cluster = -1
        
        for i, c in enumerate(clusters):
            sim = cosine_sim(emb, c['centroid'])
            if sim > best_sim:
                best_sim = sim
                best_cluster = i
                
        if best_sim > 0.55:
            clusters[best_cluster]['members'].append(face)
            all_embs = [m['embedding'] for m in clusters[best_cluster]['members']]
            clusters[best_cluster]['centroid'] = np.mean(all_embs, axis=0)
            
            ts_str = str(datetime.timedelta(seconds=int(face['timestamp'])))
            print(f"[{face['cam_id']} | {ts_str}] Face detected — added to cluster_{best_cluster+1:02d}")
        else:
            clusters.append({'centroid': emb, 'members': [face]})
            new_id = len(clusters)
            ts_str = str(datetime.timedelta(seconds=int(face['timestamp'])))
            print(f"[{face['cam_id']} | {ts_str}] Face detected — new cluster_{new_id:02d}")

    valid_clusters = [c for c in clusters if len(c['members']) >= 3]
    os.makedirs(FACES_DIR, exist_ok=True)
    cross_camera_count = 0
    valid_clusters.sort(key=lambda x: len(x['members']), reverse=True)
    
    for pid, c in enumerate(valid_clusters):
        c_id = pid + 1
        person_dir = os.path.join(FACES_DIR, f"person_{c_id:02d}")
        os.makedirs(person_dir, exist_ok=True)
        
        sorted_members = sorted(c['members'], key=lambda x: x['confidence'], reverse=True)
        top5 = sorted_members[:5]
        
        for idx, m in enumerate(top5):
            crop_img = cv2.imread(m['crop_path'])
            if crop_img is not None:
                cv2.imwrite(os.path.join(person_dir, f"best_{idx}.jpg"), crop_img)
            
        cam_timestamps = {}
        for m in sorted_members:
            cam = m['cam_id']
            ts_str = str(datetime.timedelta(seconds=int(m['timestamp'])))
            if cam not in cam_timestamps:
                cam_timestamps[cam] = []
            if ts_str not in cam_timestamps[cam]:
                cam_timestamps[cam].append(ts_str)
            
        cameras_seen = list(cam_timestamps.keys())
        if len(cameras_seen) >= 2:
            cross_camera_count += 1
            
        c['cameras_seen'] = cameras_seen
        c['cam_timestamps'] = cam_timestamps
        c['pid'] = c_id
        
        with open(os.path.join(person_dir, "info.txt"), "w") as f:
            f.write(f"Cluster ID: {c_id}\n")
            f.write(f"Total appearances: {len(c['members'])}\n")
            f.write(f"Cameras seen: {', '.join(cameras_seen)}\n")
            f.write("Timestamps:\n")
            for cam, ts_list in cam_timestamps.items():
                f.write(f"  {cam}: {', '.join(ts_list)}\n")
                
    print("\nClustering complete.")
    print(f"Total faces extracted: {len(extracted_data)}")
    print(f"Unique persons found: {len(valid_clusters)}")
    print(f"Cross-camera persons (seen in 2+ videos): {cross_camera_count}")
    return valid_clusters

def label_persons(clusters):
    print("STAGE 3: Labeling...")
    saved_persons = 0
    skipped_persons = 0
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM persons;")
    cur.execute("DELETE FROM zones;")
    cur.execute("DELETE FROM zone_access;")
    conn.commit()
    
    enrolled_persons = []
    
    try:
        for c in clusters:
            pid = c['pid']
            appearances = len(c['members'])
            cams = ", ".join(c['cameras_seen'])
            
            print("══════════════════════════════════════")
            print(f"Person {pid} of {len(clusters)}")
            print(f"Appearances : {appearances}")
            print(f"Cameras     : {cams}")
            print("Timestamps  :")
            for cam, tsl in c['cam_timestamps'].items():
                short_ts = ", ".join(tsl)
                if len(short_ts) > 60:
                    short_ts = short_ts[:60] + "..."
                print(f"  {cam}: {short_ts}")
            
            person_dir = os.path.join(FACES_DIR, f"person_{pid:02d}/")
            print(f"Photos      : {person_dir}")
            print("══════════════════════════════════════")
            
            viewer_proc = subprocess.Popen(["eog", person_dir])
            time.sleep(1.5)
            
            name = input("Name (or 'skip' to skip, 'quit' to stop): ").strip()
            
            if name.lower() == 'quit':
                viewer_proc.terminate()
                break
                
            if name.lower() == 'skip' or name == "":
                skipped_persons += 1
                viewer_proc.terminate()
                continue
                
            print("Role:")
            for k, v in ROLES.items():
                print(f"{k}. {v}")
            role_in = input("Enter number: ").strip()
            role = ROLES.get(int(role_in) if role_in.isdigit() else 7, "Unknown")
            
            emb_list = c['centroid'].tolist()
            cur.execute("""
                INSERT INTO persons (name, role, embedding, appearances, cameras_seen, enrolled_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, role, json.dumps(emb_list), appearances, json.dumps(c['cameras_seen']), str(datetime.datetime.now())))
            conn.commit()
            
            pid_db = cur.lastrowid
            enrolled_persons.append({
                'id': pid_db,
                'name': name,
                'role': role,
                'appearances': appearances,
                'cameras': cams
            })
            
            first_cam = c['cameras_seen'][0] if c['cameras_seen'] else "cam_0"
            first_ts = c['cam_timestamps'][first_cam][0] if c['cam_timestamps'] else "0:00:00"
            print(f"[{first_cam} | {first_ts}] Cluster_{pid:02d} → {name} ({role}) enrolled")
            print(f"Saved: {name} | {role} | {appearances} appearances | {cams}")
            
            saved_persons += 1
            viewer_proc.terminate()
            
    except KeyboardInterrupt:
        print("\nProgress saved. Run again to continue.")
        sys.exit(0)
        
    print("Enrollment complete.")
    print(f"Persons saved: {saved_persons}")
    print(f"Persons skipped: {skipped_persons}")
    
    conn.close()
    return enrolled_persons

def draw_zones():
    print("STAGE 4: Drawing zones...")
    cam_path = None
    for k, v in VIDEOS.items():
        if os.path.exists(v):
            cam_path = v
            break
            
    if not cam_path:
        print("No videos found to draw zones. Skipping.")
        return []
        
    cap = cv2.VideoCapture(cam_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, total//2))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
         print("Failed to read frame.")
         return []
         
    frame = cv2.resize(frame, (1280, 720))
    zones = []
    points = []
    
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            
    cv2.namedWindow("Zones")
    cv2.setMouseCallback("Zones", mouse_cb)
    
    while True:
        display = frame.copy()
        overlay = display.copy()
        
        # Color palette for different zones
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        for i, z in enumerate(zones):
            pts = np.array(z['polygon_points'], np.int32)
            c = colors[i % len(colors)]
            cv2.fillPoly(overlay, [pts], c)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
            
            # Put text in the center
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(overlay, z['name'], (cX - 40, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                
        # Blend the translucent overlay with the background
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
            
        if len(points) > 0:
            for p in points:
                cv2.circle(display, tuple(p), 3, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.polylines(display, [np.array(points, np.int32)], False, (0, 0, 255), 2)
                
        cv2.putText(display, "LEFT CLICK = add point | ENTER = save zone", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(display, "R = reset | S = skip | Q = quit zone drawing", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        cv2.imshow("Zones", display)
        key = cv2.waitKey(20) & 0xFF
        
        if key == 13 or key == 10: # Enter
            if len(points) > 2:
                cv2.destroyAllWindows()
                zname = input("Zone name: ").strip()
                if not zname: zname = f"Zone {len(zones)+1}"
                
                print("Which roles are RESTRICTED from this zone?")
                for k,v in ROLES.items():
                    print(f"{k}. {v}")
                restr_in = input("Enter numbers separated by comma (e.g. 5,6): ").strip()
                restricted_roles = []
                if restr_in:
                    parts = [p.strip() for p in restr_in.split(',')]
                    for p in parts:
                        if p.isdigit() and int(p) in ROLES:
                            restricted_roles.append(ROLES[int(p)])
                            
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                actual_ids = []
                for cam in ['cam_0', 'cam_1', 'cam_2', 'cam_3']:
                    cur.execute("""
                        INSERT INTO zones (name, cam_label, polygon_points, restricted_roles, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (zname, cam, json.dumps(points), json.dumps(restricted_roles), str(datetime.datetime.now())))
                    actual_ids.append(cur.lastrowid)
                conn.commit()
                conn.close()
                
                zones.append({
                    'name': zname,
                    'polygon_points': points.copy(),
                    'restricted_roles': restricted_roles,
                    'db_ids': actual_ids
                })
                
                print(f"[ALL CAMS | zone] {zname} drawn — restricted: {', '.join(restricted_roles) if restricted_roles else 'None'}")
                print("Saved for all 4 cameras.")
                
                points = []
                ans = input("Add another zone? (y/n): ").strip()
                if ans.lower() != 'y':
                    break
                else:
                    cv2.namedWindow("Zones")
                    cv2.setMouseCallback("Zones", mouse_cb)
        elif key == ord('r'):
            points = []
        elif key == ord('s'):
            print("Skipped current zone.")
            points = []
        elif key == ord('q'):
            break
            
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    
    print(f"Zones configured: {len(zones)}")
    if len(zones) > 0:
        print("Applied to: cam_0, cam_1, cam_2, cam_3")
        
    return zones

def assign_access(enrolled_persons, zones):
    print("STAGE 5: Assigning access...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    access_rules = 0
    
    for p in enrolled_persons:
        print("══════════════════════════════════════")
        print(f"Assigning zones for: {p['name']} ({p['role']})")
        print("Available zones:")
        
        for i, z in enumerate(zones):
            restr = ", ".join(z['restricted_roles']) if z['restricted_roles'] else "None"
            print(f"{i+1}. {z['name']} \t→ restricted: {restr}")
            
        ans = input(f"Which zones can {p['name']} access? (comma separated, e.g. 1,3)\nOr press ENTER for default (all non-restricted zones): ").strip()
        
        granted_zone_indices = []
        if ans:
            parts = [x.strip() for x in ans.split(',')]
            for part in parts:
                if part.isdigit():
                    idx = int(part)-1
                    if 0 <= idx < len(zones):
                        granted_zone_indices.append(idx)
        else:
            for i, z in enumerate(zones):
                if p['role'] not in z['restricted_roles']:
                    granted_zone_indices.append(i)
                    
        granted_names = []
        restricted_names = []
        for i, z in enumerate(zones):
            if i in granted_zone_indices:
                granted_names.append(z['name'])
                # granted for all cameras mapping to this logical zone
                for zid in z['db_ids']:
                    cur.execute("INSERT OR REPLACE INTO zone_access (person_id, zone_id, granted_at) VALUES (?, ?, ?)",
                               (p['id'], zid, str(datetime.datetime.now())))
                    access_rules += 1
            else:
                restricted_names.append(z['name'])
                
        if granted_names:
            print(f"{p['name']} ({p['role']}) → auto-granted: {', '.join(granted_names)}")
        if restricted_names:
            print(f"{p['name']} ({p['role']}) → restricted from: {', '.join(restricted_names)}")
            
    conn.commit()
    conn.close()
    
    return access_rules

if __name__ == "__main__":
    init_db()
    try:
        data = extract_faces()
        clusters = cluster_faces(data)
        persons = label_persons(clusters)
        zones = draw_zones()
        access = assign_access(persons, zones)
        
        print("DONE.")
        print(f"Persons enrolled: {len(persons)}")
        print(f"Zones created: {len(zones)} (x4 cameras = {len(zones)*4} records)")
        print(f"Access rules: {access}")
    finally:
        # Cleanup temp directory
        if os.path.exists(TMP_CROP_DIR):
            shutil.rmtree(TMP_CROP_DIR)
