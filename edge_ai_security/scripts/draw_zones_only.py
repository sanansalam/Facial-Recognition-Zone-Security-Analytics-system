import cv2
import numpy as np
import sqlite3
import json
import datetime
import os
import sys

# Use same config as enroll_multi
DB_PATH = "data/enrollment.db"

VIDEOS = {
    "cam_0": "data/videos/219_8_Jewellery_IPC6_aef7458404ad4b2c89a99ec41882bdca_20260214182622.avi",
    "cam_1": "data/videos/219_8_Jewellery_IPC6_ef337a9e9bdf415ab7137c2db0852780_20260330125041.avi",
    "cam_2": "data/videos/219_8_Jewellery_IPC6_a56efa055afe4b17b9564f37355d6077_20260214120324.avi",
    "cam_3": "data/videos/219_8_Jewellery_IPC6_8891a760756e4bd580aec41b059e2f28_20260214192449.avi"
}

ROLES = {
    1: "Manager",
    2: "Staff",
    3: "Cashier",
    4: "Security",
    5: "Cleaner",
    6: "Customer",
    7: "Unknown"
}

def wipe_zones():
    print("Wiping existing zones from database...")
    os.makedirs(os.path.dirname(DB_PATH) or '.', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS zones (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, cam_label TEXT, polygon_points TEXT, restricted_roles TEXT, created_at TEXT)")
    cur.execute("DELETE FROM zones;")
    cur.execute("DELETE FROM zone_access;")
    conn.commit()
    conn.close()
    print("Zones wiped.")

def draw_zones():
    print("Starting zone drawing...")
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

if __name__ == "__main__":
    try:
        wipe_zones()
        draw_zones()
        print("\nAll done! New zones saved.")
        
        # We must sync zones to security.db!
        shutil = __import__('shutil')
        db_path_security = "../video_ingestion_standalone/sop_state_machine/security.db"
        if os.path.exists(db_path_security):
            print("Syncing zones to unified security database...")
            # We don't overwrite the whole DB, just the zones. 
            pass # Actually we synced the db file in reset before. Let's just do it directly.
            conn1 = sqlite3.connect(DB_PATH)
            conn2 = sqlite3.connect(db_path_security)
            rows = conn1.execute("SELECT name, cam_label, polygon_points, restricted_roles FROM zones").fetchall()
            conn2.execute("DELETE FROM zones")
            for r in rows:
                conn2.execute("INSERT INTO zones (name, cam_id, polygon_points, restricted_roles) VALUES (?, ?, ?, ?)", (r[0], r[1], r[2], r[3]))
            conn2.commit()
            conn1.close()
            conn2.close()
            print("Sync complete.")
            
    except KeyboardInterrupt:
        print("\nAborted.")
