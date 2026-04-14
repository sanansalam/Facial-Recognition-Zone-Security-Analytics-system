#!/usr/bin/env python3
"""
draw_zones.py — Interactive Zone Drawing Tool
----------------------------------------------
Run this script to draw security zones on a video frame.
Controls:
  LEFT CLICK  — add a point to the current polygon
  RIGHT CLICK — undo last point
  ENTER       — finish current zone (will prompt for name)
  'r'         — restart current zone
  'q'         — quit and save all zones
  's'         — skip frame (jump forward 100 frames)
"""

import cv2, json, os, sys, sqlite3
sys.path.insert(0, os.path.dirname(__file__))

VIDEO_PATH = os.getenv("CAM_0_SOURCE", "")
DB_PATH    = "sop_state_machine/security.db"
CAM_ID     = "cam_0"

# ─── Load .env ───────────────────────────────────────────────────────────────
if os.path.exists(".env"):
    for line in open(".env"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

VIDEO_PATH = os.getenv("CAM_0_SOURCE", VIDEO_PATH)
print(f"Video: {VIDEO_PATH}")

# ─── State ───────────────────────────────────────────────────────────────────
current_pts   = []   # points for the zone being drawn
all_zones     = []   # list of {"name": str, "pts": list, "restricted": list}
COLORS        = [(0,200,100),(0,100,255),(255,100,0),(180,0,255),(0,220,220),(255,180,0)]
frame_display = None
frame_clean   = None

def draw_overlay(img):
    out = img.copy()
    # Draw complete zones
    for i, z in enumerate(all_zones):
        color = COLORS[i % len(COLORS)]
        pts_arr = [(int(p[0]), int(p[1])) for p in z["pts"]]
        cv2.polylines(out, [cv2.UMat([(p,) for p in pts_arr])], True, color, 2)
        # Label at centroid
        cx = int(sum(p[0] for p in pts_arr) / len(pts_arr))
        cy = int(sum(p[1] for p in pts_arr) / len(pts_arr))
        restricted = "🔒" if z["restricted"] else ""
        cv2.putText(out, f"{z['name']} {restricted}", (cx-60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # Draw current in-progress zone
    for p in current_pts:
        cv2.circle(out, (int(p[0]), int(p[1])), 5, (0,255,255), -1)
    if len(current_pts) >= 2:
        for j in range(len(current_pts)-1):
            cv2.line(out, (int(current_pts[j][0]), int(current_pts[j][1])),
                         (int(current_pts[j+1][0]), int(current_pts[j+1][1])),
                         (0,255,255), 1)
    # HUD
    lines = [
        f"Zones defined: {len(all_zones)}",
        "LEFT CLICK=add point | RIGHT CLICK=undo | ENTER=finish zone",
        "R=restart zone | S=next frame | Q=save & quit",
    ]
    for i, l in enumerate(lines):
        cv2.putText(out, l, (10, 25+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(out, l, (10, 25+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    return out

def mouse_cb(event, x, y, flags, param):
    global current_pts, frame_display
    cap_w, cap_h = param
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_pts:
            current_pts.pop()
    frame_display = draw_overlay(frame_clean)

def save_zones():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(f"DELETE FROM zones WHERE cam_id='{CAM_ID}'")
    for z in all_zones:
        cur.execute("INSERT INTO zones (name, cam_id, polygon_points, restricted_roles) VALUES (?,?,?,?)",
                    (z["name"], CAM_ID, json.dumps(z["pts"]), json.dumps(z["restricted"])))
    conn.commit()
    conn.close()
    print(f"\n✅ Saved {len(all_zones)} zones to {DB_PATH}")
    for z in all_zones:
        print(f"  [{z['name']}] restricted={z['restricted']} pts={len(z['pts'])}")

def main():
    global current_pts, frame_display, frame_clean

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        print("Set CAM_0_SOURCE env var or edit draw_zones.py")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_no = 0
    ret, frame = cap.read()
    if not ret:
        print("Could not read video")
        sys.exit(1)

    # Downscale for display if very large
    h, w = frame.shape[:2]
    scale = min(1.0, 1400/w, 900/h)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    dw, dh = frame.shape[1], frame.shape[0]
    print(f"Frame size in tool: {dw}x{dh}  (original {w}x{h}, scale={scale:.2f})")
    print("NOTE: coordinates will be automatically scaled back to original resolution.\n")

    frame_clean   = frame.copy()
    frame_display = draw_overlay(frame_clean)

    WIN = "Zone Drawing Tool"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh)
    cv2.setMouseCallback(WIN, mouse_cb, param=(dw, dh))

    while True:
        cv2.imshow(WIN, frame_display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            # Skip 100 frames
            for _ in range(100):
                cap.read()
                frame_no += 100
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if scale < 1.0:
                frame = cv2.resize(frame, (dw, dh))
            frame_clean   = frame.copy()
            frame_display = draw_overlay(frame_clean)
            print(f"[frame {frame_no}]")

        elif key == ord('r'):
            current_pts = []
            frame_display = draw_overlay(frame_clean)

        elif key == 13:  # ENTER
            if len(current_pts) < 3:
                print("Need at least 3 points to define a zone. Keep clicking!")
                continue
            # Ask for name
            cv2.destroyWindow(WIN)
            name = input(f"\nName for zone {len(all_zones)+1} (e.g. 'Locker Room'): ").strip()
            if not name:
                name = f"Zone {len(all_zones)+1}"
            restricted_input = input("Who is restricted? (comma-separated roles, e.g. 'visitor,customer', or ENTER for none): ").strip()
            restricted = [r.strip() for r in restricted_input.split(",") if r.strip()]
            # Scale coordinates back to original resolution
            scaled_pts = [[int(p[0]/scale), int(p[1]/scale)] for p in current_pts]
            all_zones.append({"name": name, "pts": scaled_pts, "restricted": restricted})
            print(f"Zone '{name}' saved. {len(all_zones)} zones total.")
            current_pts = []
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, dw, dh)
            cv2.setMouseCallback(WIN, mouse_cb, param=(dw, dh))
            frame_display = draw_overlay(frame_clean)

    cv2.destroyAllWindows()
    cap.release()

    if all_zones:
        save_zones()
    else:
        print("No zones drawn. Nothing saved.")

if __name__ == "__main__":
    main()
