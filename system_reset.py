import os
import shutil
import sqlite3

BASE_DIR = "/home/sana/Bank project/video_ingestion_standalone"
DATA_DIR = os.path.join(BASE_DIR, "data")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
EVENTS_DB = os.path.join(DATA_DIR, "events.db")

def reset_system():
    print("🚀 Starting System Reset (Preserving Enrollment)...")
    
    # 1. Clear Events Database
    if os.path.exists(EVENTS_DB):
        try:
            os.remove(EVENTS_DB)
            print(f"  ✅ Removed event logs: {EVENTS_DB}")
        except Exception as e:
            print(f"  ❌ Error removing {EVENTS_DB}: {e}")
            
    # 2. Clear Evidence Folder
    if os.path.isdir(EVIDENCE_DIR):
        try:
            for filename in os.listdir(EVIDENCE_DIR):
                file_path = os.path.join(EVIDENCE_DIR, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"  ✅ Cleared evidence folder: {EVIDENCE_DIR}")
        except Exception as e:
            print(f"  ❌ Error clearing {EVIDENCE_DIR}: {e}")

    # 3. Clear Reports Folder
    if os.path.isdir(REPORTS_DIR):
        try:
            for filename in os.listdir(REPORTS_DIR):
                file_path = os.path.join(REPORTS_DIR, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            print(f"  ✅ Cleared reports folder: {REPORTS_DIR}")
        except Exception as e:
            print(f"  ❌ Error clearing {REPORTS_DIR}: {e}")

    print("\n✨ System Reset Complete. You can now run 'python3 run_all.py' for a clean session.")

if __name__ == "__main__":
    reset_system()
