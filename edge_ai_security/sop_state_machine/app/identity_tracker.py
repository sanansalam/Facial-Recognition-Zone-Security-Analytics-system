"""
identity_tracker.py — Cross-camera person identity tracking.
Uses cosine similarity on ArcFace 512-dim embeddings.
"""

import os
import time
from typing import Optional, Tuple

import numpy as np

# ── Thresholds ─────────────────────────────────────────────────────────────────

KNOWN_PERSON_THRESHOLD = float(os.getenv("KNOWN_PERSON_THRESHOLD", "0.45"))  # minimum score to match against enrolled persons
                                                                           # (live CCTV crops vs enrolled face: typically 0.30-0.55)
SAME_PERSON_THRESHOLD  = float(os.getenv("SAME_PERSON_THRESHOLD", "0.45"))   # minimum score to merge cross-camera identities

# ── In-memory cross-camera identity table ──────────────────────────────────────

# key: global_id ("UNK_001", "KNOWN_001")
global_identities: dict = {}


# ── Similarity ─────────────────────────────────────────────────────────────────

def cosine_similarity(emb_a: list, emb_b: list) -> float:
    """Return cosine similarity in [-1, 1]. Higher = more similar."""
    if not emb_a or not emb_b:
        return 0.0
    a = np.array(emb_a, dtype=np.float32)
    b = np.array(emb_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Database matching ──────────────────────────────────────────────────────────

def identify_person(embedding: list, db_persons: list) -> Tuple[Optional[object], float]:
    """
    Find best matching enrolled person.
    Returns (person_obj, score) or (None, 0.0).
    """
    if not embedding:
        return None, 0.0

    best_match = None
    best_score = 0.0

    for person in db_persons:
        try:
            person_emb = person.get_embedding()
        except Exception:
            continue

        score = cosine_similarity(embedding, person_emb)
        if score > best_score:
            best_match = person
            best_score = score

    # Only return match if it meets threshold
    if best_score >= KNOWN_PERSON_THRESHOLD:
        return best_match, best_score
    return None, best_score


# ── Global identity management ─────────────────────────────────────────────────

def get_or_create_global_id(embedding: list,
                             person_match,
                             cam_id: str) -> str:
    """
    Return an existing global_id if this embedding matches a tracked person
    (cross-camera), otherwise create a new entry.
    """
    # Check against existing tracked identities
    if embedding:
        for gid, identity in global_identities.items():
            stored_emb = identity.get("embedding", [])
            score = cosine_similarity(embedding, stored_emb)
            if score > SAME_PERSON_THRESHOLD:
                return gid

    # Assign new global ID
    if person_match:
        prefix = "KNOWN"
        count = sum(1 for g in global_identities if g.startswith("KNOWN"))
        gid = f"KNOWN_{count + 1:03d}"
    else:
        prefix = "UNK"
        count = sum(1 for g in global_identities if g.startswith("UNK"))
        gid = f"UNK_{count + 1:03d}"

    now = time.time()
    global_identities[gid] = {
        "global_id":       gid,
        "person_id":       person_match.id if person_match else None,
        "person_name":     person_match.name if person_match else "Unknown",
        "person_role":     person_match.role if person_match else "Unknown",
        "embedding":       embedding,
        "first_seen_cam":  cam_id,
        "first_seen_time": now,
        "last_seen_cam":   cam_id,
        "last_seen_time":  now,
        "trail":           [],
        "violations":      [],
    }
    return gid


def update_trail(global_id: str, cam_id: str,
                 timestamp: float, zone_name: str):
    """Append a new location record to this identity's movement trail."""
    if global_id not in global_identities:
        return
    identity = global_identities[global_id]
    identity["trail"].append({
        "cam_id":    cam_id,
        "timestamp": timestamp,
        "zone":      zone_name,
    })
    identity["last_seen_cam"]  = cam_id
    identity["last_seen_time"] = timestamp


def record_violation(global_id: str, status: str,
                     zone_name: str, cam_id: str, timestamp: float):
    """Append a violation record to this identity."""
    if global_id not in global_identities:
        return
    global_identities[global_id]["violations"].append({
        "status":    status,
        "zone":      zone_name,
        "cam_id":    cam_id,
        "timestamp": timestamp,
    })


# ── Track-to-Identity Mapping (YOLO Track ID Persistence) ───────────────────

# key: (cam_id, track_id), value: {person_id, person_name, person_role, last_update}
track_map: dict = {}
track_votes: dict = {}

def get_track_identity(cam_id: str, track_id: int) -> Optional[Tuple[str, str, str]]:
    """Return (id, name, role) if this track is already linked to a person."""
    key = (cam_id, track_id)
    if key in track_map:
        data = track_map[key]
        return data["person_id"], data["person_name"], data["person_role"]
    return None

def add_identity_vote(cam_id: str, track_id: int, person_match, score: float):
    """Add a vote for an identity prediction on a track."""
    if track_id == -1: return
    key = (cam_id, track_id)
    if key not in track_votes:
        track_votes[key] = {
            "votes": [],
            "last_update": time.time()
        }
        
    track_votes[key]["last_update"] = time.time()
    track_votes[key]["votes"].append(person_match)
    
    # Keep last 15 votes
    if len(track_votes[key]["votes"]) > 15:
        track_votes[key]["votes"].pop(0)

def get_majority_identity(cam_id: str, track_id: int, min_votes=3):
    """Return the majority identity if minimum votes threshold is met."""
    key = (cam_id, track_id)
    if key not in track_votes:
        return None
        
    votes = track_votes[key]["votes"]
    valid_votes = [v for v in votes if v is not None]
    
    if len(valid_votes) >= min_votes:
        from collections import Counter
        counts = Counter([v.id for v in valid_votes])
        top_id, count = counts.most_common(1)[0]
        
        if count >= min_votes:
            for v in valid_votes:
                if v.id == top_id:
                    return v
                    
    return None

def link_track_to_person(cam_id: str, track_id: int, person_obj):
    """Associate a YOLO track with a specific enrolled person."""
    if not person_obj or track_id == -1:
        return
    track_map[(cam_id, track_id)] = {
        "person_id":   person_obj.id,
        "person_name": person_obj.name,
        "person_role": person_obj.role,
        "last_update": time.time()
    }

def clear_old_tracks(max_age_sec=30):
    """Cleanup mapping for tracks that haven't been updated recently."""
    now = time.time()
    to_delete = [k for k, v in track_map.items() 
                 if now - v["last_update"] > max_age_sec]
    for k in to_delete:
        del track_map[k]
        
    to_delete_votes = [k for k, v in track_votes.items() 
                       if now - v["last_update"] > max_age_sec]
    for k in to_delete_votes:
        del track_votes[k]


def get_identity(global_id: str) -> Optional[dict]:
    return global_identities.get(global_id)


def list_identities() -> dict:
    return dict(global_identities)


def count_unknown() -> int:
    return sum(1 for g in global_identities if g.startswith("UNK"))
