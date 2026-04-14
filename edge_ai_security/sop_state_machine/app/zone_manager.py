"""
zone_manager.py — Polygon zone assignment using OpenCV pointPolygonTest.
"""

import numpy as np
import cv2

from database import get_zones_for_camera, seed_default_zone, get_db


def find_zone_for_person(cam_id: str, person_bbox: list):
    """
    Determine which zone a detected person occupies.

    Uses the person's feet position (bottom-center of bbox)
    and tests each zone's polygon for containment.

    Returns the first matching Zone ORM object, or None.
    """
    if not person_bbox or len(person_bbox) < 4:
        return None

    x1, y1, x2, y2 = person_bbox
    feet_x = (x1 + x2) / 2.0
    feet_y = float(y2)

    db = get_db()
    try:
        zones = get_zones_for_camera(db, cam_id)

        # Seed a default zone if none exist yet for this camera
        if not zones:
            seed_default_zone(db, cam_id)
            zones = get_zones_for_camera(db, cam_id)

        for zone in zones:
            points = zone.get_polygon()
            if len(points) < 3:
                continue
            polygon = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
            result = cv2.pointPolygonTest(
                polygon, (feet_x, feet_y), False
            )
            if result >= 0:
                return zone
    finally:
        db.close()

    return None


def is_role_allowed(zone, person_role: str) -> bool:
    """Return True if person_role is NOT in the zone's restricted_roles list."""
    restricted = zone.get_restricted_roles()
    return person_role not in restricted


def get_severity_upgrade(zone_name: str, base_severity: str) -> str:
    """Upgrade severity to CRITICAL for sensitive zones."""
    sensitive_keywords = ("vault", "server", "restricted")
    if any(kw in zone_name.lower() for kw in sensitive_keywords):
        return "CRITICAL"
    return base_severity
