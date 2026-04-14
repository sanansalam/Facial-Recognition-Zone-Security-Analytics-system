# Edge AI Security System — v5 (Unified Web Management)

A comprehensive edge-based security system featuring multi-camera AI analysis, personnel identity management, and secure remote access.

## 🚀 Key Features (v5)

### 1. Security Web Dashboard
*   **Multi-Video Analysis:** Process up to 4 video streams simultaneously.
*   **Staging Area:** Queue and review videos before starting the full AI pipeline.
*   **Real-time Analysis Status:** Visual progress indicators and job history tracking.

### 2. Personnel & Identity Management
*   **Smart Enrollment:** Add authorized personnel using photo uploads or live webcam capture.
*   **Face Recognition:** Powered by InsightFace Buffalo_L for high-accuracy identity verification.
*   **Zone Management:** Define restricted areas and assign access per role (Authorized, Unauthorized, or Restricted).

### 3. Intelligent Reporting
*   **Self-Contained HTML Reports:** Detailed analysis logs with **embedded Base64 evidence photos**.
*   **Severity Highlighting:** Instant visual cues for unknown persons or restricted zone breaches.
*   **Portable Data:** Reports can be downloaded or viewed directly via the webapp.

### 4. Secure Remote Access
*   **Cloudflare Tunnel:** Securely access your local dashboard from the internet without port forwarding.
*   **One-Click Launch:** Use the included `./run_tunnel.sh` to get a public URL instantly.

---

## 🛠 Project Structure

*   `/security_webapp`: The primary management interface (FastAPI + Vanilla JS).
*   `/edge_ai_security`: Core AI inference engines (YOLOv11 + InsightFace).
*   `/video_ingestion_standalone`: Modular pipeline components for forensics.

## ⚙️ Quick Start

### 1. Launch the Backend
```bash
cd "/home/sana/Bank project/security_webapp"
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 9000
```

### 2. Start the Pipeline (In Background)
The webapp handles this automatically when you click the **"Start Analysis"** button on the dashboard.

### 3. Enable Remote Access (Optional)
```bash
cd "/home/sana/Bank project/security_webapp"
./run_tunnel.sh
```

---

## 🔒 Security Policy
*   All data is processed locally. No video or biometric data is sent to the cloud (except for web interface delivery via the secure Cloudflare tunnel).
