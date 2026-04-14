/**
 * Security Video Analysis — Frontend Logic
 * Handles: upload, status polling, result rendering, download
 */

const API = '';   // same origin
let currentJobId = null;
let pollTimer    = null;

// ── DOM refs ────────────────────────────────────────────────────────────────
const uploadZone    = document.getElementById('upload-zone');
const fileInput     = document.getElementById('file-input');
const statusSection = document.getElementById('status-section');
const resultsSection= document.getElementById('results-section');
const statusBadge   = document.getElementById('status-badge');
const statusMessage = document.getElementById('status-message');
const progressBar   = document.getElementById('progress-bar');
const fileInfoName  = document.getElementById('file-info-name');
const fileInfoSize  = document.getElementById('file-info-size');
const lightbox      = document.getElementById('lightbox');
const lightboxImg   = document.getElementById('lightbox-img');

// ── Tabs ────────────────────────────────────────────────────────────────────
const tabAnalysis = document.getElementById('tab-analysis');
const tabEnroll   = document.getElementById('tab-enroll');
const viewAnalysis= document.getElementById('view-analysis');
const viewEnroll  = document.getElementById('view-enroll');

tabAnalysis.addEventListener('click', () => {
    tabAnalysis.classList.add('btn-primary');
    tabAnalysis.classList.remove('btn-outline');
    tabEnroll.classList.add('btn-outline');
    tabEnroll.classList.remove('btn-primary');
    viewAnalysis.style.display = 'block';
    viewEnroll.style.display = 'none';
});

tabEnroll.addEventListener('click', () => {
    tabEnroll.classList.add('btn-primary');
    tabEnroll.classList.remove('btn-outline');
    tabAnalysis.classList.add('btn-outline');
    tabAnalysis.classList.remove('btn-primary');
    viewAnalysis.style.display = 'none';
    viewEnroll.style.display = 'block';
    
    // Load identity data when opening the tab
    loadZones();
    loadPeople();
});

// ── Drag & drop & Staging ─────────────────────────────────────────────────────────────
let stagedVideos = [];
const stagingArea  = document.getElementById('staging-area');
const stagingList  = document.getElementById('staging-list');
const btnStart     = document.getElementById('start-analysis-btn');
const btnBrowse    = document.getElementById('btn-browse');

function updateStagingUI() {
    if (stagedVideos.length > 0) {
        stagingArea.style.display = 'block';
        stagingList.innerHTML = '';
        stagedVideos.forEach(v => {
            const li = document.createElement('li');
            li.textContent = `${v.name} (${(v.size / (1024*1024)).toFixed(1)} MB)`;
            stagingList.appendChild(li);
        });
    } else {
        stagingArea.style.display = 'none';
    }
}

function addFilesToStaging(files) {
    const allowed = ['.avi', '.mp4', '.mkv', '.mov', '.wmv', '.flv'];
    for (const file of files) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowed.includes(ext)) {
            alert(`Unsupported format "${ext}" for ${file.name}. Accepted: ${allowed.join(', ')}`);
        } else {
            stagedVideos.push(file);
        }
    }
    updateStagingUI();
}

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) addFilesToStaging(Array.from(e.dataTransfer.files));
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) addFilesToStaging(Array.from(fileInput.files));
});

// Trigger file input
btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});
uploadZone.addEventListener('click', (e) => {
    if (e.target.closest('button')) return; // Don't trigger twice if button clicked
    fileInput.click();
});

window.triggerAnalysis = (e) => {
    if (e) e.stopPropagation();
    console.log("triggerAnalysis called, staged videos:", stagedVideos.length);
    if (stagedVideos.length > 0) {
        const btn = document.getElementById('start-analysis-btn');
        btn.disabled = true;
        btn.textContent = "⌛ Uploading Videos...";
        handleUpload(stagedVideos);
    }
};

// ── Upload ──────────────────────────────────────────────────────────────────
async function handleUpload(files) {
    console.log("handleUpload called with", files.length, "files");
    const allowed = ['.avi', '.mp4', '.mkv', '.mov', '.wmv', '.flv'];
    
    for (const file of files) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowed.includes(ext)) {
            alert(`Unsupported format "${ext}" for ${file.name}. Accepted: ${allowed.join(', ')}`);
            return;
        }
    }

    // Show status section
    statusSection.style.display = 'block';
    resultsSection.style.display = 'none';
    uploadZone.style.display = 'none';

    const totalSize = files.reduce((sum, f) => sum + f.size, 0);
    const names = files.map(f => f.name).join(', ');
    fileInfoName.textContent = `${files.length} video(s): ${names}`;
    fileInfoSize.textContent = (totalSize / (1024 * 1024)).toFixed(1) + ' MB';

    setStatus('queued', `Uploading ${files.length} video(s)…`);
    progressBar.classList.add('indeterminate');

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    console.log("Dispatching upload request...");
    try {
        const res = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) {
            setStatus('error', data.detail || 'Upload failed.');
            return;
        }

        currentJobId = data.job_id;
        setStatus('processing', data.message);
        startPolling();

    } catch (err) {
        console.error("Upload Error:", err);
        const btn = document.getElementById('start-analysis-btn');
        if (btn) {
            btn.disabled = false;
            btn.textContent = "🚀 Start Analysis";
        }
        setStatus('error', `Network error: ${err.message}`);
    }
}

// ── Status polling ──────────────────────────────────────────────────────────
function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(checkStatus, 3000);
}

async function checkStatus() {
    if (!currentJobId) return;

    try {
        const res = await fetch(`${API}/api/status/${currentJobId}`);
        const data = await res.json();

        setStatus(data.status, data.message);

        if (data.status === 'done' || data.status === 'error') {
            clearInterval(pollTimer);
            progressBar.classList.remove('indeterminate');
            progressBar.style.width = '100%';

            if (data.status === 'done') {
                loadResults();
            }
        }
    } catch (err) {
        console.error('Poll error:', err);
    }
}

function setStatus(status, message) {
    const labels = {
        queued: '⏳ Queued',
        processing: '⚙️ Processing',
        done: '✅ Complete',
        error: '❌ Error',
    };

    statusBadge.className = `status-badge ${status}`;
    statusBadge.textContent = labels[status] || status;
    statusMessage.textContent = message || '';
}

// ── Results ─────────────────────────────────────────────────────────────────
async function loadResults() {
    try {
        const res = await fetch(`${API}/api/results/${currentJobId}`);
        const data = await res.json();

        resultsSection.style.display = 'block';

        // Stats
        document.getElementById('stat-evidence').textContent = data.evidence_count || 0;
        document.getElementById('stat-report').textContent = data.has_report ? 'Yes' : 'No';
        document.getElementById('stat-video').textContent = data.video_name || '-';

        // Action buttons
        const actionsDiv = document.getElementById('action-buttons');
        actionsDiv.innerHTML = '';

        if (data.has_report) {
            const reportBtn = document.createElement('a');
            reportBtn.className = 'btn btn-primary';
            reportBtn.href = `${API}/api/report/${currentJobId}`;
            reportBtn.target = '_blank';
            reportBtn.innerHTML = '📄 View Report';
            actionsDiv.appendChild(reportBtn);
        }

        const dlBtn = document.createElement('a');
        dlBtn.className = 'btn btn-success';
        dlBtn.href = `${API}/api/download/${currentJobId}`;
        dlBtn.innerHTML = '⬇️ Download ZIP';
        actionsDiv.appendChild(dlBtn);

        const newBtn = document.createElement('button');
        newBtn.className = 'btn btn-outline';
        newBtn.innerHTML = '🔄 New Analysis';
        newBtn.onclick = resetUI;
        actionsDiv.appendChild(newBtn);

        // Evidence gallery
        const grid = document.getElementById('evidence-grid');
        grid.innerHTML = '';

        if (data.evidence_files && data.evidence_files.length) {
            data.evidence_files.forEach(fname => {
                const item = document.createElement('div');
                item.className = 'evidence-item';
                item.onclick = () => openLightbox(`${API}/api/evidence/${currentJobId}/${fname}`);

                const img = document.createElement('img');
                img.src = `${API}/api/evidence/${currentJobId}/${fname}`;
                img.alt = fname;
                img.loading = 'lazy';

                const label = document.createElement('div');
                label.className = 'evidence-label';
                label.textContent = fname;

                item.appendChild(img);
                item.appendChild(label);
                grid.appendChild(item);
            });
        } else {
            grid.innerHTML = '<p style="color:var(--text-muted); grid-column:1/-1; text-align:center; padding:32px;">No evidence images captured.</p>';
        }

    } catch (err) {
        console.error('Results error:', err);
    }
}

// ── Lightbox ────────────────────────────────────────────────────────────────
function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.classList.add('active');
}

lightbox.addEventListener('click', () => {
    lightbox.classList.remove('active');
    lightboxImg.src = '';
});

// ── Reset ───────────────────────────────────────────────────────────────────
function resetUI() {
    currentJobId = null;
    if (pollTimer) clearInterval(pollTimer);
    statusSection.style.display = 'none';
    resultsSection.style.display = 'none';
    uploadZone.style.display = 'block';
    progressBar.style.width = '0%';
    progressBar.classList.remove('indeterminate');
    fileInput.value = '';
    stagedVideos = [];
    updateStagingUI();
}

// ── Enrollment UI ───────────────────────────────────────────────────────────
async function loadZones() {
    try {
        const res = await fetch(`${API}/api/zones`);
        const zones = await res.json();
        const container = document.getElementById('enroll-zones-container');
        container.innerHTML = '';
        
        if (zones.length === 0) {
            container.innerHTML = '<span style="color:var(--text-muted); font-size:0.9rem;">No zones configured yet</span>';
            return;
        }

        zones.forEach(z => {
            const label = document.createElement('label');
            label.style.display = 'flex';
            label.style.alignItems = 'center';
            label.style.gap = '6px';
            label.style.cursor = 'pointer';
            label.style.background = 'var(--bg-card)';
            label.style.padding = '6px 12px';
            label.style.borderRadius = 'var(--radius-sm)';
            label.style.border = '1px solid var(--border)';
            label.style.fontSize = '0.85rem';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = z.id;
            checkbox.className = 'zone-checkbox';

            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(`${z.name} (Cam ${z.cam_id})`));
            container.appendChild(label);
        });
    } catch (err) {
        console.error('Failed to load zones', err);
    }
}

async function loadPeople() {
    try {
        const res = await fetch(`${API}/api/persons`);
        const peopleEl = document.getElementById('people-list');
        peopleEl.innerHTML = '';

        if (!res.ok) throw new Error('API Error');

        const persons = await res.json();
        
        if (persons.length === 0) {
            peopleEl.innerHTML = '<span style="color:var(--text-muted); font-size:0.9rem;">Database is empty.</span>';
            return;
        }

        persons.forEach(p => {
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.justifyContent = 'space-between';
            row.style.alignItems = 'center';
            row.style.padding = '12px 16px';
            row.style.background = 'var(--bg-secondary)';
            row.style.borderRadius = 'var(--radius-sm)';
            row.style.border = '1px solid var(--border)';

            const info = document.createElement('div');
            
            const nameEl = document.createElement('div');
            nameEl.style.fontWeight = '600';
            nameEl.style.color = 'var(--text-primary)';
            nameEl.textContent = p.name;
            
            const meta = document.createElement('div');
            meta.style.fontSize = '0.8rem';
            meta.style.color = 'var(--text-secondary)';
            meta.style.marginTop = '4px';
            meta.innerHTML = `<span style="color:var(--accent-light)">[${p.role}]</span> • Enrolled: ${p.enrolled_at.substring(0,10)}`;

            if (p.zones && p.zones.length > 0) {
                meta.innerHTML += ` • Zones: ${p.zones.join(', ')}`;
            }

            info.appendChild(nameEl);
            info.appendChild(meta);

            const delBtn = document.createElement('button');
            delBtn.innerHTML = '🗑️';
            delBtn.style.background = 'transparent';
            delBtn.style.border = 'none';
            delBtn.style.cursor = 'pointer';
            delBtn.style.filter = 'grayscale(1)';
            delBtn.style.transition = 'filter 0.2s';
            
            delBtn.onmouseover = () => delBtn.style.filter = 'grayscale(0)';
            delBtn.onmouseout = () => delBtn.style.filter = 'grayscale(1)';

            delBtn.onclick = async () => {
                if(confirm(`Remove ${p.name} from Security Database?`)) {
                    await fetch(`${API}/api/persons/${p.id}`, { method: 'DELETE' });
                    loadPeople();
                }
            };

            row.appendChild(info);
            row.appendChild(delBtn);
            peopleEl.appendChild(row);
        });

    } catch (err) {
        console.error('Failed to load people', err);
    }
}

// Photo Handling Variables
let enrollPhotoBlob = null;
let webcamStream = null;

const btnChoosePhoto = document.getElementById('btn-choose-photo');
const btnUseWebcam = document.getElementById('btn-use-webcam');
const enrollPhotoInput = document.getElementById('enroll-photo');
const webcamContainer = document.getElementById('webcam-container');
const webcamVideo = document.getElementById('webcam-video');
const btnSnap = document.getElementById('btn-snap');
const webcamCanvas = document.getElementById('webcam-canvas');
const photoPreviewContainer = document.getElementById('photo-preview-container');
const photoPreview = document.getElementById('photo-preview');

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    webcamContainer.style.display = 'none';
}

btnChoosePhoto.addEventListener('click', () => {
    stopWebcam();
    enrollPhotoInput.click();
});

enrollPhotoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        enrollPhotoBlob = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            photoPreview.src = e.target.result;
            photoPreviewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

btnUseWebcam.addEventListener('click', async () => {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = webcamStream;
        webcamContainer.style.display = 'block';
        photoPreviewContainer.style.display = 'none';
        enrollPhotoBlob = null;
    } catch (err) {
        alert("Webcam access denied or unavailable.");
    }
});

btnSnap.addEventListener('click', () => {
    if (!webcamStream) return;
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
    
    webcamCanvas.toBlob((blob) => {
        enrollPhotoBlob = blob;
        photoPreview.src = URL.createObjectURL(blob);
        photoPreviewContainer.style.display = 'block';
        stopWebcam();
    }, 'image/jpeg', 0.95);
});


document.getElementById('enroll-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const statusEl = document.getElementById('enroll-status');
    
    if (!enrollPhotoBlob) {
        statusEl.style.color = 'var(--danger)';
        statusEl.innerHTML = `❌ Please provide a photo.`;
        return;
    }

    statusEl.style.color = 'var(--text-secondary)';
    statusEl.innerHTML = 'Extracting AI face embedding... <span class="spinner" style="display:inline-block; width:12px; height:12px; vertical-align:middle; margin-left:8px;"></span>';

    const name = document.getElementById('enroll-name').value;
    const role = document.getElementById('enroll-role').value;
    const zoneIds = Array.from(document.querySelectorAll('.zone-checkbox:checked')).map(cb => parseInt(cb.value));

    const fd = new FormData();
    fd.append('name', name);
    fd.append('role', role);
    fd.append('zone_ids', JSON.stringify(zoneIds));
    fd.append('file', enrollPhotoBlob, 'photo.jpg');

    try {
        const res = await fetch(`${API}/api/enroll`, { method: 'POST', body: fd });
        const data = await res.json();

        if (res.ok) {
            statusEl.style.color = 'var(--success)';
            statusEl.innerHTML = `✅ ${data.message}`;
            document.getElementById('enroll-form').reset();
            enrollPhotoBlob = null;
            photoPreviewContainer.style.display = 'none';
            stopWebcam();
            loadPeople(); // Reload identity list
        } else {
            statusEl.style.color = 'var(--danger)';
            statusEl.innerHTML = `❌ Error: ${data.detail || 'Unknown error'}`;
        }
    } catch(err) {
        statusEl.style.color = 'var(--danger)';
        statusEl.innerHTML = `❌ Failed to connect: ${err.message}`;
    }
});
