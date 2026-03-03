(function () {
    'use strict';

    // --- Elements ---
    const apiToggle     = document.getElementById('apiToggle');
    const apiPanel      = document.getElementById('apiPanel');
    const apiUrlInput   = document.getElementById('apiUrlInput');
    const apiCheckBtn   = document.getElementById('apiCheckBtn');
    const apiStatus     = document.getElementById('apiStatus');
    const uploadZone    = document.getElementById('uploadZone');
    const fileInput     = document.getElementById('fileInput');
    const fileBar       = document.getElementById('fileBar');
    const fileName      = document.getElementById('fileName');
    const fileSize      = document.getElementById('fileSize');
    const fileRemove    = document.getElementById('fileRemove');
    const audioPlayer   = document.getElementById('audioPlayer');
    const analyzeBtn    = document.getElementById('analyzeBtn');
    const loader        = document.getElementById('loader');
    const result        = document.getElementById('result');
    const resetBtn      = document.getElementById('resetBtn');

    let selectedFile = null;

    // ===========================
    // Particles
    // ===========================
    (function initParticles() {
        const box = document.getElementById('bgParticles');
        for (let i = 0; i < 25; i++) {
            const p = document.createElement('div');
            p.className = 'particle';
            const s = Math.random() * 2.5 + 1;
            p.style.width = s + 'px';
            p.style.height = s + 'px';
            p.style.left = Math.random() * 100 + '%';
            p.style.animationDuration = (Math.random() * 14 + 10) + 's';
            p.style.animationDelay = (Math.random() * 10) + 's';
            p.style.background = `hsl(${Math.random() > .5 ? 239 : 270}, 75%, 65%)`;
            box.appendChild(p);
        }
    })();

    // ===========================
    // API Config
    // ===========================
    apiToggle.addEventListener('click', () => apiPanel.classList.toggle('hidden'));

    function apiUrl() {
        return (apiUrlInput.value || 'http://localhost:8000').replace(/\/+$/, '');
    }

    apiCheckBtn.addEventListener('click', async () => {
        apiStatus.textContent = 'Checking…';
        apiStatus.className = 'api-status wait';
        try {
            const r = await fetch(apiUrl() + '/health', { mode: 'cors' });
            const d = await r.json();
            if (d.status === 'ok' && d.models_loaded) {
                apiStatus.textContent = '✓ Connected — models ready';
                apiStatus.className = 'api-status ok';
            } else {
                apiStatus.textContent = '⚠ Connected — models not loaded';
                apiStatus.className = 'api-status fail';
            }
        } catch {
            apiStatus.textContent = '✗ Cannot reach API';
            apiStatus.className = 'api-status fail';
        }
    });

    // ===========================
    // File handling
    // ===========================
    function fmtSize(b) {
        if (b < 1024) return b + ' B';
        if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
        return (b / 1048576).toFixed(1) + ' MB';
    }

    const AUDIO_EXTS = ['.wav','.mp3','.flac','.ogg','.m4a','.aac','.wma','.opus','.webm','.mp4','.mpeg'];

    function isAudioFile(file) {
        if (file.type && file.type.startsWith('audio/')) return true;
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        return AUDIO_EXTS.includes(ext);
    }

    function pickFile(file) {
        if (!file) return;
        if (!isAudioFile(file)) { alert('Please select an audio file (WAV, MP3, FLAC, OGG, etc.)'); return; }
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = fmtSize(file.size);
        audioPlayer.src = URL.createObjectURL(file);
        uploadZone.classList.add('hidden');
        fileBar.classList.remove('hidden');
        audioPlayer.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
        result.classList.add('hidden');
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        audioPlayer.src = '';
        uploadZone.classList.remove('hidden');
        fileBar.classList.add('hidden');
        audioPlayer.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        result.classList.add('hidden');
        loader.classList.add('hidden');
    }

    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => pickFile(e.target.files[0]));
    fileRemove.addEventListener('click', clearFile);

    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', e => {
        e.preventDefault(); uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
    });

    // ===========================
    // Analyze
    // ===========================
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        analyzeBtn.classList.add('hidden');
        fileBar.classList.add('hidden');
        audioPlayer.classList.add('hidden');
        result.classList.add('hidden');
        loader.classList.remove('hidden');

        try {
            const fd = new FormData();
            fd.append('file', selectedFile);

            const resp = await fetch(apiUrl() + '/predict', { method: 'POST', body: fd, mode: 'cors' });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || 'Server error ' + resp.status);
            }
            const data = await resp.json();
            showResult(data);
        } catch (err) {
            loader.classList.add('hidden');
            fileBar.classList.remove('hidden');
            audioPlayer.classList.remove('hidden');
            analyzeBtn.classList.remove('hidden');
            alert('Analysis failed: ' + err.message);
        }
    });

    // ===========================
    // Show result
    // ===========================
    function esc(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

    function showResult(data) {
        loader.classList.add('hidden');
        result.classList.remove('hidden');

        const label = (data.label || '').toUpperCase();
        const score = data.score || 0;
        const reason = data.reason || '';
        const expl = data.explanation || {};

        const badge = document.getElementById('resultBadge');
        const emoji = document.getElementById('resultEmoji');
        const lbl   = document.getElementById('resultLabel');
        const expEl = document.getElementById('resultExplanation');

        badge.className = 'result-badge';
        lbl.className = 'result-label';

        if (label.includes('HUMAN')) {
            badge.classList.add('human'); emoji.textContent = '🧑';
            lbl.classList.add('human'); lbl.textContent = 'Human Voice';
        } else if (label.includes('AI')) {
            badge.classList.add('ai'); emoji.textContent = '🤖';
            lbl.classList.add('ai'); lbl.textContent = 'AI-Generated';
        } else {
            badge.classList.add('unknown'); emoji.textContent = '❓';
            lbl.classList.add('unknown'); lbl.textContent = 'Inconclusive';
        }

        // The backend text explanation might contain the confidence percentage appended to the end like ". Confidence ~ 99%."
        // Or phrases like "HUMAN — breathing found". We'll clean it up to just be the reason text.
        let rawText = expl.text || '';
        if (rawText.includes(' — ')) {
            rawText = rawText.split(' — ')[1] || rawText; // Removes the prefix "AI — " or "HUMAN — "
        }
        rawText = rawText.replace(/\.?\s*Confidence(.*?)(?:\.|$)/ig, '.'); // Strips out "Confidence ~ X%"
        
        // Capitalize the first letter
        if (rawText.length > 0) {
            rawText = rawText.charAt(0).toUpperCase() + rawText.slice(1);
        }

        expEl.textContent = rawText.trim();

        result.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // ===========================
    // Reset
    // ===========================
    resetBtn.addEventListener('click', clearFile);

})();
