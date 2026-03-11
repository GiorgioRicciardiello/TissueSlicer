/**
 * Interactive Tissue Selection GUI
 *
 * Single entry point: open http://localhost:5000 after running:
 *   python gui/backend/app.py
 *
 * Drawing state machine:
 *   IDLE    → no polygon, no drawing
 *   DRAWING → user is adding vertices (dashed outline, rubber-band visible)
 *   CLOSED  → polygon complete (solid outline, ready to save)
 *
 * Keyboard shortcuts (canvas focused):
 *   Escape        DRAWING, ≥1 vertex  → undo last vertex
 *   Escape        DRAWING, 0 vertices → cancel / back to IDLE
 *   Escape        CLOSED              → clear polygon back to IDLE
 *   Enter         DRAWING, ≥3 pts    → close polygon → CLOSED
 *   Delete/Bksp   DRAWING             → undo last vertex
 *   Right-click   DRAWING             → undo last vertex
 *   Middle-click  any                 → pan
 */

const API_BASE = '';   // same origin — Flask serves both API and frontend

// ============================================================================
// API Client
// ============================================================================

const api = {
    async _request(method, endpoint, body) {
        const opts = { method, headers: { 'Content-Type': 'application/json' } };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(API_BASE + endpoint, opts);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        return data;
    },
    loadImage: (path, ch = 0) =>
        api._request('POST', '/api/load-image', { image_path: path, channel_idx: ch }),
    getChannel: (ts, ch) =>
        api._request('POST', '/api/get-channel', { timestamp: ts, channel_idx: ch }),
    previewMask: (ts, coords, pad, sq) =>
        api._request('POST', '/api/preview-mask', {
            timestamp: ts, polygon_coords: coords, padding_px: pad, force_square: sq }),
    saveSelection: (ts, coords, name, pad, sq) =>
        api._request('POST', '/api/save-selection', {
            timestamp: ts, polygon_coords: coords, name, padding_px: pad, force_square: sq }),
    getSelections: (ts) =>
        api._request('GET', `/api/selections?timestamp=${ts}`),
    deleteSelection: (ts, id) =>
        api._request('DELETE', `/api/selection/${id}?timestamp=${ts}`),
    extractAll: (ts) =>
        api._request('POST', '/api/extract-all', { timestamp: ts }),
    clearSession: (ts) =>
        api._request('POST', '/api/clear-session', { timestamp: ts }),
    extractionProgress: (jobId) =>
        api._request('GET', `/api/extraction-progress/${jobId}`),
};

// ============================================================================
// Canvas Renderer  (simple offset+scale model — no matrix stacking)
// ============================================================================

class CanvasRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.img = null;           // HTMLImageElement
        this.maskImg = null;       // HTMLImageElement overlay
        this.polygon = [];         // [[x,y], ...]  in IMAGE pixels
        this.previewLine = null;   // current mouse position for rubber-band
        this.drawingState = 'IDLE'; // 'IDLE' | 'DRAWING' | 'CLOSED'

        // Viewport: top-left corner of image in canvas pixels
        this.offsetX = 0;
        this.offsetY = 0;
        this.scale = 1.0;          // canvas_px = image_px * scale

        this._resizeCanvas();
        window.addEventListener('resize', () => { this._resizeCanvas(); this.render(); });
    }

    _resizeCanvas() {
        const PADDING = 12;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width  = Math.max(100, rect.width  - PADDING * 2) || 800;
        this.canvas.height = Math.max(100, rect.height - PADDING * 2) || 600;
    }

    // Fit image to canvas, centered
    fitImage() {
        if (!this.img) return;
        const scaleX = this.canvas.width  / this.img.naturalWidth;
        const scaleY = this.canvas.height / this.img.naturalHeight;
        this.scale = Math.min(scaleX, scaleY) * 0.95;
        this.offsetX = (this.canvas.width  - this.img.naturalWidth  * this.scale) / 2;
        this.offsetY = (this.canvas.height - this.img.naturalHeight * this.scale) / 2;
    }

    // Convert canvas px → image px
    toImage(cx, cy) {
        return [(cx - this.offsetX) / this.scale, (cy - this.offsetY) / this.scale];
    }

    // Convert image px → canvas px
    toCanvas(ix, iy) {
        return [ix * this.scale + this.offsetX, iy * this.scale + this.offsetY];
    }

    loadImage(base64) {
        return new Promise(resolve => {
            const im = new Image();
            im.onload = () => {
                this.img = im;
                this.maskImg = null;
                this.polygon = [];
                this.previewLine = null;
                this.drawingState = 'IDLE';
                this.fitImage();
                this.render();
                resolve();
            };
            im.src = base64;
        });
    }

    setMask(base64) {
        const im = new Image();
        im.onload = () => { this.maskImg = im; this.render(); };
        im.src = base64;
    }

    clearMask() { this.maskImg = null; this.render(); }

    setPolygon(pts) { this.polygon = pts; this.render(); }

    setPreviewLine(pt) { this.previewLine = pt; this.render(); }

    zoom(factor, cx, cy) {
        const oldScale = this.scale;
        this.scale = Math.max(0.1, Math.min(20, this.scale * factor));
        const sf = this.scale / oldScale;
        this.offsetX = cx - sf * (cx - this.offsetX);
        this.offsetY = cy - sf * (cy - this.offsetY);
        this.render();
    }

    pan(dx, dy) {
        this.offsetX += dx;
        this.offsetY += dy;
        this.render();
    }

    render() {
        const { ctx, canvas, img, maskImg, polygon, scale, offsetX, offsetY, drawingState } = this;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        if (!img) {
            ctx.fillStyle = '#555';
            ctx.font = '18px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Load an image to begin', canvas.width / 2, canvas.height / 2);
            return;
        }

        // Draw image
        ctx.drawImage(img, offsetX, offsetY, img.naturalWidth * scale, img.naturalHeight * scale);

        // Draw mask overlay (transparency baked into RGBA PNG from backend)
        if (maskImg) {
            ctx.drawImage(maskImg, offsetX, offsetY, maskImg.naturalWidth * scale, maskImg.naturalHeight * scale);
        }

        // Draw polygon
        if (polygon.length > 0) {
            ctx.save();
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';

            // DRAWING → dashed cyan; CLOSED → solid cyan
            ctx.strokeStyle = '#00e5ff';
            ctx.setLineDash(drawingState === 'DRAWING' ? [6, 4] : []);

            ctx.beginPath();
            const [x0, y0] = this.toCanvas(polygon[0][0], polygon[0][1]);
            ctx.moveTo(x0, y0);
            for (let i = 1; i < polygon.length; i++) {
                const [xi, yi] = this.toCanvas(polygon[i][0], polygon[i][1]);
                ctx.lineTo(xi, yi);
            }
            // Close path only when polygon is complete
            if (drawingState === 'CLOSED' && polygon.length > 2) ctx.closePath();
            ctx.stroke();

            // Rubber-band: only visible while DRAWING
            if (drawingState === 'DRAWING' && this.previewLine && polygon.length >= 1) {
                const last = this.toCanvas(polygon[polygon.length - 1][0], polygon[polygon.length - 1][1]);
                ctx.setLineDash([4, 4]);
                ctx.strokeStyle = '#aaaaaa';
                ctx.beginPath();
                ctx.moveTo(last[0], last[1]);
                ctx.lineTo(this.previewLine[0], this.previewLine[1]);
                ctx.stroke();

                // Line from mouse back to first vertex when ≥3 pts (shows what the closed shape will look like)
                if (polygon.length >= 3) {
                    ctx.setLineDash([2, 6]);
                    ctx.strokeStyle = '#666666';
                    ctx.beginPath();
                    ctx.moveTo(this.previewLine[0], this.previewLine[1]);
                    ctx.lineTo(x0, y0);
                    ctx.stroke();
                }
            }

            // Vertices
            ctx.setLineDash([]);
            polygon.forEach(([ix, iy], i) => {
                const [cx, cy] = this.toCanvas(ix, iy);
                ctx.beginPath();
                const isFirst = i === 0;
                const radius = isFirst ? 7 : 5;
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);

                if (isFirst) {
                    // First vertex: yellow when DRAWING (target to close), green when CLOSED
                    ctx.fillStyle = drawingState === 'CLOSED' ? '#00e676' : '#ffeb3b';
                } else {
                    ctx.fillStyle = '#00e5ff';
                }
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            });

            // Hint: "click ○ to close" near first vertex when ≥3 pts and DRAWING
            if (drawingState === 'DRAWING' && polygon.length >= 3) {
                const [hx, hy] = this.toCanvas(polygon[0][0], polygon[0][1]);
                ctx.font = '11px monospace';
                ctx.fillStyle = '#ffeb3b';
                ctx.textAlign = 'left';
                ctx.fillText('dbl-click or Enter to close', hx + 10, hy - 8);
            }

            ctx.restore();
        }

        // State label in top-left corner
        if (drawingState !== 'IDLE') {
            ctx.save();
            const label = drawingState === 'DRAWING' ? '● DRAWING' : '■ CLOSED — ready to save';
            const color = drawingState === 'DRAWING' ? '#ffeb3b' : '#00e676';
            ctx.font = 'bold 12px monospace';
            ctx.fillStyle = 'rgba(0,0,0,0.55)';
            ctx.fillRect(8, 8, ctx.measureText(label).width + 12, 22);
            ctx.fillStyle = color;
            ctx.fillText(label, 14, 24);
            ctx.restore();
        }
    }
}

// ============================================================================
// App State
// ============================================================================

const state = {
    timestamp: null,
    scaleY: 1,
    scaleX: 1,
    imagePath: '',
    selections: [],
};

// ============================================================================
// DOM helpers
// ============================================================================

const $  = id  => document.getElementById(id);

/** Escape user-supplied strings before inserting into innerHTML templates. */
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
const log = (msg, cls = 'info') => {
    const el = document.createElement('div');
    el.className = `log-entry ${cls}`;
    el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    $('status-log').appendChild(el);
    $('status-log').scrollTop = 9999;
};

function toast(msg, cls = 'info') {
    const icons = { success: '[OK]', error: '[ERR]', info: '[i]', warning: '[!]' };
    const t = document.createElement('div');
    t.className = `toast ${cls}`;
    const icon = document.createElement('span');
    icon.className = 'toast-icon';
    icon.textContent = icons[cls] || '[i]';
    const text = document.createElement('span');
    text.className = 'toast-message';
    text.textContent = msg;   // textContent prevents XSS from server error strings
    t.appendChild(icon);
    t.appendChild(text);
    $('toast-container').appendChild(t);
    setTimeout(() => t.remove(), 3500);
}

function setStatus(el, msg, cls) {
    el.textContent = msg;
    el.className = `status-message ${cls}`;
}

// ============================================================================
// Selections list
// ============================================================================

function renderSelections() {
    const list = $('selections-list');
    $('selection-count').textContent = state.selections.length;
    if (!state.selections.length) {
        list.innerHTML = '<div class="empty-message">No selections yet</div>';
        return;
    }
    list.innerHTML = state.selections.map(s => `
        <div class="selection-item">
            <div class="selection-item-header">
                <span class="selection-item-name">${escapeHtml(s.name)}</span>
                <button class="selection-item-delete" data-id="${escapeHtml(s.id)}" title="Delete">x</button>
            </div>
            <div class="selection-item-info">
                <p>Vertices: ${Number(s.polygon_coords_count) || 0}</p>
                <p>Padding: ${Number(s.padding_px) || 0}px</p>
                <p>Square: ${s.force_square ? 'Yes' : 'No'}</p>
                <p>${new Date(s.created_at).toLocaleTimeString()}</p>
            </div>
        </div>`).join('');

    list.querySelectorAll('.selection-item-delete').forEach(btn =>
        btn.addEventListener('click', () => deleteSelection(btn.dataset.id)));
}

async function reloadSelections() {
    if (!state.timestamp) return;
    try {
        const r = await api.getSelections(state.timestamp);
        state.selections = r.selections;
        renderSelections();
    } catch (e) {
        log(`Could not reload selections: ${e.message}`, 'error');
    }
}

async function deleteSelection(id) {
    if (!confirm('Delete this selection?')) return;
    try {
        await api.deleteSelection(state.timestamp, id);
        log(`Deleted selection ${id}`);
        await reloadSelections();
    } catch (e) {
        toast(e.message, 'error');
    }
}

// ============================================================================
// Main App
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const canvas = $('canvas');
    const renderer = new CanvasRenderer(canvas);

    // ---- drawing state ----
    let tool = 'polygon';   // 'polygon' | 'rectangle'
    let polygon = [];       // [[x,y], ...] in IMAGE pixels
    let drawingState = 'IDLE'; // 'IDLE' | 'DRAWING' | 'CLOSED'
    let rectStart = null;
    let isPanning = false;
    let lastMouse = [0, 0];
    let previewDebounce = null;

    // ---- padding / force-square ----
    const getPad = () => parseInt($('padding-slider').value, 10);
    const getSq  = () => $('force-square-check').checked;

    // ============================================================
    // Drawing state machine
    // ============================================================

    function setDrawingState(newState) {
        drawingState = newState;
        renderer.drawingState = newState;
        updateDrawingUI();
        renderer.render();
    }

    function updateDrawingUI() {
        const btn = $('btn-clear-polygon');
        const vcEl = $('vertex-count');

        if (drawingState === 'IDLE') {
            btn.textContent = 'Clear';
            btn.className = 'tool-btn btn-muted';
            vcEl.style.color = '';
        } else if (drawingState === 'DRAWING') {
            btn.textContent = 'Undo Vertex (Esc)';
            btn.className = 'tool-btn btn-warning';
            vcEl.style.color = '#ffeb3b';
        } else { // CLOSED
            btn.textContent = 'Clear Polygon';
            btn.className = 'tool-btn btn-danger';
            vcEl.style.color = '#00e676';
        }
    }

    function undoLastVertex() {
        if (polygon.length === 0) return;
        polygon.pop();
        renderer.setPolygon([...polygon]);
        $('vertex-count').textContent = polygon.length;
        if (polygon.length === 0) {
            setDrawingState('IDLE');
            renderer.clearMask();
            $('area-px').textContent  = '—';
            $('area-um2').textContent = '—';
        } else {
            renderer.render();
        }
    }

    function clearPolygon() {
        polygon = [];
        rectStart = null;
        renderer.setPolygon([]);
        renderer.clearMask();
        renderer.previewLine = null;
        $('vertex-count').textContent = '0';
        $('area-px').textContent  = '—';
        $('area-um2').textContent = '—';
        setDrawingState('IDLE');
    }

    function closePolygon() {
        if (polygon.length < 3) {
            toast('Need at least 3 vertices to close', 'warning');
            return;
        }
        renderer.previewLine = null;
        setDrawingState('CLOSED');
        triggerPreview();
        log(`Polygon closed (${polygon.length} vertices)`);
    }

    // ============================================================
    // Keyboard shortcuts (global — works without canvas focus)
    // ============================================================

    document.addEventListener('keydown', e => {
        // Don't intercept when typing in an input field
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;

        if (e.key === 'Escape') {
            e.preventDefault();
            if (tool === 'rectangle' && rectStart) {
                // Cancel rectangle in progress
                rectStart = null;
                renderer.setPolygon([]);
                polygon = [];
                $('vertex-count').textContent = '0';
                log('Rectangle cancelled');
                return;
            }
            if (drawingState === 'DRAWING') {
                if (polygon.length > 0) {
                    undoLastVertex();
                    log('Undo: removed last vertex');
                } else {
                    setDrawingState('IDLE');
                    log('Drawing cancelled');
                }
            } else if (drawingState === 'CLOSED') {
                clearPolygon();
                log('Polygon cleared');
            }

        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (drawingState === 'DRAWING') closePolygon();

        } else if (e.key === 'Delete' || e.key === 'Backspace') {
            if (drawingState === 'DRAWING') {
                e.preventDefault();
                undoLastVertex();
                log('Undo: removed last vertex');
            }
        }
    });

    // ============================================================
    // Image loading
    // ============================================================

    async function loadImage() {
        const path = $('image-path').value.trim();
        const statusEl = $('load-status');
        if (!path) { setStatus(statusEl, 'Enter a file path', 'error'); return; }

        setStatus(statusEl, 'Loading...', 'info');
        $('btn-load-image').disabled = true;

        try {
            const ch = parseInt($('channel-select').value, 10);
            const r = await api.loadImage(path, ch);

            state.timestamp = r.timestamp;
            state.scaleY   = r.scale_y;
            state.scaleX   = r.scale_x;
            state.imagePath = path;

            await renderer.loadImage(r.image_base64);
            polygon = [];
            setDrawingState('IDLE');

            $('image-info').textContent =
                `${path.split('\\').pop()} | Display: ${r.shape_downsampled[1]}x${r.shape_downsampled[0]}` +
                ` | Full res: ${r.shape_full[2]}x${r.shape_full[1]}` +
                ` | Scale: ${r.scale_y.toFixed(2)}x`;

            if (r.channels && r.channels.length) {
                const sel = $('channel-select');
                sel.innerHTML = r.channels.map((ch, i) =>
                    `<option value="${i}">${ch}</option>`).join('');
            }

            setStatus(statusEl, 'Image loaded', 'success');
            log(`Loaded: ${path.split('\\').pop()} — ${r.num_channels} channels`);
            toast('Image loaded', 'success');

            await reloadSelections();

        } catch (e) {
            setStatus(statusEl, `Error: ${e.message}`, 'error');
            log(`Load error: ${e.message}`, 'error');
            toast(e.message, 'error');
        } finally {
            $('btn-load-image').disabled = false;
        }
    }

    // ============================================================
    // Channel switching
    // ============================================================

    async function switchChannel(ch) {
        if (!state.timestamp) return;
        try {
            const r = await api.getChannel(state.timestamp, ch);
            await renderer.loadImage(r.image_base64);
            // Restore polygon and state after channel switch
            renderer.setPolygon(polygon);
            renderer.drawingState = drawingState;
            renderer.render();
            log(`Channel ${ch} loaded`);
        } catch (e) {
            toast(`Channel error: ${e.message}`, 'error');
        }
    }

    // ============================================================
    // Mask preview (debounced)
    // ============================================================

    function triggerPreview() {
        if (!state.timestamp || polygon.length < 3) return;
        clearTimeout(previewDebounce);
        previewDebounce = setTimeout(async () => {
            try {
                const r = await api.previewMask(state.timestamp, polygon, getPad(), getSq());
                renderer.setMask(r.mask_base64);
                $('area-px').textContent  = r.area_px.toLocaleString();
                $('area-um2').textContent = r.area_um2.toFixed(1);
                log(`Preview: ${r.area_px} px, ${r.area_um2.toFixed(1)} um2`);
            } catch (e) {
                log(`Preview error: ${e.message}`, 'error');
            }
        }, 300);
    }

    // ============================================================
    // Save selection
    // ============================================================

    async function saveSelection() {
        if (!state.timestamp) { toast('Load an image first', 'warning'); return; }
        if (drawingState !== 'CLOSED') {
            toast(polygon.length < 3
                ? 'Draw a polygon first (min 3 points)'
                : 'Close the polygon first (double-click or Enter)', 'warning');
            return;
        }
        const name = ($('selection-name').value || 'Region').trim();

        try {
            const r = await api.saveSelection(state.timestamp, polygon, name, getPad(), getSq());
            $('selection-name').value = '';
            clearPolygon();
            log(`Saved: ${name} (${r.selection_id})`);
            toast(`"${name}" saved`, 'success');
            await reloadSelections();
        } catch (e) {
            toast(e.message, 'error');
        }
    }

    // ============================================================
    // Extract all
    // ============================================================

    async function extractAll() {
        if (!state.timestamp || !state.selections.length) {
            toast('No selections to extract', 'warning'); return;
        }
        if (!confirm(`Extract ${state.selections.length} region(s)? This may take several minutes.`)) return;

        $('btn-extract-all').disabled = true;
        const modal = $('extraction-modal');
        modal.classList.remove('hidden');

        // Build per-region progress rows (escape name to prevent XSS)
        const prog = $('extraction-progress');
        prog.innerHTML = state.selections.map(s =>
            `<div class="progress-item" id="prog-${escapeHtml(s.id)}">
                <div class="progress-item-name">${escapeHtml(s.name)}</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="fill-${escapeHtml(s.id)}" style="width:0%"></div>
                    <span class="progress-label" id="label-${escapeHtml(s.id)}">Queued</span>
                </div>
                <div class="progress-status" id="status-${escapeHtml(s.id)}">Waiting...</div>
            </div>`).join('');

        // Overall progress row
        const overallHtml = `
            <div class="progress-item" id="prog-overall">
                <div class="progress-item-name"><strong>Overall</strong></div>
                <div class="progress-bar">
                    <div class="progress-fill progress-fill-overall" id="fill-overall" style="width:0%"></div>
                    <span class="progress-label" id="label-overall">0%</span>
                </div>
                <div class="progress-status" id="status-overall">Starting...</div>
            </div>`;
        prog.insertAdjacentHTML('afterbegin', overallHtml);

        let jobId = null;

        try {
            const r = await api.extractAll(state.timestamp);
            jobId = r.job_id;
            log(`Extraction job started: ${jobId}`);
        } catch (e) {
            toast(`Failed to start extraction: ${e.message}`, 'error');
            log(`Extraction error: ${e.message}`, 'error');
            $('btn-extract-all').disabled = false;
            return;
        }

        // Poll every 1 second; abort after 10 consecutive failures (server down).
        let pollFailures = 0;
        const MAX_POLL_FAILURES = 10;
        const pollInterval = setInterval(async () => {
            try {
                const status = await api.extractionProgress(jobId);
                pollFailures = 0;   // reset on success
                updateExtractionUI(status);

                if (status.status === 'complete' || status.status === 'failed') {
                    clearInterval(pollInterval);
                    const ok = status.regions.filter(r => r.status === 'success').length;
                    toast(`${ok}/${status.total} regions extracted`, ok === status.total ? 'success' : 'warning');
                    log(`Extraction complete: ${ok}/${status.total} succeeded in ${status.elapsed_sec}s`);
                    $('btn-extract-all').disabled = false;
                    setTimeout(() => modal.classList.add('hidden'), 5000);
                }
            } catch (e) {
                pollFailures++;
                log(`Progress poll error (${pollFailures}/${MAX_POLL_FAILURES}): ${e.message}`, 'error');
                if (pollFailures >= MAX_POLL_FAILURES) {
                    clearInterval(pollInterval);
                    toast('Lost contact with server during extraction', 'error');
                    $('btn-extract-all').disabled = false;
                }
            }
        }, 1000);
    }

    function updateExtractionUI(status) {
        // Overall bar
        const fillOverall = $('fill-overall');
        const labelOverall = $('label-overall');
        const statusOverall = $('status-overall');
        if (fillOverall) {
            fillOverall.style.width = `${status.overall_pct}%`;
            labelOverall.textContent = `${status.overall_pct}%`;
            const elapsed = status.elapsed_sec;
            const eta = status.overall_pct > 2
                ? Math.round(elapsed / (status.overall_pct / 100) - elapsed)
                : '…';
            statusOverall.textContent =
                `${status.completed}/${status.total} regions | ${elapsed}s elapsed` +
                (status.overall_pct > 2 ? ` | ~${eta}s remaining` : '');
        }

        // Per-region bars
        for (const region of status.regions) {
            const fill   = $(`fill-${region.selection_id}`);
            const label  = $(`label-${region.selection_id}`);
            const statusEl = $(`status-${region.selection_id}`);
            if (!fill) continue;

            fill.style.width = `${region.progress_pct}%`;
            label.textContent = `${region.progress_pct}%`;

            if (region.status === 'pending') {
                statusEl.textContent = 'Queued';
            } else if (region.status === 'running') {
                const ch = region.channel_done;
                const tot = region.channel_total;
                statusEl.textContent = `Reading channel ${ch}/${tot}`;
                fill.classList.add('progress-fill-active');
            } else if (region.status === 'success') {
                statusEl.textContent = `Done → ${region.output_folder}`;
                fill.classList.remove('progress-fill-active');
                fill.classList.add('progress-fill-success');
            } else if (region.status === 'failed') {
                statusEl.textContent = `Failed: ${region.error}`;
                fill.style.backgroundColor = '#ef4444';
            }
        }
    }

    // ============================================================
    // Canvas event handlers
    // ============================================================

    canvas.addEventListener('click', e => {
        if (isPanning) return;
        if (!renderer.img) return;
        // Ignore clicks that ended a pan
        if (e.button !== 0) return;

        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const [ix, iy] = renderer.toImage(cx, cy);

        if (tool === 'polygon') {
            // Don't add vertices when polygon is already closed
            if (drawingState === 'CLOSED') return;

            polygon.push([ix, iy]);
            renderer.setPolygon([...polygon]);
            $('vertex-count').textContent = polygon.length;

            if (drawingState === 'IDLE') setDrawingState('DRAWING');
            else renderer.render();

            if (polygon.length >= 3) triggerPreview();
        }
    });

    canvas.addEventListener('dblclick', e => {
        e.preventDefault();
        if (tool === 'polygon' && drawingState === 'DRAWING') {
            // Remove the extra vertex added by the second click of dblclick
            if (polygon.length > 1) polygon.pop();
            renderer.setPolygon([...polygon]);
            $('vertex-count').textContent = polygon.length;
            closePolygon();
        }
    });

    // mousedown: pan (middle-click always) or undo vertex (right-click when DRAWING)
    canvas.addEventListener('mousedown', e => {
        if (e.button === 1) {
            // Middle-click → pan always
            isPanning = true;
            lastMouse = [e.clientX, e.clientY];
            canvas.style.cursor = 'grabbing';
            e.preventDefault();
        } else if (e.button === 2) {
            if (tool === 'polygon' && drawingState === 'DRAWING') {
                // Right-click → undo last vertex
                undoLastVertex();
                log('Undo: removed last vertex');
            } else {
                // Right-click → pan when not actively drawing
                isPanning = true;
                lastMouse = [e.clientX, e.clientY];
                canvas.style.cursor = 'grabbing';
            }
            e.preventDefault();
        } else if (tool === 'rectangle' && e.button === 0 && renderer.img && drawingState !== 'CLOSED') {
            const rect = canvas.getBoundingClientRect();
            const [ix, iy] = renderer.toImage(e.clientX - rect.left, e.clientY - rect.top);
            rectStart = [ix, iy];
        }
    });

    canvas.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        if (isPanning) {
            renderer.pan(e.clientX - lastMouse[0], e.clientY - lastMouse[1]);
            lastMouse = [e.clientX, e.clientY];
        } else if (tool === 'polygon' && drawingState === 'DRAWING' && polygon.length > 0) {
            renderer.previewLine = [cx, cy];
            renderer.render();
        } else if (tool === 'rectangle' && rectStart) {
            const [ix, iy] = renderer.toImage(cx, cy);
            const pts = [
                rectStart,
                [ix, rectStart[1]],
                [ix, iy],
                [rectStart[0], iy],
            ];
            renderer.setPolygon(pts);
        }

        // Update coordinate display
        if (renderer.img) {
            const [ix, iy] = renderer.toImage(cx, cy);
            $('zoom-level').textContent = renderer.scale.toFixed(2);
            $('pan-info').textContent = `img (${Math.round(ix)}, ${Math.round(iy)})`;
        }
    });

    canvas.addEventListener('mouseup', e => {
        if (e.button === 1 || (e.button === 2 && isPanning)) {
            isPanning = false;
            canvas.style.cursor = tool === 'rectangle' ? 'cell' : 'crosshair';
        } else if (tool === 'rectangle' && rectStart && renderer.img) {
            const rect = canvas.getBoundingClientRect();
            const [ix, iy] = renderer.toImage(e.clientX - rect.left, e.clientY - rect.top);
            polygon = [
                rectStart, [ix, rectStart[1]], [ix, iy], [rectStart[0], iy],
            ];
            rectStart = null;
            renderer.setPolygon([...polygon]);
            $('vertex-count').textContent = polygon.length;
            setDrawingState('CLOSED');
            triggerPreview();
        }
    });

    canvas.addEventListener('contextmenu', e => e.preventDefault());

    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        renderer.zoom(e.deltaY < 0 ? 1.15 : 1 / 1.15, cx, cy);
        $('zoom-level').textContent = renderer.scale.toFixed(2);
    }, { passive: false });

    canvas.addEventListener('mouseleave', () => {
        renderer.previewLine = null;
        renderer.render();
    });

    // ============================================================
    // Tool switching
    // ============================================================

    function setTool(t) {
        tool = t;
        canvas.style.cursor = t === 'rectangle' ? 'cell' : 'crosshair';
        document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
        $(`btn-${t}`).classList.add('active');
        // Clear any in-progress drawing when switching tools
        if (drawingState === 'DRAWING') {
            clearPolygon();
            log(`Switched to ${t} tool — drawing cleared`);
        }
    }

    // ============================================================
    // UI event bindings
    // ============================================================

    $('btn-load-image').addEventListener('click', loadImage);
    $('image-path').addEventListener('keydown', e => { if (e.key === 'Enter') loadImage(); });

    $('channel-select').addEventListener('change', e =>
        switchChannel(parseInt(e.target.value, 10)));

    $('btn-polygon').addEventListener('click', () => setTool('polygon'));
    $('btn-rectangle').addEventListener('click', () => setTool('rectangle'));

    $('btn-clear-polygon').addEventListener('click', () => {
        if (drawingState === 'DRAWING') {
            undoLastVertex();
        } else {
            clearPolygon();
            log('Polygon cleared');
        }
    });

    $('padding-slider').addEventListener('input', e => {
        $('padding-value').textContent = e.target.value;
        triggerPreview();
    });

    $('force-square-check').addEventListener('change', () => triggerPreview());

    $('btn-save-selection').addEventListener('click', saveSelection);
    $('selection-name').addEventListener('keydown', e => { if (e.key === 'Enter') saveSelection(); });

    $('btn-extract-all').addEventListener('click', extractAll);

    $('btn-clear-session').addEventListener('click', async () => {
        if (!state.timestamp) return;
        if (!confirm('Clear all selections?')) return;
        try {
            await api.clearSession(state.timestamp);
            state.selections = [];
            renderSelections();
            log('Session cleared');
            toast('Cleared', 'info');
        } catch (e) {
            toast(e.message, 'error');
        }
    });

    // Start with polygon tool active
    setTool('polygon');
    log('App ready — load an OME-TIFF image to begin', 'info');
    log('Shortcuts: Esc=undo vertex | Enter=close polygon | Del=undo | Right-click=undo', 'info');
});
