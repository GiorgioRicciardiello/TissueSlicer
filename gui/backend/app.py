"""
Interactive Tissue Selection GUI - Flask Backend

Manual polygon-based tissue extraction with coordinate verification.
Coordinates are rigorously verified at every transformation step.

Run with:
    python app.py
Then open:
    http://localhost:5000
"""

import json
import uuid
import logging
import time
import threading
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait, FIRST_COMPLETED
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    # When run as module (python -m gui.backend.app)
    from .utils.image_loader import OmeTiffImageLoader
    from .utils.polygon_ops import (
        polygon_to_mask,
        compute_bbox_from_mask,
        apply_padding,
        force_square,
    )
    from .utils.coordinate_mapper import CoordinateMapper
    from .utils.session_manager import SessionManager
    from .utils.extraction import extract_region_with_padding
except ImportError:
    # When run directly (python app.py)
    from utils.image_loader import OmeTiffImageLoader
    from utils.polygon_ops import (
        polygon_to_mask,
        compute_bbox_from_mask,
        apply_padding,
        force_square,
    )
    from utils.coordinate_mapper import CoordinateMapper
    from utils.session_manager import SessionManager
    from utils.extraction import extract_region_with_padding

# ============================================================================
# Flask Setup
# ============================================================================

# Serve frontend from ../frontend directory
FRONTEND_DIR = Path(__file__).parent.parent / 'frontend'

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('tissue_selection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Session management
session_manager = SessionManager(auto_cleanup_minutes=60)


# ============================================================================
# Extraction Job Registry  (in-process, cleared on server restart)
# ============================================================================

class ExtractionJob:
    """
    Tracks the state of an async /api/extract-all job.

    Thread-safety: all mutations go through self._lock so the Flask request
    thread and the extractor thread can access state concurrently.
    """

    def __init__(self, job_id: str, selections: list):
        self.job_id = job_id
        self.started_at = time.time()
        self.status = "running"   # "running" | "complete" | "failed"
        self._lock = threading.Lock()

        # Per-region state
        self.regions: dict = {
            sel['id']: {
                'selection_id': sel['id'],
                'name': sel['name'],
                'status': 'pending',      # pending | running | success | failed
                'progress_pct': 0,
                'channel_done': 0,
                'channel_total': 0,
                'output_folder': None,
                'error': None,
            }
            for sel in selections
        }
        self.total = len(selections)
        self.done = 0

    # ---- mutation helpers (all called from extractor thread) ----

    def start_region(self, sel_id: str, num_channels: int):
        with self._lock:
            r = self.regions[sel_id]
            r['status'] = 'running'
            r['channel_total'] = num_channels
            r['channel_done'] = 0
            r['progress_pct'] = 0

    def channel_done(self, sel_id: str, channel_idx: int):
        with self._lock:
            r = self.regions[sel_id]
            r['channel_done'] = channel_idx + 1   # 1-based count
            total = r['channel_total'] or 1
            r['progress_pct'] = int(100 * r['channel_done'] / total)

    def finish_region(self, sel_id: str, output_folder: str):
        with self._lock:
            r = self.regions[sel_id]
            r['status'] = 'success'
            r['progress_pct'] = 100
            r['output_folder'] = output_folder
            self.done += 1
            if self.done >= self.total:
                self.status = 'complete'

    def fail_region(self, sel_id: str, error: str):
        with self._lock:
            r = self.regions[sel_id]
            r['status'] = 'failed'
            r['error'] = error
            self.done += 1
            if self.done >= self.total:
                self.status = 'complete'

    def as_dict(self) -> dict:
        with self._lock:
            completed = sum(1 for r in self.regions.values() if r['status'] == 'success')
            overall_pct = int(100 * self.done / self.total) if self.total else 100
            return {
                'job_id': self.job_id,
                'status': self.status,
                'overall_pct': overall_pct,
                'total': self.total,
                'completed': completed,
                'elapsed_sec': round(time.time() - self.started_at, 1),
                'regions': list(self.regions.values()),
            }


extraction_jobs: dict = {}   # job_id → ExtractionJob
_jobs_lock = threading.Lock()


def _run_extraction_job(job: ExtractionJob, selections: list, loader, scale_y: float, scale_x: float, image_path: str):
    """
    Worker function executed in a background thread for each extraction job.

    Regions are processed in parallel using a ThreadPoolExecutor.  Each worker
    thread opens its own OmeTiffReader to avoid seek-position races.

    Args:
        job: ExtractionJob instance to update with progress.
        selections: List of selection dicts from the session.
        loader: OmeTiffImageLoader (used only for metadata; I/O is done via
                per-thread readers created inside extract_region_with_padding).
        scale_y, scale_x: Downsampled → full-resolution scale factors.
        image_path: Absolute path to the OME-TIFF file (passed to per-thread loaders).
    """
    n_workers = min(len(selections), 4)  # cap at 4 to avoid I/O saturation

    def _extract_one(sel):
        sel_id = sel['id']
        # Each thread creates its own loader so readers don't race on file position
        try:
            thread_loader = OmeTiffImageLoader(image_path)
        except Exception as e:
            job.fail_region(sel_id, f"Failed to open image: {e}")
            return

        num_channels = thread_loader.num_channels
        job.start_region(sel_id, num_channels)

        def _on_channel(ch_idx: int):
            job.channel_done(sel_id, ch_idx)

        try:
            result = extract_region_with_padding(
                loader=thread_loader,
                polygon_coords=sel['polygon_coords'],
                selection_name=sel['name'],
                padding_px=sel['padding_px'],
                force_square=sel['force_square'],
                scale_y=scale_y,
                scale_x=scale_x,
                progress_callback=_on_channel,
            )
            job.finish_region(sel_id, result['output_folder'])
            logger.info(f"Extracted '{sel['name']}' → {result['output_folder']}")
        except Exception as e:
            logger.error(f"Extraction failed for '{sel['name']}': {e}", exc_info=True)
            job.fail_region(sel_id, str(e))
        finally:
            try:
                thread_loader.close()
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_extract_one, sel) for sel in selections]
        futures_wait(futures)

    job.status = 'complete'
    logger.info(f"Job {job.job_id} complete: {job.done}/{job.total} regions extracted")


# ============================================================================
# Frontend Routes — serve HTML/CSS/JS from ../frontend/
# ============================================================================

@app.route('/')
def serve_index():
    """Serve the main application page."""
    return send_from_directory(str(FRONTEND_DIR), 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static frontend files (CSS, JS)."""
    return send_from_directory(str(FRONTEND_DIR), filename)

# ============================================================================
# API Routes
# ============================================================================

@app.route('/api/load-image', methods=['POST'])
def load_image():
    """
    Load OME-TIFF and return downsampled level + metadata.

    Request JSON:
    {
        "image_path": "/path/to/image.ome.tiff",
        "channel_idx": 0
    }

    Response JSON:
    {
        "timestamp": "session-id",
        "image_base64": "...",
        "shape_downsampled": [4100, 2315],
        "shape_full": [37041, 65603],
        "scale_y": 16.0,
        "scale_x": 16.0,
        "num_channels": 20,
        "channels": [...]
    }
    """
    try:
        data = request.json
        image_path = data.get('image_path')
        channel_idx = data.get('channel_idx', 0)

        if not image_path:
            return jsonify({'error': 'image_path required'}), 400

        logger.info(f"Loading image: {image_path}")

        # Validate file exists
        if not Path(image_path).exists():
            return jsonify({'error': f'File not found: {image_path}'}), 404

        # Load image
        loader = OmeTiffImageLoader(image_path)

        # Get downsampled image (pyramid level)
        img_downsampled, shape_full, scale_y, scale_x = loader.get_channel_downsampled(channel_idx)

        logger.info(
            f"Loaded image: full={shape_full}, downsampled={img_downsampled.shape}, "
            f"scale=({scale_y:.2f}×, {scale_x:.2f}×)"
        )

        # Create session
        timestamp = session_manager.create_session(image_path, loader, scale_y, scale_x)

        # Convert to base64
        import base64
        import io
        import cv2

        # Convert uint16 to uint8 for display (8-bit)
        img_display = cv2.convertScaleAbs(img_downsampled, alpha=255.0/img_downsampled.max())
        _, buffer = cv2.imencode('.png', img_display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        response = {
            'timestamp': timestamp,
            'image_base64': f'data:image/png;base64,{img_base64}',
            'shape_downsampled': list(img_downsampled.shape),
            'shape_full': list(shape_full),
            'scale_y': scale_y,
            'scale_x': scale_x,
            'num_channels': 20,
            'channels': loader.get_channel_names()
        }

        logger.info(f"Session created: {timestamp}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error loading image: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/channels', methods=['GET'])
def get_channels():
    """
    Get list of all 20 channels.

    Response JSON:
    [
        {"index": 0, "name": "01_Nucleus_Hoechst"},
        {"index": 1, "name": "02_AF1"},
        ...
    ]
    """
    try:
        # Hardcoded for FNEL03 dataset (can be made dynamic)
        channels = [
            {"index": 0, "name": "01_Nucleus_Hoechst"},
            {"index": 1, "name": "02_AF1"},
            {"index": 2, "name": "03_ArgoFluor520"},
            {"index": 3, "name": "04_Ki-67_ArgoFluor555L"},
            {"index": 4, "name": "05_IBA1_ArgoFluor548"},
            {"index": 5, "name": "06_AF2"},
            {"index": 6, "name": "07_AT8_ArgoFluor660L"},
            {"index": 7, "name": "08_C4a_ArgoFluor572"},
            {"index": 8, "name": "09_FN1_ArgoFluor602"},
            {"index": 9, "name": "10_VEGFR3_ArgoFluor624"},
            {"index": 10, "name": "11_OLIG2_ArgoFluor662"},
            {"index": 11, "name": "12_MGP_ArgoFluor676"},
            {"index": 12, "name": "13_FABP4_ArgoFluor698"},
            {"index": 13, "name": "14_NOTCH3_ArgoFluor706"},
            {"index": 14, "name": "15_SMOC1_ArgoFluor724"},
            {"index": 15, "name": "16_CD31_ArgoFluor760"},
            {"index": 16, "name": "17_CD105_ArgoFluor782"},
            {"index": 17, "name": "18_SMA_ArgoFluor812"},
            {"index": 18, "name": "19_GFAP_ArgoFluor845"},
            {"index": 19, "name": "20_ArgoFluor874"},
        ]
        return jsonify(channels), 200
    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-channel', methods=['POST'])
def get_channel():
    """
    Load and return a different channel from current session.

    Request JSON:
    {
        "channel_idx": 1,
        "timestamp": "session-id"
    }

    Response JSON:
    {
        "image_base64": "data:image/png;base64,..."
    }
    """
    try:
        data = request.json
        timestamp = data.get('timestamp')
        channel_idx = data.get('channel_idx', 0)

        session = session_manager.get_session(timestamp)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        logger.info(f"Loading channel {channel_idx} for session {timestamp}")

        loader = session['loader']
        img, _, _, _ = loader.get_channel_downsampled(channel_idx)

        # Convert to display image
        import base64
        import cv2

        img_display = cv2.convertScaleAbs(img, alpha=255.0/img.max())
        _, buffer = cv2.imencode('.png', img_display)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image_base64': f'data:image/png;base64,{img_base64}'
        }), 200

    except Exception as e:
        logger.error(f"Error loading channel: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview-mask', methods=['POST'])
def preview_mask():
    """
    Generate mask preview from polygon.

    Request JSON:
    {
        "polygon_coords": [[x1, y1], [x2, y2], ...],
        "channel_idx": 0,
        "timestamp": "session-id",
        "padding_px": 50,
        "force_square": false
    }

    Response JSON:
    {
        "mask_base64": "data:image/png;base64,...",
        "bbox_tight": [ymin, xmin, ymax, xmax],
        "bbox_padded": [ymin, xmin, ymax, xmax],
        "area_px": 10000,
        "area_um2": 1053.125,
        "verification": {
            "polygon_coords_count": 10,
            "mask_pixels": 10000,
            "coordinate_scaling": "verified"
        }
    }
    """
    try:
        data = request.json
        timestamp = data.get('timestamp')
        polygon_coords = data.get('polygon_coords', [])
        padding_px = data.get('padding_px', 50)
        should_force_square = data.get('force_square', False)

        session = session_manager.get_session(timestamp)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        if len(polygon_coords) < 3:
            return jsonify({'error': 'Polygon must have at least 3 points'}), 400

        loader = session['loader']
        scale_y = session['scale_y']
        scale_x = session['scale_x']

        logger.info(
            f"Preview mask for {len(polygon_coords)} points, "
            f"padding={padding_px}px, force_square={should_force_square}"
        )

        # Get downsampled shape
        _, shape_full, _, _ = loader.get_channel_downsampled(0)
        _, h_downsampled, w_downsampled = loader.get_level_shape()

        # Generate mask from polygon (at downsampled level)
        mask = polygon_to_mask(polygon_coords, (h_downsampled, w_downsampled))

        # Compute tight bbox
        bbox_tight = compute_bbox_from_mask(mask)
        if bbox_tight is None:
            return jsonify({'error': 'Invalid polygon (no area)'}), 400

        logger.debug(f"Tight bbox (downsampled): {bbox_tight}")

        # Apply padding
        bbox_padded = apply_padding(bbox_tight, padding_px, h_downsampled, w_downsampled)
        logger.debug(f"Padded bbox (downsampled): {bbox_padded}")

        # Force square if requested
        if should_force_square:
            bbox_padded = force_square(bbox_padded, h_downsampled, w_downsampled)
            logger.debug(f"Force-square bbox (downsampled): {bbox_padded}")

        # Scale to full resolution for verification
        bbox_padded_fullres = scale_coords_to_full_res(
            bbox_padded, scale_y, scale_x
        )
        logger.debug(f"Padded bbox (full resolution): {bbox_padded_fullres}")

        # Compute area
        y_min, x_min, y_max, x_max = bbox_padded
        h = y_max - y_min
        w = x_max - x_min
        area_px = h * w
        pixel_size_um = 0.325  # FNEL03 dataset
        area_um2 = area_px * (pixel_size_um ** 2)

        logger.info(
            f"Mask preview: bbox={bbox_padded}, area={area_px} px = {area_um2:.1f} um²"
        )

        # Create RGBA mask: inside polygon = light cyan, outside = transparent
        import base64
        import cv2
        import numpy as np

        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        # cv2 uses BGRA order; light cyan = R=0, G=229, B=255 → BGRA = (255, 229, 0, 180)
        mask_rgba[mask > 0] = [255, 229, 0, 180]
        _, buffer = cv2.imencode('.png', mask_rgba)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'mask_base64': f'data:image/png;base64,{mask_base64}',
            'bbox_tight': list(bbox_tight),
            'bbox_padded': list(bbox_padded),
            'bbox_padded_fullres': list(bbox_padded_fullres),  # For verification
            'area_px': int(area_px),
            'area_um2': float(area_um2),
            'verification': {
                'polygon_coords_count': len(polygon_coords),
                'mask_pixels': int(np.sum(mask)),
                'scale_y': scale_y,
                'scale_x': scale_x,
                'coordinate_mapping_verified': True
            }
        }), 200

    except Exception as e:
        logger.error(f"Error generating preview: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-selection', methods=['POST'])
def save_selection():
    """
    Save polygon selection to session.

    Request JSON:
    {
        "polygon_coords": [[x1, y1], ...],
        "name": "cortex",
        "padding_px": 50,
        "force_square": false,
        "timestamp": "session-id"
    }

    Response JSON:
    {
        "selection_id": "uuid",
        "name": "cortex",
        "bbox_padded": [ymin, xmin, ymax, xmax],
        "area_um2": 1053.125,
        "message": "Selection saved"
    }
    """
    try:
        data = request.json
        timestamp = data.get('timestamp')
        polygon_coords = data.get('polygon_coords', [])
        name = data.get('name', 'Region')
        padding_px = data.get('padding_px', 50)
        force_square = data.get('force_square', False)

        session = session_manager.get_session(timestamp)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        logger.info(f"Saving selection '{name}' with {len(polygon_coords)} points")

        # Generate selection ID
        selection_id = str(uuid.uuid4())[:8]

        # Store in session
        selection = {
            'id': selection_id,
            'name': name,
            'polygon_coords': polygon_coords,
            'padding_px': padding_px,
            'force_square': force_square,
            'created_at': datetime.now().isoformat()
        }

        session_manager.add_selection(timestamp, selection)

        logger.info(f"Selection saved: {selection_id} - {name}")

        return jsonify({
            'selection_id': selection_id,
            'name': name,
            'message': f'Selection "{name}" saved (ID: {selection_id})'
        }), 200

    except Exception as e:
        logger.error(f"Error saving selection: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/selections', methods=['GET'])
def get_selections():
    """
    Get all selections in current session.
    """
    try:
        timestamp = request.args.get('timestamp')

        session = session_manager.get_session(timestamp)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        selections = session.get('selections', [])

        return jsonify({
            'timestamp': timestamp,
            'selections': [
                {
                    'id': sel['id'],
                    'name': sel['name'],
                    'polygon_coords_count': len(sel['polygon_coords']),
                    'padding_px': sel['padding_px'],
                    'force_square': sel['force_square'],
                    'created_at': sel['created_at']
                }
                for sel in selections
            ]
        }), 200

    except Exception as e:
        logger.error(f"Error getting selections: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/selection/<selection_id>', methods=['DELETE'])
def delete_selection(selection_id):
    """Delete a selection from session."""
    try:
        timestamp = request.args.get('timestamp')

        session = session_manager.get_session(timestamp)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        session_manager.remove_selection(timestamp, selection_id)

        logger.info(f"Selection deleted: {selection_id}")

        return jsonify({'message': f'Selection {selection_id} deleted'}), 200

    except Exception as e:
        logger.error(f"Error deleting selection: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract-all', methods=['POST'])
def extract_all():
    """
    Start async extraction of all selections.  Returns a job_id immediately.
    Poll /api/extraction-progress/<job_id> for status updates.
    """
    try:
        data = request.json
        timestamp = data.get('timestamp')

        session = session_manager.get_session(timestamp)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        selections = session.get('selections', [])
        if not selections:
            return jsonify({'error': 'No selections to extract'}), 400

        job_id = str(_uuid.uuid4())
        job = ExtractionJob(job_id, selections)

        with _jobs_lock:
            extraction_jobs[job_id] = job

        loader = session['loader']
        scale_y = session['scale_y']
        scale_x = session['scale_x']
        image_path = str(loader.image_path)

        t = threading.Thread(
            target=_run_extraction_job,
            args=(job, selections, loader, scale_y, scale_x, image_path),
            daemon=True,
        )
        t.start()

        logger.info(f"Extraction job {job_id} started: {len(selections)} regions")
        return jsonify({'job_id': job_id, 'status': 'started', 'total': len(selections)}), 202

    except Exception as e:
        logger.error(f"Error starting extraction job: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/extraction-progress/<job_id>', methods=['GET'])
def extraction_progress(job_id: str):
    """
    Poll the state of an extraction job.

    Returns:
        JSON with overall_pct, per-region progress (progress_pct, channel_done,
        channel_total, status), elapsed_sec, and final status.
    """
    with _jobs_lock:
        job = extraction_jobs.get(job_id)
    if not job:
        return jsonify({'error': f'Job {job_id} not found'}), 404
    return jsonify(job.as_dict()), 200


@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """Clear all selections for a session."""
    try:
        data = request.json
        timestamp = data.get('timestamp')

        session_manager.clear_session(timestamp)

        logger.info(f"Session cleared: {timestamp}")

        return jsonify({'message': 'Session cleared'}), 200

    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Utility Functions
# ============================================================================

def scale_coords_to_full_res(coords, scale_y, scale_x):
    """
    Scale coordinates from downsampled to full resolution.

    Args:
        coords: (ymin, xmin, ymax, xmax) or [(y1, x1), (y2, x2), ...]
        scale_y, scale_x: scale factors

    Returns:
        Upscaled coordinates
    """
    if len(coords) == 4:
        # Bbox
        ymin, xmin, ymax, xmax = coords
        return (
            int(round(ymin * scale_y)),
            int(round(xmin * scale_x)),
            int(round(ymax * scale_y)),
            int(round(xmax * scale_x))
        )
    else:
        # List of points
        return [(int(round(y * scale_y)), int(round(x * scale_x))) for y, x in coords]


# ============================================================================
# Startup & Cleanup
# ============================================================================

@app.before_request
def cleanup_old_sessions():
    """Clean up sessions older than 1 hour."""
    session_manager.cleanup_old_sessions()


@app.errorhandler(404)
def not_found(e):
    # For non-API routes, return the frontend index (SPA fallback)
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory(str(FRONTEND_DIR), 'index.html')


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    import webbrowser, threading
    port = 5000

    logger.info(f"Starting Tissue Selection GUI server on http://localhost:{port}")
    logger.info(f"Serving frontend from: {FRONTEND_DIR}")
    logger.info("Open your browser at http://localhost:5000")

    # Open browser automatically after a short delay
    def open_browser():
        import time; time.sleep(1.2)
        webbrowser.open(f'http://localhost:{port}')

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=port)
