"""
Tissue Selection GUI — single entry point.

Usage:
    python run_gui.py

Opens http://localhost:5000 in your browser automatically.
No separate frontend server needed.
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

def check_deps():
    missing = []
    for pkg, imp in [
        ('flask',       'flask'),
        ('flask_cors',  'flask_cors'),
        ('cv2',         'cv2'),
        ('numpy',       'numpy'),
        ('scipy',       'scipy'),
        ('skimage',     'skimage'),
        ('h5py',        'h5py'),
        ('tifffile',    'tifffile'),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print(f"Install with:\n  pip install {' '.join(missing)}")
        sys.exit(1)

    print("[OK] All dependencies found")

if __name__ == '__main__':
    print("=" * 60)
    print("  Tissue Selection GUI")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Root:   {ROOT}\n")

    check_deps()

    backend = ROOT / 'gui' / 'backend' / 'app.py'
    print(f"\nStarting Flask server...")
    print(f"Open: http://localhost:5000\n")

    # Run app.py directly so it can use absolute imports
    sys.path.insert(0, str(ROOT / 'gui' / 'backend'))
    exec(open(backend).read(), {'__name__': '__main__', '__file__': str(backend)})

    # input image path
    # C:\Users\riccig01\Documents\vascbrain\OrionImages\data\raw\FNEL03_CAD001_001\FNEL03_2026_V1_001703\FNEL03_CAD001_001_FNEL03_2026_V1_001703.ome.tiff