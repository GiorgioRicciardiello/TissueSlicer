# TissueSlicer Installation Guide

Cross-platform installation instructions for Windows, macOS, and Linux.

---

## Option 1: Conda (Recommended for reproducibility)

Best for ensuring exact dependency versions across systems.

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

### Installation

**Create environment from YAML file:**
```bash
conda env create -f environment.yml
conda activate tissueslice
```

**Verify installation:**
```bash
python run_gui.py
```

The GUI will open at `http://localhost:5000`

---

## Option 2: pip (Lightweight)

For Python-only environments without conda.

### Prerequisites
- Python 3.11+ ([Download](https://www.python.org/downloads/))

### Installation

**Create virtual environment (recommended):**

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python run_gui.py
```

---

## Option 3: System Package Manager

### macOS (Homebrew)
```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_gui.py
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_gui.py
```

### Linux (Fedora/RHEL)
```bash
sudo dnf install python3.11 python3.11-devel
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_gui.py
```

---

## Dependency Overview

| Package | Purpose | Min Version |
|---------|---------|-------------|
| **flask** | REST API server | 3.0.0 |
| **flask-cors** | Cross-origin requests | 4.0.0 |
| **numpy** | Numerical arrays | 1.24.0 |
| **scipy** | Scientific computing | 1.10.0 |
| **scikit-image** | Image processing | 0.22.0 |
| **opencv-python** | Computer vision | 4.8.0 |
| **tifffile** | OME-TIFF I/O | 2024.2.0 |
| **zarr** | Chunked array storage | 2.16.0 |
| **ome-types** | OME-XML parsing | 0.5.0 |
| **h5py** | HDF5 file I/O | 3.0.0 |
| **hdf5plugin** | HDF5 compression (LZ4, Zstd) | 4.0.0 |
| **Pillow** | Image utilities | 9.0.0 |

---

## Troubleshooting

### "Module not found" errors
- **pip install:** Ensure virtual environment is activated
- **conda install:** Verify environment name with `conda env list`
- **Missing HDF5:** On Linux, install system library:
  ```bash
  sudo apt-get install libhdf5-dev  # Ubuntu/Debian
  sudo dnf install hdf5-devel       # Fedora/RHEL
  ```

### Port 5000 already in use
Edit `gui/backend/app.py`, line ~804:
```python
app.run(debug=False, port=5001)  # Change to unused port
```

### OpenCV issues (Windows)
If `cv2` fails to import:
```bash
pip install --upgrade opencv-python
```

### HDF5 compression not available
Install optional compression plugins:
```bash
pip install hdf5plugin
```

---

## Running the Application

### Start the server:
```bash
python run_gui.py
```

The application will:
1. Check all dependencies
2. Start Flask server on `http://localhost:5000`
3. Automatically open browser

### Frontend API
- **Host:** `http://localhost:5000`
- **Default channel:** Hoechst (nuclear stain)

---

## Development Setup

### Clone and install in editable mode:
```bash
git clone <repository>
cd TissueSlicer
conda env create -f environment.yml
conda activate tissueslice
pip install -e .
```

### Run tests (if available):
```bash
pytest
```

---

## Platform-Specific Notes

### Windows
- Use `.venv\Scripts\activate` (not `source`)
- May require Visual C++ build tools for some packages
- Paths use backslashes in error messages (normal)

### macOS
- M1/M2 Macs: Use `conda-forge` channel (default in `environment.yml`)
- May need Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Linux
- Most distributions need HDF5 system library
- Check GPU support for OpenCV if using CUDA

---

## Version Management

### Pin exact versions (production):
```bash
pip freeze > requirements-lock.txt
pip install -r requirements-lock.txt
```

### Update dependencies:
```bash
conda update --all  # Conda
pip list --outdated  # Check for updates
```

---

## Docker (Optional)

If Docker is available:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_gui.py"]
```

Build and run:
```bash
docker build -t tissueslice .
docker run -p 5000:5000 tissueslice
```
